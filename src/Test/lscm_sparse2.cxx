#include "catch.hpp"

#include <unsupported/Eigen/LevenbergMarquardt>

#include "optimize_function.hxx"

#include "Geo/vector.hh"
#include "Import/load_obj.hh"
#include "Import/save_obj.hh"
#include "Topology/iterator.hh"

#include "splm/lm.hxx"

namespace
{

static auto move_to_local_coord(const Geo::VectorD3 _tri[3], Geo::VectorD2 _loc_tri[2])
{
  auto v_01 = _tri[1] - _tri[0];
  _loc_tri[0][0] = Geo::length(v_01);
  _loc_tri[0][1] = 0;
  auto v_02 = _tri[2] - _tri[0];
  _loc_tri[1][0] = v_02 * v_01 / _loc_tri[0][0];
  auto area_time_2 = Geo::length(v_02 % v_01);
  _loc_tri[1][1] = area_time_2 / _loc_tri[0][0];
  return area_time_2;
}

static double area(
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf,
  const std::function<Geo::Point(Topo::Wrap<Topo::Type::VERTEX>&)>& _v_pt)
{
  double a = 0;
  for (auto face : _bf)
  {
    // 3 equations per face.
    Topo::Iterator<Topo::Type::FACE, Topo::Type::VERTEX> fv(face);
    Geo::Point pt[3];
    for (int i = 0; i < 3; ++i)
      pt[i] = _v_pt(fv.get(i));
    a += Geo::length((pt[1] - pt[0]) % (pt[2] - pt[0]));
  }
  return a / 2;
}

using VertexIndMpap = std::map<Topo::Wrap<Topo::Type::VERTEX>, size_t>;

struct EnergyFunction : LM::IMultiFunction
{
  struct DataOfFace
  {
    Geo::VectorD2 loc_tri_[2];
    size_t idx_[3];
    double area_sqrt_;
  };

  EnergyFunction(const VertexIndMpap& _vrt_inds,
    Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf, int _m, int _n) :
    n_(_n), m_(_m), vrt_inds_(_vrt_inds)
  {
    for (auto face : _bf)
    {
      auto& fd = data_of_faces_.emplace_back();
      // 3 equations per face.
      Topo::Iterator<Topo::Type::FACE, Topo::Type::VERTEX> fv(face);
      Geo::VectorD3 pts[3];
      size_t i = 0;
      for (auto v : fv)
      {
        v->geom(pts[i]);
        fd.idx_[i++] = 2 * vrt_inds_.find(v)->second;
      }
      fd.area_sqrt_ = sqrt(move_to_local_coord(pts, fd.loc_tri_) * 0.5);
    }
  }

  bool evaluate(const Eigen::VectorXd& _x, Eigen::VectorXd& _f) const override
  {
    return compute(_x, &_f, nullptr);
  }
  bool jacobian(const Eigen::VectorXd& _x, Eigen::SparseMatrix<double>& _fj) const override
  {
    return compute(_x, nullptr, &_fj);
  }
  MKL_INT rows() const override
  {
    return m_;
  }
  MKL_INT cols() const override
  {
    return n_;
  }


protected:
  const VertexIndMpap& vrt_inds_;
  std::vector<DataOfFace> data_of_faces_;
  int m_, n_;

  int compute(const Eigen::VectorXd& _x,
    Eigen::VectorXd* _fvec, Eigen::SparseMatrix<double>* _fjac) const;
};

int EnergyFunction::compute(const Eigen::VectorXd& _x,
  Eigen::VectorXd* _fvec, Eigen::SparseMatrix<double>* _fjac) const
{
  std::vector<Eigen::Triplet<double>> triplets;

  auto X = [this, &_x](int _i)
  {
    if (_i < n_)
      return _x[_i];
    else
      return 0.;
  };

  size_t i_eq_loop = 0;
  for (auto& fd : data_of_faces_)
  {
    // 3 equations per face.
    size_t i_eq = i_eq_loop;
    i_eq_loop += 3;
    auto u10 = X(fd.idx_[1]) - X(fd.idx_[0]);
    auto u20 = X(fd.idx_[2]) - X(fd.idx_[0]);
    auto v10 = X(fd.idx_[1] + 1) - X(fd.idx_[0] + 1);
    auto v20 = X(fd.idx_[2] + 1) - X(fd.idx_[0] + 1);
    double a = u10 / fd.loc_tri_[0][0];
    double b = (u20 - fd.loc_tri_[1][0] * u10 / fd.loc_tri_[0][0]) / fd.loc_tri_[1][1];
    double c = v10 / fd.loc_tri_[0][0];
    double d = (v20 - fd.loc_tri_[1][0] * v10 / fd.loc_tri_[0][0]) / fd.loc_tri_[1][1];
    auto det = a * d - b * c;
    if (_fvec != nullptr)
    {
      (*_fvec)(i_eq) = fd.area_sqrt_ * (a - d);
      (*_fvec)(i_eq + 1) = fd.area_sqrt_ * (b + c);
      (*_fvec)(i_eq + 2) = fd.area_sqrt_ * (std::log(det));
    }
    if (_fjac == nullptr)
      continue;
    Eigen::Matrix<double, 3, 4> dfa;
    dfa << 1, 0, 0, -1,
      0, 1, 1, 0,
      d / det, -c / det, -b / det, a / det;

    Eigen::Matrix<double, 4, 6> da_uv;
    auto c0 = 1. / fd.loc_tri_[0][0]; // 1 / x_1
    auto c1 = 1. / fd.loc_tri_[1][1]; // 1 / y_2
    auto c2 = fd.loc_tri_[1][0] / (fd.loc_tri_[0][0] * fd.loc_tri_[1][1]); // x_2 / (x_1 * y_2)
    da_uv <<
      -c0, 0, c0, 0, 0, 0,
      c2 - c1, 0, -c2, 0, c1, 0,
      0, -c0, 0, c0, 0, 0,
      0, c2 - c1, 0, -c2, 0, c1;
    Eigen::Matrix<double, 3, 6> df_uv = fd.area_sqrt_ * dfa * da_uv;
    for (size_t i = 0; i < 3; ++i)
    {
      for (size_t j = 0; j < 6; ++j)
      {
        auto val = df_uv(i, j);
        if (val != 0)
        {
          auto c = fd.idx_[j / 2] + j % 2;
          if (c < n_)
            triplets.emplace_back(i_eq + i, c, val);
        }
      }
    }
  }
  if (_fjac != nullptr)
    _fjac->setFromTriplets(triplets.begin(), triplets.end());

  return 0;
}

static void flatten(Topo::Wrap<Topo::Type::BODY> _body, bool _consformal)
{
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(_body);
  auto vertex_point = [](const Topo::Wrap<Topo::Type::VERTEX>& _v)
  {
    Geo::Point pt;
    _v->geom(pt);
    return pt;
  };

  auto a0 = area(bf, vertex_point);

  std::map<Topo::Wrap<Topo::Type::VERTEX>, size_t> vrt_inds;
  using Triplet = Eigen::Triplet<double, size_t>;
  using Matrix = Eigen::SparseMatrix<double>;
  std::vector<Triplet> coffs;
  size_t rows = 0, pt_nmbr = 0;
  for (auto face : bf)
  {
    std::array<Geo::VectorD2, 3> w;
    Topo::Iterator<Topo::Type::FACE, Topo::Type::VERTEX> fv(face);
    Geo::VectorD3 pts[3];
    size_t idx[3];
    size_t i = 0;

    for (auto v : fv)
    {
      v->geom(pts[i]);
      auto pos = vrt_inds.emplace(v, pt_nmbr);
      idx[i] = pos.first->second;
      if (pos.second)
        ++pt_nmbr;
      ++i;
    }
    Geo::VectorD2 loc_tri[2];
    auto area_time_2_sqrt = sqrt(move_to_local_coord(pts, loc_tri));
    w[0] = (loc_tri[1] - loc_tri[0]) / area_time_2_sqrt;
    w[1] = -loc_tri[1] / area_time_2_sqrt;
    w[2] = loc_tri[0] / area_time_2_sqrt;

    for (size_t j = 0; j < 3; ++j)
    {
      auto idx_base = idx[j] * 2;
      coffs.emplace_back(rows, idx_base, w[j][0]);
      coffs.emplace_back(rows, idx_base + 1, -w[j][1]);
      coffs.emplace_back(rows + 1, idx_base, w[j][1]);
      coffs.emplace_back(rows + 1, idx_base + 1, w[j][0]);
    }
    rows += 2;
  }
  const auto cols = 2 * pt_nmbr;
  Matrix M(rows, cols);
  M.setFromTriplets(coffs.begin(), coffs.end());
  const auto FIXED_VAR = 4;
  auto split_idx = cols - FIXED_VAR;
  auto A = M.block(0, 0, rows, split_idx);
  auto B = M.block(0, split_idx, rows, FIXED_VAR);
  Eigen::Vector4d fixed;
  fixed(0, 0) = -100;
  fixed(1) = fixed(2) = fixed(3) = 0;

  auto b = B * fixed;
  //Eigen::SparseQR <Matrix, Eigen::COLAMDOrdering<int>> solver;
  Eigen::LeastSquaresConjugateGradient<Matrix> solver;
  solver.compute(A);
  Eigen::VectorXd X = solver.solve(b);
  X.conservativeResize(cols);
  for (size_t i = 0; i < 4; ++i)
    X(split_idx + i, 0) = -fixed(i, 0);

  auto set_x = [&](const double* _x)
  {
    for (const auto& v_id : vrt_inds)
    {
      auto base_ind = 2 * v_id.second;
      Geo::VectorD3 pt;
      pt = { _x[base_ind], _x[base_ind + 1], 0 };
      const_cast<Topo::Wrap<Topo::Type::VERTEX>&>(v_id.first)->set_geom(pt);
    }
  };
  auto vertex_flat_point = [&X, &vrt_inds](const Topo::Wrap<Topo::Type::VERTEX>& _v)
  {
    auto ind = vrt_inds[_v];
    return Geo::Point{ X(2 * ind), X(2 * ind + 1), 0 };
  };
  auto a1 = area(bf, vertex_flat_point);
  X *= sqrt(a0 / a1);

  if (!_consformal)
  {
    auto unk_nmbr = cols - 3;
    auto eq_nmbr = 3 * bf.size();
    X.conservativeResize(unk_nmbr);
    EnergyFunction ef(vrt_inds, bf, eq_nmbr, unk_nmbr);
    auto solver = LM::ISparseLM::make(ef);
    solver->compute(X, 300);

    X.conservativeResize(cols);
    X(cols - 1) = X(cols - 2) = X(cols - 3) = 0;

    auto a1 = area(bf, vertex_flat_point);
    X *= sqrt(a0 / a1);
  }
  set_x(X.data());
}

} // namespace

TEST_CASE("flat_sp2_00", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa0.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb0.obj", body);
}

TEST_CASE("flat_sp2_00_conf", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa0.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb0_conf.obj", body);
}

TEST_CASE("flat_sp2_01", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa1.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb1.obj", body);
}

TEST_CASE("flat_sp2_01_conf", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa1.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb1_conf.obj", body);
}

TEST_CASE("flat_sp2_02", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa2.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb2.obj", body);
}

TEST_CASE("flat_sp2_02_conf", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa2.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb2_conf.obj", body);
}

TEST_CASE("flat_sp2_03", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa3.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb3.obj", body);
}

TEST_CASE("flat_sp2_03_conf", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa3.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb3_conf.obj", body);
}

TEST_CASE("flat_sp2_04", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa4.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb4.obj", body);
}

TEST_CASE("flat_sp2_04_conf", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa4.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb4_conf.obj", body);
}

TEST_CASE("flat_sp2_05", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa5.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb5.obj", body);
}

TEST_CASE("flat_sp2_05_conf", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa5.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb5_conf.obj", body);
}

TEST_CASE("flat_sp2_06", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa6.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb6.obj", body);
}

TEST_CASE("flat_sp2_06_conf", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa6.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb6_conf.obj", body);
}

TEST_CASE("flat_sp2_07", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa7.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb7.obj", body);
}

TEST_CASE("flat_sp2_07_conf", "[FlatteningSP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa7.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb7_conf.obj", body);
}
