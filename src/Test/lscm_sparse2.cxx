#include "catch.hpp"

#include "Geo/vector.hh"
#include "Import/load_obj.hh"
#include "Import/save_obj.hh"
#include "Topology/iterator.hh"

#include <Eigen/PardisoSupport>

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
    MKL_INT idx_[3];
    double area_sqrt_;
    double c0_, c1_, c2_;
    void compute_coeff(double _area_sqrt)
    {
      area_sqrt_ = _area_sqrt;
      c0_ = _area_sqrt / loc_tri_[0][0]; // 1 / x_1
      c1_ = _area_sqrt / loc_tri_[1][1]; // 1 / y_2
      c2_ = _area_sqrt * loc_tri_[1][0] / (loc_tri_[0][0] * loc_tri_[1][1]); // x_2 / (x_1 * y_2)
    }
    void fill_matrix(Eigen::Matrix<double, 4, 6>& _da_uv) const
    {
      _da_uv <<
             -c0_,         0,  c0_,    0,   0,   0,
        c2_ - c1_,         0, -c2_,    0, c1_,   0,
                0,      -c0_,    0,  c0_,   0,   0,
                0, c2_ - c1_,    0, -c2_,   0, c1_;

    }
  };

  EnergyFunction(Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf):
    tri_nmbr_(_bf.size()), n_(0)
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
        auto pos_it = vrt_inds_.emplace(v, n_);
        fd.idx_[i++] = 2 * pos_it.first->second;
        if (pos_it.second)
          ++n_;
      }
      fd.compute_coeff(sqrt(move_to_local_coord(pts, fd.loc_tri_) * 0.5));
    }
    m2_ = 2 * tri_nmbr_;
    m3_ = 3 * tri_nmbr_;
    n3_ = 2 * n_ - 3;
  }

  bool evaluate(const LM::ColumnVector& _x, LM::ColumnVector& _f) const override
  {
    return compute(_x, &_f, nullptr);
  }
  bool jacobian(const LM::ColumnVector& _x, LM::Matrix& _fj) const override
  {
    return compute(_x, nullptr, &_fj);
  }
  void jacobian_conformal(LM::Matrix& _fj) const;

  MKL_INT rows() const override
  {
    return m3_;
  }
  MKL_INT cols() const override
  {
    return n3_;
  }

  const VertexIndMpap& veterx_map() const
  {
    return vrt_inds_;
  }

protected:
  std::vector<DataOfFace> data_of_faces_;
  MKL_INT tri_nmbr_, n_;
  MKL_INT m2_, m3_;
  MKL_INT n3_;
  VertexIndMpap vrt_inds_;

  int compute(const LM::ColumnVector& _x,
    LM::ColumnVector* _fvec, LM::Matrix* _fjac) const;
};

int EnergyFunction::compute(const Eigen::VectorXd& _x,
  LM::ColumnVector* _fvec, LM::Matrix* _fjac) const
{
  std::vector<Eigen::Triplet<double>> triplets;
  Eigen::Matrix<double, 4, 6> da_uv;
  MKL_INT i_eq_loop = 0;
  const MKL_INT rows_per_face = 3;
  Eigen::Matrix<double, rows_per_face, 4> dfa;
  for (auto& fd : data_of_faces_)
  {
    // 3 equations per face.
    MKL_INT i_eq = i_eq_loop;
    i_eq_loop += rows_per_face;
    auto u10 = _x(fd.idx_[1]) - _x(fd.idx_[0]);
    auto u20 = _x(fd.idx_[2]) - _x(fd.idx_[0]);
    auto v10 = _x(fd.idx_[1] + 1) - _x(fd.idx_[0] + 1);
    auto v20 = _x(fd.idx_[2] + 1) - _x(fd.idx_[0] + 1);
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
    dfa << 1, 0, 0, -1,
           0, 1, 1,  0,
           d / det, -c / det, -b / det, a / det;

    auto c0 = 1. / fd.loc_tri_[0][0]; // 1 / x_1
    auto c1 = 1. / fd.loc_tri_[1][1]; // 1 / y_2
    auto c2 = fd.loc_tri_[1][0] / (fd.loc_tri_[0][0] * fd.loc_tri_[1][1]); // x_2 / (x_1 * y_2)
    fd.fill_matrix(da_uv);
    Eigen::Matrix<double, rows_per_face, 6> df_uv = dfa * da_uv;
    for (MKL_INT i = 0; i < rows_per_face; ++i)
    {
      for (MKL_INT j = 0; j < 6; ++j)
      {
        auto val = df_uv(i, j);
        if (val != 0)
        {
          auto c = fd.idx_[j / 2] + j % 2;
          if (c < _fjac->cols())
            triplets.emplace_back(
              static_cast<int>(i_eq + i),
              static_cast<int>(c), val);
        }
      }
    }
  }
  if (_fjac != nullptr)
    _fjac->setFromTriplets(triplets.begin(), triplets.end());

  return 0;
}

void EnergyFunction::jacobian_conformal(LM::Matrix& _fj) const
{
  _fj.resize(m2_, 2 * n_);
  std::vector<Eigen::Triplet<double>> triplets;
  MKL_INT i_eq_loop = 0;
  Eigen::Matrix<double, 2, 4> dfa;
  dfa << 1, 0, 0, -1, 0, 1, 1, 0;
  Eigen::Matrix<double, 4, 6> da_uv;
  for (auto& fd : data_of_faces_)
  {
    // 3 equations per face.
    MKL_INT i_eq = i_eq_loop;
    i_eq_loop += 2;
    fd.fill_matrix(da_uv);
    Eigen::Matrix<double, 2, 6> df_uv = dfa * da_uv;
    for (MKL_INT i = 0; i < 2; ++i)
    {
      for (MKL_INT j = 0; j < 6; ++j)
      {
        auto val = df_uv(i, j);
        if (val != 0)
        {
          auto c = fd.idx_[j / 2] + j % 2;
          triplets.emplace_back(
            static_cast<int>(i_eq + i),
            static_cast<int>(c), val);
        }
      }
    }
  }
  _fj.setFromTriplets(triplets.begin(), triplets.end());
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

  EnergyFunction ef(bf);
  LM::Matrix M;
  ef.jacobian_conformal(M);

  const auto FIXED_VAR = 4;
  auto split_idx = M.cols() - FIXED_VAR;
  auto A = M.block(0, 0, M.rows(), split_idx);
  auto B = M.block(0, split_idx, M.rows(), FIXED_VAR);
  Eigen::Vector4d fixed;
  fixed(0) = -100;
  fixed(1) = fixed(2) = fixed(3) = 0;

  auto b = B * fixed;
#if 0
  //Eigen::SparseQR <Matrix, Eigen::COLAMDOrdering<int>> solver;
  Eigen::LeastSquaresConjugateGradient<Matrix> solver;
  solver.compute(A);
  Eigen::VectorXd X = solver.solve(b);
#else
  Eigen::PardisoLDLT<LM::Matrix> lsolver;
  lsolver.compute(A.transpose() * A);
  Eigen::VectorXd rhs = A.transpose() * b;
  Eigen::VectorXd X = lsolver.solve(rhs);
#endif
  X.conservativeResize(M.cols());
  for (size_t i = 0; i < 4; ++i)
    X(split_idx + i, 0) = -fixed(i, 0);

  auto& vrt_inds = ef.veterx_map();

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
    auto it = vrt_inds.find(_v);
    if (it == vrt_inds.end())
      throw "Vertex veraible not found";
    auto ind = vrt_inds.find(_v)->second;
    return Geo::Point{ X(2 * ind), X(2 * ind + 1), 0 };
  };
  auto a1 = area(bf, vertex_flat_point);
  X *= sqrt(a0 / a1);

  if (!_consformal)
  {
    auto solver = LM::ISparseLM::make(ef);
    solver->compute(X, 300);
    auto a1 = area(bf, vertex_flat_point);
    X *= sqrt(a0 / a1);
  }
  set_x(X.data());
}

} // namespace

TEST_CASE("flat_sp2_00", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa0.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb0.obj", body);
}

TEST_CASE("flat_sp2_00_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa0.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb0_conf.obj", body);
}

TEST_CASE("flat_sp2_01", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa1.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb1.obj", body);
}

TEST_CASE("flat_sp2_01_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa1.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb1_conf.obj", body);
}

TEST_CASE("flat_sp2_02", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa2.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb2.obj", body);
}

TEST_CASE("flat_sp2_02_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa2.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb2_conf.obj", body);
}

TEST_CASE("flat_sp2_03", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa3.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb3.obj", body);
}

TEST_CASE("flat_sp2_03_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa3.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb3_conf.obj", body);
}

TEST_CASE("flat_sp2_04", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa4.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb4.obj", body);
}

TEST_CASE("flat_sp2_04_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa4.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb4_conf.obj", body);
}

TEST_CASE("flat_sp2_05", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa5.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb5.obj", body);
}

TEST_CASE("flat_sp2_05_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa5.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb5_conf.obj", body);
}

TEST_CASE("flat_sp2_06", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa6.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb6.obj", body);
}

TEST_CASE("flat_sp2_06_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa6.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb6_conf.obj", body);
}

TEST_CASE("flat_sp2_07", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa7.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb7.obj", body);
}

TEST_CASE("flat_sp2_07_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa7.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb7_conf.obj", body);
}

TEST_CASE("flat_sp2_08", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa8.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb8.obj", body);
}

TEST_CASE("flat_sp2_08_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa8.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb8_conf.obj", body);
}

TEST_CASE("flat_sp2_09", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa9.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb9.obj", body);
}

TEST_CASE("flat_sp2_09_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa9.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb9_conf.obj", body);
}

TEST_CASE("flat_sp2_a00", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaaa00.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbbb00.obj", body);
}

TEST_CASE("flat_sp2_a00_conf", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaaa00.obj");
  flatten(body, true);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbbb00_conf.obj", body);
}

TEST_CASE("flat_sp2_a01", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaaa01.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbbb01.obj", body);
}

TEST_CASE("flat_sp2_a02", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaaa02.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbbb02.obj", body);
}

TEST_CASE("flat_sp2_a03", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaaa03.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbbb03.obj", body);
}

TEST_CASE("flat_sp2_a04", "[Flattening2SP]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaaa04.obj");
  flatten(body, false);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbbb04.obj", body);
}
