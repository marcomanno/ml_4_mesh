//#pragma optimize("", off)
#include "catch.hpp"

#include "optimize_function.hxx"

#include "Geo/vector.hh"
#include "Import/load_obj.hh"
#include "Import/save_obj.hh"
#include "Topology/iterator.hh"

#include <Eigen/Sparse>
#include <Eigen/SparseQR>


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

using VertexIndMpap = std::map<Topo::Wrap<Topo::Type::VERTEX>, size_t>;

struct OptimizeNonLinear : public IFunction
{
  OptimizeNonLinear(const VertexIndMpap& _vrt_inds, Topo::Wrap<Topo::Type::BODY> _body):
    vrt_inds_(_vrt_inds), bf_(_body) { }

  void compute(Eigen::VectorXd& _X)
  {
    auto q_solver = IQuadraticSolver::make();
    rows_ = 3 * bf_.size();
    cols_ = _X.size();
    q_solver->init(rows_, cols_, _X.data());
    q_solver->compute(*this);
  }

  bool operator()(const double* _x, double* _f, double* _fj) const override
  {
    const size_t var_nmbr = vrt_inds_.size() * 2;
    auto INDEX = [var_nmbr](size_t _i, size_t _j) { return _i * var_nmbr + _j; };
    size_t i_eq_loop = 0;
    if (_fj != nullptr)
      std::fill_n(_fj, rows_ * cols_, 0.);
    for (auto face : bf_)
    {
      // 3 equations per face.
      size_t i_eq = i_eq_loop;
      i_eq_loop += 3;
      Topo::Iterator<Topo::Type::FACE, Topo::Type::VERTEX> fv(face);
      size_t idx[3];
      Geo::VectorD3 pts[3];
      size_t i = 0;
      for (auto v : fv)
      {
        v->geom(pts[i]);
        idx[i++] = 2 * vrt_inds_.find(v)->second;
      }
      Geo::VectorD2 loc_tri[2];
      auto area_time_2 = move_to_local_coord(pts, loc_tri);
      auto u10 = _x[idx[1]] - _x[idx[0]];
      auto u20 = _x[idx[2]] - _x[idx[0]];
      auto v10 = _x[idx[1] + 1] - _x[idx[0] + 1];
      auto v20 = _x[idx[2] + 1] - _x[idx[0] + 1];
      double a = u10 / loc_tri[0][0];
      double b = (u20 - loc_tri[1][0] * u20 / loc_tri[0][0]) / loc_tri[1][1];
      double c = v10 / loc_tri[0][0];
      double d = (v20 - loc_tri[1][0] * v20 / loc_tri[0][0]) / loc_tri[1][1];
      _f[i_eq] = a + d;
      _f[i_eq + 1] = b - c;
      _f[i_eq + 2] = std::log(a * d - b * c);
      if (_fj == nullptr)
        continue;
      auto det = a * d - b * c;
      Eigen::Matrix<double, 3, 4> dfa;
      dfa << 1,          0,        0,       1,
             0,          1,       -1,       0,
             d / det, -c / det, -b / det, a / det;

      Eigen::Matrix<double, 4, 6> da_uv;
      double coe0 = 1. / loc_tri[0][0];
      double coe1 = (loc_tri[1][0] / loc_tri[0][0] - 1) / loc_tri[1][1];
      da_uv << 
        -coe0,     0, coe0,    0,     0,     0,
         coe1,     0,    0,    0, -coe1,     0,
            0, -coe0,    0, coe0,     0,     0,
            0,  coe1,    0,    0,     0, -coe1;
      Eigen::Matrix<double, 3, 6> df_uv = dfa * da_uv;
      for (size_t i = 0; i < 3; ++i)
      {
        for (size_t j = 0; j < 6; ++j)
          _fj[INDEX(i_eq + i, idx[j / 2] + j % 2)] = df_uv(i, j);
      }
    }
    return true;
  }

  const VertexIndMpap& vrt_inds_;
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf_;
  size_t rows_, cols_;
};

// Function to minimize 



void flatten(Topo::Wrap<Topo::Type::BODY> _body)
{
  std::map<Topo::Wrap<Topo::Type::VERTEX>, size_t> vrt_inds;
  using Triplet = Eigen::Triplet<double, size_t>;
  using Matrix = Eigen::SparseMatrix<double>;
  std::vector<Triplet> coffs;
  size_t rows = 0, pt_nmbr = 0;
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(_body);
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
    auto area_time_2_sqr = sqrt(move_to_local_coord(pts, loc_tri));
    w[0] = (loc_tri[1] - loc_tri[0]) / area_time_2_sqr;
    w[1] =  -loc_tri[1] / area_time_2_sqr;
    w[2] = loc_tri[0] / area_time_2_sqr;

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
  fixed(0, 0) = fixed(1, 0) = fixed(2, 0) = 0;
  fixed(3, 0) = -1;
  auto b = B * fixed;
  //Eigen::SparseQR <Matrix, Eigen::COLAMDOrdering<int>> solver;
  Eigen::LeastSquaresConjugateGradient<Matrix> solver;
  solver.compute(A);
  Eigen::VectorXd X = solver.solve(b);
  std::cout << X.rows() << " " << X.cols() << std::endl;
  X.conservativeResize(cols);
  for (size_t i = 0; i < 4; ++i)
    X(split_idx + i, 0) = fixed(i, 0);

  OptimizeNonLinear onl(vrt_inds, _body);
  onl.compute(X);

  for (const auto& v_id : vrt_inds)
  {
    auto base_ind = 2 * v_id.second;
    Geo::VectorD3 pt;
    pt = { X(base_ind, 0), X(base_ind + 1, 0), 0 };
    const_cast<Topo::Wrap<Topo::Type::VERTEX>&>(v_id.first)->set_geom(pt);
  }



}

TEST_CASE("flat_00", "[Flattening]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa0.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb0.obj", body);
}

TEST_CASE("flat_01", "[Flattening]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa1.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb1.obj", body);
}

TEST_CASE("flat_02", "[Flattening]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa2.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb2.obj", body);
}
