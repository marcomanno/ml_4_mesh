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

struct OptimizeNonLinear
{
  OptimizeNonLinear(const VertexIndMpap& _vrt_inds, Topo::Wrap<Topo::Type::BODY> _body):
    vrt_inds_(_vrt_inds), body_(_body) { }

  void compute(Eigen::VectorXd& _X)
  {
    Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(body_);
    auto q_solver = IQuadraticSolver::make();
    q_solver->init(_X.size(), bf.size(), _X.data());
    q_solver->compute(*this);
  }

  bool operator()(double* _x, double* _f, double* _fj)
  {
    return true;
  }

  const VertexIndMpap& vrt_inds_;
  Topo::Wrap<Topo::Type::BODY> body_;
};

// Function to minimize 



void flatten(Topo::Wrap<Topo::Type::BODY> _body)
{
  std::map<Topo::Wrap<Topo::Type::VERTEX>, size_t> vrt_inds;
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(_body);
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
    Geo::VectorD2 _loc_tri[2];
    auto area_time_2_sqr = sqrt(move_to_local_coord(pts, _loc_tri));
    w[0] = (_loc_tri[1] - _loc_tri[0]) / area_time_2_sqr;
    w[1] =  -_loc_tri[1] / area_time_2_sqr;
    w[2] = _loc_tri[0] / area_time_2_sqr;

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
  X.resize(cols);
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
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa1.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb1.obj", body);
}

TEST_CASE("flat_01", "[Flattening]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa2.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb2.obj", body);
}
