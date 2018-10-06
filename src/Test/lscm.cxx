//#pragma optimize("", off)
#include "catch.hpp"

#include "Geo/vector.hh"
#include "Import/load_obj.hh"
#include "Import/save_obj.hh"
#include "Topology/iterator.hh"

#include <Eigen/Sparse>
#include <Eigen/SparseQR>


auto move_to_local_coord(const Geo::VectorD3 _tri[3], Geo::VectorD2 _loc_tri[2])
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

void flatten(Topo::Wrap<Topo::Type::BODY> _in_body)
{
  std::map<Topo::Wrap<Topo::Type::VERTEX>, size_t> vrt_inds;
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(_in_body);
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
  Eigen::SparseQR <Matrix, Eigen::COLAMDOrdering<int>> solver;
  solver.compute(A);
  Eigen::VectorXd X = solver.solve(b);
  std::cout << X.rows() << " " << X.cols() << std::endl;
  for (const auto& v_id : vrt_inds)
  {
    auto base_ind = 2 * v_id.second;
    Geo::VectorD3 pt;
    if (base_ind < split_idx)
      pt = { X(base_ind, 0), X(base_ind + 1, 0), 0 };
    else
    {
      auto idx = base_ind - split_idx;
      pt = { -fixed(idx, 0), -fixed(idx + 1, 0), 0 };
    }
    const_cast<Topo::Wrap<Topo::Type::VERTEX>&>(v_id.first)->set_geom(pt);
  }
}

TEST_CASE("flat_00", "[Flattening]")
{
  auto body = IO::load_obj("C:\\Users\\USER\\source\\repos\\ml_4_mesh\\src\\Test\\Data\\aaa1.obj");
  flatten(body);
  IO::save_obj("C:\\Users\\USER\\source\\repos\\ml_4_mesh\\src\\Test\\Data\\bbb1.obj", body);
}
