#pragma once

#include <Geo/vector.hh>
#include <Eigen/dense>

namespace MeshOp {

/// Find the best rotottranslation that moves points in _pqs[i][0]
/// to points in _pqs[i][1].
void find_optimal_rotation(
  const std::vector<std::array<Geo::VectorD2, 2>>& _pqs,
  Eigen::Matrix2d& _R, Eigen::Vector2d& _T);

} // namespace MeshOp
