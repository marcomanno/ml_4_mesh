
#include "optimal_rotation.hxx"

namespace MeshOp {

void find_optimal_rotation(
  const std::vector<std::array<Geo::VectorD2, 2>>& _pqs,
  Eigen::Matrix2d& _R, Eigen::Vector2d& _T)
{
  const auto N = _pqs.size();
  std::array<Geo::VectorD2, 2> p_m = { Geo::VectorD2{0, 0}, Geo::VectorD2{0, 0} };
  for (const auto& pq : _pqs)
  {
    p_m[0] += pq[0];
    p_m[1] += pq[1];
  }
  for (auto& v : p_m)
    v /= static_cast<double>(N);

  using Matrix = Eigen::MatrixXd;
  Matrix X(2, N), Y(2, N);
  for (size_t col = 0; col < N; ++col)
  {
    auto& p = _pqs[col][0];
    auto& q = _pqs[col][1];
    for (auto row : { 0, 1})
    {
      X(row, col) = p[row] - p_m[0][row];
      Y(row, col) = q[row] - p_m[1][row];
    }
  }
  auto S = X * Y.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
  auto U = svd.matrixU();
  auto V = svd.matrixV();
  Matrix m(2, 2);
  m.setIdentity();
  m(1, 1) = (U * V.transpose()).determinant();
  _R = V * m * U.transpose();
  Eigen::Vector2d P, Q;
  P << p_m[0][0], p_m[0][1];
  Q << p_m[1][0], p_m[1][1];
  _T = Q - _R * P;
}

} // namespace MeshOp
