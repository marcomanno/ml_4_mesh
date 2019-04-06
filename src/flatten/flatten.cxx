#pragma optimize("", off)
#include "flatten.hxx"
#include "optimal_rotation.hxx"

#include "energy_function.hxx"
#include "Topology/iterator.hh"


//#include <Eigen/PardisoSupport>
//#include <Eigen/SparseQR>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <Spectra/SymGEigsSolver.h>
#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/MatOp/SparseCholesky.h>

#include <Import/save_obj.hh>

#include <fstream>

namespace MeshOp {

namespace {

static double area(
  const Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf,
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

} // namespace


struct Flatten : public IFlatten
{
  void set_body(Topo::Wrap<Topo::Type::BODY>& _body) override
  {
    body_ = _body;
  }
  void add_fixed_group(std::vector<FixedPositions>& _fixed) override
  {
    fixed_pos_groups_.emplace_back(std::move( _fixed));
  }
  void compute(bool _conformal = false) override;

private:
  Topo::Wrap<Topo::Type::BODY> body_;
  std::vector<std::vector<FixedPositions>> fixed_pos_groups_;
};

std::unique_ptr<IFlatten> IFlatten::make()
{
  return std::make_unique<Flatten>();
}

namespace {
struct ComputeData
{
  ComputeData(const Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf,
    MapVertPos& _mvp);

  void set_vertex_position(
    Topo::Wrap<Topo::Type::VERTEX>& _vert, const Geo::VectorD2& _uv);

  void set_constraints(std::vector<std::vector<FixedPositions>>& _constr);

  void compute(bool _conformal, bool _apply_constraints);

  void apply_result();

  const Eigen::VectorXd& X() const { return X_; }

  MKL_INT constrain_vertices() const { return ef_.constrain_vertices(); }

private:
  EnergyFunction ef_;
  Eigen::VectorXd X_;
  double area0_;
  const Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& bf_;
};

ComputeData::ComputeData(const Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf,
  MapVertPos& _mvp) : bf_(_bf)
{
  auto vertex_point = [](const Topo::Wrap<Topo::Type::VERTEX>& _v)
  {
    Geo::Point pt;
    _v->geom(pt);
    return pt;
  };
  area0_ = area(bf_, vertex_point);
  ef_.init(bf_, _mvp, area0_ / _bf.size());
}

void ComputeData::set_vertex_position(
  Topo::Wrap<Topo::Type::VERTEX>& _vert, const Geo::VectorD2& _uv)
{
  auto vert_it = ef_.veterx_map().find(_vert);
  if (vert_it == ef_.veterx_map().end())
    throw "Vertex not found";
  X_(2 * vert_it->second) = _uv[0];
  X_(2 * vert_it->second + 1) = _uv[1];
}

void ComputeData::set_constraints(std::vector<std::vector<FixedPositions>>& _constrs)
{
  auto& vert_map = ef_.veterx_map();
  for (auto& constr_set : _constrs)
  {
    std::vector<Geo::VectorD2> dd[3];

    std::vector<std::array<Geo::VectorD2, 2>> from_to_pts;
    for (auto& constr : constr_set)
    {
      auto it = vert_map.find(constr.vert_);
      if (it == vert_map.end())
        throw "Vertex not found";
      auto ind = 2 * it->second;
      from_to_pts.push_back({constr.pos_, {X_(ind), X_(ind + 1)}});

      dd[0].push_back(constr.pos_);
      dd[1].push_back({X_(ind), X_(ind + 1)});
    }
    Eigen::Matrix2d R;
    Eigen::Vector2d T;
    find_optimal_rotation(from_to_pts, R, T);
    Eigen::Vector2d x;
    for (auto& constr : constr_set)
    {
      auto ind = 2 * vert_map.find(constr.vert_)->second;
      x << constr.pos_[0], constr.pos_[1];
      X_.segment<2>(ind) = R * x + T;
      dd[2].push_back({X_(ind), X_(ind + 1)});
    }
    for (auto& xx : dd)
    {
      static int nn;
      IO::save_polyline((std::string("C:/t/") + std::to_string(nn++)).c_str(), xx);
    }
  }
}

void ComputeData::apply_result()
{
  for (const auto& v_id : ef_.veterx_map())
  {
    auto base_ind = 2 * v_id.second;
    Geo::VectorD3 pt;
    pt = { X_(base_ind), X_(base_ind + 1), 0 };
    const_cast<Topo::Wrap<Topo::Type::VERTEX>&>(v_id.first)->set_geom(pt);
  }
}


void ComputeData::compute(bool _conformal, bool _apply_constraints)
{
  MKL_INT unkn_nmbr = ef_.compute_unkown_nmbr(_apply_constraints);
  LM::Matrix M;
  ef_.jacobian_conformal(M);

  MKL_INT fixed_var = M.cols() - unkn_nmbr;
  auto split_idx = unkn_nmbr;
  auto A = M.block(0, 0, M.rows(), split_idx);
  auto B = M.block(0, split_idx, M.rows(), fixed_var);
  Eigen::VectorXd fixed(fixed_var);
  if (_apply_constraints)
    fixed = X_.bottomRows(fixed_var);
  else
  {
    fixed.setZero();
    fixed[0] = -100;
  }

  auto b = B * fixed;
  {
    std::ofstream pipo("piripippo.txt");
#define CH 4
#if CH == 0
    Eigen::SparseQR<LM::Matrix, Eigen::COLAMDOrdering<__int64>> lsolver;
    lsolver.compute(A);
    X_ = lsolver.solve(b);
#elif CH == 1
    LM::Matrix AA = A.transpose() * A;
    Eigen::Matrix<double,-1,1,0,-1,1> B = A.transpose() * b;
    for (auto i = AA.rows(); i-- > 0;)
    {
      auto n = AA.row(i).norm();
      AA.row(i) /= n;
      B.row(i) /= n;
    }
    //auto keep_fun = [](int, int, double v) { return abs(v) > 1e-15; };
    //AA.prune(keep_fun);
    Eigen::SparseLU<LM::Matrix> lsolver;
    lsolver.compute(AA);
    //pipo << AA << std::endl;
    //pipo << "determinant" << lsolver.determinant();
    X_ = lsolver.solve(B);

#elif CH == 2
    LM::Matrix AA = A.transpose() * A;
    Eigen::Matrix<double,-1,1,0,-1,1>  B = A.transpose() * b;
    for (auto i = AA.rows(); i-- > 0;)
    {
      auto n = AA.row(i).norm();
      AA.row(i) /= n;
      B(i) /= n;
    }
    auto keep_fun = [](int, int, double v) { return abs(v) > 1e-15; };
    AA.prune(keep_fun);
    pipo << AA << std::endl;
    pipo << "determinant";
    Eigen::PardisoLU<LM::Matrix> lsolver;
    lsolver.compute(AA);
    X_ = lsolver.solve(B);
#elif CH == 3
    Eigen::SuperLU<LM::Matrix> lsolver;
    lsolver.compute(A.transpose() * A);
    X_ = lsolver.solve(A.transpose() * b);
#elif CH == 4
    LM::Matrix area_matrix;
    ef_.area_matrix(area_matrix);
    LM::Matrix AA = A.transpose() * A;

    using SpectraOp = Spectra::SparseSymMatProd<double, Eigen::Lower, 0, MKL_INT>;
    using SpectraBOp = Spectra::SparseCholesky<double, Eigen::Lower, 0, MKL_INT>;
    SpectraOp op(area_matrix);
    SpectraBOp  Bop(AA);

    // Construct generalized eigen solver object, requesting the largest three generalized eigenvalues
    Spectra::SymGEigsSolver<double, Spectra::LARGEST_ALGE, SpectraOp, SpectraBOp, Spectra::GEIGS_CHOLESKY>
        geigs(&op, &Bop, 1, 6);
    // Initialize and compute
    geigs.init();
    int nconv = geigs.compute();
    auto res = geigs.info();
    if (res != Spectra::SUCCESSFUL)
    {
      std::cout << res << " Computed:" << nconv << std::endl;
      throw "Error";
    }
    auto eig_val = geigs.eigenvalues();
    std::cout << std::endl << "eig_val:"<< std::endl << eig_val << std::endl;
    X_ = geigs.eigenvectors();
    //std::cout << X_ << std::endl;
#else
    Eigen::LeastSquaresConjugateGradient<LM::Matrix> lsolver;
    lsolver.compute(A);
    lsolver.setMaxIterations(10000);
    X_ = lsolver.solve(b);
#endif
  }
  X_.conservativeResize(M.cols());
  //X_.bottomRows(fixed_var) = -fixed;

  auto& vrt_inds = ef_.veterx_map();

  auto vertex_flat_point = [this, &vrt_inds](const Topo::Wrap<Topo::Type::VERTEX>& _v)
  {
    auto it = vrt_inds.find(_v);
    if (it == vrt_inds.end())
      throw "Vertex veraible not found";
    auto ind = vrt_inds.find(_v)->second;
    return Geo::Point{ X_(2 * ind), X_(2 * ind + 1), 0 };
  };
  if (!_apply_constraints)
  {
    auto a1 = area(bf_, vertex_flat_point);
    X_ *= sqrt(area0_ / a1);
  }
  if (_conformal)
    return;

  auto solver = LM::ISparseLM::make(ef_);
  solver->compute(X_, 300);
  if (!_apply_constraints)
  {
    auto a1 = area(bf_, vertex_flat_point);
    X_ *= sqrt(area0_ / a1);
  }
}

} // namespace

void Flatten::compute(bool _conformal)
{
  MapVertPos mvp;
  for (auto& fixed_pos : fixed_pos_groups_)
  {
    for (auto& vert_pos : fixed_pos)
    {
      auto it = mvp.emplace(vert_pos.vert_, vert_pos.pos_);
      if (!it.second)
        throw "DUPLICATED_VERTEX_CONSTRAINT";
    }
  }
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(body_);

  ComputeData cmp_data(bf, mvp);
  cmp_data.compute(_conformal, false);

  if (cmp_data.constrain_vertices() > 1)
  {
    cmp_data.set_constraints(fixed_pos_groups_);
    cmp_data.compute(_conformal, true);
  }
  cmp_data.apply_result();
}

} // namespace MeshOp
