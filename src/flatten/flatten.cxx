#include "flatten.hxx"
#include "energy_function.hxx"
#include "Topology/iterator.hh"

#include <Eigen/PardisoSupport>

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

  void compute_internal(
    bool _conformal, EnergyFunction& _ef, bool _apply_constraints);
};

std::unique_ptr<IFlatten> IFlatten::make()
{
  return std::make_unique<Flatten>();
}

namespace {
struct ComputeData
{
  ComputeData(const Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf,
    MapVertPos& _mvp) : bf_(_bf)
  {
  auto vertex_point = [](const Topo::Wrap<Topo::Type::VERTEX>& _v)
  {
    Geo::Point pt;
    _v->geom(pt);
    return pt;
  };
  area0_ = area(bf_, vertex_point);
  ef_.init(bf_, _mvp);
  }

  void compute(bool _conformal, bool _apply_constraints);

  void apply_result()
  {
    auto& vrt_inds = ef_.veterx_map();
    auto set_x = [&vrt_inds](const double* _x)
    {
      for (const auto& v_id : vrt_inds)
      {
        auto base_ind = 2 * v_id.second;
        Geo::VectorD3 pt;
        pt = { _x[base_ind], _x[base_ind + 1], 0 };
        const_cast<Topo::Wrap<Topo::Type::VERTEX>&>(v_id.first)->set_geom(pt);
      }
    };
    set_x(X_.data());
  }

  Eigen::VectorXd& X()
  {
    return X_;
  }

  MKL_INT constrain_number() const { return ef_.constrain_number(); }

private:
  EnergyFunction ef_;
  Eigen::VectorXd X_;
  double area0_;
  const Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& bf_;
};

void ComputeData::compute(bool _conformal, bool _apply_constraints)
{
  MKL_INT unkn_nmbr = ef_.compute_unkown_nmbr(_apply_constraints);
  LM::Matrix M;
  ef_.jacobian_conformal(M);

  MKL_INT fixed_var = M.cols() - unkn_nmbr;
  auto split_idx = unkn_nmbr;
  auto A = M.block(0, 0, M.rows(), split_idx);
  auto B = M.block(0, split_idx, M.rows(), fixed_var);
  Eigen::Vector4d fixed;
  fixed(0) = -100;
  fixed(1) = fixed(2) = fixed(3) = 0;

  auto b = B * fixed;
  Eigen::PardisoLDLT<LM::Matrix> lsolver;
  lsolver.compute(A.transpose() * A);
  Eigen::VectorXd rhs = A.transpose() * b;
  X_ = lsolver.solve(rhs);
  X_.conservativeResize(M.cols());
  for (size_t i = 0; i < 4; ++i)
    X_(split_idx + i, 0) = -fixed(i, 0);

  auto& vrt_inds = ef_.veterx_map();

  auto vertex_flat_point = [this, &vrt_inds](const Topo::Wrap<Topo::Type::VERTEX>& _v)
  {
    auto it = vrt_inds.find(_v);
    if (it == vrt_inds.end())
      throw "Vertex veraible not found";
    auto ind = vrt_inds.find(_v)->second;
    return Geo::Point{ X_(2 * ind), X_(2 * ind + 1), 0 };
  };
  auto a1 = area(bf_, vertex_flat_point);
  X_ *= sqrt(area0_ / a1);

  if (!_conformal)
  {
    auto solver = LM::ISparseLM::make(ef_);
    solver->compute(X_, 300);
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

  if (cmp_data.constrain_number() > 3)
    cmp_data.compute(_conformal, true);
  cmp_data.apply_result();
}

} // namespace FLAT
