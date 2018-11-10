#pragma once

#include <Topology/topology.hh>
#include <Geo/vector.hh>

#include <map>
#include <memory>
#include <vector>

namespace MeshOp {

struct FixedPositions
{
  Topo::Wrap<Topo::Type::VERTEX> vert_;
  Geo::VectorD2 pos_;
};

struct IFlatten
{
  virtual ~IFlatten() {}
  virtual void set_body(Topo::Wrap<Topo::Type::BODY>& _body) = 0;
  virtual void add_fixed_group(std::vector<FixedPositions>& _fixed) = 0;
  virtual void compute(bool _conformal = false) = 0;
  static std::unique_ptr<IFlatten> make();
};

} // namespace FLAT
