#pragma once

#include <Topology/topology.hh>
#include <map>

namespace IO {
Topo::Wrap<Topo::Type::BODY> load_obj(
  const char* _flnm, 
  std::map<Topo::Wrap<Topo::Type::FACE>, int>* _face_groups = nullptr);
} // namespace IO