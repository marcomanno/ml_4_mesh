#pragma once

#include <Topology/topology.hh>

namespace IO {
Topo::Wrap<Topo::Type::BODY> load_obj(const char* _flnm);
} // namespace IO