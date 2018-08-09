
#include "decompose.hh"
#include "iterator.hh"


namespace Topo
{
std::vector<Wrap<Type::BODY>> decompose(Wrap<Type::BODY> _body)
{
  Iterator<Type::BODY, Type::EDGE> edges(_body);
  for (auto ed : edges)
  {
    Iterator<Type::VERTEX, Type::EDGE> ed_it(_body);

  }

}

} // namespace Topo
