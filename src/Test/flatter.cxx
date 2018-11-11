#include "catch.hpp"
#include "flatten/flatten.hxx"
#include "Import/load_obj.hh"
#include "Import/save_obj.hh"
#include "Topology/iterator.hh"

#include <filesystem>
#include <string>
#include <functional>

namespace {

using constr_function = std::function<
  std::vector<std::vector<MeshOp::FixedPositions>>(Topo::Wrap<Topo::Type::BODY>)>;

constr_function empty_constr_function = [](Topo::Wrap<Topo::Type::BODY>)
{
  return std::vector<std::vector<MeshOp::FixedPositions>>();
};

static void flatten_complete(const char* file, bool _conformal,
  constr_function& _constr_func = empty_constr_function)
{
  namespace fs = std::filesystem;
  fs::path out_dir(OUTDIR);
  if (!fs::is_directory(out_dir) || !fs::exists(out_dir))
    fs::create_directory(out_dir); // create src folder

  auto body = IO::load_obj((std::string(INDIR) + "/" + file).c_str());
  std::vector<std::vector<MeshOp::FixedPositions>> constr = _constr_func(body);
  auto flatter = MeshOp::IFlatten::make();
  flatter->set_body(body);
  if (!constr.empty())
  {
    for (auto& group : constr)
      flatter->add_fixed_group(group);
  }
  flatter->compute(_conformal);
  auto flnm = std::string(OUTDIR) + "/" + (_conformal ? "conf_" : "exact_") + file;
  IO::save_obj(flnm.c_str(), body);
}

} // namespace

TEST_CASE("my_flat_00", "[FlatteningFinal]")
{
  flatten_complete("aaa0.obj", false);
}

TEST_CASE("my_flat_00_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa0.obj", true);
}

TEST_CASE("my_flat_01", "[FlatteningFinal]")
{
  flatten_complete("aaa1.obj", false);
}

TEST_CASE("my_flat_01_constr", "[FlatteningFinal]")
{
  constr_function a_constr_function = [](Topo::Wrap<Topo::Type::BODY> _body)
  {
    Topo::Iterator<Topo::Type::BODY, Topo::Type::VERTEX> bv_it(_body);
    Geo::Point pt_min{ 100, 100, 100 };
    Topo::Wrap<Topo::Type::VERTEX> vmin;
    for (auto v : bv_it)
    {
      Geo::Point pt;
      v->geom(pt);
      if (pt_min > pt)
      {
        vmin = v;
        pt_min = pt;
      }
    }
    std::vector<Topo::Wrap<Topo::Type::VERTEX>> seq;
    for (seq.push_back(vmin);;)
    {
      Topo::Iterator<Topo::Type::VERTEX, Topo::Type::EDGE> ve_it(seq.back());
      for (auto e : ve_it)
      {
        e;
      }

    }
    return std::vector<std::vector<MeshOp::FixedPositions>>();
  };
  flatten_complete("aaa1.obj", false);
}

TEST_CASE("my_flat_01_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa1.obj", true);
}

TEST_CASE("my_flat_02", "[FlatteningFinal]")
{
  flatten_complete("aaa2.obj", false);
}

TEST_CASE("my_flat_02_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa2.obj", true);
}

TEST_CASE("my_flat_03", "[FlatteningFinal]")
{
  flatten_complete("aaa3.obj", false);
}

TEST_CASE("my_flat_03_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa3.obj", true);
}

TEST_CASE("my_flat_04", "[FlatteningFinal]")
{
  flatten_complete("aaa4.obj", false);
}

TEST_CASE("my_flat_04_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa4.obj", true);
}

TEST_CASE("my_flat_05", "[FlatteningFinal]")
{
  flatten_complete("aaa5.obj", false);
}

TEST_CASE("my_flat_05_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa5.obj", true);
}

TEST_CASE("my_flat_06", "[FlatteningFinal]")
{
  flatten_complete("aaa6.obj", false);
}

TEST_CASE("my_flat_06_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa6.obj", true);
}

TEST_CASE("my_flat_07", "[FlatteningFinal]")
{
  flatten_complete("aaa7.obj", false);
}

TEST_CASE("my_flat_07_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa7.obj", true);
}

TEST_CASE("my_flat_08", "[FlatteningFinal]")
{
  flatten_complete("aaa8.obj", false);
}

TEST_CASE("my_flat_08_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa8.obj", true);
}

TEST_CASE("my_flat_09", "[FlatteningFinal]")
{
  flatten_complete("aaa9.obj", false);
}

TEST_CASE("my_flat_09_conf", "[FlatteningFinal]")
{
  flatten_complete("aaa9.obj", true);
}

TEST_CASE("my_flat_10", "[FlatteningFinal]")
{
  flatten_complete("aaaa00.obj", false);
}

TEST_CASE("my_flat_10_conf", "[FlatteningFinal]")
{
  flatten_complete("aaaa00.obj", true);
}

TEST_CASE("my_flat_11", "[FlatteningFinal]")
{
  flatten_complete("aaaa01.obj", false);
}

TEST_CASE("my_flat_12", "[FlatteningFinal]")
{
  flatten_complete("aaaa02.obj", false);
}

TEST_CASE("my_flat_13", "[FlatteningFinal]")
{
  flatten_complete("aaaa03.obj", false);
}

TEST_CASE("my_flat_14", "[FlatteningFinal]")
{
  flatten_complete("aaaa04.obj", false);
}
