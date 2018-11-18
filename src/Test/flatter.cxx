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
  const constr_function& _constr_func = empty_constr_function)
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
  constr_function a_constr_function = [](Topo::Wrap<Topo::Type::BODY> _body)
  {
    std::vector<std::vector<MeshOp::FixedPositions>> constraints(1);
    Topo::Iterator<Topo::Type::BODY, Topo::Type::VERTEX> bv_it(_body);
    Geo::VectorD2 p_uv{ 0 };
    for (auto v : bv_it)
    {
      Geo::Point pt;
      v->geom(pt);
      if (fabs(pt[0] - 30) > 0.1 || fabs(pt[2] + 25) > 0.1)
        continue;
      constraints[0].push_back({v, p_uv});
      p_uv[0] += 50;
    }
    return constraints;
  };
  flatten_complete("aaa0.obj", false, a_constr_function);
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
    std::vector<Topo::Wrap<Topo::Type::VERTEX>> seq{vmin};
    for (bool new_vert = true; new_vert;)
    {
      new_vert = false;
      Topo::Iterator<Topo::Type::VERTEX, Topo::Type::EDGE> ve_it(seq.back());
      for (auto e : ve_it)
      {
        Topo::Wrap<Topo::Type::VERTEX> oth_v;
        Topo::Iterator<Topo::Type::EDGE, Topo::Type::VERTEX> ev_it(e);
        for (auto v : ev_it)
        {
          if (v != seq.back())
          {
            oth_v = v;
            break;
          }
        }
        if (seq.size() > 1 && seq[seq.size() - 2] == oth_v)
          continue;
        Geo::Point pt;
        oth_v->geom(pt);
        if (pt[2] < 1e-5)
        {
        seq.push_back(oth_v);
        new_vert = true;
        break;
        }
      }
    }
    std::vector<std::vector<MeshOp::FixedPositions>> constraints(1);
    Geo::VectorD2 p_uv{ 0 };
    bool first = true;
    Geo::Point pt_prev;
    for (auto v : seq)
    {
      Geo::Point pt;
      v->geom(pt);
      if (first)
        first = false;
      else
        p_uv[0] += Geo::length(pt - pt_prev);
      pt_prev = pt;
      constraints[0].push_back({ v, p_uv });
    }
    return constraints;
  };
  flatten_complete("aaa1.obj", false, a_constr_function);
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

TEST_CASE("my_flat_15", "[FlatteningFinal]")
{
  flatten_complete("aaaa05.obj", false);
}

TEST_CASE("my_flat_15_constr", "[FlatteningFinal]")
{
  constr_function a_constr_function = [](Topo::Wrap<Topo::Type::BODY> _body)
  {
    std::vector<std::vector<MeshOp::FixedPositions>> constraints(2);
    Topo::Iterator<Topo::Type::BODY, Topo::Type::VERTEX> bv_it(_body);
    bool open = false;
    for (auto v : bv_it)
    {
      Geo::Point pt;
      v->geom(pt);
      if (pt[0] > -18)
        continue;
      Topo::Iterator<Topo::Type::VERTEX, Topo::Type::EDGE> ve_it(v);
      for (auto e : ve_it)
      {
        Topo::Iterator<Topo::Type::EDGE, Topo::Type::FACE> ef_it(e);
        if (open = (ef_it.size() == 1))
          break;
      }
      if (!open)
        continue;
      auto const_idx = pt[1] > 0;
      constraints[const_idx].push_back({ v, {-pt[1], pt[2]} });
    }
    return constraints;
  };
  flatten_complete("aaaa05.obj", false, a_constr_function);
}
