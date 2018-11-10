#include "catch.hpp"
#include "flatten/flatten.hxx"
#include "Import/load_obj.hh"
#include "Import/save_obj.hh"

#include <filesystem>
#include <string>

static void flatten_complete(const char* file, bool _conformal)
{
  namespace fs = std::filesystem;
  fs::path out_dir(OUTDIR);
  if (!fs::is_directory(out_dir) || !fs::exists(out_dir))
      fs::create_directory(out_dir); // create src folder

  auto body = IO::load_obj((std::string(INDIR) + "/" + file).c_str());
  auto flatter = MeshOp::IFlatten::make();
  flatter->set_body(body);
  flatter->add_fixed_group();
  flatter->compute(_conformal);
  auto flnm = std::string(OUTDIR) + "/" + (_conformal ? "conf_" : "exact_") + file;
  IO::save_obj(flnm.c_str(), body);
}

TEST_CASE("my_flat_00", "[FlatteningFinal]")
{
  std::vector<std::vector<MeshOp::FixedPositions>> constr;
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
