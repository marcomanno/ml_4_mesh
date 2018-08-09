
#include "Import/import.hh"
#include "Topology/iterator.hh"

#include <array>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <sstream>

namespace fs = std::filesystem;

static std::string convert(const fs::path& _path)
{
  std::wstring fn(_path.c_str());
  std::string fn1;
  for (auto c : fn)
    fn1 += c;
  return fn1;
}

static void process(const fs::path& _mesh_file)
{
  auto body = IO::load_obj(convert(_mesh_file).c_str());
  std::vector<Topo::Wrap<Topo::Type::VERTEX>> boundary_vertices;
  {
    std::vector<size_t> bndrs;
    auto boundaries = _mesh_file;
    boundaries.replace_extension(".bnd");
    std::ifstream bndr_stream(convert(boundaries).c_str());
    while (!bndr_stream.eof() && !bndr_stream.bad())
      bndr_stream >> bndrs.emplace_back();
    std::sort(bndrs.begin(), bndrs.end());
    bndrs.erase(std::unique(bndrs.begin(), bndrs.end()), bndrs.end());
    Topo::Iterator<Topo::Type::BODY, Topo::Type::VERTEX> bv(body);
    std::vector<Topo::Wrap<Topo::Type::VERTEX>> all_vertices;
    for (auto v : bv)
      all_vertices.push_back(v);
    std::sort(all_vertices.begin(), all_vertices.end());
    for (auto v_idx : bndrs)
      boundary_vertices.push_back(all_vertices[v_idx]);
  }

}

void train_mesh_segmentation(const fs::path& _folder)
{
  if (!fs::exists(_folder))
    return;
  const fs::directory_iterator end_itr;
  for (fs::directory_iterator itr(_folder); itr != end_itr; ++itr)
  {
    if (fs::is_directory(itr->status()))
      train_mesh_segmentation(itr->path());
    else if (itr->path().extension() == ".obj1")
      process(itr->path());
  }
}

void train_mesh_segmentation(const char* _folder)
{
  train_mesh_segmentation(fs::path(_folder));
}
