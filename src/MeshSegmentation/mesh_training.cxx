
#include "Import/import.hh"
#include "Topology/iterator.hh"
#include "Topology/geom.hh"

#include <array>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <map>
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

struct Angles
{
  double init_angle_ = -1; // between 0 and Pi
  double edge_angle_ = -1; // between 0 and 2 * Pi
};

Geo::VectorD3 get_direction(
  Topo::Wrap<Topo::Type::COEDGE> _a)
{
  Topo::Iterator<Topo::Type::COEDGE, Topo::Type::VERTEX> cv;
  Geo::VectorD3 pnts[2];
  cv.get(1)->geom(pnts[1]);
  cv.get(0)->geom(pnts[0]);
  return pnts[1] - pnts[0];
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
  std::map<Topo::Wrap<Topo::Type::COEDGE>, Angles> mesh_angles;
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(body);
  for (auto f : bf)
  {
    Topo::Iterator<Topo::Type::FACE, Topo::Type::COEDGE> fc(f);
    const auto coed_nmbr = fc.size();
    Geo::VectorD3 prev_dir = get_direction(fc.get(coed_nmbr - 1));
    for (auto c : fc)
    {
      auto& coe_dat = mesh_angles[c];
      auto curr_dir = get_direction(c);
      auto ang = Geo::angle(prev_dir, curr_dir);
      prev_dir = curr_dir;
      coe_dat.init_angle_ = ang;
      if (coe_dat.edge_angle_ >= 0)
        continue;
      Topo::Iterator<Topo::Type::COEDGE, Topo::Type::EDGE> ce(c);
      Topo::Iterator<Topo::Type::EDGE, Topo::Type::FACE> ef(ce.get(0));
      auto n0 = Topo::face_normal(ef.get(0));
      auto n1 = Topo::face_normal(ef.get(1));
      if (f == ef.get(1))
        std::swap(n0, n1);
      auto ang1 = Geo::signed_angle(n0, n1, curr_dir);
      if (ang1 < 0) ang1 += 2 * M_PI;
      coe_dat.edge_angle_ = ang1;
    }
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
