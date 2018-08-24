#pragma optimize ("", off)
#include "Import/import.hh"
#include "Topology/iterator.hh"

#include "MeshSegmentation/machine.hxx"

#include "Topology/geom.hh"
#include "Topology/shared.hh"

#include <array>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <map>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

namespace
{
static const size_t INPUT_SIZE = 1016;

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
  Topo::Iterator<Topo::Type::COEDGE, Topo::Type::VERTEX> cv(_a);
  Geo::VectorD3 pnts[2];
  cv.get(1)->geom(pnts[1]);
  cv.get(0)->geom(pnts[0]);
  return pnts[1] - pnts[0];
}

struct FaceInfo
{
  FaceInfo() : valid_(false) {}
  FaceInfo(const Topo::Wrap<Topo::Type::FACE>& _f,
           const Topo::Wrap<Topo::Type::COEDGE>& _c,
           bool _valid = true) :
    face_(_f), coe_(_c), valid_(_valid) {}
  Topo::Wrap<Topo::Type::FACE> face_;
  Topo::Wrap<Topo::Type::COEDGE> coe_;
  bool valid_;
};

struct MachineData
{
  std::set<Topo::Wrap<Topo::Type::EDGE>> edges_;
  std::vector<FaceInfo> faces_;
  std::vector<double> input_var_;
  std::map<Topo::Wrap<Topo::Type::COEDGE>, Angles>& mesh_angles_;

  MachineData(std::map<Topo::Wrap<Topo::Type::COEDGE>, Angles>& _mesh_angles) :
    mesh_angles_(_mesh_angles) {}
  void init(Topo::Wrap<Topo::Type::EDGE> _ed)
  {
    //edges_.insert(_ed);
    Topo::Iterator<Topo::Type::EDGE, Topo::Type::COEDGE> ec(_ed);
    for (auto c : ec)
    {
      Topo::Iterator<Topo::Type::COEDGE, Topo::Type::FACE> cf(c);
      faces_.emplace_back(cf.get(0), c, true);
    }
  }

  void add_invalid(std::vector<FaceInfo>& _new_faces)
  {
    _new_faces.emplace_back();
    _new_faces.emplace_back();
    input_var_.insert(input_var_.end(), 4, 0.);
  };

  bool add_element(const FaceInfo& _fi)
  {
    if (!_fi.valid_)
    {
      input_var_.insert(input_var_.end(), 4, 0.);
      return false;
    }
    Topo::Iterator<Topo::Type::COEDGE, Topo::Type::EDGE> ce(_fi.coe_);
    auto edge = ce.get(0);
    if (!edges_.insert(edge).second)
    {
      input_var_.insert(input_var_.end(), 4, 0.);
      return false;
    }
    const auto& angs = mesh_angles_[_fi.coe_];
    input_var_.push_back(angs.edge_angle_);
    input_var_.push_back(angs.init_angle_);
    return true;
  }

  void process()
  {
    if (input_var_.size() >= INPUT_SIZE)
      return;
    std::vector<FaceInfo> new_faces;
    for (auto& fi : faces_)
    {
      if (!add_element(fi))
      {
        new_faces.emplace_back();
        new_faces.emplace_back();
        continue;
      }

      Topo::Iterator<Topo::Type::FACE, Topo::Type::COEDGE> fc(fi.face_);

      std::vector<Topo::Wrap<Topo::Type::COEDGE>> coeds;
      bool rev = false;
      for (auto c : fc)
      {
        if (c != fi.coe_)
          coeds.push_back(c);
        else
          rev = coeds.size() == 1;
      }
      if (rev)
        std::swap(coeds[0], coeds[1]);
      for (auto cc : coeds)
      {
        const auto& angs = mesh_angles_[cc];
        input_var_.push_back(angs.init_angle_);
        Topo::Iterator<Topo::Type::COEDGE, Topo::Type::EDGE> ce(cc);
        if (!edges_.insert(ce.get(0)).second)
        {
          new_faces.emplace_back();
          continue;
        }
        Topo::Iterator<Topo::Type::COEDGE, Topo::Type::FACE> cf(cc);
        new_faces.emplace_back(cf.get(0), cc);
      }
    }
    faces_ = std::move(new_faces);
    process();
  }
  void train(bool _is_on_boundary, ML::IMachine<double>& _machine)
  {
    std::vector<double> out(1, _is_on_boundary ? 1. : 0.);
    _machine.train(input_var_, out);
  }
};

static void process(const fs::path& _mesh_file, ML::IMachine<double>& _machine)
{
  auto body = IO::load_obj(convert(_mesh_file).c_str());
  std::set<Topo::Wrap<Topo::Type::EDGE>> boundary_edges;
  {
    std::vector<Topo::Wrap<Topo::Type::VERTEX>> boundary_vertices;
    std::vector<size_t> bndrs;
    auto boundaries = _mesh_file;
    boundaries.replace_extension(".bnd");
    std::ifstream bndr_stream(convert(boundaries).c_str());
    while (!bndr_stream.eof() && !bndr_stream.bad())
      bndr_stream >> bndrs.emplace_back();
    std::vector<size_t> bndr_edges = bndrs;
    std::sort(bndrs.begin(), bndrs.end());
    bndrs.erase(std::unique(bndrs.begin(), bndrs.end()), bndrs.end());
    Topo::Iterator<Topo::Type::BODY, Topo::Type::VERTEX> bv(body);
    std::vector<Topo::Wrap<Topo::Type::VERTEX>> all_vertices;
    for (auto v : bv)
      all_vertices.push_back(v);
    std::sort(all_vertices.begin(), all_vertices.end());
    for (auto v_idx : bndrs)
      boundary_vertices.push_back(all_vertices[v_idx]);
    for (size_t i = 1; i < bndr_edges.size(); i += 2)
    {
      auto v0 = boundary_vertices[bndr_edges[i - 1]];
      auto v1 = boundary_vertices[bndr_edges[i]];
      auto eds = Topo::shared_entities<Topo::Type::VERTEX, Topo::Type::EDGE>
        (v0, v1);
      for (auto ed : eds)
        boundary_edges.insert(ed);
    }
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
  Topo::Iterator<Topo::Type::BODY, Topo::Type::EDGE> be(body);
  for (auto ed : be)
  {
    bool is_boundary =
      boundary_edges.find(ed) != boundary_edges.end();
    MachineData md(mesh_angles);
    md.init(ed);
    md.process();
    md.train(is_boundary, _machine);
  }
}

void train_mesh_segmentation_on_folder(
  const fs::path& _folder, ML::IMachine<double>& _machine)
{
  if (!fs::exists(_folder))
    return;
  const fs::directory_iterator end_itr;
  for (fs::directory_iterator itr(_folder); itr != end_itr; ++itr)
  {
    if (fs::is_directory(itr->status()))
      train_mesh_segmentation_on_folder(itr->path(), _machine);
    else if (itr->path().extension() == ".obj")
      process(itr->path(), _machine);
  }
}
} // namespace

namespace MeshSegmentation
{
void train_mesh_segmentation(const char* _folder)
{
  auto machine = ML::IMachine<double>::make();
  auto x = machine->make_input(INPUT_SIZE);
  auto y = machine->make_output(1);
  auto w0 = machine->add_weight(INPUT_SIZE, 1);
  auto b0 = machine->add_weight(1, 1);
  auto layer0 = machine->add_layer(x, w0, b0);
  machine->set_targets(layer0);
  train_mesh_segmentation_on_folder(fs::path(_folder), *machine);
}
} // namespace MeshSegmentation
