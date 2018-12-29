//#pragma optimize ("", off)
#include "Import/load_obj.hh"
#include "Import/save_obj.hh"
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
static const size_t INPUT_SIZE = 762;

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

struct TrainData
{
  std::vector<double> in_;
  std::vector<double> out_;
  void print_output()
  {
    std::cout << "Predictions:" << std::endl;
    int i = 0;
    for (const auto& v : out_)
    {
      std::cout << " " << std::round(9 * v);
      if (++i % 64 == 0)
        std::cout << std::endl;
    }
    std::cout << std::endl;
  }
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

  bool add_element(const FaceInfo& _fi)
  {
    if (!_fi.valid_)
    {
      input_var_.insert(input_var_.end(), 3, 0.);
      return false;
    }
    const auto& angs = mesh_angles_[_fi.coe_];
    input_var_.push_back(angs.edge_angle_);
    input_var_.push_back(angs.init_angle_);
    Topo::Iterator<Topo::Type::FACE, Topo::Type::COEDGE> fc(_fi.face_);
    size_t i = 0;
    for (; i < 3 && fc.get(i) != _fi.coe_; ++i);
    if (++i >= 3)
      i = 0;
    const auto& next_angs = mesh_angles_[fc.get(i)];
    input_var_.push_back(next_angs.init_angle_);
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
        Topo::Iterator<Topo::Type::COEDGE, Topo::Type::EDGE> ce(cc);
        auto ed = ce.get(0);
        if (!edges_.insert(ed).second)
        {
          new_faces.emplace_back();
          continue;
        }
        Topo::Iterator<Topo::Type::EDGE, Topo::Type::COEDGE> ec(ed);
        auto opp_coe = ec.get(0);
        if (opp_coe == cc)
          opp_coe = ec.get(1);
        Topo::Iterator<Topo::Type::COEDGE, Topo::Type::FACE> cf(opp_coe);
        new_faces.emplace_back(cf.get(0), opp_coe);
      }
    }
    faces_ = std::move(new_faces);
    process();
  }
  void train(bool _is_on_boundary, TrainData& _tr_dat)
  {
    _tr_dat.in_.insert(_tr_dat.in_.end(), input_var_.begin(), input_var_.end());
    _tr_dat.out_.push_back(_is_on_boundary ? 1. : 0.);
  }
  bool predict(ML::IMachine<double>& _machine)
  {
    std::vector<double> res;
    _machine.predict1(input_var_, res);
    return res[0] > 0.5;
  }
};

std::map<Topo::Wrap<Topo::Type::COEDGE>, Angles> 
compute_mesh_angles(Topo::Wrap<Topo::Type::BODY> _body)
{
  std::map<Topo::Wrap<Topo::Type::COEDGE>, Angles> mesh_angles;
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(_body);
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
      coe_dat.edge_angle_ = ang1;
    }
  }
  return mesh_angles;
}

static void
process(const fs::path& _mesh_file, TrainData& _tr_dat)
{
  std::map<Topo::Wrap<Topo::Type::FACE>, int> face_groups;
  auto body = IO::load_obj(convert(_mesh_file).c_str(), &face_groups);
  std::set<Topo::Wrap<Topo::Type::EDGE>> boundary_edges;
  {
    Topo::Iterator<Topo::Type::BODY, Topo::Type::EDGE> be(body);
    for (auto ed : be)
    {
      Topo::Iterator<Topo::Type::EDGE, Topo::Type::FACE> ef(ed);
      if (ef.size() == 2)
      {
        if (face_groups[ef.get(0)] != face_groups[ef.get(1)])
          boundary_edges.insert(ed);
      }
    }
  }
  auto mesh_angles = compute_mesh_angles(body);
  Topo::Iterator<Topo::Type::BODY, Topo::Type::EDGE> be(body);
  for (auto ed : be)
  {
    bool is_boundary =
      boundary_edges.find(ed) != boundary_edges.end();
    MachineData md(mesh_angles);
    md.init(ed);
    md.process();
    md.train(is_boundary, _tr_dat);
  }
}

static void save_edges(std::set<Topo::Wrap<Topo::Type::EDGE>>& _boundry_edges,
  const std::string& _bndrs_mesh_name)
{
  std::ofstream out(_bndrs_mesh_name);
  auto save_point = [&out](const Geo::VectorD3& _pt)
  {
    out << "v " << _pt[0] << " " << _pt[1] << " " << _pt[2] << std::endl;
  };
  for (auto& ed : _boundry_edges)
  {
    Topo::Iterator<Topo::Type::EDGE, Topo::Type::VERTEX> ev(ed);
    Geo::VectorD3 pt[2];
    ev.get(0)->geom(pt[0]);
    ev.get(1)->geom(pt[1]);
    save_point(pt[0]);
    save_point(pt[1]);
    save_point(pt[0]);
  }
  for (size_t n = _boundry_edges.size(), i = 1; n-- > 0; i += 3)
    out << "f " << i << " " << i + 1 << " " << i + 2 << std::endl;
}

void make_segmented_mesh(
  const fs::path& _mesh_filename, ML::IMachine<double>& _machine)
{
  auto body = IO::load_obj(_mesh_filename.string().c_str());
  auto mesh_angles = compute_mesh_angles(body);
  Topo::Iterator<Topo::Type::BODY, Topo::Type::EDGE> be(body);
  std::set<Topo::Wrap<Topo::Type::EDGE>> boundry_edges;
  for (auto ed : be)
  {
    MachineData md(mesh_angles);
    md.init(ed);
    md.process();
    if (md.predict(_machine))
      boundry_edges.insert(ed);
  }
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(body);
  std::map<Topo::Wrap<Topo::Type::FACE>, int> faces_group;
  for (auto f : bf)
    faces_group.emplace(f, -1);
  int new_group = -1;
  for (auto& fg : faces_group)
  {
    if (fg.second >= 0)
      continue;
    ++new_group;
    std::vector<Topo::Wrap<Topo::Type::FACE>> f_to_proc = { fg.first };
    while (!f_to_proc.empty())
    {
      auto curr = f_to_proc.back();
      f_to_proc.pop_back();
      auto& fg = faces_group[curr];
      if (fg >= 0)
        continue;
      fg = new_group;
      Topo::Iterator<Topo::Type::FACE, Topo::Type::EDGE> fe(curr);
      for (auto e : fe)
      {
        if (boundry_edges.find(e) != boundry_edges.end())
          continue;
        Topo::Iterator<Topo::Type::EDGE, Topo::Type::FACE> ef(e);
        for (auto f : ef)
        {
          if (f != curr)
            f_to_proc.push_back(f);
        }
      }
    }
  }
  std::map<int, std::vector<Topo::Wrap<Topo::Type::FACE>>> groups_faces;
  for (auto& fg : faces_group)
  {
    groups_faces[fg.second].push_back(fg.first);
  }
  fs::path new_file_dir(OUTDIR);
  new_file_dir.append("TestData");
  new_file_dir.append("seg_mesh");
  if (!fs::exists(new_file_dir))
    fs::create_directory(new_file_dir);
  auto new_file = new_file_dir;
  new_file.append(_mesh_filename.filename().string());

  IO::save_obj(new_file.string().c_str(), body, true, &groups_faces);
  auto fname = new_file.string();
  fname.erase(fname.end() - 4, fname.end());
  fname += "-bndr.obj";
  save_edges(boundry_edges, fname);
}

void train_mesh_segmentation_on_folder(
  const fs::path& _folder, TrainData& _tr_dat)
{
  if (!fs::exists(_folder))
    return;
  const fs::directory_iterator end_itr;
  for (fs::directory_iterator itr(_folder); itr != end_itr; ++itr)
  {
    if (fs::is_directory(itr->status()))
      train_mesh_segmentation_on_folder(itr->path(), _tr_dat);
    else if (itr->path().extension() == ".obj")
      process(itr->path(), _tr_dat);
  }
}

} // namespace

namespace MeshSegmentation
{

static std::string data_file()
{
  return OUTDIR"/data";
}


void train_mesh_segmentation(const char* _folder)
{
  auto machine = ML::IMachine<double>::make();
  auto x = machine->make_input(INPUT_SIZE);
  auto y = machine->make_output(1);

  const int INTERM_STEP1 = 3;

  auto w0 = machine->add_weight(INPUT_SIZE, INTERM_STEP1);
  auto b0 = machine->add_weight(1, INTERM_STEP1);
  auto layer0 = machine->add_layer(x, w0, b0);

  if constexpr(INTERM_STEP1 == 1)
    machine->set_target(layer0, 1.e-6);
  else
  {
    const int INTERM_STEP2 = 1;
    auto w1 = machine->add_weight(INTERM_STEP1, INTERM_STEP2, 1.e-5);
    auto b1 = machine->add_weight(1, INTERM_STEP2, 1.e-5);
    auto layer1 = machine->add_layer(tensorflow::Input(layer0), w1, b1);
    if constexpr(INTERM_STEP2 == 1)
      machine->set_target(layer1, 2.e-5);
    else
    {
    auto w2 = machine->add_weight(INTERM_STEP2, 1, -1e-3);
    auto b2 = machine->add_weight(1, 1, -1e-3);
    auto layer2 = machine->add_layer(tensorflow::Input(layer1), w2, b2);
    machine->set_target(layer2, 1e-6);
    }
  }

  TrainData tr_dat;
  train_mesh_segmentation_on_folder(fs::path(_folder), tr_dat);
  for (int i = 0; i < tr_dat.out_.size(); ++i)
  {
    if ((tr_dat.out_[i] > 0) ^ (fabs(tr_dat.in_[INPUT_SIZE * i]) > 0.1))
      std::cout << "Error " << tr_dat.out_[i] << " " << tr_dat.in_[INPUT_SIZE * i] << std::endl;
  }
  machine->train(tr_dat.in_, tr_dat.out_, 100000);
  auto flnm = data_file();
  tr_dat.out_.resize(tr_dat.in_.size() / INPUT_SIZE);
  machine->predictN(tr_dat.in_, tr_dat.out_);
  tr_dat.print_output();
  machine->save(flnm.c_str());
  auto machine2 = ML::IMachine<double>::make();
  machine2->load(flnm.c_str());
  tr_dat.in_.resize(INPUT_SIZE);
  machine2->predictN(tr_dat.in_, tr_dat.out_);
  tr_dat.print_output();
}

void apply_mesh_segmentation(const char* _folder)
{
  if (!fs::exists(_folder))
    return;

  auto machine = ML::IMachine<double>::make();
  machine->load(data_file().c_str());

  const fs::directory_iterator end_itr;
  for (fs::directory_iterator itr(_folder); itr != end_itr; ++itr)
  {
    if (fs::is_directory(itr->status()))
      apply_mesh_segmentation(itr->path().string().c_str());
    else if (itr->path().extension() == ".obj")
      make_segmented_mesh(itr->path(), *machine);
  }
}

} // namespace MeshSegmentation
