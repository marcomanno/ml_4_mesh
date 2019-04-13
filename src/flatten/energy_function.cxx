#pragma optimize ("", off)
#include "energy_function.hxx"

namespace MeshOp {

namespace {

static auto move_to_local_coord(const Geo::VectorD3 _tri[3], Geo::VectorD2 _loc_tri[2])
{
  auto v_01 = _tri[1] - _tri[0];
  _loc_tri[0][0] = Geo::length(v_01);
  _loc_tri[0][1] = 0;
  auto v_02 = _tri[2] - _tri[0];
  _loc_tri[1][0] = v_02 * v_01 / _loc_tri[0][0];
  auto area_time_2 = Geo::length(v_02 % v_01);
  _loc_tri[1][1] = area_time_2 / _loc_tri[0][0];
  return area_time_2 / 2;
}

} // namespace

void EnergyFunction::DataOfFace::compute_coeff(double _area_sqrt)
{
  area_sqrt_ = _area_sqrt;
  c0_ = _area_sqrt / loc_tri_[0][0]; // 1 / x_1
  c1_ = _area_sqrt / loc_tri_[1][1]; // 1 / y_2
  c2_ = _area_sqrt * loc_tri_[1][0] / (loc_tri_[0][0] * loc_tri_[1][1]); // x_2 / (x_1 * y_2)

  w_[0] = (loc_tri_[1] - loc_tri_[0]) / _area_sqrt;
  w_[1] =  -loc_tri_[1] / _area_sqrt;
  w_[2] = loc_tri_[0] / _area_sqrt;
}

void EnergyFunction::DataOfFace::fill_matrix(
    Eigen::Matrix<double, 4, 6>& _da_uv) const
  {
      _da_uv <<
             -c0_,         0,  c0_,    0,   0,   0,
        c2_ - c1_,         0, -c2_,    0, c1_,   0,
                0,      -c0_,    0,  c0_,   0,   0,
                0, c2_ - c1_,    0, -c2_,   0, c1_;
  }

const std::array<Geo::VectorD2, 3>& 
EnergyFunction::DataOfFace::get_lscm_coeff() const
{
  return w_;
}

void EnergyFunction::init(
  const Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf,
  const MapVertPos& _mvp, double _a0)
{
  tri_nmbr_ = _bf.size();
  n_ = 0;
  fixed_nmbr_ = -1;

  for (auto face : _bf)
  {
    auto& fd = data_of_faces_.emplace_back();
    // 3 equations per face.
    Topo::Iterator<Topo::Type::FACE, Topo::Type::VERTEX> fv(face);
    Geo::VectorD3 pts[3];
    size_t i = 0;
    for (auto v : fv)
    {
      v->geom(pts[i]);
      auto ifxed_vert_it = _mvp.find(v);
      if (ifxed_vert_it == _mvp.end())
      {
        auto pos_it = vrt_inds_.emplace(v, n_);
        fd.idx_[i++] = 2 * pos_it.first->second;
        if (pos_it.second)
          ++n_;
      }
      else
      {
        auto pos_it = vrt_inds_.emplace(v, fixed_nmbr_);
        fd.idx_[i++] = 2 * pos_it.first->second;
        if (pos_it.second)
          --fixed_nmbr_;
      }
    }
    fd.compute_coeff(sqrt(move_to_local_coord(pts, fd.loc_tri_) / _a0));
  }
  fixed_nmbr_ = -fixed_nmbr_ - 1;
  const auto vert_nmbr = n_ + fixed_nmbr_;
  std::map<MKL_INT, MKL_INT> remaps;
  static bool sisi = true;
  if (sisi && _mvp.empty())
  {
    auto find_reorder_map = [this, vert_nmbr, &remaps]()
    {
      auto reorder = 1;
      double len_sq = 0; // std::numeric_limits<double>::max();
      Topo::Wrap<Topo::Type::EDGE> ed_max;
      for (auto&[v, ind] : vrt_inds_)
      {
        Topo::Iterator<Topo::Type::VERTEX, Topo::Type::EDGE> ve(v);
        for (auto e : ve)
        {
          Topo::Iterator<Topo::Type::EDGE, Topo::Type::FACE> ef(e);
          if (ef.size() == 1)
          {
            Topo::Iterator<Topo::Type::EDGE, Topo::Type::VERTEX> ev(e);
            Geo::Point pts[2];
            int j = 0;
            for (auto v : ev)
            {
              v->geom(pts[j++]);
            }
            auto len_sq_tmp = Geo::length_square(pts[0] - pts[1]);
            if (len_sq_tmp > len_sq)
            {
              len_sq = len_sq_tmp;
              ed_max = e;
              goto here;
            }
          }
        }
      }
      here:
      Topo::Iterator<Topo::Type::EDGE, Topo::Type::VERTEX> ev(ed_max);
      for (auto v : ev)
      {
        auto ind1 = vrt_inds_[v];
        remaps[ind1] = vert_nmbr - reorder;
        remaps[vert_nmbr - reorder] = ind1;
        reorder++;
      }
    };
    find_reorder_map();
  }

  auto fix_id = [&remaps, vert_nmbr](MKL_INT& _ind, double _coeff)
  {
    if (_ind < 0)
    {
      _ind += _coeff * vert_nmbr;
      return true;
    }
    else
    {
      auto it = remaps.find(_ind / _coeff);
      if (it != remaps.end())
      {
        _ind = _coeff * it->second;
        return true;
      }
    }
    return false;
  };

  for (auto&[v, ind] : vrt_inds_)
    fix_id(ind, 1);

  for (auto& fdit : data_of_faces_)
  {
    int changes = 0;
    for (auto& ind : fdit.idx_)
    {
      changes += fix_id(ind, 2);
    }
    if (changes > 0)
      std::cout << changes;
  }

  m2_ = 2 * tri_nmbr_;
  m3_ = 3 * tri_nmbr_;
  n2_ = 2 * vert_nmbr;
}

MKL_INT EnergyFunction::compute_unkown_nmbr(bool _apply_constraints)
{
  if (_apply_constraints)
  {
    n3_ = 2 * n_;
    return n3_;
  }
  else
  {
    fixed_zero_ = 3;
    n3_ = 2 * (n_ + fixed_nmbr_) - 3;
    return n3_;
  }
}

bool EnergyFunction::evaluate(
  const LM::ColumnVector& _x, LM::ColumnVector& _f) const
{
  return compute(_x, &_f, nullptr);
}
bool EnergyFunction::jacobian(const LM::ColumnVector& _x, LM::Matrix& _fj) const
{
  return compute(_x, nullptr, &_fj);
}
MKL_INT EnergyFunction::rows() const
{
  return m3_;
}

MKL_INT EnergyFunction::cols() const
{
  return n3_;
}
void EnergyFunction::jacobian_conformal(LM::Matrix& _fj) const
{
  _fj.resize(m2_, n2_);
  std::vector<Eigen::Triplet<double>> triplets;
  MKL_INT i_eq_loop = 0;
  Eigen::Matrix<double, 2, 4> dfa;
  dfa << 1, 0, 0, -1, 0, 1, 1, 0;
  Eigen::Matrix<double, 4, 6> da_uv;
  for (auto& fd : data_of_faces_)
  {
    // 2 equations per face.
    MKL_INT i_eq = i_eq_loop;
    i_eq_loop += 2;
#if 0
    const auto& w = fd.get_lscm_coeff();
    for (size_t j = 0; j < 3; ++j)
    {
      auto idx_base = fd.idx_[j];
      triplets.emplace_back(i_eq, idx_base, w[j][0]);
      triplets.emplace_back(i_eq, idx_base + 1, -w[j][1]);
      triplets.emplace_back(i_eq + 1, idx_base, w[j][1]);
      triplets.emplace_back(i_eq + 1, idx_base + 1, w[j][0]);
    }

#else
    fd.fill_matrix(da_uv);
    Eigen::Matrix<double, 2, 6> df_uv = dfa * da_uv;
    for (MKL_INT i = 0; i < 2; ++i)
    {
      for (MKL_INT j = 0; j < 6; ++j)
      {
        auto val = df_uv(i, j);
        if (val != 0)
        {
          auto c = fd.idx_[j / 2] + j % 2;
          triplets.emplace_back(
            static_cast<int>(i_eq + i),
            static_cast<int>(c), val);
        }
      }
    }
#endif
  }
  _fj.setFromTriplets(triplets.begin(), triplets.end());
}

// Return the matrix that can be used to compute the area by multiplying
// x' * _area_matrix * x
void EnergyFunction::area_matrix(LM::Matrix& _area_matrix)
{
  MKL_INT n_var = n2_ - fixed_zero_;
  _area_matrix.resize(n_var, n_var);
  std::vector<Eigen::Triplet<double>> triplets;
  for (auto& fd : data_of_faces_)
  {
    // Area of a face = 
    // u0 * v1 + u1 * v2 + u2 * v0 - u0 * v2 - u1 * v0 - u2 * v1
    auto insert = [n_var, &triplets, &fd](int _a, int _b, double _v)
    {
      auto get_idx = [&fd](int _i) { return static_cast<int>(fd.idx_[_i]); };
      auto i = get_idx(_a);
      auto j = get_idx(_b) + 1;
      if (i < n_var && j < n_var)
        triplets.emplace_back(i, j, _v);
    };
    insert(0, 1, 1.);
    insert(1, 2, 1.);
    insert(2, 0, 1.);
    insert(0, 2, -1.);
    insert(1, 0, -1.);
    insert(2, 1, -1.);
  }
  _area_matrix.setFromTriplets(triplets.begin(), triplets.end());
}

int EnergyFunction::compute(const Eigen::VectorXd& _x,
  LM::ColumnVector* _fvec, LM::Matrix* _fjac) const
{
  std::vector<Eigen::Triplet<double>> triplets;
  Eigen::Matrix<double, 4, 6> da_uv;
  MKL_INT i_eq_loop = 0;
  const MKL_INT rows_per_face = 3;
  Eigen::Matrix<double, rows_per_face, 4> dfa;
  for (auto& fd : data_of_faces_)
  {
    // 3 equations per face.
    MKL_INT i_eq = i_eq_loop;
    i_eq_loop += rows_per_face;
    auto u10 = _x(fd.idx_[1]) - _x(fd.idx_[0]);
    auto u20 = _x(fd.idx_[2]) - _x(fd.idx_[0]);
    auto v10 = _x(fd.idx_[1] + 1) - _x(fd.idx_[0] + 1);
    auto v20 = _x(fd.idx_[2] + 1) - _x(fd.idx_[0] + 1);
    double a = u10 / fd.loc_tri_[0][0];
    double b = (u20 - fd.loc_tri_[1][0] * u10 / fd.loc_tri_[0][0]) / fd.loc_tri_[1][1];
    double c = v10 / fd.loc_tri_[0][0];
    double d = (v20 - fd.loc_tri_[1][0] * v10 / fd.loc_tri_[0][0]) / fd.loc_tri_[1][1];
    auto det = a * d - b * c;
    if (_fvec != nullptr)
    {
      (*_fvec)(i_eq) = fd.area_sqrt() * (a - d);
      (*_fvec)(i_eq + 1) = fd.area_sqrt() * (b + c);
      (*_fvec)(i_eq + 2) = fd.area_sqrt() * (std::log(det));
    }
    if (_fjac == nullptr)
      continue;
    dfa << 1, 0, 0, -1,
      0, 1, 1, 0,
      d / det, -c / det, -b / det, a / det;

    auto c0 = 1. / fd.loc_tri_[0][0]; // 1 / x_1
    auto c1 = 1. / fd.loc_tri_[1][1]; // 1 / y_2
    auto c2 = fd.loc_tri_[1][0] / (fd.loc_tri_[0][0] * fd.loc_tri_[1][1]); // x_2 / (x_1 * y_2)
    fd.fill_matrix(da_uv);
    Eigen::Matrix<double, rows_per_face, 6> df_uv = dfa * da_uv;
    for (MKL_INT i = 0; i < rows_per_face; ++i)
    {
      for (MKL_INT j = 0; j < 6; ++j)
      {
        auto val = df_uv(i, j);
        if (val != 0)
        {
          auto c = fd.idx_[j / 2] + j % 2;
          if (c < _fjac->cols())
            triplets.emplace_back(
              static_cast<int>(i_eq + i),
              static_cast<int>(c), val);
        }
      }
    }
  }
  if (_fjac != nullptr)
    _fjac->setFromTriplets(triplets.begin(), triplets.end());

  return 0;
}

} // namespace MeshOp
