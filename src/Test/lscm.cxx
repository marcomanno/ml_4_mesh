//#pragma optimize("", off)
#include "catch.hpp"

#include <unsupported/Eigen/LevenbergMarquardt>

#include "optimize_function.hxx"

#include "Geo/vector.hh"
#include "Import/load_obj.hh"
#include "Import/save_obj.hh"
#include "Topology/iterator.hh"

#include <Eigen/Sparse>
#include <Eigen/SparseQR>

#include "xxxxx.hxx"

static auto move_to_local_coord(const Geo::VectorD3 _tri[3], Geo::VectorD2 _loc_tri[2])
{
  auto v_01 = _tri[1] - _tri[0];
  _loc_tri[0][0] = Geo::length(v_01);
  _loc_tri[0][1] = 0;
  auto v_02 = _tri[2] - _tri[0];
  _loc_tri[1][0] = v_02 * v_01 / _loc_tri[0][0];
  auto area_time_2 = Geo::length(v_02 % v_01);
  _loc_tri[1][1] = area_time_2 / _loc_tri[0][0];
  return area_time_2;
}

using VertexIndMpap = std::map<Topo::Wrap<Topo::Type::VERTEX>, size_t>;
using Functor = Eigen::SparseFunctor<double, int>;

struct EnergyFunction : public Functor
{
  struct DataOfFace
  {
    Geo::VectorD2 loc_tri_[2];
    size_t idx_[3];
    double area_time_2_;
  };

  EnergyFunction(const VertexIndMpap& _vrt_inds, 
    Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf, int _m, int _n) : 
    Functor(_n, _m), vrt_inds_(_vrt_inds)
  {
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
        fd.idx_[i++] = 2 * vrt_inds_.find(v)->second;
      }
      fd.area_time_2_ = move_to_local_coord(pts, fd.loc_tri_);
    }
  }

  int operator()(const Functor::InputType &_x, Functor::ValueType& _fvec) const
  {
    return compute(_x, &_fvec, nullptr);
  }

  int df(const Functor::InputType& _x, Functor::JacobianType& _fjac) const
  {
    return compute(_x, nullptr, &_fjac);
  }

protected:
  const VertexIndMpap& vrt_inds_;
  std::vector<DataOfFace> data_of_faces_;

  int compute(const Functor::InputType& _x,
    Functor::ValueType* _fvec, Functor::JacobianType* _fjac) const
  {
    //if (_fjac != nullptr) _fjac->setZero();

    auto X = [this, &_x](int _i)
    {
      if (_i < inputs())
        return _x(_i);
      else
        return 0.;
    };

    size_t i_eq_loop = 0;
    for (auto& fd : data_of_faces_)
    {
      // 3 equations per face.
      size_t i_eq = i_eq_loop;
      i_eq_loop += 3;
      auto u10 = X(fd.idx_[1])     - X(fd.idx_[0]);
      auto u20 = X(fd.idx_[2])     - X(fd.idx_[0]);
      auto v10 = X(fd.idx_[1] + 1) - X(fd.idx_[0] + 1);
      auto v20 = X(fd.idx_[2] + 1) - X(fd.idx_[0] + 1);
      double a = u10 / fd.loc_tri_[0][0];
      double b = (u20 - fd.loc_tri_[1][0] * u10 / fd.loc_tri_[0][0]) / fd.loc_tri_[1][1];
      double c = v10 / fd.loc_tri_[0][0];
      double d = (v20 - fd.loc_tri_[1][0] * v10 / fd.loc_tri_[0][0]) / fd.loc_tri_[1][1];
      auto det = a * d - b * c;
      if (_fvec != nullptr)
      {
        (*_fvec)(i_eq) = a - d;
        (*_fvec)(i_eq + 1) = b + c;
        (*_fvec)(i_eq + 2) = std::log(det);
      }
      if (_fjac == nullptr)
        continue;
      Eigen::Matrix<double, 3, 4> dfa;
      dfa << 1,        0,        0,      -1,
             0,        1,        1,       0,
             d / det, -c / det, -b / det, a / det;

      Eigen::Matrix<double, 4, 6> da_uv;
      auto c0 = 1. / fd.loc_tri_[0][0]; // 1 / x_1
      auto c1 = 1. / fd.loc_tri_[1][1]; // 1 / y_2
      auto c2 = fd.loc_tri_[1][0] / (fd.loc_tri_[0][0] * fd.loc_tri_[1][1]); // x_2 / (x_1 * y_2)
      da_uv <<
        -c0, 0, c0, 0, 0, 0,
        c2 - c1, 0, -c2, 0, c1, 0,
        0, -c0, 0, c0, 0, 0,
        0, c2 - c1, 0, -c2, 0, c1;
      Eigen::Matrix<double, 3, 6> df_uv = dfa * da_uv;
      for (size_t i = 0; i < 3; ++i)
      {
        for (size_t j = 0; j < 6; ++j)
        {
          auto val = df_uv(i, j);
          if (val != 0)
          {
            auto c = fd.idx_[j / 2] + j % 2;
            if (c < inputs())
              _fjac->coeffRef(i_eq + i, c) = val;
          }
        }
      }
    }
    if (_fjac != nullptr) _fjac->makeCompressed();
    if (_fvec != nullptr)
    {
      double sum = 0;
      for (int i = 0; i < _fvec->rows(); ++i)
        sum += (*_fvec)(i) * (*_fvec)(i);
      std::cout << "F: " << sum << "\n\n";
    }
#ifdef PRINT
    if (_fvec != nullptr)
      std::cout << "F: " << *_fvec << "\n\n";
    if (_fjac != nullptr)
      std::cout << "df: " << *_fjac << "\n\n";
#endif
    return 0;
  }
};


struct EnergySquare : public  IFunctionXXX, EnergyFunction
{
  using EnergyFunction::EnergyFunction;

  EnergySquare(const VertexIndMpap& _vrt_inds,
    Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf, int _m, int _n) :
    EnergyFunction(_vrt_inds, _bf, _m, _n)
  {
    f_val_.resize(values(), 1);
    jac_.resize(values(), inputs());
  }

  bool valuate(const Eigen::VectorXd& _x, double* _f, Eigen::VectorXd* _df) const
  {
    compute(_x, &f_val_, &jac_);
    if (_f != nullptr)
      *_f = (f_val_.transpose() * f_val_)(0, 0);
    if (_df != nullptr)
      *_df = 2 * (f_val_.transpose() * jac_);
    return true;
  }

  mutable Eigen::VectorXd f_val_;
  mutable Eigen::SparseMatrix<double> jac_;

};

struct OptimizeNonLinear : public IFunction
{
  OptimizeNonLinear(const VertexIndMpap& _vrt_inds, Topo::Wrap<Topo::Type::BODY> _body):
    vrt_inds_(_vrt_inds), bf_(_body) { }

  const double* compute(Eigen::VectorXd& _X)
  {
    if (!q_solver_)
      q_solver_ = IQuadraticSolver::make();
    rows_ = 3 * bf_.size();
    cols_ = _X.size();
    q_solver_->init(rows_, cols_, _X.data());
    q_solver_->compute(*this);
    return q_solver_->get_x();
  }

  bool operator()(const double* _x, double* _f, double* _fj) const override
  {
    const size_t var_nmbr = vrt_inds_.size() * 2;
    auto INDEX = [var_nmbr](size_t _i, size_t _j) { return _i * var_nmbr + _j; };
    size_t i_eq_loop = 0;
    if (_fj != nullptr)
      std::fill_n(_fj, rows_ * cols_, 0.);
    for (auto face : bf_)
    {
      // 3 equations per face.
      size_t i_eq = i_eq_loop;
      i_eq_loop += 3;
      Topo::Iterator<Topo::Type::FACE, Topo::Type::VERTEX> fv(face);
      size_t idx[3];
      Geo::VectorD3 pts[3];
      size_t i = 0;
      for (auto v : fv)
      {
        v->geom(pts[i]);
        idx[i++] = 2 * vrt_inds_.find(v)->second;
      }
      Geo::VectorD2 loc_tri[2];
      auto area2 = move_to_local_coord(pts, loc_tri);
      auto u10 = _x[idx[1]] - _x[idx[0]];
      auto u20 = _x[idx[2]] - _x[idx[0]];
      auto v10 = _x[idx[1] + 1] - _x[idx[0] + 1];
      auto v20 = _x[idx[2] + 1] - _x[idx[0] + 1];
      double a = u10 / loc_tri[0][0];
      double b = (u20 - loc_tri[1][0] * u10 / loc_tri[0][0]) / loc_tri[1][1];
      double c = v10 / loc_tri[0][0];
      double d = (v20 - loc_tri[1][0] * v10 / loc_tri[0][0]) / loc_tri[1][1];
      auto det = a * d - b * c;
      _f[i_eq] = a - d;
      _f[i_eq + 1] = b + c;
      _f[i_eq + 2] = std::log(det);
      std::cout << "F: " << _f[i_eq] << _f[i_eq + 1] << _f[i_eq + 2] << std::endl;
      if (_fj == nullptr)
        continue;
      Eigen::Matrix<double, 3, 4> dfa;
      dfa <<       1,        0,        0,      -1,
                   0,        1,        1,       0,
             d / det, -c / det, -b / det, a / det;

      Eigen::Matrix<double, 4, 6> da_uv;
      auto c0 = 1. / loc_tri[0][0]; // 1 / x_1
      auto c1 = 1. / loc_tri[1][1]; // 1 / y_2
      auto c2 = loc_tri[1][0] / (loc_tri[0][0] * loc_tri[1][1]); // x_2 / (x_1 * y_2)
      da_uv << 
           -c0,      0,   c0,    0,     0,  0,
         c2-c1,      0,  -c2,    0,    c1,  0,
             0,    -c0,    0,   c0,     0,  0,
             0,  c2-c1,    0,  -c2,     0, c1;
      Eigen::Matrix<double, 3, 6> df_uv = dfa * da_uv;
      std::cout << df_uv << std::endl << std::endl;
      for (size_t i = 0; i < 3; ++i)
      {
        for (size_t j = 0; j < 6; ++j)
        {
          auto val = df_uv(i, j);
          if (val != 0)
            _fj[INDEX(i_eq + i, idx[j / 2] + j % 2)] = val;
        }
      }
    }
    return true;
  }

  const VertexIndMpap& vrt_inds_;
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf_;
  size_t rows_, cols_;
  std::unique_ptr<IQuadraticSolver> q_solver_;
};

// Function to minimize 



void flatten(Topo::Wrap<Topo::Type::BODY> _body)
{
  std::map<Topo::Wrap<Topo::Type::VERTEX>, size_t> vrt_inds;
  using Triplet = Eigen::Triplet<double, size_t>;
  using Matrix = Eigen::SparseMatrix<double>;
  std::vector<Triplet> coffs;
  size_t rows = 0, pt_nmbr = 0;
  Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE> bf(_body);
  for (auto face : bf)
  {
    std::array<Geo::VectorD2, 3> w;
    Topo::Iterator<Topo::Type::FACE, Topo::Type::VERTEX> fv(face);
    Geo::VectorD3 pts[3];
    size_t idx[3];
    size_t i = 0;

    for (auto v : fv)
    {
      v->geom(pts[i]);
      auto pos = vrt_inds.emplace(v, pt_nmbr);
      idx[i] = pos.first->second;
      if (pos.second)
        ++pt_nmbr;
      ++i;
    }
    Geo::VectorD2 loc_tri[2];
    auto area_time_2_sqr = sqrt(move_to_local_coord(pts, loc_tri));
    w[0] = (loc_tri[1] - loc_tri[0]) / area_time_2_sqr;
    w[1] =  -loc_tri[1] / area_time_2_sqr;
    w[2] = loc_tri[0] / area_time_2_sqr;

    for (size_t j = 0; j < 3; ++j)
    {
      auto idx_base = idx[j] * 2;
      coffs.emplace_back(rows, idx_base, w[j][0]);
      coffs.emplace_back(rows, idx_base + 1, -w[j][1]);
      coffs.emplace_back(rows + 1, idx_base, w[j][1]);
      coffs.emplace_back(rows + 1, idx_base + 1, w[j][0]);
    }
    rows += 2;
  }
  const auto cols = 2 * pt_nmbr;
  Matrix M(rows, cols);
  M.setFromTriplets(coffs.begin(), coffs.end());
  const auto FIXED_VAR = 4;
  auto split_idx = cols - FIXED_VAR;
  auto A = M.block(0, 0, rows, split_idx);
  auto B = M.block(0, split_idx, rows, FIXED_VAR);
  Eigen::Vector4d fixed;
  fixed(0, 0) = -100;
  fixed(1) = fixed(2) = fixed(3) = 0;
  
  auto b = B * fixed;
  //Eigen::SparseQR <Matrix, Eigen::COLAMDOrdering<int>> solver;
  Eigen::LeastSquaresConjugateGradient<Matrix> solver;
  solver.compute(A);
  Eigen::VectorXd X = solver.solve(b);
  X.conservativeResize(cols);
  for (size_t i = 0; i < 4; ++i)
    X(split_idx + i, 0) = -fixed(i, 0);

  auto set_x = [&](const double* _x)
  {
    for (const auto& v_id : vrt_inds)
    {
      auto base_ind = 2 * v_id.second;
      Geo::VectorD3 pt;
      pt = { _x[base_ind], _x[base_ind + 1], 0 };
      const_cast<Topo::Wrap<Topo::Type::VERTEX>&>(v_id.first)->set_geom(pt);
    }
  };

  static bool conformal = false;
  if (!conformal)
  {
    int unknowns = cols - 3;
    X.conservativeResize(unknowns);

#if 1
    EnergySquare es(vrt_inds, bf, 3 * bf.size(), unknowns);
    minimize(X, es);
#else
    EnergyFunction ef(vrt_inds, bf, 3 * bf.size(), unknowns);
    Eigen::LevenbergMarquardt_xxx lm(ef);
    auto info = lm.lmder1(X);
#endif

    X.conservativeResize(cols);
    X(cols - 1) = X(cols - 2) = X(cols - 3) = 0;

#if 0
    OptimizeNonLinear onl(vrt_inds, _body);
    const double* x = onl.compute(X);
    REQUIRE(x != nullptr);
    set_x(x);
#endif
  }
  set_x(X.data());
}

TEST_CASE("flat_00", "[Flattening]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa0.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb0.obj", body);
}

TEST_CASE("flat_01", "[Flattening]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa1.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb1.obj", body);
}

TEST_CASE("flat_02", "[Flattening]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa2.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb2.obj", body);
}

TEST_CASE("flat_03", "[Flattening]")
{
  auto body = IO::load_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/aaa3.obj");
  flatten(body);
  IO::save_obj("C:/Users/USER/source/repos/ml_4_mesh/src/Test/Data/bbb3.obj", body);
}
