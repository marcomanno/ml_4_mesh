
#include "optimize_function.hxx"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <functional>

#include "mkl_rci.h"
#include "mkl_types.h"
#include "mkl_service.h"

namespace {

struct wrap_mkl_array
{
  ~wrap_mkl_array() { clean(); }
  bool init(MKL_INT _size)
  {
    resize(_size);
    std::fill_n(arr_, _size, 0);
    return arr_ != nullptr;
  }
  bool init(MKL_INT _size, double* _data)
  {
    resize(_size);
    std::copy_n(_data, _size, arr_);
    return arr_ != nullptr;
  }
  double* operator()() const { return arr_; }

private:
  void clean()
  {
    if (arr_ != nullptr)
      mkl_free(arr_);
  }
  void resize(MKL_INT _size)
  {
    clean();
    arr_ = (double *)mkl_malloc(sizeof(double) * _size, 64);
  }
  double* arr_ = nullptr;
};

class QuadraticSolver : public IQuadraticSolver
{
public:
  ~QuadraticSolver() override
  {
    if (result_ == TR_SUCCESS)
      result_ = dtrnlsp_delete(&handle_);
    if (result_ != TR_SUCCESS)
      MKL_Free_Buffers();
  }

  bool init(size_t _rows, size_t _cols, double* _x) override
  {
    var_nmbr_ = _cols;
    equat_nmbr_ = _rows;
    if (!x_.init(var_nmbr_, _x) || !fvec_.init(equat_nmbr_) || !fjac_.init(var_nmbr_ * equat_nmbr_))
    {
      result_ = 1;
      return false;
    }
    eps_[0] = 0.0000001;    //  Δ < eps[0]
    eps_[1] = 0.0000001;    // ||F(x)||2 < eps[1]
    eps_[2] = 0.0000001;   // The Jacobian matrix is singular
    eps_[3] = 0.0000001;   // || s || 2 < eps[3]
    eps_[4] = 0.0000001;   // || F(x) || 2 - || F(x) - J(x)s || 2 < eps[4]
    eps_[5] = 0.0000001; // The trial step precision.If eps[5] = 0, then the trial step 
                         // meets the required precision(≤ 1.0 * 10 - 10)
    std::fill_n(eps_, 6, 1.e-7);
    result_ = dtrnlsp_init(&handle_, &var_nmbr_, &equat_nmbr_, x_(), eps_, &iter1_, &iter2_, &rs_);
    if (result_ != TR_SUCCESS)
      return false;
#if 0
    MKL_INT info[6] = {};
    result_ = dtrnlsp_check(&handle_, &var_nmbr_, &equat_nmbr_, fjac_(), fvec_(), eps_, info);
    if (result_ != TR_SUCCESS)
      return false;

    if (info[0] != 0 || // The handle is not valid.
      info[1] != 0 || // The fjac array is not valid.
      info[2] != 0 || // The fvec array is not valid.
      info[3] != 0)  // The eps array is not valid.
    {
      result_ = 1;
      return false;
    }
#endif
    return true;
  }

  bool compute(const IFunction& _mat_functon) override
  {
    MKL_INT RCI_Request = 0;
    MKL_INT successful = 0;
    /* rci cycle */
    while (successful == 0)
    {
      /* call tr solver
         handle               in/out: tr solver handle
         fvec         in:     vector
         fjac         in:     jacobi matrix
         RCI_request in/out:  return number which denote next step for performing */
      result_ = dtrnlsp_solve(&handle_, fvec_(), fjac_(), &RCI_Request);
      if (result_ != TR_SUCCESS)
        return false;
      if (RCI_Request < 0)
        break;

      if (RCI_Request > 0)
      {
        double* f_jac = RCI_Request == 1 ? nullptr : fjac_();
        _mat_functon(x_(), fvec_(), f_jac);
      }
    }
    return true;
  }

  bool get_result_info(size_t& _iter_nmbr, size_t& _stop_crit,
    double& residual_0, double& residual_1)  override
  {
    if (result_ == TR_SUCCESS)
    {
      MKL_INT iter_nmbr, stop_crit;
      result_ = dtrnlsp_get(&handle_, &iter_nmbr, &stop_crit, &residual_0, &residual_1);
      _iter_nmbr = iter_nmbr;
      _stop_crit = stop_crit;
    }
    return  result_ == TR_SUCCESS;
  }

  const double* get_x() const override
  {
    if (result_ == TR_SUCCESS)
      return x_();
    else
      return nullptr;
  }

private:
  _TRNSP_HANDLE_t handle_;

  MKL_INT var_nmbr_;
  MKL_INT equat_nmbr_;

  const MKL_INT iter1_ = 1000; // precisions for stop-criteria
  const MKL_INT iter2_ = 100;  // maximum number of iterations of calculation of trial-step
  const double rs_ = 0.0;      // initial step bound
  double eps_[6]; /* set precisions for stop-criteria */

  wrap_mkl_array x_;    // variable array
  wrap_mkl_array fvec_; // function (f(x)) value vector
  wrap_mkl_array fjac_; // jacobi matrix

  MKL_INT result_;
};

} // namespace

std::unique_ptr<IQuadraticSolver> IQuadraticSolver::make()
{
  return std::make_unique<QuadraticSolver>();
}

#include "cppoptlib/meta.h"
#include "cppoptlib/problem.h"
#include "cppoptlib/solver/bfgssolver.h"

void minimize(Eigen::VectorXd& _x, const IFunctionXXX& _func)
{
  using Problem = cppoptlib::Problem<double>;
  struct FunctionWrap : public Problem
  {
    FunctionWrap(const IFunctionXXX& _func) : f_(_func) {}
    using typename Problem::TVector;

    // this is just the objective (NOT optional)
    double value(const TVector& _x)
    {
      double val;
      f_.valuate(_x, &val, nullptr);
      return val;
    }

    // if you calculated the derivative by hand
    // you can implement it here (OPTIONAL)
    // otherwise it will fall back to (bad) numerical finite differences
    void gradient(const TVector &_x, TVector &_grad)
    {
      f_.valuate(_x, nullptr, &_grad);
    }
    const IFunctionXXX& f_;
  };

  cppoptlib::BfgsSolver<FunctionWrap> solver;
  solver.minimize(FunctionWrap(_func), _x);

}

