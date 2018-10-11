
#include "optimize_function.hxx"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <functional>

#include "mkl_rci.h"
#include "mkl_types.h"
#include "mkl_service.h"

namespace {

struct mkl_array_wrap
{
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
  ~mkl_array_wrap() { clean(); }
  double* operator()() { return arr_; }

private:
  void clean()
  {
    if (arr_ != nullptr)
      mkl_free(arr_);
  }
  void resize(MKL_INT _size)
  {
    clean();
    if (arr_ != nullptr)
      mkl_free(arr_);
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
    result_ = dtrnlsp_init(&handle_, &var_nmbr_, &equat_nmbr_, _x, eps_, &iter1_, &iter2_, &rs_);
    if (result_ != TR_SUCCESS)
      return false;

    MKL_INT info[6];
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
    return true;
  }

  const double* get_x()  override { return x_(); }

  bool compute(const IFunction& _mat_functon) override
  {
    MKL_INT RCI_Request = 0;
    MKL_INT successful = 0;
    _mat_functon(x_(), fvec_(), fjac_());
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

private:
  _TRNSP_HANDLE_t handle_;

  MKL_INT var_nmbr_;
  MKL_INT equat_nmbr_;

  const double eps_[6] = { 0.00001,0.00001,0.00001,0.00001,0.00001,0.00001 }; /* set precisions for stop-criteria */
  const MKL_INT iter1_ = 1000; // precisions for stop-criteria
  const MKL_INT iter2_ = 100;  // maximum number of iterations of calculation of trial-step
  const double rs_ = 0.0;      // initial step bound

  mkl_array_wrap x_;    // variable array
  mkl_array_wrap fvec_; // function (f(x)) value vector
  mkl_array_wrap fjac_; // jacobi matrix

  MKL_INT result_;
};

} // namespace

std::unique_ptr<IQuadraticSolver> IQuadraticSolver::make()
{
  return std::make_unique<QuadraticSolver>();
}


#if 0

/* nonlinear least square problem without boundary constraints */
int solve_non_linear_function(size_t _cols, size_t _rows)
{
  /* user's objective function */
  /* n - number of function variables
     m - dimension of function value */
  const MKL_INT n = 4;
  const MKL_INT m = 4;
  /* precisions for stop-criteria (see manual for more details) */
  const double eps[6] = { 0.00001,0.00001,0.00001,0.00001,0.00001,0.00001 }; /* set precisions for stop-criteria */
  /* precision of the Jacobian matrix calculation */
  double jac_eps;
  /* solution vector. contains values x for f(x) */
  double *x = NULL;
  /* iter1 - maximum number of iterations
     iter2 - maximum number of iterations of calculation of trial-step */
  const MKL_INT iter1 = 1000;
  const MKL_INT iter2 = 100;
  /* initial step bound */
  double rs = 0.0;
  /* reverse communication interface parameter */
  MKL_INT RCI_Request;      // reverse communication interface variable
  MKL_INT successful;  /* controls of rci cycle */
  double *fvec = NULL; /* function (f(x)) value vector */
  double *fjac = NULL; /* jacobi matrix */
  MKL_INT iter;        /* number of iterations */
  /* number of stop-criterion */
  MKL_INT st_cr;
  /* initial and final residuals */
  double r1, r2;
  /* TR solver handle */
  _TRNSP_HANDLE_t handle;   // TR solver handle
  /* cycle's counter */
  MKL_INT i;
  /* results of input parameter checking */
  MKL_INT info[6];
  /* memory allocation flags */
  MKL_INT mem_error, error;

  error = 0;
  /* memory allocation */
  mem_error = 1;
  x = (double *)mkl_malloc(sizeof(double) * n, 64);
  if (x == NULL) goto end;
  fvec = (double *)mkl_malloc(sizeof(double) * m, 64);
  if (fvec == NULL) goto end;
  fjac = (double *)mkl_malloc(sizeof(double) * m * n, 64);
  if (fjac == NULL) goto end;
  /* memory allocated correctly */
  mem_error = 0;
  /* set precision of the Jacobian matrix calculation */
  jac_eps = 0.00000001;
  /* set the initial guess */
  for (i = 0; i < n / 4; i++)
  {
    x[4 * i] = 3.0;
    x[4 * i + 1] = -1.0;
    x[4 * i + 2] = 0.0;
    x[4 * i + 3] = 1.0;
  }
  /* set initial values */
  for (i = 0; i < m; i++)
    fvec[i] = 0.0;
  for (i = 0; i < m * n; i++)
    fjac[i] = 0.0;
  /* initialize solver (allocate memory, set initial values)
     handle       in/out: TR solver handle
     n       in:     number of function variables
     m       in:     dimension of function value
     x       in:     solution vector. contains values x for f(x)
     eps     in:     precisions for stop-criteria
     iter1   in:     maximum number of iterations
     iter2   in:     maximum number of iterations of calculation of trial-step
     rs      in:     initial step bound */
  if (dtrnlsp_init(&handle, &n, &m, x, eps, &iter1, &iter2, &rs) != TR_SUCCESS)
  {
    /* if function does not complete successfully then print error message */
    printf("| error in dtrnlsp_init\n");
    /* Release internal Intel(R) MKL memory that might be used for computations.        */
    /* NOTE: It is important to call the routine below to avoid memory leaks   */
    /* unless you disable Intel(R) MKL Memory Manager                                   */
    MKL_Free_Buffers();
    /* and exit */
    error = 1;
    goto end;
  }
  /* Checks the correctness of handle and arrays containing Jacobian matrix,
     objective function, lower and upper bounds, and stopping criteria. */
  if (dtrnlsp_check(&handle, &n, &m, fjac, fvec, eps, info) != TR_SUCCESS)
  {
    /* if function does not complete successfully then print error message */
    printf("| error in dtrnlspbc_init\n");
    /* Release internal Intel(R) MKL memory that might be used for computations.        */
    /* NOTE: It is important to call the routine below to avoid memory leaks   */
    /* unless you disable Intel(R) MKL Memory Manager                                   */
    MKL_Free_Buffers();
    /* and exit */
    error = 1;
    goto end;
  }
  else
  {
    if (info[0] != 0 || // The handle is not valid.
      info[1] != 0 || // The fjac array is not valid.
      info[2] != 0 || // The fvec array is not valid.
      info[3] != 0    // The eps array is not valid.
      )
    {
      printf("| input parameters for dtrnlsp_solve are not valid\n");
      /* Release internal Intel(R) MKL memory that might be used for computations.        */
      /* NOTE: It is important to call the routine below to avoid memory leaks   */
      /* unless you disable Intel(R) MKL Memory Manager                                   */
      MKL_Free_Buffers();
      /* and exit */
      error = 1;
      goto end;
    }
  }
  /* set initial rci cycle variables */
  RCI_Request = 0;
  successful = 0;
  /* rci cycle */
  while (successful == 0)
  {
    /* call tr solver
       handle               in/out: tr solver handle
       fvec         in:     vector
       fjac         in:     jacobi matrix
       RCI_request in/out:  return number which denote next step for performing */
    if (dtrnlsp_solve(&handle, fvec, fjac, &RCI_Request) != TR_SUCCESS)
    {
      /* if function does not complete successfully then print error message */
      printf("| error in dtrnlsp_solve\n");
      /* Release internal Intel(R) MKL memory that might be used for computations.        */
      /* NOTE: It is important to call the routine below to avoid memory leaks   */
      /* unless you disable Intel(R) MKL Memory Manager                                   */
      MKL_Free_Buffers();
      /* and exit */
      error = 1;
      goto end;
    }
    /* according with rci_request value we do next step */
    if (RCI_Request == -1 ||
      RCI_Request == -2 ||
      RCI_Request == -3 ||
      RCI_Request == -4 || RCI_Request == -5 || RCI_Request == -6)
      /* exit rci cycle */
      successful = 1;
    if (RCI_Request == 1)
    {
      /* recalculate function value
         m            in:     dimension of function value
         n            in:     number of function variables
         x            in:     solution vector
         fvec    out:    function value f(x) */
      extended_powell(&m, &n, x, fvec);
    }
    if (RCI_Request == 2)
    {
      /* compute jacobi matrix
         extended_powell      in:     external objective function
         n               in:     number of function variables
         m               in:     dimension of function value
         fjac            out:    jacobi matrix
         x               in:     solution vector
         jac_eps         in:     jacobi calculation precision */
      if (djacobi(extended_powell, &n, &m, fjac, x, &jac_eps) != TR_SUCCESS)
      {
        /* if function does not complete successfully then print error message */
        printf("| error in djacobi\n");
        /* Release internal Intel(R) MKL memory that might be used for computations.        */
        /* NOTE: It is important to call the routine below to avoid memory leaks   */
        /* unless you disable Intel(R) MKL Memory Manager                                   */
        MKL_Free_Buffers();
        /* and exit */
        error = 1;
        goto end;
      }
    }
  }
  /* get solution statuses
     handle            in:        TR solver handle
     iter              out:       number of iterations
     st_cr             out:       number of stop criterion
     r1                out:       initial residuals
     r2                out:       final residuals */
  if (dtrnlsp_get(&handle, &iter, &st_cr, &r1, &r2) != TR_SUCCESS)
  {
    /* if function does not complete successfully then print error message */
    printf("| error in dtrnlsp_get\n");
    /* Release internal Intel(R) MKL memory that might be used for computations.        */
    /* NOTE: It is important to call the routine below to avoid memory leaks   */
    /* unless you disable Intel(R) MKL Memory Manager                                   */
    MKL_Free_Buffers();
    /* and exit */
    error = 1;
    goto end;
  }
  /* free handle memory */
  if (dtrnlsp_delete(&handle) != TR_SUCCESS)
  {
    /* if function does not complete successfully then print error message */
    printf("| error in dtrnlsp_delete\n");
    /* Release internal Intel(R) MKL memory that might be used for computations.        */
    /* NOTE: It is important to call the routine below to avoid memory leaks   */
    /* unless you disable Intel(R) MKL Memory Manager                                   */
    MKL_Free_Buffers();
    /* and exit */
    error = 1;
    goto end;
  }
  /* free allocated memory */
end:
  mkl_free(fjac);
  mkl_free(fvec);
  mkl_free(x);
  if (error != 0)
  {
    return 1;
  }
  if (mem_error == 1)
  {
    printf("| insufficient memory \n");
    return 1;
  }
  /* Release internal Intel(R) MKL memory that might be used for computations.        */
  /* NOTE: It is important to call the routine below to avoid memory leaks   */
  /* unless you disable Intel(R) MKL Memory Manager                                   */
  MKL_Free_Buffers();
  /* if final residual less then required precision then print pass */
  if (r2 < 0.00001)
  {
    printf("|         dtrnlsp Powell............PASS\n");
    return 0;
  }
  /* else print failed */
  else
  {
    printf("|         dtrnlsp Powell............FAILED\n");
    return 1;
  }
}

/* nonlinear system equations without constraints */
/* routine for extended Powell function calculation
   m     in:     dimension of function value
   n     in:     number of function variables
   x     in:     vector for function calculating
   f     out:    function value f(x) */
void extended_powell(MKL_INT * m, MKL_INT * n, double *x, double *f)
{
  MKL_INT i;

  for (i = 0; i < (*n) / 4; i++)
  {
    f[4 * i] = x[4 * i] + 10.0 * x[4 * i + 1];
    f[4 * i + 1] = 2.2360679774998 * (x[4 * i + 2] - x[4 * i + 3]);
    f[4 * i + 2] = (x[4 * i + 1] - 2.0 * x[4 * i + 2]) *
      (x[4 * i + 1] - 2.0 * x[4 * i + 2]);
    f[4 * i + 3] = 3.1622776601684 * (x[4 * i] - x[4 * i + 3]) *
      (x[4 * i] - x[4 * i + 3]);
  }
  return;
}
#endif

