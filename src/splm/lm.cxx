#include "lm.hxx"

#include <Eigen/PardisoSupport>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

namespace LM
{
struct SparseLM : public ISparseLM
{
  enum class Stop {
    CONT,
    RESIDUAL_ZERO,
    JAC_ZERO,
    ZERO_DX,
    BAD_NUMBER,
    SINGULAR,
    ANOVERFLOW
  };
  SparseLM(const IMultiFunction& _fun) : fun_(_fun), m_(_fun.rows()), n_(_fun.cols()) {}
  bool compute(ColumnVector& _x, MKL_INT _max_iterations) override;
  void compute_internal(ColumnVector& _x, MKL_INT _max_iterations);
private:
  const IMultiFunction& fun_;
  MKL_INT m_, n_;
};


std::unique_ptr<ISparseLM> ISparseLM::make(IMultiFunction& _fun)
{
  return std::make_unique<SparseLM>(_fun);
}

#define SPLM_EPSILON       1E-12
#define SPLM_EPSILON_SQ    ( (SPLM_EPSILON)*(SPLM_EPSILON) )

bool SparseLM::compute(ColumnVector& _x, MKL_INT _max_iterations)
{
  try
  {
    compute_internal(_x, _max_iterations);
  }
  catch (const Stop& _stop)
  {
    return _stop < Stop::BAD_NUMBER;
  }
  return false;
}


void SparseLM::compute_internal(ColumnVector& _x, MKL_INT _max_iterations)
{
  double tau = 0.5;
  double eps1 = 1e-12;
  double eps2 = 1e-5;
  double eps2_sq = square(eps2);
  double eps3 = 1e-12;
  double delta = 1e-6;

  ColumnVector dx;
  ColumnVector f_val(m_);
  MKL_INT nu = 2;

  fun_.evaluate(_x, f_val);
  auto p_eL2 = f_val.squaredNorm();
  Stop stop = Stop::CONT;
  std::cout << "Err=" << p_eL2 << std::endl;
  if (std::isnan(p_eL2) || std::isinf(p_eL2))
    stop = Stop::BAD_NUMBER;
  Matrix jac(m_, n_);
  auto init_p_eL2 = p_eL2;
  double mu = 0; // damping constant
  for (MKL_INT iter = 0; iter < _max_iterations; ++iter)
  {
    if (p_eL2 <= eps3)
    {
      stop = Stop::RESIDUAL_ZERO;
      break;
    }
    jac.setZero();
    fun_.jacobian(_x, jac);
    Matrix jacTjac = jac.transpose() * jac;
    ColumnVector jacTe = -jac.transpose() * f_val;
    auto jacTe_inf = jacTe.lpNorm<Eigen::Infinity>();

    if ((jacTe_inf <= eps1))
      throw Stop::JAC_ZERO;

    auto p_L2 = _x.squaredNorm();
    auto diagHess = jacTjac.diagonal();
    if (iter == 0)
    {
      auto tmp = std::max(diagHess.lpNorm<Eigen::Infinity>(), DBL_MIN);
      mu = tau * tmp;
    }
    /* determine increment using adaptive damping */
    /* NOTE: during the following loop, e might contain the error corresponding to *unaccepted* dampings! */
    auto muincr = mu;
    for (;;)
    {
      diagHess.array() += muincr;
#if 0
      Eigen::ConjugateGradient<Matrix, 
        Eigen::Lower | Eigen::Upper> cg;
      cg.compute(jacTjac);
      dx = cg.solve(jacTe);
#else
      Eigen::PardisoLDLT<Matrix> lsolver;
      lsolver.compute(jacTjac);
      dx = lsolver.solve(jacTe);
#endif
      auto dp_L2 = dx.squaredNorm();

      if (dp_L2 <= eps2_sq * p_L2)
        throw Stop::ZERO_DX;  // relative change in p is small, stop

      if (dp_L2 * SPLM_EPSILON_SQ >= (p_L2 + eps2))
        throw Stop::SINGULAR; // almost singular
      auto pdx = _x + dx;
      fun_.evaluate(pdx, f_val);
      auto pdp_eL2 = f_val.squaredNorm();
      std::cout << std::setprecision(17) << "Err=" << pdp_eL2 << std::endl;
      if (std::isnan(pdp_eL2) || std::isinf(pdp_eL2))
        throw Stop::BAD_NUMBER;

      auto dF = p_eL2 - pdp_eL2;
      if (dF > 0.0)
      {
        auto dL = dx.dot(mu * dx + jacTe);
        if (dL > 0.0)
        {
          // reduction in error, increment is accepted
          auto tmp = 2 * dF / dL - 1;
          tmp = 1.0 - cube(tmp);
          const double SPLM_ONE_THIRD = 1. / 3.;
          if (tmp <= SPLM_ONE_THIRD)
            tmp = SPLM_ONE_THIRD;
          mu *= tmp;
          nu = 2;
          _x = pdx;
          p_eL2 = pdp_eL2; /* update ||e||_2 */
          break;
        }
      }
      muincr = -mu;
      mu *= nu;
      muincr += mu; // muincr:=new_mu - old_mu
      auto nu2 = nu * 2;
      if (nu2 <= nu)
        throw Stop::ANOVERFLOW; // nu has wrapped around (overflown)
      nu = nu2;
    }
  }
}

} // namespace LM
