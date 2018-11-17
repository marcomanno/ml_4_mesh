#include "catch.hpp"

#include <unsupported/Eigen/LevenbergMarquardt>

struct lmder_functor : public Eigen::DenseFunctor<double>
{
  lmder_functor(void) : Eigen::DenseFunctor<double>(3, 15) {}
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
  {
    double tmp1, tmp2, tmp3;
    static const double y[15] = { 1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
        3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39 };

    for (int i = 0; i < values(); i++)
    {
      tmp1 = i + 1;
      tmp2 = 16 - i - 1;
      tmp3 = (i >= 8) ? tmp2 : tmp1;
      fvec[i] = y[i] - (x[0] + tmp1 / (x[1] * tmp2 + x[2] * tmp3));
    }
    return 0;
  }

  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) const
  {
    double tmp1, tmp2, tmp3, tmp4;
    for (int i = 0; i < values(); i++)
    {
      tmp1 = i + 1;
      tmp2 = 16 - i - 1;
      tmp3 = (i >= 8) ? tmp2 : tmp1;
      tmp4 = (x[1] * tmp2 + x[2] * tmp3); tmp4 = tmp4 * tmp4;
      fjac(i, 0) = -1;
      fjac(i, 1) = tmp1 * tmp2 / tmp4;
      fjac(i, 2) = tmp1 * tmp3 / tmp4;
    }
    return 0;
  }
};


TEST_CASE("OPT_basci", "[NLOPT]")
{
  int n = 3, info;

  Eigen::VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  lmder_functor functor;
  Eigen::LevenbergMarquardt<lmder_functor> lm(functor);
  info = lm.lmder1(x);
}
