
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <functional>
#include <memory>

#include <Eigen/Dense>

struct IFunction
{
  virtual bool operator()(const double* _x, double* _f, double* _fj) const = 0;
};

class IQuadraticSolver
{
public:
  static std::unique_ptr<IQuadraticSolver> make();
  virtual ~IQuadraticSolver() {}
  virtual bool init(size_t _rows, size_t _cols, double* _x) = 0;
  virtual bool compute(const IFunction& _mat_functon) = 0;
  virtual bool get_result_info(size_t& _iter_nmbr, size_t& _stop_crit,
    double& residual_0, double& residual_1) = 0;
  virtual const double* get_x() const = 0;
};

struct IFunctionXXX
{
  virtual bool valuate(const Eigen::VectorXd& _x, double* _f, Eigen::VectorXd* _df) const = 0;
};

void minimize(Eigen::VectorXd& _x, const IFunctionXXX& _func);

