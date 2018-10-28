#pragma once

#include <Eigen/dense>
#include <Eigen/sparse>
#include <mkl.h>
#include <memory>

namespace LM
{
template<typename valT> static valT square(const valT& _val)
{
  return _val * _val;
}

template<typename valT> static valT cube(const valT& _val)
{
  return _val * _val * _val;
}


struct IMultiFunction
{
  virtual bool evaluate(const Eigen::VectorXd& _x, Eigen::VectorXd& _f) const = 0;
  virtual bool jacobian(const Eigen::VectorXd& _x, Eigen::SparseMatrix<double>& _fj) const = 0;
  virtual MKL_INT rows() const = 0; // Number of equations
  virtual MKL_INT cols() const = 0; // Number of unknown
};

struct ISparseLM
{
  virtual ~ISparseLM() {}
  virtual bool compute(Eigen::VectorXd& _x, MKL_INT _max_iterations) = 0;
  static std::unique_ptr<ISparseLM> make(IMultiFunction& _fun);
};

} // namespace LM
