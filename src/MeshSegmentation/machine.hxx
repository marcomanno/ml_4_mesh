#pragma once

#include "tensorflow/include/tensorflow/cc/framework/ops.h"

#include <memory>
#include <vector>

namespace tensorflow 
{
class Input;
namespace ops
{
class Placeholder;
class Variable;
class Add;
class Tanh;
}
}

namespace ML
{
template <typename RealT>
struct IMachine
{
  static std::shared_ptr<IMachine<RealT>> make();
  virtual tensorflow::Input make_input(int _rows) = 0;
  virtual tensorflow::Input make_output(int _rows) = 0;
  virtual tensorflow::Input add_weight(int _m, int _n, const RealT& _init_val = 0) = 0;
  virtual tensorflow::Output add_layer(
    tensorflow::Input& _X,
    tensorflow::Input& _A,
    tensorflow::Input& _B) = 0;
  virtual tensorflow::Input set_target(tensorflow::Output& _layer,
    const RealT& _grad_coeff = 1.e-5, const RealT& _reg_coeff = 0) = 0;
  virtual void train(const std::vector<RealT>& _in,
                     const std::vector<RealT>& _out,
                     int _iterations = 10000) = 0;
  virtual void predict1(const std::vector<RealT>& _in, std::vector<RealT> &_out) = 0;
  virtual void predictN(const std::vector<RealT>& _in, std::vector<RealT> &_out) = 0;
  virtual void save(const char* _flnm) = 0;
  virtual void load(const char* _flnm) = 0;
};
} // namespace ML

