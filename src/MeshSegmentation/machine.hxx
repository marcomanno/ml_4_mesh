#pragma once


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
  virtual tensorflow::Input add_weight(int _m, int _n) = 0;
  virtual tensorflow::Input add_layer(
    tensorflow::Input& _X,
    tensorflow::Input& _A,
    tensorflow::Input& _B) = 0;
  virtual tensorflow::Input set_targets(tensorflow::Input& _layer) = 0;
  virtual void train(const std::vector<RealT>& _in,
                     const std::vector<RealT>& _out) = 0;
  virtual void predict(
    const std::vector<RealT>& _in,
    std::vector<RealT> &_out) = 0;
  virtual void save(const char* _flnm) = 0;
  virtual void load(const char* _flnm) = 0;
};
} // namespace ML
