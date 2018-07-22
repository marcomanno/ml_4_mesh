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
struct IMachine
{
  static std::shared_ptr<IMachine> make();
  virtual tensorflow::Input make_input(int _rows) = 0;
  virtual tensorflow::Input make_output(int _rows) = 0;
  virtual tensorflow::Input add_weight(int _m, int _n) = 0;
  virtual tensorflow::ops::Tanh add_layer(
    tensorflow::Input& _X,
    tensorflow::Input& _A,
    tensorflow::Input& _B) = 0;
  virtual tensorflow::Input set_targets(tensorflow::Input& _layer) = 0;
  virtual void train(const std::vector<double>& _in,
                     const std::vector<double>& _out) = 0;
  virtual void predict(
    const std::vector<double>& _in, 
    std::vector<double> &_out) = 0;
  virtual void store(const char* _flnm) = 0;
};
} // namespace ML

