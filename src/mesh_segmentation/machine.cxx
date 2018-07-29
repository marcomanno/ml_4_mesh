#pragma optimize ("", off)
#include "machine.hxx"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/core/kernels/matmul_op.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/training_ops.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace ML
{
struct Machine : public IMachine
{
  Machine() : session_(scope_){}
  tensorflow::Input make_input(int _rows) override;
  tensorflow::Input make_output(int _rows) override;
  tensorflow::Input add_weight(int _m, int _n) override;
  tensorflow::Input add_layer(
    tensorflow::Input& _X,
    tensorflow::Input& _A,
    tensorflow::Input& _B) override;
  tensorflow::Input set_targets(tensorflow::Input& _layer) override;
  void train(const std::vector<double>& _in, const std::vector<double>& _out) override;

  void predict(const std::vector<double>& _in, std::vector<double> &_out) override;
  void store(const char* _flnm) override;
private:
  tensorflow::Scope scope_ = tensorflow::Scope::NewRootScope();
  tensorflow::ClientSession session_;

  std::shared_ptr<tensorflow::ops::Placeholder> x_, y_;
  tensorflow::int64 x_size_ = 0, y_size_ = 0;

  tensorflow::ops::Cast val_0_01_ = 
    tensorflow::ops::Cast(scope_, 0.01, tensorflow::DT_DOUBLE);

  tensorflow::OutputList weights_;
  tensorflow::Output loss_;
  std::vector<tensorflow::Output> apply_grad_;
  tensorflow::Output out_layer_;

  void add_place_holder(int _rows, std::shared_ptr<tensorflow::ops::Placeholder>& _obj);
  auto x_size() const { return x_size_; }
  auto y_size() const { return y_size_; }
};

std::shared_ptr<IMachine> IMachine::make()
{
  return std::make_unique<Machine>();
}

tensorflow::Input Machine::make_input(int _rows)
{
  add_place_holder(x_size_ = _rows, x_);
  return *x_;
}

tensorflow::Input Machine::make_output(int _rows)
{
  add_place_holder(y_size_ = _rows, y_);
  return *y_;
}

void Machine::add_place_holder(
  int _rows, std::shared_ptr<tensorflow::ops::Placeholder>& _obj)
{
  tensorflow::ops::Placeholder::Attrs attr;
  _obj.reset(new tensorflow::ops::Placeholder(
      scope_, 
      tensorflow::DT_DOUBLE,
      attr.Shape({ 1, _rows })));
}

tensorflow::Input Machine::add_weight(int _m, int _n)
{
  auto w = tensorflow::ops::Variable(scope_, { _m, _n }, tensorflow::DT_DOUBLE);
  auto assign = tensorflow::ops::Assign(
    scope_, w,
    tensorflow::ops::RandomNormal(scope_, { _m, _n }, tensorflow::DT_DOUBLE));
  TF_CHECK_OK(session_.Run({assign}, nullptr));
  weights_.push_back(w);
  return w;
}

tensorflow::Input Machine::add_layer(
  tensorflow::Input& _X,
  tensorflow::Input& _A,
  tensorflow::Input& _B)
{
  auto layer = tensorflow::ops::Tanh(
    scope_,
    tensorflow::ops::Add(scope_, tensorflow::ops::MatMul(scope_, _X, _A), _B));
  return layer;
}

tensorflow::Input Machine::set_targets(tensorflow::Input& _layer)
{
  // regularization
  tensorflow::OutputList l2_losses;
  for (auto& w : weights_)
    l2_losses.push_back(tensorflow::ops::L2Loss(scope_, w));
  auto regularization = tensorflow::ops::AddN(
    scope_, l2_losses);

  loss_ = tensorflow::ops::Add(
    scope_,
    tensorflow::ops::ReduceMean(
      scope_,
      tensorflow::ops::Square(
        scope_, 
        tensorflow::ops::Sub(scope_, _layer, *y_)), 
      { 0, 1 }),
    tensorflow::ops::Mul(scope_, val_0_01_, regularization));

  // add the gradients operations to the graph
  std::vector<tensorflow::Output> grad_outputs;
  tensorflow::AddSymbolicGradients(scope_, { loss_ }, weights_, &grad_outputs);
  for (int i = 0; i < std::size(weights_); ++i)
  {
    apply_grad_.push_back(tensorflow::ops::ApplyGradientDescent(
      scope_, weights_[i], val_0_01_, { grad_outputs[i] }));
  }
  out_layer_ =  tensorflow::Output(_layer.node());
  return loss_;
}

void Machine::train(const std::vector<double>& _in, const std::vector<double>& _out)
{
  auto in_rows = x_size();
  tensorflow::Tensor x_data(
    tensorflow::DT_DOUBLE,
    tensorflow::TensorShape{ static_cast<int>(_in.size()) / in_rows, in_rows });
  std::copy(_in.begin(), _in.end(), x_data.flat<double>().data());

  auto out_rows = y_size();
  tensorflow::Tensor y_data(
    tensorflow::DT_DOUBLE,
    tensorflow::TensorShape{ static_cast<int>(_out.size() / out_rows), out_rows });
  std::copy(_out.begin(), _out.end(), y_data.flat<double>().data());

  // training steps
  for (int i = 0; i < 5000; ++i) {
    if (i % 100 == 0)
    {
      std::vector<tensorflow::Tensor> outputs;
      TF_CHECK_OK(session_.Run({ { *x_, x_data }, { *y_, y_data } }, { loss_ }, &outputs));
      std::cout << "Loss after " << i << " steps " << outputs[0].scalar<double>() << std::endl;
    }
    // nullptr because the output from the run is useless
    TF_CHECK_OK(session_.Run({ { *x_, x_data }, { *y_, y_data } }, apply_grad_, nullptr));
  }
}

void Machine::predict(const std::vector<double>& _in, std::vector<double>& _out)
{
  tensorflow::Tensor x_0(tensorflow::DataTypeToEnum<double>::v(), tensorflow::TensorShape{ 1, x_size() });
  std::copy(_in.begin(), _in.end(), x_0.flat<double>().data());
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(session_.Run({ { *x_, x_0 } }, { out_layer_ }, &outputs));
  _out.resize(y_size());
  std::copy_n(outputs[0].scalar<double>().data(), outputs[0].dim_size(0), _out.begin());
}

void Machine::store(const char* _flnm)
{

}


} // namespace ML 

