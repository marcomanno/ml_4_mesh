//#pragma optimize ("", off)
#include "machine.hxx"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/ops/random_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/training_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/direct_session.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/kernels/matmul_op.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"

#include <fstream>

namespace ML
{
const std::string machine_attr("MLMachine");

template <class RealT>
struct Machine : public IMachine<RealT>
{
  const tensorflow::DataType TfType = tensorflow::DataTypeToEnum<RealT>::value;
  Machine() : IMachine<RealT>(), client_session_(scope_){}
  tensorflow::Input make_input(int _rows) override;
  tensorflow::Input make_output(int _rows) override;
  tensorflow::Input add_weight(int _m, int _n, const RealT& _init_val = 0) override;
  tensorflow::Output add_layer(
    tensorflow::Input& _X,
    tensorflow::Input& _A,
    tensorflow::Input& _B) override;
  tensorflow::Input set_target(tensorflow::Output& _layer,
    const RealT& _grad_coeff = 1.e-5, const RealT& _reg_coeff = 0) override;
  void train(
    const std::vector<RealT>& _in, const std::vector<RealT>& _out,
    int _iterations = 10000) override;

  void predictN(const std::vector<RealT>& _in, std::vector<RealT> &_out) override
  {
    return predict(_in, _out, x_size_);
  }
  void predict1(const std::vector<RealT>& _in, std::vector<RealT> &_out) override
  {
    return predict(_in, _out, _in.size());
  }
  void save(const char* _flnm) override;
  void load(const char* _flnm) override;
private:
  tensorflow::Scope scope_ = tensorflow::Scope::NewRootScope();
  tensorflow::ClientSession client_session_;

  std::shared_ptr<tensorflow::ops::Placeholder> y_;
  tensorflow::int64 y_size_ = 0;
  tensorflow::Output x_;
  tensorflow::int64 x_size_ = 0;

  std::vector<tensorflow::Output> weights_;
  std::vector<std::array<int, 2>> weights_size_;

  tensorflow::Output loss_, real_loss_;
  std::vector<tensorflow::Output> apply_grad_;
  std::unique_ptr<tensorflow::Output> out_layer_;

  // Op, A, X, B
  std::vector<std::array<int, 4>> layers_;

  void add_place_holder(int _rows, std::shared_ptr<tensorflow::ops::Placeholder>& _obj);
  auto x_size() const { return x_size_; }
  auto y_size() const { return y_size_; }
  void predict(const std::vector<RealT>& _in, std::vector<RealT> &_out, 
               tensorflow::int64 _x_size);
};

template <typename RealT> std::shared_ptr<IMachine<RealT>>
IMachine<RealT>::make()
{
  IMachine<RealT>* m = new Machine<RealT>;
  std::shared_ptr<IMachine<RealT>> x(m);
  return x;
}

template struct IMachine<double>;
template struct IMachine<float>;


template <class RealT> tensorflow::Input
Machine<RealT>::make_input(int _rows)
{
  std::shared_ptr<tensorflow::ops::Placeholder> x;
  add_place_holder(x_size_ = _rows, x);
  x_ = *x;
  return x_;
}

template <class RealT> tensorflow::Input 
Machine<RealT>::make_output(int _rows)
{
  add_place_holder(y_size_ = _rows, y_);
  return *y_;
}

template <class RealT> void 
Machine<RealT>::add_place_holder(
  int _rows, std::shared_ptr<tensorflow::ops::Placeholder>& _obj)
{
  tensorflow::ops::Placeholder::Attrs attr;
  _obj.reset(new tensorflow::ops::Placeholder(
      scope_, 
      TfType,
      attr.Shape({ 1, _rows })));
}

template <class RealT> tensorflow::Input 
Machine<RealT>::add_weight(int _m, int _n, const RealT& _init_val)
{
  auto w = tensorflow::ops::Variable(scope_, { _m, _n }, TfType);
  auto int_tensor = tensorflow::ops::Const(scope_, _init_val, {_m, _n});
  auto assign = tensorflow::ops::Assign(
    scope_, w,
    int_tensor);
  TF_CHECK_OK(client_session_.Run({assign}, nullptr));
  weights_.push_back(w);
  weights_size_.push_back({ _m, _n });
  return w;
}

template <class RealT> tensorflow::Output
Machine<RealT>::add_layer(
  tensorflow::Input& _X,
  tensorflow::Input& _A,
  tensorflow::Input& _B)
{
  auto layer = tensorflow::ops::Tanh(
    scope_,
    tensorflow::ops::Add(scope_, tensorflow::ops::MatMul(scope_, _X, _A), _B));
  layers_.push_back({ layer.node()->id(), _X.node()->id(), _A.node()->id(), _B.node()->id() });
  return layer;
}

template <class RealT> tensorflow::Input 
Machine<RealT>::set_target(tensorflow::Output& _layer,
    const RealT& _grad_coeff, const RealT& _reg_coeff)
{
  // regularization
  tensorflow::OutputList l2_losses;
  for (auto& w : weights_)
    l2_losses.push_back(tensorflow::ops::L2Loss(scope_, w));
  auto regularization = tensorflow::ops::AddN(scope_, l2_losses);

#if 0
  real_loss_ = tensorflow::ops::ReduceMean(scope_,
    tensorflow::ops::Square(scope_,
      tensorflow::ops::Sub(scope_, _layer, *y_)),
    { 0, 1 });
#else
  real_loss_ = tensorflow::ops::ReduceSum(
    scope_,
    tensorflow::ops::Square(
      scope_,
      tensorflow::ops::Sub(scope_, _layer, *y_)),
    { 0, 1 });
#endif
#if 1
  loss_ = real_loss_;
#else
  tensorflow::ops::Cast reg_coeff = tensorflow::ops::Cast(scope_, _reg_coeff, TfType);
  loss_ = tensorflow::ops::Add(
    scope_,
    real_loss_,
    tensorflow::ops::Mul(scope_, reg_coeff, regularization));
#endif

  // add the gradients operations to the graph
  tensorflow::ops::Cast grad_coeff = tensorflow::ops::Cast(scope_, _grad_coeff, TfType);
  std::vector<tensorflow::Output> grad_outputs;
  tensorflow::AddSymbolicGradients(scope_, { loss_ }, weights_, &grad_outputs);
  for (int i = 0; i < std::size(weights_); ++i)
  {
    apply_grad_.push_back(tensorflow::ops::ApplyGradientDescent(
      scope_, weights_[i], grad_coeff, { grad_outputs[i] }));
  }
  out_layer_.reset(new tensorflow::Output(_layer));
  return loss_;
}

template <class RealT> void 
Machine<RealT>::train(
  const std::vector<RealT>& _in, const std::vector<RealT>& _out,
  int _iterations)
{
  auto in_rows = x_size();
  tensorflow::Tensor x_data(
    TfType,
    tensorflow::TensorShape{ static_cast<int>(_in.size()) / in_rows, in_rows });
  std::copy(_in.begin(), _in.end(), x_data.flat<RealT>().data());

  auto out_rows = y_size();
  tensorflow::Tensor y_data(
    TfType,
    tensorflow::TensorShape{ static_cast<int>(_out.size() / out_rows), out_rows });
  std::copy(_out.begin(), _out.end(), y_data.flat<RealT>().data());

  // training steps
  for (int i = 0; i <= _iterations; ++i) {
    if (i % 100 == 0)
    {
      std::vector<tensorflow::Tensor> outputs;
      TF_CHECK_OK(client_session_.Run({ { x_, x_data }, { *y_, y_data } }, { loss_ }, &outputs));
      std::cout << "Loss after " << i << " steps " << outputs[0].scalar<RealT>() << std::endl;
    }
    // nullptr because the output from the run is useless
    TF_CHECK_OK(client_session_.Run({ { x_, x_data }, { *y_, y_data } }, apply_grad_, nullptr));
  }
  TF_CHECK_OK(client_session_.Run({ { x_, x_data }, { *y_, y_data } }, { loss_ }, &outputs));
  std::cout << "Final loss " << outputs[0].scalar<RealT>() << std::endl;
}

template <class RealT> void 
Machine<RealT>::predict(
  const std::vector<RealT>& _in, std::vector<RealT>& _out, tensorflow::int64 _x_size)
{
  auto res_nmbr = static_cast<tensorflow::int64>(_in.size()) / _x_size;
  tensorflow::Tensor x_0(tensorflow::DataTypeToEnum<RealT>::v(), 
                         tensorflow::TensorShape{ res_nmbr, _x_size });
  std::copy(_in.begin(), _in.end(), x_0.flat<RealT>().data());
  std::vector<tensorflow::Tensor> outputs;
  TF_CHECK_OK(client_session_.Run({ { x_, x_0 } }, { *out_layer_ }, &outputs));
  auto size = outputs[0].NumElements();
  auto data = outputs[0].flat<RealT>().data();
  _out.resize(size);
  std::copy_n(data, size, _out.begin());
}

std::string make_fiLename(const char* _flnm, int _var_id = -1)
{
  if (_var_id >= 0)
    return std::string(_flnm) + "_" + std::to_string(_var_id) + ".pb";
  else
    return std::string(_flnm) + "_graph" + ".pb";
}

template <class RealT>
void Machine<RealT>::save(const char* _flnm)
{
  auto flnm = make_fiLename(_flnm);
  std::ofstream graph_stream(flnm);
  graph_stream << "X " << x_.node()->id() << " " << x_size_ << std::endl;
  graph_stream << "Y " << y_->node()->id() << " " << y_size_ << std::endl;
  for (size_t i = 0; i < weights_.size(); ++i)
  {
    const auto& w = weights_[i];
    const auto& ws = weights_size_[i];
    graph_stream << "W " << w.node()->id() << " " << ws[0] << " " << ws[1] << std::endl;

    std::vector<tensorflow::Tensor> t(1);
    client_session_.Run({ w }, &t);
    tensorflow::TensorProto tensor_proto;
    t[0].AsProtoTensorContent(&tensor_proto);
    tensorflow::WriteTextProto(
      tensorflow::Env::Default(), make_fiLename(_flnm, w.node()->id()).c_str(),
      tensor_proto);
  }
  for (auto& l : layers_)
  {
    if (l[0] == out_layer_->node()->id())
      graph_stream << 'O';
    else
      graph_stream << 'L';
    for (auto id_ref : l)
      graph_stream << " " << id_ref;
    graph_stream << std::endl;
  }
}

template <class RealT>
void Machine<RealT>::load(const char* _flnm)
{
  auto flnm = make_fiLename(_flnm);
  std::ifstream graph_stream(flnm);
  std::string line;
  std::map<int, tensorflow::Node*> map;
  while (std::getline(graph_stream, line))
  {
    int id, m, n;
    std::istringstream str_stream(line);
    char opt;
    str_stream >> opt;
    switch (opt) 
    {
      case 'X':
      {
        str_stream >> id >> m;
        auto inp = make_input(m);
        map.emplace(id, make_input(m).node());
      }
      break;
      case 'Y':
      {
        str_stream >> id >> m;
        auto inp = make_output(m);
      }
      break;
      case 'W':
      {
        str_stream >> id >> m >> n;
        auto node = add_weight(m, n).node();
        map.emplace(id, node);

        tensorflow::TensorProto tensor_proto;
        tensorflow::ReadTextProto(
          tensorflow::Env::Default(), make_fiLename(_flnm, id).c_str(),
          &tensor_proto);
        tensorflow::Tensor new_tensor;
        if (!new_tensor.FromProto(tensor_proto))
          std::cout << "Error init_var.FromProto(tensor_proto)/n";
        auto var = ::tensorflow::Input(::tensorflow::Output(node));
        tensorflow::ops::Assign assign(scope_, var, new_tensor);
        TF_CHECK_OK(client_session_.Run({ assign }, nullptr));
      }
      break;
      case 'L':
      case 'O':
      {
        int A_id, X_id, B_id;
        str_stream >> id >> A_id >> X_id >> B_id;
        tensorflow::Output new_layer = 
          add_layer(
            tensorflow::Input(tensorflow::Output(map[A_id])), 
            tensorflow::Input(tensorflow::Output(map[X_id])),
            tensorflow::Input(tensorflow::Output(map[B_id])));
        map.emplace(id, new_layer.node());
        if (opt == 'O')
          set_target(new_layer);
        break;
      }
    }
  }
#if 0
  // restore
  tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
  checkpointPathTensor.scalar<std::string>()() = _flnm;
  tensor_dict feed_dict = { { graph_def.saver_def().filename_tensor_name(), checkpointPathTensor } };
  session_->Run(feed_dict, {}, { graph_def.saver_def().restore_op_name() }, nullptr);
#endif
}

} // namespace ML 

