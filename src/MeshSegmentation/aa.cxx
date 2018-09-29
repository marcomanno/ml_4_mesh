// CMakeProject1.cpp : Defines the entry point for the application.
//
#pragma optimize ("", off)

#include "aa.hxx"

#include "MeshSegmentation/mesh_training.hxx"

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

#include <iostream>
#include <fstream>

// https://matrices.io/training-a-deep-neural-network-using-only-tensorflow-c/
// https://www.tensorflow.org/api_guides/cc/guide

struct Weight
{
  Weight(tensorflow::Scope& scope, int M, int N) : 
    w_(scope, { M, N }, tensorflow::DT_FLOAT),
    assign_w_(
      scope, w_,
      tensorflow::ops::RandomNormal(scope, { M, N }, tensorflow::DT_FLOAT))
  {}
  tensorflow::ops::Variable w_;
  tensorflow::ops::Assign assign_w_;
};

template <class InputT>
auto make_layer(tensorflow::Scope& _scope,
                InputT& _X,
                tensorflow::Input& _A,
                tensorflow::Input& _B)
{
  return tensorflow::ops::Tanh(
    _scope,
    tensorflow::ops::Add(_scope, tensorflow::ops::MatMul(_scope, _X, _A), _B));
}

template <class InputT>
auto make_layer(tensorflow::Scope& _scope,
                InputT& _X,
                tensorflow::ops::Variable& _A,
                tensorflow::ops::Variable& _B)
{
  return tensorflow::ops::Tanh(
    _scope,
    tensorflow::ops::Add(_scope, tensorflow::ops::MatMul(_scope, _X, _A), _B));
}

template <typename typeT>
void load_data(std::vector<typeT> & _xx, std::vector<typeT>& _yy)
{
  std::ifstream in_data("C:/Users/marco/Project/ml_4_mesh/src/mesh_segmentation/data.txt");
  while (in_data.good() && !in_data.eof())
  {
    for (int i = 3; --i >= 0; in_data >> _xx.emplace_back());
    in_data >> _yy.emplace_back();
  }
};

#include "machine.hxx"

void compute()
{
  using RealType = float;
  auto machine = ML::IMachine<RealType>::make();
  auto x = machine->make_input(3);
  auto y = machine->make_output(1);
  auto w0 = machine->add_weight(3, 3);
  auto b0 = machine->add_weight(1, 3);
  auto w1 = machine->add_weight(3, 2);
  auto b1 = machine->add_weight(1, 2);
  auto w2 = machine->add_weight(2, 1);
  auto b2 = machine->add_weight(1, 1);
  auto layer0 = machine->add_layer(x, w0, b0);
  auto layer1 = machine->add_layer(tensorflow::Input(layer0), w1, b1);
  auto layer2 = machine->add_layer(tensorflow::Input(layer1), w2, b2);
  machine->set_target(layer2);

  std::vector<RealType> xx, yy;
  load_data(xx, yy);
  machine->train(xx, yy);
  std::vector<RealType> x1 = {1.25, -1, 7}, y1;
  machine->predictN(x1, y1);

  std::cout << x1[0] << " " << x1[1] << " " << x1[2] << " ---> " << y1[0];
}

int main()
{
  MeshSegmentation::train_mesh_segmentation(INDIR);
#if 0
  compute();
  //tensorflow::port::InitMain(nullptr, 0, nullptr);

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto x = tensorflow::ops::Placeholder(scope, tensorflow::DT_FLOAT);
  tensorflow::Input xx1 = x;
  auto y = tensorflow::ops::Placeholder(scope, tensorflow::DT_FLOAT);

  // weights init
  Weight w1(scope, 3, 3);
  Weight w2(scope, 3, 2);
  Weight w3(scope, 2, 1);

  // bias init
  Weight b1(scope, 1, 3);
  Weight b2(scope, 1, 2);
  Weight b3(scope, 1, 1);

  std::vector<tensorflow::Output> all_weights =
  { w1.w_, w2.w_, w3.w_, b1.w_, b2.w_, b3.w_ };

  // layers
  tensorflow::Input www1 = w1.w_;
  tensorflow::Input bbb1 = b1.w_;
  auto layer_1 = make_layer(scope, xx1, www1, bbb1);
  auto layer_2 = make_layer(scope, layer_1, w2.w_, b2.w_);
  auto layer_3 = make_layer(scope, layer_2, w3.w_, b3.w_);

  auto val_0_01 = tensorflow::ops::Cast(scope, 0.01, tensorflow::DT_FLOAT);

  // regularization
  auto regularization = tensorflow::ops::AddN(
      scope,
      std::initializer_list<tensorflow::Input>{
        tensorflow::ops::L2Loss(scope, w1.w_),
        tensorflow::ops::L2Loss(scope, w2.w_),
        tensorflow::ops::L2Loss(scope, w3.w_)});

  // loss calculation
  auto loss = tensorflow::ops::Add(
    scope,
    tensorflow::ops::ReduceMean(scope,
      tensorflow::ops::Square(scope, tensorflow::ops::Sub(scope, layer_3, y)), { 0, 1 }),
    tensorflow::ops::Mul(scope, val_0_01, regularization));

  // update the weights and bias using gradient descent
  std::vector<tensorflow::ops::ApplyGradientDescent> apply_grad;
  auto compute_apply_grad = [&apply_grad, &all_weights, &scope, &loss]()
  {
    auto val_0_01 = tensorflow::ops::Cast(scope, 0.01, tensorflow::DT_FLOAT);
    // add the gradients operations to the graph
    std::vector<tensorflow::Output> grad_outputs;
    tensorflow::AddSymbolicGradients(scope, { loss }, all_weights, &grad_outputs);
    for (int i = 0; i < std::size(all_weights); ++i)
    {
      apply_grad.push_back(tensorflow::ops::ApplyGradientDescent(
        scope, all_weights[i], val_0_01, { grad_outputs[i] }));
    }
  };
  compute_apply_grad();

  tensorflow::ClientSession session(scope);
  std::vector<tensorflow::Tensor> outputs;

  std::vector<float> xx, yy;
  load_data(xx, yy);

  tensorflow::Tensor x_data(
    tensorflow::DataTypeToEnum<float>::v(),
    tensorflow::TensorShape{ static_cast<int>(xx.size()) / 3, 3 });
  std::copy(xx.begin(), xx.end(), x_data.flat<float>().data());

  tensorflow::Tensor y_data(
    tensorflow::DataTypeToEnum<float>::v(),
    tensorflow::TensorShape{ static_cast<int>(yy.size()), 1 });
  std::copy(yy.begin(), yy.end(), y_data.flat<float>().data());

  // init the weights and biases by running the assigns nodes once
  TF_CHECK_OK(session.Run(
    { w1.assign_w_, w2.assign_w_, w3.assign_w_, 
      b1.assign_w_, b2.assign_w_, b3.assign_w_ }, nullptr));

  // training steps
  for (int i = 0; i < 5000; ++i) {
    TF_CHECK_OK(session.Run({ { x, x_data },{ y, y_data } }, { loss }, &outputs));
    if (i % 100 == 0) {
      std::cout << "Loss after " << i << " steps " << outputs[0].scalar<float>() << std::endl;
    }
    // nullptr because the output from the run is useless
    TF_CHECK_OK(session.Run(
      { { x, x_data },{ y, y_data } }, 
      { apply_grad[0], apply_grad[1], apply_grad[2], 
        apply_grad[3], apply_grad[4], apply_grad[5], layer_3 }, nullptr));
  }

  // prediction using the trained neural net
  tensorflow::Tensor x_0(tensorflow::DataTypeToEnum<float>::v(), tensorflow::TensorShape{ 1, 3 });
  x_0.flat<float>().data()[0] = 1.1f;
  x_0.flat<float>().data()[1] = -1;
  x_0.flat<float>().data()[2] = 7.f;

  TF_CHECK_OK(session.Run({ { x, x_0} }, { layer_3 }, &outputs));
  std::cout << "DNN output: " << *outputs[0].scalar<float>().data() << std::endl;
  std::cout << "Price predicted " << *outputs[0].scalar<float>().data() << " euros" << std::endl;
#endif
  return 0;
}
