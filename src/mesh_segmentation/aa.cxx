// CMakeProject1.cpp : Defines the entry point for the application.
//
#pragma optimize ("", off)

#include "aa.hxx"

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

struct Weight00
{
  Weight00(tensorflow::Scope& scope, int M, int N) :
    w_(scope, { M, N }, tensorflow::DT_FLOAT)
  {}
  tensorflow::ops::Variable w_;
};

struct Weight : public Weight00
{
  Weight(tensorflow::Scope& scope, int M, int N) : Weight00(scope, M, N),
    assign_w_(
      scope, w_,
      tensorflow::ops::RandomNormal(scope, { M, N }, tensorflow::DT_FLOAT))
  {}
  tensorflow::ops::Assign assign_w_;
};

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


// https://matrices.io/training-a-deep-neural-network-using-only-tensorflow-c/
// https://www.tensorflow.org/api_guides/cc/guide
int main()
{
  //tensorflow::port::InitMain(nullptr, 0, nullptr);

  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  auto x = tensorflow::ops::Placeholder(scope, tensorflow::DT_FLOAT);
  auto y = tensorflow::ops::Placeholder(scope, tensorflow::DT_FLOAT);

  Weight00 ff(scope, 3, 3);
  //tensorflow::ops::RandomNormal val(scope, { 3., 3. }, tensorflow::DT_FLOAT);

  tensorflow::Tensor vv(tensorflow::DT_FLOAT, tensorflow::TensorShape{ 3, 3 });
  for(int i = 0; i < 9; ++i)
    vv.flat<float>().data()[i] = 0;
  auto c = tensorflow::ops::Const(scope, vv);

  tensorflow::ops::Assign assign_ff(
    scope, ff.w_, c);




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
  auto layer_1 = make_layer(scope, x, w1.w_, b1.w_);
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

  // add the gradients operations to the graph
  std::vector<tensorflow::Output> grad_outputs;
  tensorflow::AddSymbolicGradients(scope, { loss }, all_weights, &grad_outputs);

  // update the weights and bias using gradient descent
  std::vector<tensorflow::ops::ApplyGradientDescent> apply_grad;
  for (int i = 0; i < std::size(all_weights); ++i)
  {
    apply_grad.push_back(tensorflow::ops::ApplyGradientDescent(
      scope, all_weights[i], val_0_01, { grad_outputs[i] }));
  }

  tensorflow::ClientSession session(scope);
  std::vector<tensorflow::Tensor> outputs;

  std::vector<float> xx, yy;
  auto load_data = [&xx, &yy]()
  {
    const char* flnm = R"(C:\Users\marco\Project\mesh_and_machine_learning\src\mesh_segmentation\data.txt)";
    std::ifstream file(flnm);
    auto add_element = [&file](std::vector<float>& _vv)
    {
      float data;
      file >> data;
      _vv.push_back(data);
    };
    char newline;
    while(!file.eof() && file.good())
    {
      add_element(xx);
      add_element(xx);
      add_element(xx);
      add_element(yy);
      file >> newline;
    }
  };
  load_data();

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
  x_0.vec<float>()(0) = 110000.f;
  x_0.vec<float>()(1) = -1;
  x_0.vec<float>()(2) = 7.f;

  TF_CHECK_OK(session.Run({ { x, { x_0 } } }, { layer_3 }, &outputs));
  std::cout << "DNN output: " << *outputs[0].scalar<float>().data() << std::endl;
  std::cout << "Price predicted " << *outputs[0].scalar<float>().data() << " euros" << std::endl;
  return 0;
}
