// This file is MACHINE GENERATED! Do not edit.

#ifndef C__USERS_USER_SOURCE_REPOS_TENSORFLOW_TENSORFLOW_CONTRIB_CMAKE_BUILD_TENSORFLOW_CC_OPS_TRAINING_OPS_INTERNAL_H_
#define C__USERS_USER_SOURCE_REPOS_TENSORFLOW_TENSORFLOW_CONTRIB_CMAKE_BUILD_TENSORFLOW_CC_OPS_TRAINING_OPS_INTERNAL_H_

// This file is MACHINE GENERATED! Do not edit.

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace tensorflow {
namespace ops {
namespace internal {
// NOTE: This namespace has internal TensorFlow details that
// are not part of TensorFlow's public API.

/// @defgroup training_ops_internal Training Ops Internal
/// @{

/// Update '*var' according to the AdaMax algorithm.
///
/// m_t <- beta1 * m_{t-1} + (1 - beta1) * g
/// v_t <- max(beta2 * v_{t-1}, abs(g))
/// variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)
///
/// Arguments:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * v: Should be from a Variable().
/// * beta1_power: Must be a scalar.
/// * lr: Scaling factor. Must be a scalar.
/// * beta1: Momentum factor. Must be a scalar.
/// * beta2: Momentum factor. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, m, and v tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * `Output`: Same as "var".
class ApplyAdaMax {
 public:
  /// Optional attribute setters for ApplyAdaMax
  struct Attrs {
    /// If `True`, updating of the var, m, and v tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ApplyAdaMax(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
            ::tensorflow::Input m, ::tensorflow::Input v, ::tensorflow::Input
            beta1_power, ::tensorflow::Input lr, ::tensorflow::Input beta1,
            ::tensorflow::Input beta2, ::tensorflow::Input epsilon,
            ::tensorflow::Input grad);
  ApplyAdaMax(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
            ::tensorflow::Input m, ::tensorflow::Input v, ::tensorflow::Input
            beta1_power, ::tensorflow::Input lr, ::tensorflow::Input beta1,
            ::tensorflow::Input beta2, ::tensorflow::Input epsilon,
            ::tensorflow::Input grad, const ApplyAdaMax::Attrs& attrs);
  operator ::tensorflow::Output() const { return out; }
  operator ::tensorflow::Input() const { return out; }
  ::tensorflow::Node* node() const { return out.node(); }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  ::tensorflow::Output out;
};

/// Update '*var' according to the AdaMax algorithm.
///
/// m_t <- beta1 * m_{t-1} + (1 - beta1) * g
/// v_t <- max(beta2 * v_{t-1}, abs(g))
/// variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)
///
/// Arguments:
/// * scope: A Scope object
/// * var: Should be from a Variable().
/// * m: Should be from a Variable().
/// * v: Should be from a Variable().
/// * beta1_power: Must be a scalar.
/// * lr: Scaling factor. Must be a scalar.
/// * beta1: Momentum factor. Must be a scalar.
/// * beta2: Momentum factor. Must be a scalar.
/// * epsilon: Ridge term. Must be a scalar.
/// * grad: The gradient.
///
/// Optional attributes (see `Attrs`):
/// * use_locking: If `True`, updating of the var, m, and v tensors will be protected
/// by a lock; otherwise the behavior is undefined, but may exhibit less
/// contention.
///
/// Returns:
/// * the created `Operation`
class ResourceApplyAdaMax {
 public:
  /// Optional attribute setters for ResourceApplyAdaMax
  struct Attrs {
    /// If `True`, updating of the var, m, and v tensors will be protected
    /// by a lock; otherwise the behavior is undefined, but may exhibit less
    /// contention.
    ///
    /// Defaults to false
    TF_MUST_USE_RESULT Attrs UseLocking(bool x) {
      Attrs ret = *this;
      ret.use_locking_ = x;
      return ret;
    }

    bool use_locking_ = false;
  };
  ResourceApplyAdaMax(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                    ::tensorflow::Input m, ::tensorflow::Input v,
                    ::tensorflow::Input beta1_power, ::tensorflow::Input lr,
                    ::tensorflow::Input beta1, ::tensorflow::Input beta2,
                    ::tensorflow::Input epsilon, ::tensorflow::Input grad);
  ResourceApplyAdaMax(const ::tensorflow::Scope& scope, ::tensorflow::Input var,
                    ::tensorflow::Input m, ::tensorflow::Input v,
                    ::tensorflow::Input beta1_power, ::tensorflow::Input lr,
                    ::tensorflow::Input beta1, ::tensorflow::Input beta2,
                    ::tensorflow::Input epsilon, ::tensorflow::Input grad,
                    const ResourceApplyAdaMax::Attrs& attrs);
  operator ::tensorflow::Operation() const { return operation; }

  static Attrs UseLocking(bool x) {
    return Attrs().UseLocking(x);
  }

  Operation operation;
};

}  // namespace internal
}  // namespace ops
}  // namespace tensorflow

#endif  // C__USERS_USER_SOURCE_REPOS_TENSORFLOW_TENSORFLOW_CONTRIB_CMAKE_BUILD_TENSORFLOW_CC_OPS_TRAINING_OPS_INTERNAL_H_
