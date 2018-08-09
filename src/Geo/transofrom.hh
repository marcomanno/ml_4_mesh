#pragma once
#include "range.hh"
#include <memory>

namespace Geo
{
struct Transform
{
  // By defolt identity transformation
  VectorD<3> delta_ = {};
  VectorD<3> rotation_ = {};
  VectorD<3> operator()(const VectorD<3>& _pos);
};

struct ITrajectory
{
  virtual Transform transform(double _par) = 0;
  virtual const Interval<double>& range() = 0;
  virtual VectorD<3> transform(double _par,
                               const VectorD<3>& _pos,
                               const VectorD<3>* _dir = nullptr) = 0;

  static std::shared_ptr<ITrajectory>
    make_linear(const Interval<double>& _interv,
                const VectorD<3>& _start, const VectorD<3>& _end,
                const VectorD<3>* _rot = nullptr);
  static std::shared_ptr<ITrajectory>
    make_rotation(const Interval<double>& _interv,
                  const VectorD<3>& _ax, const double& _al0, const double& _al1);
  static std::shared_ptr<ITrajectory>
    make_interpolate(const Interval<double>& _interv,
                     const Transform& _tr_beg, const Transform& _tr_end);
  static std::shared_ptr<ITrajectory>
    make_composite(const Interval<double>& _interv,
                   const Transform& _tr_beg, const Transform& _tr_end);
};


}