#include "transofrom.hh"

namespace Geo
{
  
VectorD<3> Transform::operator()(const VectorD<3>& _pos)
{
  auto len = Geo::length(rotation_);
  VectorD<3> transf_pos = {};
  if (len > 1e-12)
  {
    auto ax = rotation_ / len;
    auto alpha = 2 * M_PI * len;
    auto ax_comp = (_pos * ax) * ax;
    auto orth_comp = _pos - ax_comp;
    auto orth_comp_perp = ax % orth_comp;
    transf_pos = ax_comp + cos(alpha) * orth_comp + sin(alpha) * orth_comp_perp;
  }
  transf_pos += delta_;
  return transf_pos;
}

struct Trajectory : public ITrajectory
{
  const Interval<double>& range() override { return range_; }
  Interval<double> range_;
};
#if 0
struct TrajectoryLinear : public Trajectory
{
  Transform transform(double _par) override
  {
    Transform res;
    auto t = (_par - range_[0]) / range_.length();
    res .delta_ = trnsf_.delta_ * (1 - t) + end_pos_;
    res.rotation_ = trnsf_.rotation_;
    return res;
  }
  VectorD<3> transform(double _par,
                       const VectorD<3>& _pos,
                       const VectorD<3>* _dir = nullptr)
  {
    auto trnsf = transform(_par);
    return trnsf(_pos);
  }

  Transform trnsf_;
  VectorD3 end_pos_;
};

std::shared_ptr<ITrajectory>
ITrajectory::make_linear(const Interval<double>& _interv,
                         const VectorD<3>& _start, const VectorD<3>& _end,
                         const VectorD<3>* _rot)
{
  auto res = std::make_shared<TrajectoryLinear>();
  res->range_ = _interv;
  res->end_pos_ = _end;
  res->trnsf_.delta_ = _start;
  if (_rot != nullptr)
    res->trnsf_.rotation_ = *_rot;
  return res;
}

struct TrajectoryRotation : public ITrajectory
{
  TRAJECTORY_METHODS(, override);
};

struct TrajectoryInterpolate : public ITrajectory
{
  TRAJECTORY_METHODS(, override);
};

struct TrajectoryCompose : public ITrajectory
{
  TRAJECTORY_METHODS(, override);
};

#endif

} // nemespace Geo
