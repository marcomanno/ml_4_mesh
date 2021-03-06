#pragma once

#include "Geo/vector.hh"
#include "Topology/iterator.hh"
#include "splm/lm.hxx"


namespace MeshOp {

/** Map from vertex to index in the solution array */
using VertexIndMpap = std::map<Topo::Wrap<Topo::Type::VERTEX>, MKL_INT>;
using MapVertPos = std::map<Topo::Wrap<Topo::Type::VERTEX>, Geo::VectorD2>;

struct EnergyFunction : LM::IMultiFunction
{
  struct DataOfFace
  {
    void compute_coeff(double _area_sqrt);
    void fill_matrix(Eigen::Matrix<double, 4, 6>& _da_uv) const;
    const std::array<Geo::VectorD2, 3>& get_lscm_coeff() const;
    double area_sqrt() const { return area_sqrt_; }

    Geo::VectorD2 loc_tri_[2];
    MKL_INT idx_[3];
  private:
    double area_sqrt_;
    double c0_, c1_, c2_;
    std::array<Geo::VectorD2, 3> w_;
  };


  void init(const Topo::Iterator<Topo::Type::BODY, Topo::Type::FACE>& _bf, 
    const MapVertPos& _mvp, double _a0);
  MKL_INT compute_unkown_nmbr(bool _apply_constraints);
  bool evaluate(const LM::ColumnVector& _x, LM::ColumnVector& _f) const override;
  bool jacobian(const LM::ColumnVector& _x, LM::Matrix& _fj) const override;
  void jacobian_conformal(LM::Matrix& _fj) const;
  void area_matrix(LM::Matrix& _area_m);

  MKL_INT rows() const override;
  MKL_INT cols() const override;

  const VertexIndMpap& veterx_map() const { return vrt_inds_; }

  MKL_INT constrain_vertices() const { return fixed_nmbr_; }

protected:
  std::vector<DataOfFace> data_of_faces_;
  MKL_INT tri_nmbr_, n_;
  MKL_INT fixed_nmbr_;
  MKL_INT m2_, m3_;
  MKL_INT n2_, n3_;
  MKL_INT fixed_zero_ = 0;
  VertexIndMpap vrt_inds_;

  int compute(const LM::ColumnVector& _x,
    LM::ColumnVector* _fvec, LM::Matrix* _fjac) const;
};

} // namespace MeshOp
