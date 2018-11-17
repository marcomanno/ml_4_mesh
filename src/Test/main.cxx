#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <Geo/vector.hh>
#include <Eigen/dense>
#include "flatten/optimal_rotation.hxx"
#include <Import/save_obj.hh>

TEST_CASE("basic_00", "[RotationOptimization]")
{
  std::vector<std::array<Geo::VectorD2, 2>> ptmap =
  {
    {Geo::VectorD2{-1, -1}, Geo::VectorD2{ 1,  1}},
    {Geo::VectorD2{-1,  1}, Geo::VectorD2{ 1, -1}},
    {Geo::VectorD2{ 1,  1}, Geo::VectorD2{-1, -1}},
    {Geo::VectorD2{ 1, -1}, Geo::VectorD2{-1,  1}}
  };
  Eigen::Matrix2d R;
  Eigen::Vector2d T;
  MeshOp::find_optimal_rotation(ptmap, R, T);
  REQUIRE((R(0, 0) == R(1, 1) && R(0, 1) == -R(1, 0)));
  REQUIRE(R(0, 0) == -1);
  REQUIRE(R(0, 1) == 0);
  REQUIRE((T(0, 0) == 0 && T(1, 0) == 0));
}

TEST_CASE("basic_01", "[RotationOptimization]")
{
  std::vector<std::array<Geo::VectorD2, 2>> ptmap =
  {
    {Geo::VectorD2{-1, -1}, Geo::VectorD2{3, 3}},
    {Geo::VectorD2{-1,  1}, Geo::VectorD2{3, 1}},
    {Geo::VectorD2{ 1,  1}, Geo::VectorD2{1, 1}},
    {Geo::VectorD2{ 1, -1}, Geo::VectorD2{1, 3}}
  };
  Eigen::Matrix2d R;
  Eigen::Vector2d T;
  MeshOp::find_optimal_rotation(ptmap, R, T);
  REQUIRE((R(0, 0) == R(1, 1) && R(0, 1) == -R(1, 0)));
  REQUIRE(R(0, 0) == -1);
  REQUIRE(R(1, 0) == 0);
  REQUIRE((T(0, 0) == 2. && T(1, 0) == 2.));
}

TEST_CASE("basic_02", "[RotationOptimization]")
{
  std::vector<std::array<Geo::VectorD2, 2>> ptmap =
  {
    {Geo::VectorD2{-1, -1}, Geo::VectorD2{-1,  1}},
    {Geo::VectorD2{-1,  1}, Geo::VectorD2{ 1,  1}},
    {Geo::VectorD2{ 1,  1}, Geo::VectorD2{ 1, -1}},
    {Geo::VectorD2{ 1, -1}, Geo::VectorD2{-1, -1}}
  };
  Eigen::Matrix2d R;
  Eigen::Vector2d T;
  MeshOp::find_optimal_rotation(ptmap, R, T);
  REQUIRE((R(0, 0) == R(1, 1) && R(0, 1) == -R(1, 0)));
  REQUIRE(R(0, 0) == 0);
  REQUIRE(R(0, 1) == 1);
  REQUIRE((T(0, 0) == 0. && T(1, 0) == 0.));
}

#include <crtdbg.h>

int checckk_memeory()
{
  return _CrtCheckMemory();
}

TEST_CASE("real_01", "[RotationOptimization]")
{
  double coords[] =
  {
 0.00000000000000000 ,
 0.00000000000000000 ,
  40.202983747171203 ,
 -4.0886445922206889 ,
  15.000000000000000 ,
 0.00000000000000000 ,
  22.553999397620220 ,
-0.00000000000000000 ,
  15.603784596048127 ,
 0.00000000000000000 ,
-0.00000000000000000 ,
-0.00000000000000000 ,
  16.207570581345070 ,
 0.00000000000000000 ,
0.029891575329340784 ,
-0.95519280449439359 ,
  16.811355484518380 ,
 0.00000000000000000 ,
0.072978197384848620 ,
 -1.8446809400062452 ,
  17.415140036227019 ,
 0.00000000000000000 ,
 0.12879993762853059 ,
 -2.6695935496373546 ,
  18.018924367068752 ,
 0.00000000000000000 ,
 0.19689670847076865 ,
 -3.4308378889376550 ,
  18.622709326574203 ,
 0.00000000000000000 ,
 0.27677715956735333 ,
 -4.1291007633797090 ,
  19.226495079107334 ,
 0.00000000000000000 ,
 0.36803996882143714 ,
 -4.7648840520773366 ,
  19.830279868148100 ,
 0.00000000000000000 ,
 0.47019244528389359 ,
 -5.3384530732134099 ,
  20.434064537433432 ,
 0.00000000000000000 ,
 0.58279215960689823 ,
 -5.8498809799495088 ,
  21.037850001041988 ,
 0.00000000000000000 ,
 0.70526853763176234 ,
 -6.2990169967602103 ,
  21.641634904215294 ,
 0.00000000000000000 ,
 0.83704022875181805 ,
 -6.6855163068608441 ,
  22.245420086147579 ,
 0.00000000000000000 ,
 0.97748541632348029 ,
 -7.0088380756250626 ,
  22.849204621817382 ,
 0.00000000000000000 ,
  1.1259347232354313 ,
 -7.2682428590548849 ,
  37.849204621817378 ,
 0.00000000000000000 ,
 -12.677572177572014 ,
 -12.510416430011940 ,
  38.452990155662796 ,
 0.00000000000000000 ,
 -13.356623379623805 ,
 -12.729869316820203 ,
  39.056774353975406 ,
 0.00000000000000000 ,
 -14.032706550986621 ,
 -12.963273399511754 ,
  39.660559257148712 ,
 0.00000000000000000 ,
 -14.705128637999984 ,
 -13.210250992524925 ,
  40.264344720757272 ,
 0.00000000000000000 ,
 -15.373214906951862 ,
 -13.470416382956936 ,
  40.868128534260912 ,
 0.00000000000000000 ,
 -16.036307103689943 ,
 -13.743375422342172 ,
  41.471914110484789 ,
 0.00000000000000000 ,
 -16.693767625165680 ,
 -14.028727986550578 ,
  42.075699863017917 ,
 0.00000000000000000 ,
 -17.344977420617976 ,
 -14.326066370606455 ,
  42.679484822523371 ,
 0.00000000000000000 ,
 -17.989323745086445 ,
 -14.634973537032272 ,
  43.283269153365104 ,
 0.00000000000000000 ,
 -18.626211600113550 ,
 -14.955027692747759 ,
  43.887053705073740 ,
 0.00000000000000000 ,
 -19.255047380472462 ,
 -15.285793061478346 ,
  44.490838905751104 ,
 0.00000000000000000 ,
 -19.875246650330062 ,
 -15.626823452457387 ,
  45.094624530542717 ,
 0.00000000000000000 ,
 -20.486230223434273 ,
 -15.977660076162556 ,
  45.698409186970821 ,
 0.00000000000000000 ,
 -21.087422135508657 ,
 -16.337830089943743 ,
  60.698409186970821 ,
 0.00000000000000000 ,
 -33.710266198572427 ,
   -23.347020653699012 };
   std::vector<std::array<Geo::VectorD2, 2>> ptmap;
   auto& chain = ptmap.emplace_back();
   size_t j = 2;
   for (auto cood : coords)
   {
     if (++j == 3)
     {
       ptmap.emplace_back();
       j = 0;
     }
     ptmap.back()[j / 2][j % 2] = cood;
   }
  Eigen::Matrix2d R;
  Eigen::Vector2d T;
  MeshOp::find_optimal_rotation(ptmap, R, T);
  std::vector<Geo::VectorD2> rigid_input, rigid_target, optimized;
  for (auto& two_pt : ptmap)
  {
    rigid_input.push_back(two_pt[0]);
    rigid_target.push_back(two_pt[1]);
    Eigen::Vector2d x; x << two_pt[0][0], two_pt[0][1];
    x = R * x + T;
    optimized.push_back({x(0), x(1)});
  }
  auto flnm = std::string(OUTDIR) + "/" + Catch::getResultCapture().getCurrentTestName() + "_";
  IO::save_polyline((flnm + "rigid_input").c_str(), rigid_input);
  IO::save_polyline((flnm + "rigid_target").c_str(), rigid_target);
  IO::save_polyline((flnm + "optimized").c_str(), optimized);
}
