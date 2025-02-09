#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>

using namespace std;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg2Rad = kPi / 180.0;
constexpr double kRad2Deg = 180.0 / kPi;

template <typename Scalar = double>
inline Eigen::Matrix<Scalar, 3, 1> RMat2Ypr(const Eigen::Matrix<Scalar, 3, 3>& R) {
  Scalar y = atan2(R(1, 0), R(0, 0));
  Scalar p = atan2(R(0, 2) * sin(y) - R(1, 2) * cos(y), -R(0, 1) * sin(y) + R(1, 1) * cos(y));
  Scalar r = atan2(-R(2, 0), R(0, 0) * cos(y) + R(1, 0) * sin(y));
  return {y, p, r};
}

template <typename Scalar = double>
Eigen::Matrix<Scalar, 3, 3> Ypr2RMat(const Eigen::Matrix<Scalar, 3, 1>& ypr) {
  const Scalar y = ypr(0);
  const Scalar p = ypr(1);
  const Scalar r = ypr(2);
  const Scalar cos_y = cos(y);
  const Scalar sin_y = sin(y);
  const Scalar cos_p = cos(p);
  const Scalar sin_p = sin(p);
  const Scalar cos_r = cos(r);
  const Scalar sin_r = sin(r);
  Eigen::Matrix<Scalar, 3, 3> Rz;
  Rz << cos_y, -sin_y, 0, sin_y, cos_y, 0, 0, 0, 1;
  Eigen::Matrix<Scalar, 3, 3> Ry;
  Ry << cos_r, 0., sin_r, 0., 1., 0., -sin_r, 0., cos_r;
  Eigen::Matrix<Scalar, 3, 3> Rx;
  Rx << 1., 0., 0., 0., cos_p, -sin_p, 0., sin_p, cos_p;
  return Rz * Ry * Rx;
}

struct AccOb {
  AccOb() = default;

  AccOb(const Eigen::Vector3d& _g, const Eigen::Vector3d& _a) : g(_g), a(_a) {}

  Eigen::Vector3d g;
  Eigen::Vector3d a;
};

struct AccObGenerator {
  inline static const AccOb ob_vehicle{{0, 0, 9.8}, {10, 0, 0}};
  vector<AccOb> obs_imu;
  vector<Eigen::Matrix3d> params;

  auto& generate(const vector<Eigen::Vector3d>& yprs) {
    params.clear();
    params.reserve(yprs.size());
    obs_imu.clear();
    obs_imu.reserve(yprs.size());
    for (const auto& ypr_deg : yprs) {
      Eigen::Vector3d ypr = ypr_deg * kDeg2Rad;
      params.emplace_back(Ypr2RMat(ypr).transpose());
      auto& rmat = params.back();
      obs_imu.emplace_back(rmat * ob_vehicle.g, rmat * ob_vehicle.a);
    }
    return obs_imu;
  }
};

struct Solution {
  virtual ~Solution() = default;

  virtual string name() const { return "Solution"; }
  virtual Eigen::Quaterniond exec(const Eigen::Vector3d& g_imu, const Eigen::Vector3d& a_imu) const = 0;
};

struct Solution1 : Solution {
  string name() const { return "Solution1"; }

  Eigen::Quaterniond exec(const Eigen::Vector3d& g_imu, const Eigen::Vector3d& a_imu) const override {
    Eigen::Vector3d g_vehicle(0, 0, 1);
    Eigen::Vector3d a_vehicle(1, 0, 0);

    Eigen::Vector3d a1 = g_imu.normalized();
    Eigen::Vector3d a2 = a_imu.normalized();
    const auto& b1 = g_vehicle;
    const auto& b2 = a_vehicle;

    Eigen::Quaterniond q1 = Eigen::Quaterniond::FromTwoVectors(a1, b1);
    Eigen::Vector3d a2_prime = q1 * a2;
    Eigen::Vector3d p1 = a2_prime - a2_prime.dot(b1) * b1;
    Eigen::Vector3d p2 = b2 - b2.dot(b1) * b1;
    Eigen::Quaterniond q2 = Eigen::Quaterniond::FromTwoVectors(p1, p2);
    Eigen::Quaterniond q = q2 * q1;
    return q;
  }
};

struct Solution2 : Solution {
  string name() const { return "Solution2"; }

  Eigen::Quaterniond exec(const Eigen::Vector3d& g_imu, const Eigen::Vector3d& a_imu) const override {
    Eigen::Vector3d g_vehicle(0, 0, 1);
    Eigen::Vector3d a_vehicle(1, 0, 0);

    Eigen::Vector3d a1 = g_imu.normalized();
    Eigen::Vector3d a2 = a_imu.normalized();
    const auto& b1 = g_vehicle;
    const auto& b2 = a_vehicle;

    // Compute orthogonal vectors to complete the basis
    Eigen::Vector3d a3 = a1.cross(a2).normalized();
    Eigen::Vector3d b3 = b1.cross(b2).normalized();

    // Construct orthonormal basis matrices
    Eigen::Matrix3d A, B;
    A << a1, a2, a3;
    B << b1, b2, b3;

    // Compute rotation matrix: R = B * A^T
    Eigen::Matrix3d R = B * A.transpose();

    // Convert to quaternion (ensure it's a proper rotation)
    return Eigen::Quaterniond(R).normalized();
  }
};

void test_solution(const AccObGenerator& gen, const Solution& solution) {
  cout << "\33[33m" << "test " << solution.name() << "----------------------------\33[m\n";
  for (size_t i = 0; i < gen.params.size(); ++i) {
    Eigen::Matrix3d rmat = gen.params.at(i).transpose();
    auto gt = RMat2Ypr(rmat);
    const auto& d = gen.obs_imu.at(i);
    auto q = solution.exec(d.g, d.a);
    auto res = RMat2Ypr(q.toRotationMatrix());
    cout << "data " << i << ": g[" << d.g.transpose() << "] a[" << d.a.transpose() << "]\n";
    cout << "     gt: " << gt.transpose() * kRad2Deg << "\n";
    cout << "    res: " << res.transpose() * kRad2Deg << "\n";
  }
}

int main() {
  vector<Eigen::Vector3d> yprs;
  yprs.emplace_back(0, 0, 0);
  yprs.emplace_back(30, 2, -2);
  yprs.emplace_back(120, 97, -2);
  yprs.emplace_back(80, -90, 160);

  AccObGenerator gen;
  gen.generate(yprs);
  test_solution(gen, Solution2());
}
