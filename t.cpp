#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <vector>

using namespace std;

template<typename Scalar = double>
inline Eigen::Matrix<Scalar, 3, 1> RMat2Ypr(const Eigen::Matrix<Scalar, 3, 3> &R) {
    Scalar y = atan2(R(1, 0), R(0, 0));
    Scalar p = atan2(R(0, 2) * sin(y) - R(1, 2) * cos(y), -R(0, 1) * sin(y) + R(1, 1) * cos(y));
    Scalar r = atan2(-R(2, 0), R(0, 0) * cos(y) + R(1, 0) * sin(y));
    return {y, p, r};
}

template<typename Scalar = double>
Eigen::Matrix<Scalar, 3, 3> Ypr2RMat(const Eigen::Matrix<Scalar, 3, 1> &ypr) {
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

    AccOb(const Eigen::Vector3d &_g, const Eigen::Vector3d &_a): g(_g), a(_a) {
    }

    Eigen::Vector3d g;
    Eigen::Vector3d a;
};

struct AccObGenerator {
    const AccOb ob_vehicle{{0, 0, 9.8}, {10, 0, 0}};
    vector<AccOb> obs_imu;
    vector<Eigen::Matrix3d> params;

    auto &generate(const vector<Eigen::Vector3d> &yprs) {
        params.clear();
        params.reserve(yprs.size());
        obs_imu.clear();
        obs_imu.reserve(yprs.size());
        for (const auto &ypr_deg: yprs) {
            Eigen::Vector3d ypr = ypr_deg * M_PI / 180.0;
            params.emplace_back(Ypr2RMat(ypr).transpose());
            auto &rmat = params.back();
            obs_imu.emplace_back(rmat * ob_vehicle.g, rmat * ob_vehicle.a);
        }
        return obs_imu;
    }
};

int main() {
    vector<Eigen::Vector3d> yprs;
    yprs.emplace_back(0, 0, 0);
    yprs.emplace_back(30, 2, -2);

    AccObGenerator gen;
    gen.generate(yprs);
    for (auto &ob: gen.obs_imu) {
        cout << "[" << ob.g.transpose() << "] [" << ob.a.transpose() << "]" << endl;
    }
}
