#include "cppoptlib/meta.h"
#include "cppoptlib/problem.h"
#include "cppoptlib/solver/lbfgsbsolver.h"
#include <chrono>
#include <cmath>
#include <iostream>

using namespace cppoptlib;

template <typename Scalar> class DhParam {
  public:
  DhParam(Scalar d, Scalar theta, Scalar r, Scalar alpha) : d(d), theta(theta), r(r), alpha(alpha) {
    ft.coeffRef(2, 0) = 0;
    ft.coeffRef(2, 3) = d;
    ft.coeffRef(3, 0) = 0;
    ft.coeffRef(3, 1) = 0;
    ft.coeffRef(3, 2) = 0;
    ft.coeffRef(3, 3) = 1;
  }

  void computeFT(const Scalar jointAngle) {
    const Scalar ct = std::cos(theta + jointAngle);
    const Scalar st = std::sin(theta + jointAngle);
    const Scalar ca = std::cos(alpha);
    const Scalar sa = std::sin(alpha);
    ft.coeffRef(0, 0) = ct;
    ft.coeffRef(0, 1) = -st * ca;
    ft.coeffRef(0, 2) = st * sa;
    ft.coeffRef(0, 3) = r * ct;
    ft.coeffRef(1, 0) = st;
    ft.coeffRef(1, 1) = ct * ca;
    ft.coeffRef(1, 2) = -ct * sa;
    ft.coeffRef(1, 3) = r * st;
    ft.coeffRef(2, 1) = sa;
    ft.coeffRef(2, 2) = ca;
  }

  const Scalar d;
  const Scalar theta;
  const Scalar r;
  const Scalar alpha;
  Eigen::Matrix<Scalar, 4, 4> ft;
};

template <typename T> class IkProblem : public BoundedProblem<T> {
  public:
  using typename Problem<T>::Scalar;
  using typename Problem<T>::TVector;
  using FT = Eigen::Matrix<Scalar, 4, 4>;

  IkProblem(std::vector<DhParam<T>> dhParams, const FT target, const TVector &l, const TVector &u)
    : BoundedProblem<T>(l, u), target(std::move(target)), dhParams(std::move(dhParams)) {
  }

  T value(const TVector &jointAngles) override {
    FT tip = FT::Identity();
    for (size_t i = 0; i < dhParams.size(); ++i) {
      dhParams[i].computeFT(jointAngles[i]);
      tip *= dhParams[i].ft;
    }

    const T targetX = target.coeff(0, 3);
    const T targetY = target.coeff(1, 3);
    const T targetZ = target.coeff(2, 3);

    const T tipX = tip.coeff(0, 3);
    const T tipY = tip.coeff(1, 3);
    const T tipZ = tip.coeff(2, 3);

    return std::sqrt(std::pow(tipX - targetX, 2) + std::pow(tipY - targetY, 2) +
                     std::pow(tipZ - targetZ, 2));
  }

  void gradient(const TVector &x, TVector &grad) override {
    // TODO: Compute jacobian
    Problem<T>::finiteGradient(x, grad, 0);
  }

  const FT target;
  std::vector<DhParam<T>> dhParams;
};

double rad(const double deg) {
  return deg * (M_PI / 180.0);
}

int main(int, char const *[]) {
  Eigen::VectorXd lowerJointLimits(6);
  lowerJointLimits << -M_PI, -M_PI, -M_PI, -M_PI, -M_PI, -M_PI;

  Eigen::VectorXd upperJointLimits(6);
  upperJointLimits << M_PI, M_PI, M_PI, M_PI, M_PI, M_PI;

  Eigen::Matrix4d target;
  // 3001 home
  // target << 1, 0, 0, 175, 0, 1, 0, 1.0365410507983197e-14, 0, 0, 1, -34.28, 0, 0, 0, 1;

  // 3001 target
  // target << 1, 0, 0, 200, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

  // cmm target
  target << 1, 0, 0, 41.999999999999986, 0, 1, 0, -44, 0, 0, 1, 169, 0, 0, 0, 1;

  typedef IkProblem<double> Problem;
  Problem f({DhParam(13.0, rad(180), 32.0, rad(-90)),
             DhParam(25.0, rad(-90), 93.0, rad(180)),
             DhParam(11.0, rad(90), 24.0, rad(90)),
             DhParam(128.0, rad(-90), 0.0, rad(90)),
             DhParam(0.0, 0.0, 0.0, rad(-90)),
             DhParam(25.0, rad(90), 0.0, 0.0)},
            target,
            lowerJointLimits,
            upperJointLimits);

  Eigen::VectorXd initialJointAngles(6);
  initialJointAngles << 0, 0, 0, 0, 0, 0;

  LbfgsbSolver<Problem> solver;

  const int numIter = 1000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < numIter; ++i) {
    solver.minimize(f, initialJointAngles);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Microseconds per solve: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / numIter
            << std::endl;

  std::cout << initialJointAngles(0) * (180.0 / M_PI) << " "
            << initialJointAngles(1) * (180.0 / M_PI) << " "
            << initialJointAngles(2) * (180.0 / M_PI) << " "
            << initialJointAngles(3) * (180.0 / M_PI) << " "
            << initialJointAngles(4) * (180.0 / M_PI) << " "
            << initialJointAngles(5) * (180.0 / M_PI) << std::endl;
  std::cout << "argmin      " << initialJointAngles.transpose() << std::endl;
  std::cout << "f in argmin " << f(initialJointAngles) << std::endl;

  return 0;
}
