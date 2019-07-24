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
  }

  void computeFT(Scalar jointAngle) {
    const Scalar ct = std::cos(theta + jointAngle);
    const Scalar st = std::sin(theta + jointAngle);
    const Scalar ca = std::cos(alpha);
    const Scalar sa = std::sin(alpha);
    ft << ct, -st * ca, st * sa, r * ct, st, ct * ca, -ct * sa, r * st, 0, sa, ca, d, 0, 0, 0, 1;
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

    const T targetX = target(0, 3);
    const T targetY = target(1, 3);
    const T targetZ = target(2, 3);

    const T tipX = tip(0, 3);
    const T tipY = tip(1, 3);
    const T tipZ = tip(2, 3);

    return std::sqrt((tipX - targetX) * (tipX - targetX) + (tipY - targetY) * (tipY - targetY) +
                     (tipZ - targetZ) * (tipZ - targetZ));
  }

  void gradient(const TVector &x, TVector &grad) override {
    // TODO: Compute jacobian
    Problem<T>::finiteGradient(x, grad, 0);
  }

  const FT target;
  std::vector<DhParam<T>> dhParams;
};

int main(int, char const *[]) {
  Eigen::VectorXd lowerJointLimits(3);
  lowerJointLimits << -M_PI, -M_PI, -M_PI;

  Eigen::VectorXd upperJointLimits(3);
  upperJointLimits << M_PI, M_PI, M_PI;

  Eigen::Matrix4d target;
  //  target << 1, 0, 0, 175, 0, 1, 0, 1.0365410507983197e-14, 0, 0, 1, -34.28, 0, 0, 0, 1;
  target << 1, 0, 0, 200, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

  typedef IkProblem<double> Problem;
  Problem f({DhParam(135.0, 0.0, 0.0, -90 * (M_PI / 180.0)),
             DhParam(0.0, 0.0, 175.0, 0.0),
             DhParam(0.0, 90.0 * (M_PI / 180.0), 169.28, 0.0)},
            target,
            lowerJointLimits,
            upperJointLimits);

  Eigen::VectorXd initialJointAngles(3);
  initialJointAngles << 0, 0, 0;

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

  std::cout << initialJointAngles(0) * (180.0 / M_PI) << std::endl;
  std::cout << initialJointAngles(1) * (180.0 / M_PI) << std::endl;
  std::cout << initialJointAngles(2) * (180.0 / M_PI) << std::endl;
  std::cout << "argmin      " << initialJointAngles.transpose() << std::endl;
  std::cout << "f in argmin " << f(initialJointAngles) << std::endl;

  return 0;
}
