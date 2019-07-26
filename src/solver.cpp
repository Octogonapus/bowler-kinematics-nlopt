#include "cppoptlib/meta.h"
#include "cppoptlib/problem.h"
#include "cppoptlib/solver/lbfgsbsolver.h"
#include <cmath>
#include <jni.h>

template <typename Scalar> class DhParam {
  public:
  DhParam(Scalar d, Scalar theta, Scalar r, Scalar alpha)
    : m_d(d), m_theta(theta), m_r(r), m_alpha(alpha) {
    m_ft.coeffRef(2, 0) = 0;
    m_ft.coeffRef(2, 3) = m_d;
    m_ft.coeffRef(3, 0) = 0;
    m_ft.coeffRef(3, 1) = 0;
    m_ft.coeffRef(3, 2) = 0;
    m_ft.coeffRef(3, 3) = 1;
  }

  void computeFT(const Scalar jointAngle) {
    const Scalar ct = std::cos(m_theta + jointAngle);
    const Scalar st = std::sin(m_theta + jointAngle);
    const Scalar ca = std::cos(m_alpha);
    const Scalar sa = std::sin(m_alpha);
    m_ft.coeffRef(0, 0) = ct;
    m_ft.coeffRef(0, 1) = -st * ca;
    m_ft.coeffRef(0, 2) = st * sa;
    m_ft.coeffRef(0, 3) = m_r * ct;
    m_ft.coeffRef(1, 0) = st;
    m_ft.coeffRef(1, 1) = ct * ca;
    m_ft.coeffRef(1, 2) = -ct * sa;
    m_ft.coeffRef(1, 3) = m_r * st;
    m_ft.coeffRef(2, 1) = sa;
    m_ft.coeffRef(2, 2) = ca;
  }

  const Scalar m_d;
  const Scalar m_theta;
  const Scalar m_r;
  const Scalar m_alpha;
  Eigen::Matrix<Scalar, 4, 4> m_ft;
};

template <typename T> class IkProblem : public cppoptlib::BoundedProblem<T> {
  public:
  using typename cppoptlib::Problem<T>::Scalar;
  using typename cppoptlib::Problem<T>::TVector;
  using FT = Eigen::Matrix<Scalar, 4, 4>;

  IkProblem(std::vector<DhParam<T>> dhParams, const FT target, const TVector &l, const TVector &u)
    : cppoptlib::BoundedProblem<T>(l, u),
      m_target(std::move(target)),
      m_dhParams(std::move(dhParams)) {
  }

  T value(const TVector &jointAngles) override {
    FT tip = FT::Identity();
    for (size_t i = 0; i < m_dhParams.size(); ++i) {
      m_dhParams[i].computeFT(jointAngles[i]);
      tip *= m_dhParams[i].m_ft;
    }

    const T targetX = m_target.coeff(0, 3);
    const T targetY = m_target.coeff(1, 3);
    const T targetZ = m_target.coeff(2, 3);

    const T tipX = tip.coeff(0, 3);
    const T tipY = tip.coeff(1, 3);
    const T tipZ = tip.coeff(2, 3);

    return std::sqrt(std::pow(tipX - targetX, 2) + std::pow(tipY - targetY, 2) +
                     std::pow(tipZ - targetZ, 2));
  }

  void gradient(const TVector &x, TVector &grad) override {
    // TODO: Compute jacobian
    cppoptlib::Problem<T>::finiteGradient(x, grad, 0);
  }

  const FT m_target;
  std::vector<DhParam<T>> m_dhParams;
};

using IkProblemf = IkProblem<float>;

extern "C" {
JNIEXPORT jfloatArray JNICALL
Java_com_neuronrobotics_bowlerkinematicsnative_solver_NativeIKSolver_solve(
  JNIEnv *env,
  jobject,
  jint numberOfLinks,
  jfloatArray dhParamsJArray,
  jfloatArray upperJointLimitsJArray,
  jfloatArray lowerJointLimitsJArray,
  jfloatArray initialJointAnglesJArray,
  jfloatArray targetJArray) {
  jfloat *dhParamsData = env->GetFloatArrayElements(dhParamsJArray, nullptr);
  std::vector<DhParam<float>> dhParams;
  dhParams.reserve(numberOfLinks);
  for (int i = 0; i < numberOfLinks; ++i) {
    dhParams.emplace_back(dhParamsData[i * 4 + 0],
                          dhParamsData[i * 4 + 1],
                          dhParamsData[i * 4 + 2],
                          dhParamsData[i * 4 + 3]);
  }
  env->ReleaseFloatArrayElements(dhParamsJArray, dhParamsData, 0);

  jfloat *upperJointLimitsData = env->GetFloatArrayElements(upperJointLimitsJArray, nullptr);
  Eigen::VectorXf upperJointLimits(numberOfLinks);
  for (int i = 0; i < numberOfLinks; ++i) {
    upperJointLimits.coeffRef(i) = upperJointLimitsData[i];
  }
  env->ReleaseFloatArrayElements(upperJointLimitsJArray, upperJointLimitsData, 0);

  jfloat *lowerJointLimitsData = env->GetFloatArrayElements(lowerJointLimitsJArray, nullptr);
  Eigen::VectorXf lowerJointLimits(numberOfLinks);
  for (int i = 0; i < numberOfLinks; ++i) {
    lowerJointLimits.coeffRef(i) = lowerJointLimitsData[i];
  }
  env->ReleaseFloatArrayElements(lowerJointLimitsJArray, lowerJointLimitsData, 0);

  jfloat *initialJointAnglesData = env->GetFloatArrayElements(initialJointAnglesJArray, nullptr);
  Eigen::VectorXf initialJointAngles(numberOfLinks);
  for (int i = 0; i < numberOfLinks; ++i) {
    initialJointAngles.coeffRef(i) = initialJointAnglesData[i];
  }
  env->ReleaseFloatArrayElements(initialJointAnglesJArray, initialJointAnglesData, 0);

  jfloat *targetData = env->GetFloatArrayElements(targetJArray, nullptr);
  Eigen::Matrix4f target;
  for (int i = 0; i < 16; ++i) {
    const int row = i / 4;
    const int col = i % 4;
    target.coeffRef(row, col) = targetData[i];
  }
  env->ReleaseFloatArrayElements(targetJArray, targetData, 0);

  IkProblemf f(std::move(dhParams), std::move(target), lowerJointLimits, upperJointLimits);

  cppoptlib::LbfgsbSolver<IkProblemf> solver;
  solver.minimize(f, initialJointAngles);
  jfloatArray result = env->NewFloatArray(numberOfLinks);
  env->SetFloatArrayRegion(result, 0, numberOfLinks, initialJointAngles.data());
  return result;
}
}
