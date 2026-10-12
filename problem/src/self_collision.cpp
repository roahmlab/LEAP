#include "leap/problem/self_collision.hpp"

#include <cmath>
#include <stdexcept>

namespace leap {

namespace {
double clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }
}  // namespace

SegmentSegmentClosest closestSegmentSegment(const Eigen::Vector3d& A1, const Eigen::Vector3d& B1,
                                            const Eigen::Vector3d& A2, const Eigen::Vector3d& B2) {
  static constexpr double kMinAxis2 = 1e-18;
  const Eigen::Vector3d d1 = B1 - A1;
  const Eigen::Vector3d d2 = B2 - A2;
  const Eigen::Vector3d r = A1 - A2;
  const double a = d1.squaredNorm();
  const double e = d2.squaredNorm();
  const double f = d2.dot(r);
  SegmentSegmentClosest out;
  if (a <= kMinAxis2 && e <= kMinAxis2) {
    // two points
  } else if (a <= kMinAxis2) {
    out.s = clamp01(f / e);
    out.interiorS = f / e > 0.0 && f / e < 1.0;
  } else if (e <= kMinAxis2) {
    const double traw = -d1.dot(r) / a;
    out.t = clamp01(traw);
    out.interiorT = traw > 0.0 && traw < 1.0;
  } else {
    const double c = d1.dot(r);
    const double b = d1.dot(d2);
    const double denom = a * e - b * b;
    double traw = denom > kMinAxis2 * a * e ? (b * f - c * e) / denom : 0.0;
    out.t = clamp01(traw);
    const double sraw = (b * out.t + f) / e;
    if (sraw < 0.0) {
      out.s = 0.0;
      traw = -c / a;
      out.t = clamp01(traw);
    } else if (sraw > 1.0) {
      out.s = 1.0;
      traw = (b - c) / a;
      out.t = clamp01(traw);
    } else {
      out.s = sraw;
      out.interiorS = sraw > 0.0 && sraw < 1.0;
    }
    out.interiorT = traw > 0.0 && traw < 1.0;
  }
  out.w = r + out.t * d1 - out.s * d2;
  out.dist2 = out.w.squaredNorm();
  return out;
}

double capsulePairResidual(const Eigen::Vector3d& A1, const Eigen::Vector3d& B1, const Eigen::Vector3d& A2,
                           const Eigen::Vector3d& B2, double dsafe) {
  const SegmentSegmentClosest g = closestSegmentSegment(A1, B1, A2, B2);
  return (dsafe * dsafe - g.dist2) / (2.0 * dsafe);
}

CapsuleSelfCollisionConstraint::CapsuleSelfCollisionConstraint(const RobotModel& m,
                                                               std::vector<std::pair<int, int>> pairs, double margin,
                                                               std::string name, std::vector<double> perPairMargin)
    : pairs_(std::move(pairs)), margin_(margin), name_(std::move(name)) {
  if (pairs_.empty()) throw std::invalid_argument(name_ + ": no capsule pairs");
  if (!std::isfinite(margin_) || margin_ < 0.0) throw std::invalid_argument(name_ + ": margin must be finite and >= 0");
  if (!perPairMargin.empty() && perPairMargin.size() != pairs_.size())
    throw std::invalid_argument(name_ + ": perPairMargin size mismatch");
  const int pointBase = m.nContacts() + m.nMonitors();
  for (size_t r = 0; r < pairs_.size(); ++r) {
    const int i = pairs_[r].first, j = pairs_[r].second;
    if (i < 0 || i >= m.nCapsules() || j < 0 || j >= m.nCapsules() || i == j)
      throw std::invalid_argument(name_ + ": capsule pair index out of range");
    cacheA1_.push_back(2 * i);
    cacheB1_.push_back(2 * i + 1);
    cacheA2_.push_back(2 * j);
    cacheB2_.push_back(2 * j + 1);
    pointA1_.push_back(pointBase + 2 * i);
    pointB1_.push_back(pointBase + 2 * i + 1);
    pointA2_.push_back(pointBase + 2 * j);
    pointB2_.push_back(pointBase + 2 * j + 1);
    labels_.push_back(m.capsuleName(i) + " vs " + m.capsuleName(j));
    const double mr = perPairMargin.empty() ? margin_ : perPairMargin[r];
    if (!std::isfinite(mr) || mr < 0.0) throw std::invalid_argument(name_ + ": margin must be finite and >= 0");
    const double ds = m.capsuleRadius(i) + m.capsuleRadius(j) + mr;
    if (!(ds > 0.0)) throw std::invalid_argument(name_ + ": radii + margin must be positive");
    dsafe_.push_back(ds);
  }
}

void CapsuleSelfCollisionConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                           Eigen::Ref<Eigen::VectorXd> g) const {
  for (size_t r = 0; r < pairs_.size(); ++r) {
    const SegmentSegmentClosest s = closestSegmentSegment(
        c.capPos[static_cast<size_t>(cacheA1_[r])], c.capPos[static_cast<size_t>(cacheB1_[r])],
        c.capPos[static_cast<size_t>(cacheA2_[r])], c.capPos[static_cast<size_t>(cacheB2_[r])]);
    const double ds = dsafe_[r];
    g(static_cast<Eigen::Index>(r)) = (ds * ds - s.dist2) / (2.0 * ds);
  }
}

void CapsuleSelfCollisionConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c,
                                              MatRef J) const {
  J.setZero();
  for (size_t r = 0; r < pairs_.size(); ++r) {
    const SegmentSegmentClosest s = closestSegmentSegment(
        c.capPos[static_cast<size_t>(cacheA1_[r])], c.capPos[static_cast<size_t>(cacheB1_[r])],
        c.capPos[static_cast<size_t>(cacheA2_[r])], c.capPos[static_cast<size_t>(cacheB2_[r])]);
    const Eigen::MatrixXd& JA1 = c.capJ[static_cast<size_t>(cacheA1_[r])];
    const Eigen::MatrixXd& JB1 = c.capJ[static_cast<size_t>(cacheB1_[r])];
    const Eigen::MatrixXd& JA2 = c.capJ[static_cast<size_t>(cacheA2_[r])];
    const Eigen::MatrixXd& JB2 = c.capJ[static_cast<size_t>(cacheB2_[r])];
    J.row(static_cast<Eigen::Index>(r)).segment(d.offQ(), d.nq) =
        (-1.0 / dsafe_[r]) * (s.w.transpose() * ((1.0 - s.t) * JA1.leftCols(d.nq) + s.t * JB1.leftCols(d.nq) -
                                                 (1.0 - s.s) * JA2.leftCols(d.nq) - s.s * JB2.leftCols(d.nq)));
  }
}

void CapsuleSelfCollisionConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d,
                                                          HessianScratch& sc,
                                                          const Eigen::Ref<const Eigen::VectorXd>& w,
                                                          Eigen::MatrixXd& H) const {
  if (sc.hessians == nullptr) throw std::logic_error(name_ + ": exact Hessian requested without a provider");
  bool kinematics = false;
  bool zPacked = false;
  for (size_t r = 0; r < pairs_.size(); ++r) {
    const double wr = w(static_cast<Eigen::Index>(r));
    if (wr == 0.0) continue;
    if (!kinematics) {
      sc.model->capsuleKinematics(*sc.data, z.q, sc.capPos, sc.capJ);
      kinematics = true;
    }
    const Eigen::Vector3d& A1 = sc.capPos[static_cast<size_t>(cacheA1_[r])];
    const Eigen::Vector3d& B1 = sc.capPos[static_cast<size_t>(cacheB1_[r])];
    const Eigen::Vector3d& A2 = sc.capPos[static_cast<size_t>(cacheA2_[r])];
    const Eigen::Vector3d& B2 = sc.capPos[static_cast<size_t>(cacheB2_[r])];
    const Eigen::MatrixXd& JA1 = sc.capJ[static_cast<size_t>(cacheA1_[r])];
    const Eigen::MatrixXd& JB1 = sc.capJ[static_cast<size_t>(cacheB1_[r])];
    const Eigen::MatrixXd& JA2 = sc.capJ[static_cast<size_t>(cacheA2_[r])];
    const Eigen::MatrixXd& JB2 = sc.capJ[static_cast<size_t>(cacheB2_[r])];
    const SegmentSegmentClosest s = closestSegmentSegment(A1, B1, A2, B2);
    const double k = -wr / dsafe_[r];

    sc.Jp.resize(3, d.nq);
    sc.Jp.noalias() = (1.0 - s.t) * JA1.leftCols(d.nq);
    sc.Jp.noalias() += s.t * JB1.leftCols(d.nq);
    sc.Jp.noalias() -= (1.0 - s.s) * JA2.leftCols(d.nq);
    sc.Jp.noalias() -= s.s * JB2.leftCols(d.nq);
    H.block(d.offQ(), d.offQ(), d.nq, d.nq).noalias() += k * sc.Jp.transpose() * sc.Jp;

    const Eigen::Vector3d u1 = B1 - A1;
    const Eigen::Vector3d u2 = B2 - A2;
    if (s.interiorT || s.interiorS) {
      Eigen::VectorXd ft, fs;
      if (s.interiorT) {
        ft = 2.0 * (sc.Jp.transpose() * u1);
        ft.noalias() += 2.0 * ((JB1.leftCols(d.nq) - JA1.leftCols(d.nq)).transpose() * s.w);
      }
      if (s.interiorS) {
        fs = -2.0 * (sc.Jp.transpose() * u2);
        fs.noalias() -= 2.0 * ((JB2.leftCols(d.nq) - JA2.leftCols(d.nq)).transpose() * s.w);
      }
      const double ftt = 2.0 * u1.squaredNorm();
      const double fss = 2.0 * u2.squaredNorm();
      const double fts = -2.0 * u1.dot(u2);
      if (s.interiorT && s.interiorS) {
        const double det = ftt * fss - fts * fts;
        if (det > 1e-18 * ftt * fss)
          H.block(d.offQ(), d.offQ(), d.nq, d.nq).noalias() -=
              (0.5 * k / det) * (fss * ft * ft.transpose() - fts * ft * fs.transpose() - fts * fs * ft.transpose() +
                                 ftt * fs * fs.transpose());
      } else if (s.interiorT) {
        if (ftt > 1e-18) H.block(d.offQ(), d.offQ(), d.nq, d.nq).noalias() -= (0.5 * k / ftt) * ft * ft.transpose();
      } else {
        if (fss > 1e-18) H.block(d.offQ(), d.offQ(), d.nq, d.nq).noalias() -= (0.5 * k / fss) * fs * fs.transpose();
      }
    }

    if (!zPacked) {
      sc.z.resize(d.nP());
      z.toVector(d, sc.z);
      zPacked = true;
    }
    if (s.t < 1.0) {
      sc.w3 = (k * (1.0 - s.t)) * s.w;
      sc.hessians->addFrameHessian(FrameTerm::Position, pointA1_[r], d, sc.z, sc.w3, H);
    }
    if (s.t > 0.0) {
      sc.w3 = (k * s.t) * s.w;
      sc.hessians->addFrameHessian(FrameTerm::Position, pointB1_[r], d, sc.z, sc.w3, H);
    }
    if (s.s < 1.0) {
      sc.w3 = (-k * (1.0 - s.s)) * s.w;
      sc.hessians->addFrameHessian(FrameTerm::Position, pointA2_[r], d, sc.z, sc.w3, H);
    }
    if (s.s > 0.0) {
      sc.w3 = (-k * s.s) * s.w;
      sc.hessians->addFrameHessian(FrameTerm::Position, pointB2_[r], d, sc.z, sc.w3, H);
    }
  }
}

}  // namespace leap
