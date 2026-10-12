#include "leap/problem/obstacles.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace leap {

Eigen::Matrix3d rotationFromRpy(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

SegmentPointClosest closestOnSegment(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& c) {
  static constexpr double kMinAxis2 = 1e-18;
  SegmentPointClosest out;
  const Eigen::Vector3d u = B - A;
  const Eigen::Vector3d w = A - c;
  out.uu = u.squaredNorm();
  if (out.uu > kMinAxis2) {
    const double traw = -w.dot(u) / out.uu;
    out.t = traw < 0.0 ? 0.0 : (traw > 1.0 ? 1.0 : traw);
    out.interior = traw > 0.0 && traw < 1.0;
  }
  out.e = w + out.t * u;
  out.dist2 = out.e.squaredNorm();
  return out;
}

double capsuleSphereResidual(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& c,
                             double dsafe) {
  const SegmentPointClosest g = closestOnSegment(A, B, c);
  return (dsafe * dsafe - g.dist2) / (2.0 * dsafe);
}

bool bottomFaceUnreachable(const CuboidObstacle& box, double dsafe, double groundZ) {
  const Eigen::Vector3d down = -box.rot.col(2);
  if (!(down.z() < -1.0 + 1e-12)) return false;
  return box.center.z() - box.half.z() - dsafe <= groundZ;
}

SegmentBoxSeparation separateSegmentBox(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const CuboidObstacle& box,
                                        bool skipBottomFace) {
  const Eigen::Vector3d yA = box.rot.transpose() * (A - box.center);
  const Eigen::Vector3d yB = box.rot.transpose() * (B - box.center);
  SegmentBoxSeparation out;
  out.offset = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < 3; ++k) {
    for (double s : {1.0, -1.0}) {
      if (skipBottomFace && k == 2 && s == -1.0) continue;
      const double oa = s * yA(k) - box.half(k);
      const double ob = s * yB(k) - box.half(k);
      const int end = oa <= ob ? 0 : 1;  // tie-break: A
      const double off = end == 0 ? oa : ob;
      if (off > out.offset) {  // strict: keeps the lowest axis, then +1
        out.offset = off;
        out.axis = k;
        out.sign = s;
        out.end = end;
      }
    }
  }
  out.normal = out.sign * box.rot.col(out.axis);
  return out;
}

namespace {

void considerBoxCandidate(const Eigen::Vector3d& a, const Eigen::Vector3d& u, const CuboidObstacle& box, double t,
                          SegmentBoxClosest& best) {
  t = std::clamp(t, 0.0, 1.0);
  SegmentBoxClosest candidate;
  candidate.t = t;
  candidate.point = a + t * u;
  candidate.boxPoint = candidate.point.cwiseMax(-box.half).cwiseMin(box.half);
  candidate.delta = candidate.point - candidate.boxPoint;
  candidate.dist2 = candidate.delta.squaredNorm();
  if (candidate.dist2 < best.dist2) best = candidate;
}

bool nearlyEqual(double x, double y) { return std::abs(x - y) <= 32.0 * std::numeric_limits<double>::epsilon(); }

}  // namespace

SegmentBoxClosest closestSegmentBox(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const CuboidObstacle& box) {
  const Eigen::Vector3d a = box.rot.transpose() * (A - box.center);
  const Eigen::Vector3d u = box.rot.transpose() * (B - A);
  std::vector<double> cuts{0.0, 1.0};
  for (int k = 0; k < 3; ++k) {
    if (std::abs(u(k)) <= 1e-15) continue;
    for (double s : {1.0, -1.0}) {
      const double t = (s * box.half(k) - a(k)) / u(k);
      if (t > 0.0 && t < 1.0) cuts.push_back(t);
    }
  }
  std::sort(cuts.begin(), cuts.end());
  cuts.erase(std::unique(cuts.begin(), cuts.end(), nearlyEqual), cuts.end());

  SegmentBoxClosest best;
  best.dist2 = std::numeric_limits<double>::infinity();
  for (double t : cuts) considerBoxCandidate(a, u, box, t, best);
  for (size_t j = 0; j + 1 < cuts.size(); ++j) {
    const double lo = cuts[j], hi = cuts[j + 1];
    const double mid = 0.5 * (lo + hi);
    const Eigen::Vector3d p = a + mid * u;
    double numerator = 0.0, denominator = 0.0;
    for (int k = 0; k < 3; ++k) {
      double boundary = 0.0;
      bool active = false;
      if (p(k) > box.half(k)) {
        boundary = box.half(k);
        active = true;
      } else if (p(k) < -box.half(k)) {
        boundary = -box.half(k);
        active = true;
      }
      if (active) {
        numerator += u(k) * (a(k) - boundary);
        denominator += u(k) * u(k);
      }
    }
    if (denominator > 0.0)
      considerBoxCandidate(a, u, box, std::clamp(-numerator / denominator, lo, hi), best);
    else
      considerBoxCandidate(a, u, box, mid, best);
  }
  return best;
}

double capsuleCuboidResidual(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const CuboidObstacle& box,
                             double dsafe) {
  return dsafe - separateSegmentBox(A, B, box, bottomFaceUnreachable(box, dsafe)).offset;
}

// --- spheres ---------------------------------------------------------------------------------

CapsuleSphereConstraint::CapsuleSphereConstraint(const RobotModel& m, std::vector<SphereObstacle> obstacles,
                                                 std::vector<int> capsules, double margin, std::string name,
                                                 std::vector<double> capsuleMargins)
    : obstacles_(std::move(obstacles)),
      caps_(std::move(capsules)),
      margin_(margin),
      margins_(std::move(capsuleMargins)),
      name_(std::move(name)) {
  if (obstacles_.empty()) throw std::invalid_argument(name_ + ": no obstacles");
  if (!std::isfinite(margin_) || margin_ < 0.0) throw std::invalid_argument(name_ + ": margin must be finite and >= 0");
  if (!margins_.empty() && margins_.size() != caps_.size())
    throw std::invalid_argument(name_ + ": capsuleMargins must match the enforced-capsule list");
  for (double mg : margins_)
    if (!std::isfinite(mg) || mg < 0.0) throw std::invalid_argument(name_ + ": every capsule margin must be finite and >= 0");
  if (caps_.empty())
    for (int i = 0; i < m.nCapsules(); ++i) caps_.push_back(i);
  if (caps_.empty()) throw std::invalid_argument(name_ + ": the robot declares no collision capsules");
  for (const SphereObstacle& o : obstacles_)
    if (!(o.radius >= 0.0) || !std::isfinite(o.radius) || !o.center.allFinite())
      throw std::invalid_argument(name_ + ": obstacle '" + o.name + "' has a bad center/radius");
  for (size_t k = 0; k < caps_.size(); ++k) {
    const int ci = caps_[k];
    if (ci < 0 || ci >= m.nCapsules()) throw std::invalid_argument(name_ + ": capsule index out of range");
    cacheA_.push_back(2 * ci);
    cacheB_.push_back(2 * ci + 1);
    pointA_.push_back(m.nContacts() + m.nMonitors() + 2 * ci);
    pointB_.push_back(m.nContacts() + m.nMonitors() + 2 * ci + 1);
    labels_.push_back(m.capsuleName(ci));
    const double mg = margins_.empty() ? margin_ : margins_[k];
    for (const SphereObstacle& o : obstacles_) dsafe_.push_back(m.capsuleRadius(ci) + o.radius + mg);
  }
  for (double ds : dsafe_)
    if (!(ds > 0.0)) throw std::invalid_argument(name_ + ": capsule radius + obstacle radius + margin must be positive");
}

std::string CapsuleSphereConstraint::rowLabel(int r) const {
  const int nObs = static_cast<int>(obstacles_.size());
  return labels_.at(static_cast<size_t>(r / nObs)) + " vs " + obstacles_.at(static_cast<size_t>(r % nObs)).name;
}

void CapsuleSphereConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                    Eigen::Ref<Eigen::VectorXd> g) const {
  int r = 0;
  for (size_t i = 0; i < caps_.size(); ++i) {
    const Eigen::Vector3d& A = c.capPos[static_cast<size_t>(cacheA_[i])];
    const Eigen::Vector3d& B = c.capPos[static_cast<size_t>(cacheB_[i])];
    for (size_t o = 0; o < obstacles_.size(); ++o, ++r) {
      const double ds = dsafe_[static_cast<size_t>(r)];
      const SegmentPointClosest s = closestOnSegment(A, B, obstacles_[o].center);
      g(r) = (ds * ds - s.dist2) / (2.0 * ds);
    }
  }
}

void CapsuleSphereConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const {
  J.setZero();
  int r = 0;
  for (size_t i = 0; i < caps_.size(); ++i) {
    const Eigen::Vector3d& A = c.capPos[static_cast<size_t>(cacheA_[i])];
    const Eigen::Vector3d& B = c.capPos[static_cast<size_t>(cacheB_[i])];
    const Eigen::MatrixXd& JA = c.capJ[static_cast<size_t>(cacheA_[i])];
    const Eigen::MatrixXd& JB = c.capJ[static_cast<size_t>(cacheB_[i])];
    for (size_t o = 0; o < obstacles_.size(); ++o, ++r) {
      const double ds = dsafe_[static_cast<size_t>(r)];
      const SegmentPointClosest s = closestOnSegment(A, B, obstacles_[o].center);
      J.row(r).segment(d.offQ(), d.nq) =
          (-1.0 / ds) * (s.e.transpose() * ((1.0 - s.t) * JA.leftCols(d.nq) + s.t * JB.leftCols(d.nq)));
    }
  }
}

void CapsuleSphereConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                   const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  if (s.hessians == nullptr) throw std::logic_error(name_ + ": exact Hessian requested without a provider");
  s.model->capsuleKinematics(*s.data, z.q, s.capPos, s.capJ);
  s.z.resize(d.nP());
  z.toVector(d, s.z);
  int r = 0;
  for (size_t i = 0; i < caps_.size(); ++i) {
    const Eigen::Vector3d& A = s.capPos[static_cast<size_t>(cacheA_[i])];
    const Eigen::Vector3d& B = s.capPos[static_cast<size_t>(cacheB_[i])];
    const Eigen::MatrixXd& JA = s.capJ[static_cast<size_t>(cacheA_[i])];
    const Eigen::MatrixXd& JB = s.capJ[static_cast<size_t>(cacheB_[i])];
    for (size_t o = 0; o < obstacles_.size(); ++o, ++r) {
      const double wr = w(r);
      if (wr == 0.0) continue;
      const double ds = dsafe_[static_cast<size_t>(r)];
      const SegmentPointClosest cl = closestOnSegment(A, B, obstacles_[o].center);
      const double k = -wr / ds;
      s.Jp.resize(3, d.nq);
      s.Jp.noalias() = (1.0 - cl.t) * JA.leftCols(d.nq) + cl.t * JB.leftCols(d.nq);
      H.block(d.offQ(), d.offQ(), d.nq, d.nq).noalias() += k * s.Jp.transpose() * s.Jp;
      if (cl.interior) {
        const Eigen::Vector3d u = B - A;
        s.gq.resize(d.nq);
        s.gq.noalias() = JA.leftCols(d.nq).transpose() * Eigen::Vector3d((1.0 - cl.t) * u - cl.e);
        s.gq.noalias() += JB.leftCols(d.nq).transpose() * Eigen::Vector3d(cl.t * u + cl.e);
        H.block(d.offQ(), d.offQ(), d.nq, d.nq).noalias() -= (k / cl.uu) * s.gq * s.gq.transpose();
      }
      if (cl.t < 1.0) {
        s.w3 = (k * (1.0 - cl.t)) * cl.e;
        s.hessians->addFrameHessian(FrameTerm::Position, pointA_[i], d, s.z, s.w3, H);
      }
      if (cl.t > 0.0) {
        s.w3 = (k * cl.t) * cl.e;
        s.hessians->addFrameHessian(FrameTerm::Position, pointB_[i], d, s.z, s.w3, H);
      }
    }
  }
}

// --- cuboids ---------------------------------------------------------------------------------

CapsuleCuboidConstraint::CapsuleCuboidConstraint(const RobotModel& m, std::vector<CuboidObstacle> boxes,
                                                 std::vector<int> capsules, double margin, std::string name,
                                                 bool exactOutside, std::vector<double> capsuleMargins,
                                                 double smoothEps)
    : boxes_(std::move(boxes)),
      caps_(std::move(capsules)),
      margin_(margin),
      margins_(std::move(capsuleMargins)),
      name_(std::move(name)),
      exactOutside_(exactOutside),
      smoothEps_(smoothEps) {
  if (boxes_.empty()) throw std::invalid_argument(name_ + ": no cuboid obstacles");
  if (!std::isfinite(smoothEps_) || smoothEps_ < 0.0) throw std::invalid_argument(name_ + ": smoothEps must be finite and >= 0");
  if (!std::isfinite(margin_) || margin_ < 0.0) throw std::invalid_argument(name_ + ": margin must be finite and >= 0");
  if (!margins_.empty() && margins_.size() != caps_.size())
    throw std::invalid_argument(name_ + ": capsuleMargins must match the enforced-capsule list");
  for (double mg : margins_)
    if (!std::isfinite(mg) || mg < 0.0) throw std::invalid_argument(name_ + ": every capsule margin must be finite and >= 0");
  if (caps_.empty())
    for (int i = 0; i < m.nCapsules(); ++i) caps_.push_back(i);
  if (caps_.empty()) throw std::invalid_argument(name_ + ": the robot declares no collision capsules");
  for (const CuboidObstacle& b : boxes_) {
    if (!b.center.allFinite() || !b.half.allFinite() || !b.rot.allFinite())
      throw std::invalid_argument(name_ + ": cuboid '" + b.name + "' has a non-finite centre/size/orientation");
    if (!(b.half.minCoeff() > 0.0)) throw std::invalid_argument(name_ + ": cuboid '" + b.name + "' must have positive half-extents");
    if ((b.rot.transpose() * b.rot - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > 1e-9)
      throw std::invalid_argument(name_ + ": cuboid '" + b.name + "' orientation is not a rotation matrix");
  }
  for (size_t k = 0; k < caps_.size(); ++k) {
    const int ci = caps_[k];
    if (ci < 0 || ci >= m.nCapsules()) throw std::invalid_argument(name_ + ": capsule index out of range");
    cacheA_.push_back(2 * ci);
    cacheB_.push_back(2 * ci + 1);
    pointA_.push_back(m.nContacts() + m.nMonitors() + 2 * ci);
    pointB_.push_back(m.nContacts() + m.nMonitors() + 2 * ci + 1);
    labels_.push_back(m.capsuleName(ci));
    const double mg = margins_.empty() ? margin_ : margins_[k];
    for (const CuboidObstacle& b : boxes_) {
      const double ds = m.capsuleRadius(ci) + mg;
      dsafe_.push_back(ds);
      skipBottom_.push_back(bottomFaceUnreachable(b, ds) ? 1 : 0);
    }
  }
  for (double ds : dsafe_)
    if (!(ds > 0.0)) throw std::invalid_argument(name_ + ": capsule radius + margin must be positive");
}

std::string CapsuleCuboidConstraint::rowLabel(int r) const {
  const int nBox = static_cast<int>(boxes_.size());
  return labels_.at(static_cast<size_t>(r / nBox)) + " vs " + boxes_.at(static_cast<size_t>(r % nBox)).name;
}

SegmentBoxSeparation CapsuleCuboidConstraint::separation(const Eigen::Vector3d& A, const Eigen::Vector3d& B, size_t o,
                                                         int r) const {
  return separateSegmentBox(A, B, boxes_[o], skipBottom_[static_cast<size_t>(r)] != 0);
}

CapsuleCuboidConstraint::SmoothSeparation CapsuleCuboidConstraint::smoothSeparation(const Eigen::Vector3d& A,
                                                                                    const Eigen::Vector3d& B, size_t o,
                                                                                    int r) const {
  const CuboidObstacle& box = boxes_[o];
  const bool skipBottom = skipBottom_[static_cast<size_t>(r)] != 0;
  const double eps = smoothEps_;
  const Eigen::Vector3d yA = box.rot.transpose() * (A - box.center);
  const Eigen::Vector3d yB = box.rot.transpose() * (B - box.center);
  double m[6], wA[6];
  Eigen::Vector3d n[6];
  int K = 0;
  for (int k = 0; k < 3; ++k) {
    for (double sgn : {1.0, -1.0}) {
      if (skipBottom && k == 2 && sgn == -1.0) continue;
      const double oa = sgn * yA(k) - box.half(k);
      const double ob = sgn * yB(k) - box.half(k);
      const double lo = std::min(oa, ob);
      const double ea = std::exp(-(oa - lo) / eps);
      const double eb = std::exp(-(ob - lo) / eps);
      m[K] = lo - eps * std::log(ea + eb);
      wA[K] = ea / (ea + eb);
      n[K] = sgn * box.rot.col(k);
      ++K;
    }
  }
  double mMax = m[0];
  for (int k = 1; k < K; ++k) mMax = std::max(mMax, m[k]);
  double Z = 0.0;
  double p[6];
  for (int k = 0; k < K; ++k) {
    p[k] = std::exp((m[k] - mMax) / eps);
    Z += p[k];
  }
  SmoothSeparation out;
  out.offset = mMax + eps * std::log(Z) - eps * std::log(static_cast<double>(K));
  for (int k = 0; k < K; ++k) {
    const double pk = p[k] / Z;
    out.nA += pk * wA[k] * n[k];
    out.nB += pk * (1.0 - wA[k]) * n[k];
  }
  return out;
}

bool CapsuleCuboidConstraint::useExactOutside(const Eigen::Vector3d& A, const Eigen::Vector3d& B, size_t o,
                                              SegmentBoxClosest& closest) const {
  if (!exactOutside_) return false;
  closest = closestSegmentBox(A, B, boxes_[o]);
  return closest.dist2 > 1e-20;
}

void CapsuleCuboidConstraint::value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c,
                                    Eigen::Ref<Eigen::VectorXd> g) const {
  int r = 0;
  for (size_t i = 0; i < caps_.size(); ++i) {
    const Eigen::Vector3d& A = c.capPos[static_cast<size_t>(cacheA_[i])];
    const Eigen::Vector3d& B = c.capPos[static_cast<size_t>(cacheB_[i])];
    for (size_t o = 0; o < boxes_.size(); ++o, ++r) {
      SegmentBoxClosest closest;
      if (useExactOutside(A, B, o, closest))
        g(r) = dsafe_[static_cast<size_t>(r)] - std::sqrt(closest.dist2);
      else if (smoothEps_ > 0.0)
        g(r) = dsafe_[static_cast<size_t>(r)] - smoothSeparation(A, B, o, r).offset;
      else
        g(r) = dsafe_[static_cast<size_t>(r)] - separation(A, B, o, r).offset;
    }
  }
}

void CapsuleCuboidConstraint::jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const {
  J.setZero();
  int r = 0;
  for (size_t i = 0; i < caps_.size(); ++i) {
    const Eigen::Vector3d& A = c.capPos[static_cast<size_t>(cacheA_[i])];
    const Eigen::Vector3d& B = c.capPos[static_cast<size_t>(cacheB_[i])];
    const Eigen::MatrixXd& JA = c.capJ[static_cast<size_t>(cacheA_[i])];
    const Eigen::MatrixXd& JB = c.capJ[static_cast<size_t>(cacheB_[i])];
    for (size_t o = 0; o < boxes_.size(); ++o, ++r) {
      SegmentBoxClosest closest;
      if (useExactOutside(A, B, o, closest)) {
        const Eigen::Vector3d normal = boxes_[o].rot * closest.delta / std::sqrt(closest.dist2);
        J.row(r).segment(d.offQ(), d.nq) = -(1.0 - closest.t) * normal.transpose() * JA.leftCols(d.nq) -
                                           closest.t * normal.transpose() * JB.leftCols(d.nq);
      } else if (smoothEps_ > 0.0) {
        const SmoothSeparation sm = smoothSeparation(A, B, o, r);
        J.row(r).segment(d.offQ(), d.nq) = -sm.nA.transpose() * JA.leftCols(d.nq) - sm.nB.transpose() * JB.leftCols(d.nq);
      } else {
        const SegmentBoxSeparation sp = separation(A, B, o, r);
        const Eigen::MatrixXd& Je = sp.end == 0 ? JA : JB;
        J.row(r).segment(d.offQ(), d.nq) = -sp.normal.transpose() * Je.leftCols(d.nq);
      }
    }
  }
}

void CapsuleCuboidConstraint::addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                                                   const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const {
  if (s.hessians == nullptr) throw std::logic_error(name_ + ": exact Hessian requested without a provider");
  bool kinematics = false;
  bool zPacked = false;
  int r = 0;
  for (size_t i = 0; i < caps_.size(); ++i) {
    for (size_t o = 0; o < boxes_.size(); ++o, ++r) {
      const double wr = w(r);
      if (wr == 0.0) continue;
      if (!kinematics) {
        s.model->capsuleKinematics(*s.data, z.q, s.capPos, s.capJ);
        kinematics = true;
      }
      const Eigen::Vector3d& A = s.capPos[static_cast<size_t>(cacheA_[i])];
      const Eigen::Vector3d& B = s.capPos[static_cast<size_t>(cacheB_[i])];
      if (!zPacked) {
        s.z.resize(d.nP());
        z.toVector(d, s.z);
        zPacked = true;
      }
      SegmentBoxClosest closest;
      if (useExactOutside(A, B, o, closest)) {
        const Eigen::Vector3d normal = boxes_[o].rot * closest.delta / std::sqrt(closest.dist2);
        s.w3 = (-wr * (1.0 - closest.t)) * normal;
        s.hessians->addFrameHessian(FrameTerm::Position, pointA_[i], d, s.z, s.w3, H);
        s.w3 = (-wr * closest.t) * normal;
        s.hessians->addFrameHessian(FrameTerm::Position, pointB_[i], d, s.z, s.w3, H);
      } else if (smoothEps_ > 0.0) {
        const SmoothSeparation sm = smoothSeparation(A, B, o, r);
        s.w3 = (-wr) * sm.nA;
        s.hessians->addFrameHessian(FrameTerm::Position, pointA_[i], d, s.z, s.w3, H);
        s.w3 = (-wr) * sm.nB;
        s.hessians->addFrameHessian(FrameTerm::Position, pointB_[i], d, s.z, s.w3, H);
      } else {
        const SegmentBoxSeparation sp = separation(A, B, o, r);
        s.w3 = (-wr) * sp.normal;
        s.hessians->addFrameHessian(FrameTerm::Position, sp.end == 0 ? pointA_[i] : pointB_[i], d, s.z, s.w3, H);
      }
    }
  }
}

}  // namespace leap
