// Pure geometry helpers copied from LEAP obstacles.hpp and self_collision.hpp.
// No solver or trajectory code is imported. Keep source hashes in protocol.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
namespace benchmark_geometry {
struct CuboidObstacle {
  std::string name;
  Eigen::Vector3d center{0.0, 0.0, 0.0};
  Eigen::Vector3d half{0.0, 0.0, 0.0};
  Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
};

// Rz(yaw) * Ry(pitch) * Rx(roll) -- pinocchio's rpy convention, which is what
// the robot and problem yamls use everywhere else.
inline Eigen::Matrix3d rotationFromRpy(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

// The best separating face of one box against one capsule axis: which face, and
// which capsule endpoint attains that face's minimum offset. Everything the
// value, the Jacobian and the Hessian of the row need.
struct SegmentBoxSeparation {
  int axis = 0;        // box axis of the winning face (0, 1 or 2)
  double sign = 1.0;   // +1 = the +h_axis face, -1 = the -h_axis face
  int end = 0;         // 0 = endpoint A attains the minimum offset, 1 = B
  double offset = 0.0; // m: the winning (largest) minimum offset, in metres
  // Outward unit normal of the winning face, in WORLD: d(offset)/d(p_end).
  Eigen::Vector3d normal{1.0, 0.0, 0.0};
};

// The world floor every ground keep-out row in this codebase assumes.
inline constexpr double kObstacleGroundZ = 0.0;

// True when separating over `box`'s straight-down face would require the capsule
// to be under the floor, which makes that face unreachable and its contribution
// to the max spurious (see the header comment). Only a face whose outward normal
// is exactly -z qualifies: any other downward-tilted face still has feasible
// points above the floor, far enough to the side.
inline bool bottomFaceUnreachable(const CuboidObstacle& box, double dsafe,
                                  double groundZ = kObstacleGroundZ) {
  const Eigen::Vector3d down = -box.rot.col(2);  // outward normal of the -z face
  if (!(down.z() < -1.0 + 1e-12)) return false;  // not straight down
  return box.center.z() - box.half.z() - dsafe <= groundZ;
}

inline SegmentBoxSeparation separateSegmentBox(const Eigen::Vector3d& A,
                                              const Eigen::Vector3d& B,
                                              const CuboidObstacle& box,
                                              bool skipBottomFace = false) {
  const Eigen::Vector3d yA = box.rot.transpose() * (A - box.center);
  const Eigen::Vector3d yB = box.rot.transpose() * (B - box.center);
  SegmentBoxSeparation out;
  out.offset = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < 3; ++k)
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
  out.normal = out.sign * box.rot.col(out.axis);
  return out;
}

// Exact closest-point data for a segment and an oriented cuboid. The segment is
// transformed into the box frame, where squared distance to the AABB is a
// convex piecewise quadratic in t. Enumerating the six slab-crossing breakpoints
// and minimizing each interval gives the global answer without iteration.
struct SegmentBoxClosest {
  double t = 0.0;
  Eigen::Vector3d point{0.0, 0.0, 0.0};
  Eigen::Vector3d boxPoint{0.0, 0.0, 0.0};
  Eigen::Vector3d delta{0.0, 0.0, 0.0};
  double dist2 = 0.0;
};

inline SegmentBoxClosest closestSegmentBox(const Eigen::Vector3d& A,
                                           const Eigen::Vector3d& B,
                                           const CuboidObstacle& box) {
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
  cuts.erase(std::unique(cuts.begin(), cuts.end(), [](double x, double y) {
               return std::abs(x - y) <= 32.0 *
                      std::numeric_limits<double>::epsilon();
             }), cuts.end());

  SegmentBoxClosest best;
  best.dist2 = std::numeric_limits<double>::infinity();
  const auto consider = [&](double t) {
    t = std::clamp(t, 0.0, 1.0);
    SegmentBoxClosest candidate;
    candidate.t = t;
    candidate.point = a + t * u;
    candidate.boxPoint = candidate.point.cwiseMax(-box.half).cwiseMin(box.half);
    candidate.delta = candidate.point - candidate.boxPoint;
    candidate.dist2 = candidate.delta.squaredNorm();
    if (candidate.dist2 < best.dist2) best = candidate;
  };
  for (double t : cuts) consider(t);
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
      consider(std::clamp(-numerator / denominator, lo, hi));
    else
      consider(mid);
  }
  return best;
}

struct SegmentSegmentClosest {
  double t = 0.0, s = 0.0;
  Eigen::Vector3d w{0.0, 0.0, 0.0};  // P(t) - Q(s)
  double dist2 = 0.0;
  bool interiorT = false, interiorS = false;
};

inline SegmentSegmentClosest closestSegmentSegment(const Eigen::Vector3d& A1,
                                                   const Eigen::Vector3d& B1,
                                                   const Eigen::Vector3d& A2,
                                                   const Eigen::Vector3d& B2) {
  static constexpr double kMinAxis2 = 1e-18;
  const Eigen::Vector3d d1 = B1 - A1;
  const Eigen::Vector3d d2 = B2 - A2;
  const Eigen::Vector3d r = A1 - A2;
  const double a = d1.squaredNorm();
  const double e = d2.squaredNorm();
  const double f = d2.dot(r);
  SegmentSegmentClosest out;
  const auto clamp01 = [](double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); };
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
    // Parallel axes (denom ~ 0): pick t = 0, a representative of the
    // non-unique closest family.
    double traw = denom > kMinAxis2 * a * e ? (b * f - c * e) / denom : 0.0;
    out.t = clamp01(traw);
    double sraw = (b * out.t + f) / e;
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

// Signed clearance of one capsule pair, in metres: negative = separated by
// more than dsafe, positive = too close (the residual driven to <= 0).
inline double capsulePairResidual(const Eigen::Vector3d& A1, const Eigen::Vector3d& B1,
                                  const Eigen::Vector3d& A2, const Eigen::Vector3d& B2,
                                  double dsafe) {
  const SegmentSegmentClosest g = closestSegmentSegment(A1, B1, A2, B2);
  return (dsafe * dsafe - g.dist2) / (2.0 * dsafe);
}


}
