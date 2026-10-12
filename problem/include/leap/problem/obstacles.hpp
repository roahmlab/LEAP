#pragma once

// Obstacle avoidance: keep the robot's capsule collision model clear of static spheres
// and cuboids. A capsule is the segment A(q) -> B(q) between two body-fixed points swept
// by radius r. Rows read only q (through kCapsulePos); row order is capsule-major
// (row = capsule * nObstacles + obstacle).
//
// Sphere (centre c, radius R), dsafe = r + R + margin, e = closest axis point - c:
//     g = (dsafe^2 - |e|^2) / (2 dsafe) <= 0
// (no square root, so finite everywhere; equals dsafe - |e| to first order on the boundary).
//
// Cuboid (centre c, half extents h along its axes, rotation R whose columns are the axes),
// dsafe = r + margin. The default row is the SEPARATING-FACE CERTIFICATE
//     g = dsafe - max_{face} min_{endpoint} offset_face(endpoint) <= 0,
// offset = s y_k - h_k with y = R'(p - c): some face plane pushed out by dsafe has the
// whole axis beyond it. It is conservative (the keep-out is the box grown by dsafe per
// face), exact and affine in the winning endpoint, and its gradient is a unit box axis
// that never vanishes, which is what pulls a limb out of a fence the seed put it through.
// A box standing on the floor drops its bottom face (unreachable).
// Variants: `exactOutside` uses the exact rounded distance once the axis is outside the
// box (dsafe - dist); `smoothEps > 0` is the log-sum-exp surrogate used by the first
// restoration stage (softmax over faces, softmin over endpoints).

#include <string>
#include <vector>

#include "leap/problem/constraint.hpp"

namespace leap {

struct SphereObstacle {
  std::string name;
  Eigen::Vector3d center{0.0, 0.0, 0.0};
  double radius = 0.0;
};

struct CuboidObstacle {
  std::string name;
  Eigen::Vector3d center{0.0, 0.0, 0.0};
  Eigen::Vector3d half{0.0, 0.0, 0.0};
  Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
};

// Rz(yaw) Ry(pitch) Rx(roll).
Eigen::Matrix3d rotationFromRpy(double roll, double pitch, double yaw);

// Closest point of a segment to a point.
struct SegmentPointClosest {
  double t = 0.0;
  Eigen::Vector3d e{0.0, 0.0, 0.0};  // A + t (B - A) - c
  double dist2 = 0.0;
  double uu = 0.0;  // |B - A|^2
  bool interior = false;
};
SegmentPointClosest closestOnSegment(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& c);
double capsuleSphereResidual(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& c,
                             double dsafe);

// The best separating face of a box against a segment.
struct SegmentBoxSeparation {
  int axis = 0;
  double sign = 1.0;
  int end = 0;  // 0 = endpoint A attains the face's minimum offset, 1 = B
  double offset = 0.0;
  Eigen::Vector3d normal{1.0, 0.0, 0.0};
};
inline constexpr double kObstacleGroundZ = 0.0;
bool bottomFaceUnreachable(const CuboidObstacle& box, double dsafe, double groundZ = kObstacleGroundZ);
SegmentBoxSeparation separateSegmentBox(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const CuboidObstacle& box,
                                        bool skipBottomFace = false);

// Exact closest points of a segment and an oriented box.
struct SegmentBoxClosest {
  double t = 0.0;
  Eigen::Vector3d point{0.0, 0.0, 0.0};
  Eigen::Vector3d boxPoint{0.0, 0.0, 0.0};
  Eigen::Vector3d delta{0.0, 0.0, 0.0};
  double dist2 = 0.0;
};
SegmentBoxClosest closestSegmentBox(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const CuboidObstacle& box);
double capsuleCuboidResidual(const Eigen::Vector3d& A, const Eigen::Vector3d& B, const CuboidObstacle& box,
                             double dsafe);

class CapsuleSphereConstraint final : public Constraint {
public:
  // `capsules` indexes the model's capsule table (empty = all); `capsuleMargins`
  // optionally overrides `margin` per enforced capsule.
  CapsuleSphereConstraint(const RobotModel& m, std::vector<SphereObstacle> obstacles, std::vector<int> capsules,
                          double margin, std::string name, std::vector<double> capsuleMargins = {});
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(caps_.size() * obstacles_.size()); }
  Sense sense() const override { return Sense::Inequality; }
  unsigned modelNeeds() const override { return kCapsulePos; }
  std::string rowLabel(int r) const;
  const std::vector<SphereObstacle>& obstacles() const { return obstacles_; }
  const std::vector<int>& capsuleIndices() const { return caps_; }
  double margin() const { return margin_; }
  const std::vector<double>& capsuleMargins() const { return margins_; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  std::vector<SphereObstacle> obstacles_;
  std::vector<int> caps_;
  std::vector<int> cacheA_, cacheB_, pointA_, pointB_;
  std::vector<std::string> labels_;
  std::vector<double> dsafe_;
  double margin_;
  std::vector<double> margins_;
  std::string name_;
};

class CapsuleCuboidConstraint final : public Constraint {
public:
  CapsuleCuboidConstraint(const RobotModel& m, std::vector<CuboidObstacle> boxes, std::vector<int> capsules,
                          double margin, std::string name, bool exactOutside = false,
                          std::vector<double> capsuleMargins = {}, double smoothEps = 0.0);
  const std::string& name() const override { return name_; }
  int rows(const NodeDims&) const override { return static_cast<int>(caps_.size() * boxes_.size()); }
  Sense sense() const override { return Sense::Inequality; }
  unsigned modelNeeds() const override { return kCapsulePos; }
  std::string rowLabel(int r) const;
  const std::vector<CuboidObstacle>& boxes() const { return boxes_; }
  const std::vector<int>& capsuleIndices() const { return caps_; }
  double margin() const { return margin_; }
  const std::vector<double>& capsuleMargins() const { return margins_; }
  double dsafe(int r) const { return dsafe_.at(static_cast<size_t>(r)); }
  double smoothEps() const { return smoothEps_; }
  bool exactOutside() const { return exactOutside_; }
  void value(const NodeQuantities&, const NodeDims&, const ModelEvalCache& c, Eigen::Ref<Eigen::VectorXd> g) const override;
  void jacobian(const NodeQuantities&, const NodeDims& d, const ModelEvalCache& c, MatRef J) const override;
  void addContractedHessian(const NodeQuantities& z, const NodeDims& d, HessianScratch& s,
                            const Eigen::Ref<const Eigen::VectorXd>& w, Eigen::MatrixXd& H) const override;

private:
  struct SmoothSeparation {
    double offset = 0.0;
    Eigen::Vector3d nA{0.0, 0.0, 0.0}, nB{0.0, 0.0, 0.0};
  };
  SegmentBoxSeparation separation(const Eigen::Vector3d& A, const Eigen::Vector3d& B, size_t o, int r) const;
  SmoothSeparation smoothSeparation(const Eigen::Vector3d& A, const Eigen::Vector3d& B, size_t o, int r) const;
  bool useExactOutside(const Eigen::Vector3d& A, const Eigen::Vector3d& B, size_t o, SegmentBoxClosest& closest) const;

  std::vector<CuboidObstacle> boxes_;
  std::vector<int> caps_;
  std::vector<int> cacheA_, cacheB_, pointA_, pointB_;
  std::vector<std::string> labels_;
  std::vector<double> dsafe_;
  std::vector<char> skipBottom_;
  double margin_;
  std::vector<double> margins_;
  std::string name_;
  bool exactOutside_;
  double smoothEps_;
};

}  // namespace leap
