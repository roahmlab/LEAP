#pragma once

// Local h-refinement of a TrajectoryProblem: bisect one element in physical time and
// prolong an iterate exactly onto the refined mesh. Generic over the attachment kinds of
// the formulation API, so any problem built with it can run the adaptive pipeline.

#include <vector>

#include <Eigen/Dense>

#include "leap/problem/compiled_problem.hpp"

namespace leap {

// Where an attachment of a parent element lands on child i of k children, with unique
// junction ownership: a row reading only q and v is stated once at an interior junction
// (on the LEFT child's last node); rows reading a, lam or u stay on both junction nodes.
std::vector<Where> mapWhereUniqueOwner(Where where, int i, int k, bool qvOnly);

// Bisect `element` at the physical-time fraction. The children keep degree, contacts,
// control-point count, costs and defect cost; node terms are mapped by
// mapWhereUniqueOwner; off-node attachments are copied to both children, except hard
// dynamics rows at fixed taus (which follow their physical points) and a force pin named
// "pin_lam_liftoff" (which belongs to the right child, the physical endpoint).
TrajectoryProblem splitElement(const TrajectoryProblem& parent, int element, double fraction);

// q, v, a, u of the two children are the parent polynomials at the child nodes; the
// force polygon is subdivided by de Casteljau. Duals zero. Returns the measured
// reproduction error of the children against the parent.
double prolongElementSplit(const CompiledProblem& parent, const Eigen::VectorXd& yParent, const CompiledProblem& refined,
                           int element, double fraction, Eigen::VectorXd& yRefined);

// Several bisections of `from`'s mesh, given in the order they were applied (each element
// index refers to the mesh at that moment), replayed one at a time on their own compiled
// intermediates; yTo receives the prolonged primal with zero duals.
double prolongSplits(const CompiledProblem& from, const Eigen::VectorXd& yFrom, const std::vector<int>& elements,
                     const std::vector<double>& fractions, const CompiledProblem& to, Eigen::VectorXd& yTo);

}  // namespace leap
