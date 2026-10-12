// hopper_jump: a vertical jump in place of the planar hopper, written against the
// formulation API from scratch (no task builder, no yaml). The tutorial
// docs/TUTORIAL.md walks through this file section by section.
//
//   push-off stance (rest, squat)  --continuity-->  flight  --plastic impact-->  landing stance (rest, stand)
//
// The foot leaves and lands on the same spot; the durations and the hop are the ones of
// the paper's hopper2d fence instance (a flight of 0.391 s carries a 0.17 m hop:
// tair >= 2 sqrt(2 g hop) / g = 0.372 s).
//
// usage: build/examples/hopper_jump [--root <new_src>] [--hop 0.17] [--tstance 0.8] [--tair 0.391]
//                                   [--out results/hopper_jump] [--levels 6] [--verbose 0|1]
// Prints the solve log, the judges and a short report, and writes <out>.h5.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "leap/alm/evaluator.hpp"
#include "leap/io/solution_archive.hpp"
#include "leap/model/robot_description.hpp"
#include "leap/model/robot_model.hpp"
#include "leap/problem/compiled_problem.hpp"
#include "leap/problem/constraints.hpp"
#include "leap/problem/costs.hpp"
#include "leap/problem/mesh_refinement.hpp"
#include "leap/problem/reset_map.hpp"
#include "leap/problem/trajectory_problem.hpp"
#include "leap/solver/judge.hpp"
#include "leap/solver/pipeline.hpp"
#include "leap/solver/recipe.hpp"
#include "leap/spectral/bernstein.hpp"
#include "leap/spectral/cgl_grid.hpp"
#include "leap/tasks/static_family.hpp"

using namespace leap;

namespace {

constexpr double kGravity = 9.81;

struct Options {
  std::string root = ".";
  std::string out = "results/hopper_jump";
  double hop = 0.17, tstance = 0.8, tair = 0.391;
  int levels = 6;
  bool verbose = true;
};

Options parseOptions(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--root") o.root = v;
    else if (k == "--out") o.out = v;
    else if (k == "--hop") o.hop = std::atof(v.c_str());
    else if (k == "--tstance") o.tstance = std::atof(v.c_str());
    else if (k == "--tair") o.tair = std::atof(v.c_str());
    else if (k == "--levels") o.levels = std::atoi(v.c_str());
    else if (k == "--verbose") o.verbose = std::atoi(v.c_str()) != 0;
    else throw std::runtime_error("unknown option " + k);
  }
  return o;
}

std::vector<int> range(int first, int count) {
  std::vector<int> v;
  for (int i = 0; i < count; ++i) v.push_back(first + i);
  return v;
}

// Split element `e` into k elements whose widths shrink geometrically by `grade` toward
// the end (towardEnd) or the start of the element: the mesh is finest where the contact
// changes and the force transient lives.
void splitGraded(TrajectoryProblem& problem, int e, int k, double grade, bool towardEnd) {
  std::vector<double> w(static_cast<size_t>(k));
  double sum = 0.0;
  for (int i = 0; i < k; ++i) {
    w[static_cast<size_t>(i)] = std::pow(grade, towardEnd ? i : k - 1 - i);
    sum += w[static_cast<size_t>(i)];
  }
  double remaining = sum;
  for (int i = 0; i + 1 < k; ++i) {
    problem = splitElement(problem, e + i, w[static_cast<size_t>(i)] / remaining);
    remaining -= w[static_cast<size_t>(i)];
  }
}

// The rows of E^-1: the Bernstein coefficients of a degree-n nodal polynomial as linear
// functionals of its node values. A box on all of them bounds the polynomial between
// the nodes (it lies in the convex hull of its coefficients).
std::vector<Eigen::VectorXd> bernsteinCoefficientRows(int degree) {
  const CglGrid g(degree);
  const Eigen::MatrixXd Einv = bernsteinBasisAtDegree(degree, g).inverse();
  std::vector<Eigen::VectorXd> rows;
  for (int r = 0; r < Einv.rows(); ++r) rows.push_back(Einv.row(r).transpose());
  return rows;
}

// ---------------------------------------------------------------------------------------
// 1. The robot: robots/hopper2d.yaml names the URDF, the explicit floating base, the
//    actuated joints, the contact frame (a point foot), the ground keep-out monitors
//    and the capsule collision model. RobotModel wraps pinocchio over it.
// ---------------------------------------------------------------------------------------

// World position of the foot at q.
Eigen::Vector3d footPosition(const RobotModel& m, const Eigen::VectorXd& q) {
  ModelDataPtr data = m.makeData();
  ModelEvalCache cache;
  m.initCache(cache);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(m.nv());
  m.evalNode(*data, q, zero, zero, kFootPos, cache);
  return cache.footPos[0].head<3>();
}

// A rest pose from joint angles with the foot at `foot` on the floor: the base is placed
// so that the forward kinematics puts the foot there (a pose does not put the foot
// straight under the base, so the base position is derived, never guessed).
Eigen::VectorXd restPose(const RobotModel& m, double hip, double knee, const Eigen::Vector3d& foot) {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(m.nq());
  q(m.jointIndex("hip")) = hip;
  q(m.jointIndex("knee")) = knee;
  q.head<3>() += foot - footPosition(m, q);
  return q;
}

// ---------------------------------------------------------------------------------------
// 2. The problem: three contact modes, each one Element, then a mesh of elements.
// ---------------------------------------------------------------------------------------

struct Bundle {  // the constraint objects every mode shares (one penalty term per name)
  std::shared_ptr<Constraint> dynStance, dynAir, velFoot, posFoot, clearStance, clearAir, clearFlight;
  std::shared_ptr<Constraint> boxQ, boxV, boxU, boxAtt, pinPlanar;
  std::shared_ptr<Constraint> fric, lamFloor, lamLiftoffPin;
  std::shared_ptr<Cost> effort;
};

Bundle makeBundle(const RobotModel& m, const ContactSet& stance, const ContactSet& air, const Eigen::Vector3d& foot, double mu,
                  double lamMin, double footClear) {
  Bundle b;
  const NodeDims d = m.nodeDims(stance);
  // planar robot: the out-of-plane coordinates (y, roll, yaw) are pinned, so the
  // dynamics keeps the in-plane rows only
  const std::vector<int> planar{m.jointIndex("base_py"), m.jointIndex("base_rx"), m.jointIndex("base_rz")};
  std::vector<int> rows;
  for (int i = 0; i < m.nv(); ++i)
    if (std::find(planar.begin(), planar.end(), i) == planar.end()) rows.push_back(i);
  b.dynStance = std::make_shared<SelectedRowsDynamicsConstraint>(m, stance, rows);
  b.dynAir = std::make_shared<SelectedRowsDynamicsConstraint>(m, air, rows);
  b.pinPlanar = std::make_shared<PinConstraint>(planar, Eigen::VectorXd::Zero(3), "pin_planar");
  // the foot: no slip and on the ground while in stance
  b.velFoot = std::make_shared<FrameVelocityConstraint>(0, Eigen::Vector3d::Zero(), "vel_foot");
  b.posFoot = std::make_shared<FramePositionConstraint>(0, foot, "pos_foot");
  // ground keep-out of the monitor frames (foot, knee): the foot's own row is slack in
  // stance (it stands on the floor), active in flight; mid-flight the foot clears footClear
  Eigen::VectorXd lbStance = Eigen::VectorXd::Zero(m.nMonitors()), lbAir = Eigen::VectorXd::Zero(m.nMonitors());
  for (int i = 0; i < m.nMonitors(); ++i)
    if (m.monitorName(i) == m.contactName(0)) {
      lbStance(i) = -1.0;
      lbAir(i) = footClear;
    }
  b.clearStance = std::make_shared<MonitorHeightLowerBoundConstraint>(m, lbStance, "ineq_clear");
  b.clearFlight = std::make_shared<MonitorHeightLowerBoundConstraint>(m, Eigen::VectorXd::Zero(m.nMonitors()), "ineq_clear");
  b.clearAir = std::make_shared<MonitorHeightLowerBoundConstraint>(m, lbAir, "ineq_clear_air");
  // hardware boxes from the URDF
  const std::vector<int>& act = m.actuatedIdx();
  const Eigen::VectorXd lo = m.lowerPositionLimit(), hi = m.upperPositionLimit();
  Eigen::VectorXd qlo(m.nu()), qhi(m.nu());
  for (int j = 0; j < m.nu(); ++j) {
    qlo(j) = lo(act[static_cast<size_t>(j)]);
    qhi(j) = hi(act[static_cast<size_t>(j)]);
  }
  const Eigen::VectorXd vlim = m.actuatedVelocityLimit(), ulim = m.actuatedEffortLimit();
  b.boxQ = std::make_shared<BoxConstraint>(Field::Q, act, qlo, qhi, "box_q");
  b.boxV = std::make_shared<BoxConstraint>(Field::V, act, -vlim, vlim, "box_v");
  b.boxU = std::make_shared<BoxConstraint>(Field::U, range(0, m.nu()), -0.995 * ulim, 0.995 * ulim, "box_u");
  b.boxAtt = std::make_shared<BoxConstraint>(Field::Q, std::vector<int>{m.jointIndex("base_ry")}, Eigen::VectorXd::Constant(1, -0.5),
                                             Eigen::VectorXd::Constant(1, 0.5), "box_att");
  // contact force rows, enforced on the Bezier control points (convexity certifies them
  // along the whole curve): the friction cone, a normal-force floor, the liftoff pin
  b.fric = std::make_shared<FrictionConeConstraint>(stance, mu, 1e-2, "ineq_fric");
  const int fzIdx = d.offLam() + 2;
  auto floor = std::make_shared<LowerBoundConstraint>(std::vector<int>{fzIdx}, Eigen::VectorXd::Constant(1, lamMin), "ineq_lam_z");
  floor->setForceOnly(true);
  b.lamFloor = floor;
  auto pin = std::make_shared<PinConstraint>(range(d.offLam(), d.nlam), Eigen::VectorXd::Zero(d.nlam), "pin_lam_liftoff");
  pin->setForceOnly(true);
  b.lamLiftoffPin = pin;
  b.effort = std::make_shared<ControlEffortCost>();
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parseOptions(argc, argv);
    const auto t0 = std::chrono::steady_clock::now();

    // --- the robot -------------------------------------------------------------------
    const RobotDescription rd = loadRobotDescription(opt.root + "/robots/hopper2d.yaml");
    std::shared_ptr<RobotModel> model = std::make_shared<RobotModel>(rd);
    const RobotModel& m = *model;
    const ContactSet stance = m.allContacts();  // the foot
    const ContactSet air;                       // nothing
    const int impactSet = m.ensureImpactSet(stance);  // the touchdown impact map (register before makeData)
    const Eigen::Vector3d foot(0.0, 0.0, 0.0);                        // where the foot stands, takes off and lands
    const Eigen::VectorXd qSquat = restPose(m, -0.4982, 0.9, foot);   // rest pose at the start
    const Eigen::VectorXd qStand = restPose(m, -0.2202, 0.4, foot);   // rest pose at the end
    std::printf("[hopper_jump] nq=%d nu=%d squat base (x %.3f, z %.3f) m, stand (x %.3f, z %.3f) m, hop %.2f m\n", m.nq(), m.nu(),
                qSquat(0), qSquat(2), qStand(0), qStand(2), opt.hop);

    // --- the three modes ---------------------------------------------------------------
    const double mu = 0.85, lamMin = 5.0, footClear = 0.03, rho = 3e6;
    const int degree = 4, Nc = 4;
    const Bundle b = makeBundle(m, stance, air, foot, mu, lamMin, footClear);
    const NodeDims d = m.nodeDims(stance);
    const double vzLiftoff = std::sqrt(2.0 * kGravity * opt.hop);

    Element pushOff;
    pushOff.label = "push_off";
    pushOff.phase = 0;
    pushOff.T = opt.tstance;
    pushOff.contacts = stance;
    pushOff.degree = degree;
    pushOff.forceControlPoints = Nc;
    pushOff.addNode(b.dynStance, Where::AllNodes);
    pushOff.addNode(std::make_shared<PinConstraint>(range(0, m.nq()), qSquat, "pin_q_rest"), Where::FirstNode);
    pushOff.addNode(std::make_shared<PinConstraint>(range(d.offV(), m.nv()), Eigen::VectorXd::Zero(m.nv()), "pin_v_rest"), Where::FirstNode);
    pushOff.addNode(b.velFoot, Where::Interior);
    pushOff.addNode(b.velFoot, Where::LastNode);
    pushOff.addNode(b.posFoot, Where::Interior);
    pushOff.addNode(b.posFoot, Where::LastNode);
    pushOff.addNode(std::make_shared<LowerBoundConstraint>(std::vector<int>{d.offV() + 2}, Eigen::VectorXd::Constant(1, vzLiftoff), "ineq_vz_up"),
                    Where::LastNode);
    pushOff.addNode(b.clearStance, Where::AllNodes);
    pushOff.addOffNode(b.velFoot, uniformInteriorTaus(3));  // no slip between the nodes too
    pushOff.setDefectCost(b.dynStance, uniformInteriorTaus(6), rho);

    Element flight;
    flight.label = "flight";
    flight.phase = 1;
    flight.T = opt.tair;
    flight.contacts = air;
    flight.degree = degree;
    flight.addNode(b.dynAir, Where::AllNodes);
    flight.addNode(b.clearFlight, Where::AllNodes);
    flight.addNode(b.posFoot, Where::LastNode);  // land where it took off
    flight.addNode(std::make_shared<FrameVelocityBoundConstraint>(0, 2, -1.0, -0.5, "ineq_land_vz", 3), Where::LastNode);
    flight.setDefectCost(b.dynAir, uniformInteriorTaus(6), rho);

    Element landing;
    landing.label = "landing";
    landing.phase = 2;
    landing.T = opt.tstance;
    landing.contacts = stance;
    landing.degree = degree;
    landing.forceControlPoints = Nc;
    landing.addNode(b.dynStance, Where::AllNodes);
    landing.addNode(b.velFoot, Where::Interior);
    landing.addNode(b.posFoot, Where::Interior);
    landing.addNode(std::make_shared<PinConstraint>(range(0, m.nq()), qStand, "pin_q_rest"), Where::LastNode);
    landing.addNode(std::make_shared<PinConstraint>(range(d.offV(), m.nv()), Eigen::VectorXd::Zero(m.nv()), "pin_v_rest"), Where::LastNode);
    landing.addNode(b.clearStance, Where::AllNodes);
    landing.addOffNode(b.velFoot, uniformInteriorTaus(3));
    landing.setDefectCost(b.dynStance, uniformInteriorTaus(6), rho);

    for (Element* e : {&pushOff, &flight, &landing}) {
      e->addNode(b.boxQ, Where::AllNodes);
      e->addNode(b.boxV, Where::AllNodes);
      e->addFieldFunctionals(b.boxU, Field::U, bernsteinCoefficientRows(degree));  // holds between the nodes too
      e->addNode(b.boxAtt, Where::AllNodes);
      e->addNode(b.pinPlanar, Where::AllNodes);
      e->addCost(b.effort);
    }

    // --- the mesh: the stances graded (ratio 0.6) toward liftoff / touchdown, where the
    //     force transient lives; the flight uniform. Split back to front so indices hold.
    TrajectoryProblem problem;
    problem.model = model;
    problem.elements = {pushOff, flight, landing};
    problem.addReset(std::make_shared<ContinuityReset>(m.nq(), m.nv()));            // liftoff
    problem.addReset(std::make_shared<PlasticImpactReset>(impactSet, m.nq(), m.nv()));  // touchdown
    splitGraded(problem, 2, 6, 0.6, false);  // landing: finest at its start (touchdown)
    splitGraded(problem, 1, 4, 1.0, true);   // flight: uniform
    splitGraded(problem, 0, 6, 0.6, true);   // push-off: finest at its end (liftoff)
    // mid-flight foot clearance: off-node points on the flight elements that lie inside
    // the middle half of the flight (near liftoff and touchdown the foot is on the floor)
    {
      const std::vector<int> flightElems = problem.elementsOfPhase(1);
      double t0 = 0.0;
      for (int e = 0; e < flightElems.front(); ++e) t0 += problem.elements[static_cast<size_t>(e)].T;
      double tStart = t0;
      for (int e : flightElems) {
        Element& el = problem.elements[static_cast<size_t>(e)];
        const double f0 = (tStart - t0) / opt.tair, f1 = (tStart + el.T - t0) / opt.tair;
        if (f0 >= 0.25 - 1e-9 && f1 <= 0.75 + 1e-9) el.addOffNode(b.clearAir, uniformInteriorTaus(3));
        tStart += el.T;
      }
    }
    // force rows on the control points of every stance element; the last push-off
    // element lifts off: a tapered floor, the force pinned to zero at its last point
    // (which the cone therefore skips: a zero force sits on the smoothed cone's boundary)
    const int lastPushOff = static_cast<int>(problem.elementsOfPhase(0).back());
    for (int e = 0; e < static_cast<int>(problem.elements.size()); ++e) {
      Element& el = problem.elements[static_cast<size_t>(e)];
      if (el.contacts.active.empty()) continue;
      el.addForceControlPoints(b.fric, range(0, e == lastPushOff ? Nc - 1 : Nc));
      if (e != lastPushOff) {
        el.addForceControlPoints(b.lamFloor, range(0, Nc));
        continue;
      }
      for (int j = 0; j + 1 < Nc; ++j) {
        auto taper = std::make_shared<LowerBoundConstraint>(std::vector<int>{d.offLam() + 2},
                                                            Eigen::VectorXd::Constant(1, lamMin * (1.0 - static_cast<double>(j) / (Nc - 1))), "ineq_lam_z");
        taper->setForceOnly(true);
        el.addForceControlPoints(taper, {j});
      }
      el.addForceControlPoints(b.lamLiftoffPin, {Nc - 1});
    }
    problem.validate();
    const CompiledProblem prob = CompiledProblem::compile(problem);
    std::printf("[hopper_jump] %d elements, %d decision variables, %d rows, %d terms\n", prob.numElements(), prob.layout().np,
                prob.layout().md, prob.terms().size());

    // --- the cold seed: hold the squat, blend to the stand through the flight with a hop ----
    Eigen::VectorXd y0 = Eigen::VectorXd::Zero(prob.layout().n());
    {
      ModelDataPtr data = m.makeData();
      ModelEvalCache cache;
      m.initCache(cache);
      const Eigen::VectorXd zero = Eigen::VectorXd::Zero(m.nv());
      m.evalNode(*data, qSquat, zero, zero, kRnea, cache);
      const double weight = cache.tau(2);  // the static support force
      const std::vector<int> flightElems = problem.elementsOfPhase(1);
      const double tFlight0 = prob.layout().elem[static_cast<size_t>(flightElems.front())].t0;
      for (int e = 0; e < prob.numElements(); ++e) {
        const ElementLayout& el = prob.layout().elem[static_cast<size_t>(e)];
        ElementViewMutable X(y0.data(), el);
        const Eigen::VectorXd t = prob.grid(e).timeGrid(el.T);
        const int phase = problem.elements[static_cast<size_t>(e)].phase;
        for (int k = 0; k < el.Nn; ++k) {
          if (phase == 0) X.Q.col(k) = qSquat;
          if (phase == 2) X.Q.col(k) = qStand;
          if (phase == 1) {
            const double s = (el.t0 + t(k) - tFlight0) / opt.tair;  // 0 at liftoff, 1 at touchdown
            const double blend = s * s * (3.0 - 2.0 * s);
            X.Q.col(k) = (1.0 - blend) * qSquat + blend * qStand;
            X.Q(2, k) += opt.hop * 4.0 * s * (1.0 - s);
          }
        }
        const Eigen::MatrixXd& D = prob.grid(e).D();
        const double a = prob.timeScale(e);
        X.V = a * X.Q * D.transpose();
        X.A = a * X.V * D.transpose();
        if (el.Nc > 0) {
          Eigen::MatrixXd Cphys = Eigen::MatrixXd::Zero(el.dims.nlam, el.Nc);
          Cphys.row(2).setConstant(weight);
          if (e == lastPushOff) Cphys.col(el.Nc - 1).setZero();
          X.C = Cphys * el.sigma.cwiseInverse().asDiagonal();
        }
        const Eigen::MatrixXd lamNodes = el.Nc > 0 ? prob.nodeForces(y0, e) : Eigen::MatrixXd(0, el.Nn);
        for (int k = 0; k < el.Nn; ++k)
          X.U.col(k) = m.inverseDynamicsTorque(*data, X.Q.col(k), X.V.col(k), X.A.col(k), lamNodes.col(k), problem.elements[static_cast<size_t>(e)].contacts);
      }
    }

    // --- solve ---------------------------------------------------------------------------
    StaticFamily family("hopper_jump", problem, y0, rho);
    SolverRecipe recipe;
    recipe.applyPenaltyScale(rho);  // penalties in units of the defect weight
    recipe.levels = opt.levels;
    recipe.verbose = opt.verbose;
    PipelineResult res = runPipeline(family, recipe, "[hopper_jump]");

    // --- judge and report ----------------------------------------------------------------
    Level& L = res.level;
    const Eigen::VectorXd& y = res.y;
    const CompiledProblem& solved = *L.restoration;
    Evaluation ev;
    const double gate = L.evaluator->zeroDualViolation(y, L.penalties->muRow(), ev);
    printViolationsByGroup(solved, ev);
    const FineGridReport fr = fineGridViolations(solved, y, 0.01);
    printFineGrid(fr, 12);
    double effort = 0.0, apex = 0.0, peakU = 0.0, peakFz = 0.0;
    ModelDataPtr data = m.makeData();
    const Eigen::VectorXd ulim = m.actuatedEffortLimit();
    for (int e = 0; e < solved.numElements(); ++e) {
      const ElementLayout& el = solved.layout().elem[static_cast<size_t>(e)];
      const ElementView X(y.data(), el);
      const Eigen::VectorXd cc = clenshawCurtisWeights(solved.grid(e).degree());
      const Eigen::MatrixXd lam = el.Nc > 0 ? solved.nodeForces(y, e) : Eigen::MatrixXd(0, el.Nn);
      for (int k = 0; k < el.Nn; ++k) {
        effort += 0.5 * el.T * cc(k) * X.U.col(k).squaredNorm();
        apex = std::max(apex, X.Q(2, k));
        peakU = std::max(peakU, (X.U.col(k).cwiseAbs().cwiseQuotient(ulim)).maxCoeff());
        if (el.Nc > 0) peakFz = std::max(peakFz, lam(2, k));
      }
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n[hopper_jump] REPORT\n");
    std::printf("  criterion met            : %s (node gate %.2e, fine grid %.2e, tol %.0e)\n", res.record.met ? "yes" : "NO", gate, fr.worstGate, recipe.tol);
    std::printf("  accepted steps / levels  : %ld / %zu (final mesh %d elements)\n", res.record.totalSteps(), res.record.levels.size(), solved.numElements());
    std::printf("  control effort           : %.4e N^2 m^2 s (int ||u||^2 dt)\n", effort);
    std::printf("  base apex                : %.3f m above the squat height %.3f m\n", apex - qSquat(2), qSquat(2));
    std::printf("  peak torque / limit      : %.1f %%\n", 100.0 * peakU);
    std::printf("  peak normal force        : %.1f N (weight %.1f N)\n", peakFz, m.totalMass() * kGravity);
    std::printf("  wall                     : %.1f s\n", wall);
    std::printf("[unified] robot=%s type=custom met=%d gate=%.6e fine=%.6e total=%ld solvewall=%.1f effort=%.6e\n", rd.name.c_str(),
                res.record.met ? 1 : 0, gate, fr.worstGate, res.record.totalSteps(), res.record.solveWall, effort);

    ArchiveRunInfo info;
    info.generator = "hopper_jump";
    info.problemName = "hopper_jump";
    info.problemType = "custom";
    info.robotYamlText = rd.sourceText;
    info.urdfPath = rd.urdfPath;
    info.solverFingerprint = recipe.fingerprint();
    info.configHashHex = fnv1aHex(info.solverFingerprint);
    info.wallS = res.record.solveWall;
    info.ompThreads = std::getenv("OMP_NUM_THREADS") ? std::atoi(std::getenv("OMP_NUM_THREADS")) : 0;
    info.interNodeDynMax = fr.worstDyn;
    info.lastFlow = &res.record.lastFlow;
    writeSolutionArchive(solved, ev, y, L.penalties->muRow(), info, opt.out + ".h5");
    std::printf("solution archive written to %s.h5\n", opt.out.c_str());
    return res.record.met ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "hopper_jump: %s\n", e.what());
    return 2;
  }
}
