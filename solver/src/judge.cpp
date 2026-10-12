#include "leap/solver/judge.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>

#include "leap/spectral/bernstein.hpp"

namespace leap {

namespace {

bool startsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

bool isThroughout(const Constraint* c) {
  return !startsWith(c->name(), "pin") && !startsWith(c->name(), "ineq_swing_z");
}

struct JudgedConstraint {
  const Constraint* con = nullptr;
  int kind = 0;  // 0 node/off-node row, 1 control-point row (full coverage), 2 partial coverage
};

void addJudged(std::vector<JudgedConstraint>& list, const Constraint* c, int kind, unsigned& needs) {
  for (const JudgedConstraint& j : list)
    if (j.con == c) return;
  list.push_back({c, kind});
  needs |= c->modelNeeds();
}

bool compareTerms(const FineGridTerm& a, const FineGridTerm& b) { return a.worst > b.worst; }

}  // namespace

std::string constraintGroup(const std::string& name) {
  if (startsWith(name, "dyn")) return "dynamics";
  if (startsWith(name, "kin")) return "kinematic";
  if (name == "reset") return "reset";
  if (startsWith(name, "pos_") || startsWith(name, "vel_") || startsWith(name, "acc_") || startsWith(name, "pin_foot") ||
      startsWith(name, "pin_sole") || startsWith(name, "pin_land"))
    return "foot";
  if (startsWith(name, "pin")) return "boundary";
  if (startsWith(name, "ineq") || startsWith(name, "box")) return "ineq";
  return "other";
}

void printViolationsByGroup(const CompiledProblem& prob, const Evaluation& ev) {
  std::map<std::string, double> groups;
  std::map<std::string, std::string> worstTerm;
  for (int t = 0; t < prob.terms().size(); ++t) {
    const std::string name = TermTable::stripped(prob.terms().name(t));
    const std::string g = constraintGroup(name);
    const double v = ev.perTermViol(t);
    if (v >= groups[g]) {
      groups[g] = v;
      worstTerm[g] = name;
    }
  }
  std::printf("    violations by group:");
  for (const auto& kv : groups) std::printf("  %s %.2e (%s)", kv.first.c_str(), kv.second, worstTerm[kv.first].c_str());
  std::printf("\n");
}

FineGridReport fineGridViolations(const CompiledProblem& prob, const Eigen::VectorXd& y, double dt) {
  FineGridReport out;
  out.dt = dt;
  if (!(dt > 0.0)) return out;
  const Layout& L = prob.layout();
  const RobotModel& m = prob.model();
  ModelDataPtr data = m.makeData();
  ModelEvalCache cache;
  m.initCache(cache);
  const int M = static_cast<int>(L.elem.size());
  out.elemWorst.assign(static_cast<size_t>(M), 0.0);
  out.elemWorstGate.assign(static_cast<size_t>(M), 0.0);
  out.elemWorstTau.assign(static_cast<size_t>(M), 0.0);
  out.elemWorstGateTau.assign(static_cast<size_t>(M), 0.0);
  out.elemWorstName.assign(static_cast<size_t>(M), "");
  out.elemWorstGateName.assign(static_cast<size_t>(M), "");
  const double Ttot = prob.spec().totalTime();
  const int K = static_cast<int>(std::floor(Ttot / dt + 1e-9));
  std::map<std::string, size_t> slot;
  const std::vector<int>& act = m.actuatedIdx();
  Eigen::VectorXd q, v, a, lam, u, g;
  int k = 0;
  for (int e = 0; e < M && k <= K; ++e) {
    const ElementLayout& el = L.elem[static_cast<size_t>(e)];
    const Element& spec = prob.spec().elements[static_cast<size_t>(e)];
    const NodeDims& d = el.dims;
    const bool last = (e + 1 == M);
    // the constraints judged on this element
    std::vector<JudgedConstraint> cons;
    unsigned needs = 0;
    for (const NodeAttachment& att : spec.nodeTerms) {
      if (att.where != Where::AllNodes && att.where != Where::Interior) continue;
      if (!isThroughout(att.con.get())) continue;
      addJudged(cons, att.con.get(), 0, needs);
    }
    for (const PointAttachment& att : spec.pointTerms) {
      if (!isThroughout(att.con.get())) continue;
      if (att.kind == PointKind::OffNode) {
        addJudged(cons, att.con.get(), 0, needs);
      } else if (att.kind == PointKind::ForceControlPoints) {
        std::vector<char> covered(static_cast<size_t>(std::max(el.Nc, 1)), 0);
        for (int j : att.controlPoints) covered[static_cast<size_t>(j)] = 1;
        int nCov = 0;
        for (char c : covered) nCov += c;
        if (nCov + 1 < el.Nc) continue;
        addJudged(cons, att.con.get(), nCov < el.Nc ? 2 : 1, needs);
      }
      // field functionals (Bernstein coefficient rows): the node rows cover them
    }
    if (!el.uStored) needs |= kRnea | kFootPos;
    const ElementView X(y.data(), el);
    for (; k <= K; ++k) {
      const double t = k * dt;
      if (!last && t >= el.t0 + el.T - 1e-12) break;
      const double tau = std::clamp(-1.0 + 2.0 * (t - el.t0) / el.T, -1.0, 1.0);
      ++out.nSamples;
      if (cons.empty()) continue;
      prob.stateAt(y, e, tau, q, v, a, lam);
      if (el.uStored)
        u = X.U * prob.grid(e).interpWeights(tau);
      else
        u.setZero(d.nu);
      m.evalNode(*data, q, v, a, lam, spec.contacts, needs, cache);
      if (!el.uStored)
        for (int j = 0; j < d.nu; ++j) u(j) = cache.gdyn(act[static_cast<size_t>(j)]);
      const NodeQuantities z{q, v, a, lam, u};
      for (const JudgedConstraint& jc : cons) {
        const int R = jc.con->rows(d);
        if (R <= 0) continue;
        g.resize(R);
        jc.con->value(z, d, cache, g);
        const bool ineq = jc.con->sense() == Sense::Inequality;
        double worst = 0.0;
        int worstRow = -1;
        for (int r = 0; r < R; ++r) {
          const double gr = g(r);
          if (!std::isfinite(gr)) {
            worst = std::numeric_limits<double>::infinity();
            worstRow = r;
            break;
          }
          const double vr = ineq ? std::max(0.0, gr) : std::abs(gr);
          if (vr > worst) {
            worst = vr;
            worstRow = r;
          }
        }
        const bool gated = jc.kind != 2;
        auto it = slot.find(jc.con->name());
        if (it == slot.end()) {
          FineGridTerm term;
          term.name = jc.con->name();
          term.ineq = ineq;
          term.controlPoint = jc.kind != 0;
          out.terms.push_back(term);
          it = slot.emplace(jc.con->name(), out.terms.size() - 1).first;
        }
        FineGridTerm& term = out.terms[it->second];
        if (!gated) term.partialCoverage = true;
        if (worst > term.worst) {
          term.worst = worst;
          term.elem = e;
          term.t = t;
          term.row = worstRow;
        }
        if (worst > out.elemWorst[static_cast<size_t>(e)]) {
          out.elemWorst[static_cast<size_t>(e)] = worst;
          out.elemWorstName[static_cast<size_t>(e)] = jc.con->name();
          out.elemWorstTau[static_cast<size_t>(e)] = tau;
        }
        if (worst > out.worst) {
          out.worst = worst;
          out.worstName = jc.con->name();
          out.worstElem = e;
          out.worstT = t;
        }
        if (gated) {
          if (worst > out.elemWorstGate[static_cast<size_t>(e)]) {
            out.elemWorstGate[static_cast<size_t>(e)] = worst;
            out.elemWorstGateName[static_cast<size_t>(e)] = jc.con->name();
            out.elemWorstGateTau[static_cast<size_t>(e)] = tau;
          }
          if (worst > out.worstGate) {
            out.worstGate = worst;
            out.worstGateName = jc.con->name();
            out.worstGateElem = e;
            out.worstGateT = t;
          }
        }
        if (constraintGroup(jc.con->name()) == "dynamics")
          out.worstDyn = std::max(out.worstDyn, worst);
        else
          out.worstNonDyn = std::max(out.worstNonDyn, worst);
      }
    }
  }
  std::sort(out.terms.begin(), out.terms.end(), compareTerms);
  return out;
}

void printFineGrid(const FineGridReport& r, int maxTerms) {
  std::printf("\n=== FINE-GRID feasibility (dt = %g s, %d samples equidistant in physical time) ===\n", r.dt, r.nSamples);
  std::printf("%-24s %-5s %12s %5s %8s %4s\n", "constraint", "sense", "fine-grid max", "elem", "t (s)", "row");
  int n = 0;
  for (const FineGridTerm& t : r.terms) {
    if (n++ >= maxTerms) break;
    std::printf("%-24s %-5s %12.3e %5d %8.3f %4d%s\n", t.name.c_str(), t.ineq ? "ineq" : "eq", t.worst, t.elem, t.t, t.row,
                t.partialCoverage ? "  (control-point row on the force curve; partial coverage, not gated)"
                                  : (t.controlPoint ? "  (control-point row on the force curve)" : ""));
  }
  std::printf("  FINE-GRID max|c| = %.3e (%s, elem %d, t = %.3f s)   dyn %.3e   non-dyn %.3e\n", r.worst, r.worstName.c_str(),
              r.worstElem, r.worstT, r.worstDyn, r.worstNonDyn);
  std::printf("  FINE-GRID gated max|c| = %.3e (%s, elem %d, t = %.3f s)\n", r.worstGate, r.worstGateName.c_str(),
              r.worstGateElem, r.worstGateT);
}

Judgement judgeLevel(AlmEvaluator& ev, const Penalties& pen, const Eigen::VectorXd& y, double dt, bool gated) {
  const CompiledProblem& prob = ev.problem();
  Judgement J;
  const int M = prob.numElements();
  J.elements.assign(static_cast<size_t>(M), ElementIndicator{});
  Evaluation c;
  ev.zeroDualViolation(y, pen.muRow(), c);
  const std::vector<RowInfo>& rows = prob.rowInfo();
  for (int r = 0; r < prob.layout().md; ++r) {
    const double v = std::abs(c.phr.ceff(r));
    J.gate = std::max(J.gate, v);
    const RowInfo& ri = rows[static_cast<size_t>(r)];
    if (ri.family == RowFamily::Reset || ri.elem < 0 || ri.elem >= M) continue;
    ElementIndicator& e = J.elements[static_cast<size_t>(ri.elem)];
    e.node = std::max(e.node, v);
    if (v > e.ind) {
      e.ind = v;
      e.worst = TermTable::stripped(prob.terms().name(ri.term)) + " (row)";
    }
  }
  const FineGridReport fr = fineGridViolations(prob, y, dt);
  const std::vector<double>& ew = gated ? fr.elemWorstGate : fr.elemWorst;
  const std::vector<double>& etau = gated ? fr.elemWorstGateTau : fr.elemWorstTau;
  const std::vector<std::string>& en = gated ? fr.elemWorstGateName : fr.elemWorstName;
  for (int e = 0; e < M; ++e) {
    ElementIndicator& ei = J.elements[static_cast<size_t>(e)];
    ei.fine = ew[static_cast<size_t>(e)];
    ei.fineTau = etau[static_cast<size_t>(e)];
    if (ei.fine > ei.ind) {
      ei.ind = ei.fine;
      ei.worst = en[static_cast<size_t>(e)] + " (fine)";
    }
  }
  J.fine = gated ? fr.worstGate : fr.worst;
  J.fineAll = fr.worst;
  J.fineName = gated ? fr.worstGateName : fr.worstName;
  J.fineElem = gated ? fr.worstGateElem : fr.worstElem;
  J.fineT = gated ? fr.worstGateT : fr.worstT;
  if (J.fineElem >= 0) J.fineTau = etau[static_cast<size_t>(J.fineElem)];
  return J;
}

}  // namespace leap
