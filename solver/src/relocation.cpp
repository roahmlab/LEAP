#include "leap/solver/relocation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace leap {

namespace {

bool startsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

struct Candidate {
  double value, tau;
};
bool byValueDesc(const Candidate& a, const Candidate& b) { return a.value > b.value; }

}  // namespace

bool isBoxRowName(const std::string& name) {
  return startsWith(name, "box_u") || startsWith(name, "box_v") || startsWith(name, "ineq_motor");
}

bool fineGridWorstReachable(const CompiledProblem& prob, const std::string& fineName, int fineElem, double fineTau,
                            double fineValue, double movableTau, double ceiling, bool moveBoxRows) {
  if (fineElem < 0 || std::abs(fineTau) >= movableTau) return false;
  if (fineValue > ceiling) return false;
  for (const PointInfo& info : prob.points()) {
    if (info.kind != PointKind::OffNode || info.taus.empty()) continue;
    if (info.elem != fineElem) continue;
    if (info.con->name() != fineName) continue;
    if (!moveBoxRows && isBoxRowName(info.con->name())) continue;
    return true;
  }
  return false;
}

RelocationReport relocateOffNodePoints(CompiledProblem& prob, Eigen::VectorXd& y, const RelocationOptions& opt,
                                       const Eigen::VectorXd& termScales, bool dryRun) {
  RelocationReport rep;
  const Layout& L = prob.layout();
  const RobotModel& m = prob.model();
  ModelDataPtr data = m.makeData();
  ModelEvalCache cache;
  m.initCache(cache);
  const std::vector<int>& act = m.actuatedIdx();
  // movable infos per element (pins and the mid-swing window are never moved)
  std::vector<std::vector<int>> byElem(L.elem.size());
  for (int i = 0; i < static_cast<int>(prob.points().size()); ++i) {
    const PointInfo& info = prob.points()[static_cast<size_t>(i)];
    if (info.kind != PointKind::OffNode || info.taus.empty() || info.nPoints <= 0) continue;
    const std::string& nm = info.con->name();
    if (startsWith(nm, "pin") || startsWith(nm, "ineq_swing_z")) continue;
    byElem[static_cast<size_t>(info.elem)].push_back(i);
  }
  Eigen::VectorXd q, v, a, lam, u, g;
  for (size_t e = 0; e < L.elem.size(); ++e) {
    if (byElem[e].empty()) continue;
    const ElementLayout& el = L.elem[e];
    const Element& spec = prob.spec().elements[e];
    const NodeDims& d = el.dims;
    const ElementView X(y.data(), el);
    unsigned needs = el.uStored ? 0u : (kRnea | kFootPos);
    for (int i : byElem[e]) needs |= prob.points()[static_cast<size_t>(i)].con->modelNeeds();
    // the dense grid plus every info's current points (so a move is compared with the
    // true value at the current point)
    std::vector<double> taus;
    for (int i = 0; i < opt.nScan; ++i) taus.push_back(-1.0 + 2.0 * (i + 0.5) / opt.nScan);
    const size_t nGrid = taus.size();
    for (int i : byElem[e])
      for (double t : prob.points()[static_cast<size_t>(i)].taus) taus.push_back(t);
    Eigen::MatrixXd prof(static_cast<Eigen::Index>(byElem[e].size()), static_cast<Eigen::Index>(taus.size()));
    for (size_t t = 0; t < taus.size(); ++t) {
      prob.stateAt(y, static_cast<int>(e), taus[t], q, v, a, lam);
      if (el.uStored)
        u = X.U * prob.grid(static_cast<int>(e)).interpWeights(taus[t]);
      else
        u.setZero(d.nu);
      m.evalNode(*data, q, v, a, lam, spec.contacts, needs, cache);
      if (!el.uStored)
        for (int j = 0; j < d.nu; ++j) u(j) = cache.gdyn(act[static_cast<size_t>(j)]);
      const NodeQuantities z{q, v, a, lam, u};
      for (size_t k = 0; k < byElem[e].size(); ++k) {
        const PointInfo& info = prob.points()[static_cast<size_t>(byElem[e][k])];
        g.resize(info.rowsPerPoint);
        info.con->value(z, d, cache, g);
        const double val = info.con->sense() == Sense::Inequality ? std::max(0.0, g.maxCoeff()) : g.cwiseAbs().maxCoeff();
        prof(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(t)) = val;
      }
    }
    size_t curOff = nGrid;
    for (size_t k = 0; k < byElem[e].size(); ++k) {
      const int infoIdx = byElem[e][k];
      const PointInfo& info = prob.points()[static_cast<size_t>(infoIdx)];
      ++rep.scanned;
      const int term = info.term;
      double scale = termScales.size() == prob.terms().size() ? termScales(term) : 1.0;
      const std::string& name = info.con->name();
      const double movableTau = startsWith(name, "dyn") ? std::max(opt.movableTau, 0.995) : opt.movableTau;
      const bool box = isBoxRowName(name);
      if (opt.moveBoxRows && box) scale = 1.0;  // box rows are judged raw
      double pk = 0.0;
      int pkArg = 0;
      for (size_t t = 0; t < nGrid; ++t) {
        const double val = prof(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(t));
        if (val > pk) {
          pk = val;
          pkArg = static_cast<int>(t);
        }
      }
      if (scale * pk > rep.worstScaled) {
        rep.worstScaled = scale * pk;
        rep.worstRaw = pk;
        rep.worstName = prob.terms().name(term);
        rep.worstElem = static_cast<int>(e);
        rep.worstTau = taus[static_cast<size_t>(pkArg)];
      }
      const size_t myCur = curOff;
      curOff += info.taus.size();
      if (dryRun) continue;
      if (opt.maxMoves > 0 && rep.moved >= opt.maxMoves) continue;
      if (pk <= 0.0) continue;
      if (scale * pk <= opt.minScaled) continue;
      if (!opt.moveBoxRows && box) continue;
      // interior local maxima, descending
      std::vector<Candidate> cands;
      for (size_t t = 0; t < nGrid; ++t) {
        const double val = prof(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(t));
        if (val <= 0.0 || scale * val <= opt.minScaled) continue;
        if (opt.maxScaled > 0.0 && scale * val > opt.maxScaled) continue;
        const double vm = t > 0 ? prof(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(t - 1)) : -1.0;
        const double vp = t + 1 < nGrid ? prof(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(t + 1)) : -1.0;
        if (val >= vm && val >= vp && std::abs(taus[t]) < movableTau) cands.push_back({val, taus[t]});
      }
      std::sort(cands.begin(), cands.end(), byValueDesc);
      std::vector<Candidate> chosen;
      for (const Candidate& c : cands) {
        if (static_cast<int>(chosen.size()) >= info.nPoints) break;
        bool ok = true;
        for (const Candidate& ch : chosen) ok = ok && std::abs(ch.tau - c.tau) >= opt.minSep;
        if (ok) chosen.push_back(c);
      }
      if (chosen.empty()) continue;
      // greedy nearest-point assignment, worst target first
      std::vector<double> newTaus = info.taus;
      std::vector<char> movedPt(info.taus.size(), 0), claimed(info.taus.size(), 0);
      for (const Candidate& c : chosen) {
        int best = -1;
        double bestDist = std::numeric_limits<double>::infinity();
        for (size_t p = 0; p < info.taus.size(); ++p) {
          if (claimed[p]) continue;
          const double dist = std::abs(info.taus[p] - c.tau);
          if (dist < bestDist) {
            bestDist = dist;
            best = static_cast<int>(p);
          }
        }
        if (best < 0) break;
        claimed[static_cast<size_t>(best)] = 1;
        if (bestDist < opt.coverSep) continue;
        const double vOld = prof(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(myCur + static_cast<size_t>(best)));
        if (c.value <= opt.moveTol * std::max(vOld, 1e-300)) continue;
        newTaus[static_cast<size_t>(best)] = c.tau;
        movedPt[static_cast<size_t>(best)] = 1;
      }
      bool any = false;
      for (char c : movedPt) any = any || c;
      if (!any) continue;
      prob.relocateOffNodeTaus(infoIdx, newTaus);
      for (size_t p = 0; p < movedPt.size(); ++p) {
        if (!movedPt[p]) continue;
        ++rep.moved;
        y.segment(L.np + info.row0 + static_cast<int>(p) * info.rowsPerPoint, info.rowsPerPoint).setZero();
      }
    }
  }
  return rep;
}

}  // namespace leap
