#include "leap/io/solution_archive.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "leap/io/h5_writer.hpp"
#include "leap/problem/obstacles.hpp"
#include "leap/solver/judge.hpp"

namespace leap {

namespace {

std::string elemPath(size_t e) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "/segments/%02zu", e);
  return buf;
}

std::string esc(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    if (c == '"' || c == '\\') (o += '\\') += c;
    else if (c == '\n') o += "\\n";
    else if (c == '\r') o += "\\r";
    else if (c == '\t') o += "\\t";
    else if (static_cast<unsigned char>(c) < 0x20) {
      char b[8];
      std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
      o += b;
    } else o += c;
  }
  return o;
}

void strList(std::ostringstream& f, const std::vector<std::string>& v) {
  for (size_t i = 0; i < v.size(); ++i) f << (i ? ", " : "") << '"' << esc(v[i]) << '"';
}

// JSON has no NaN / inf: those print as null.
void num(std::ostringstream& f, double x) {
  if (std::isfinite(x)) f << x;
  else f << "null";
}

void vec3(std::ostringstream& f, const Eigen::Vector3d& p) {
  f << "[";
  num(f, p.x());
  f << ", ";
  num(f, p.y());
  f << ", ";
  num(f, p.z());
  f << "]";
}

void doubleList(std::ostringstream& f, const std::vector<double>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) f << ", ";
    num(f, v[i]);
  }
}

void flowJson(std::ostringstream& f, const FlowReport* r) {
  if (r == nullptr) {
    f << "null";
    return;
  }
  f << "{\"s\": ";
  num(f, r->s);
  f << ", \"steps\": " << r->steps << ", \"windows\": " << r->windows << ", \"max_viol\": ";
  num(f, r->maxViol);
  f << ", \"gnorm\": ";
  num(f, r->gnorm);
  f << ", \"h_final\": ";
  num(f, r->hFinal);
  f << ", \"converged\": " << (r->converged ? "true" : "false") << ", \"stalled\": " << (r->stalled ? "true" : "false")
    << ", \"diverged\": " << (r->diverged ? "true" : "false") << ", \"shrinks\": " << r->shrinks << ", \"grows\": " << r->grows
    << ", \"restarts\": " << r->restarts << ", \"backstops\": " << r->backstops << ", \"solve_fails\": " << r->solveFails
    << ", \"dual_updates\": " << r->dualUpdates << ", \"blowup_recoveries\": " << r->blowupRecoveries << ", \"trials\": " << r->trials
    << ", \"factorizations\": " << r->factorizations << ", \"stop\": \"" << esc(r->stop) << "\"}";
}

int groupColumn(const std::string& group) {
  if (group == "dynamics") return 0;
  if (group == "foot") return 1;
  if (group == "boundary") return 2;
  if (group == "ineq") return 3;
  return 4;
}

}  // namespace

std::string fnv1aHex(const std::string& s) {
  std::uint64_t h = 1469598103934665603ULL;
  for (char ch : s) {
    h ^= static_cast<unsigned char>(ch);
    h *= 1099511628211ULL;
  }
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

void writeSolutionArchive(const CompiledProblem& prob, const Evaluation& ev, const Eigen::VectorXd& y,
                          const Eigen::VectorXd& muRow, const ArchiveRunInfo& info, const std::string& path) {
  const Layout& L = prob.layout();
  const RobotModel& m = prob.model();
  const TrajectoryProblem& spec = prob.spec();
  const std::vector<int>& rowTerm = prob.rowTerm();
  const std::vector<char>& rowIneq = prob.rowIneq();
  const size_t M = L.elem.size();
  if (ev.g.size() != L.md) throw std::runtime_error("writeSolutionArchive: the evaluation does not match the problem's row space");
  if (y.size() != L.n()) throw std::runtime_error("writeSolutionArchive: y does not match the layout");

  // --- quality metrics from the judging evaluation ---
  const double gate = ev.maxViol;
  const double gnormFinal = ev.F.size() >= L.np ? ev.F.head(L.np).norm() : -1.0;
  double gDyn = 0, gFoot = 0, gKin = 0, gBc = 0, gReset = 0, gIneq = 0;
  std::vector<std::string> termGroup(static_cast<size_t>(prob.terms().size()));
  for (int t = 0; t < prob.terms().size(); ++t) termGroup[static_cast<size_t>(t)] = constraintGroup(TermTable::stripped(prob.terms().name(t)));
  for (int i = 0; i < L.md; ++i) {
    if (rowIneq[static_cast<size_t>(i)]) {
      gIneq = std::max(gIneq, std::max(0.0, ev.g(i)));
      continue;
    }
    const std::string& g = termGroup[static_cast<size_t>(rowTerm[static_cast<size_t>(i)])];
    const double v = std::abs(ev.g(i));
    if (g == "dynamics") gDyn = std::max(gDyn, v);
    else if (g == "foot") gFoot = std::max(gFoot, v);
    else if (g == "kinematic") gKin = std::max(gKin, v);
    else if (g == "boundary") gBc = std::max(gBc, v);
    else if (g == "reset") gReset = std::max(gReset, v);
  }

  H5Writer w(path);
  w.writeRootAttr("schema_version", static_cast<std::int64_t>(1));
  w.writeRootAttr("generator", info.generator.empty() ? std::string("leap") : info.generator);

  // --- per-element trajectory data ---
  ModelDataPtr data = m.makeData();
  for (size_t e = 0; e < M; ++e) {
    const ElementLayout& el = L.elem[e];
    const Element& es = spec.elements[e];
    const ElementView X(y.data(), el);
    const std::string p = elemPath(e);
    w.writeMatrix(p + "/Q", Eigen::MatrixXd(X.Q));
    w.writeMatrix(p + "/V", Eigen::MatrixXd(X.V));
    w.writeMatrix(p + "/A", Eigen::MatrixXd(X.A));
    const Eigen::MatrixXd lamNodes = el.Nc > 0 ? prob.nodeForces(y, static_cast<int>(e)) : Eigen::MatrixXd(0, el.Nn);
    if (el.uStored) {
      w.writeMatrix(p + "/U", Eigen::MatrixXd(X.U));
    } else {
      // torque eliminated: u = [M a + C v + g (+ loss) - J' lam]_actuated at every node
      Eigen::MatrixXd U(el.dims.nu, el.Nn);
      for (int k = 0; k < el.Nn; ++k)
        U.col(k) = m.inverseDynamicsTorque(*data, X.Q.col(k), X.V.col(k), X.A.col(k), lamNodes.col(k), es.contacts);
      w.writeMatrix(p + "/U", U);
    }
    w.writeMatrix(p + "/lam_nodes", lamNodes);
    if (el.Nc > 0) {
      w.writeMatrix(p + "/lam_ctrl", Eigen::MatrixXd(X.C));
      w.writeVector(p + "/lam_scale", el.sigma);
    }
    w.writeVector(p + "/taus", prob.grid(static_cast<int>(e)).nodes());
    // per-node max violation, columns [dynamics, foot, boundary, ineq, other, kin1, kin2]
    Eigen::MatrixXd cons(el.Nn, 7);
    for (int k = 0; k < el.Nn; ++k) {
      double g5[5] = {0, 0, 0, 0, 0};
      const int r0 = el.nodeRow0[static_cast<size_t>(k)];
      for (int i = r0; i < r0 + el.nodeRows[static_cast<size_t>(k)]; ++i) {
        const int col = groupColumn(termGroup[static_cast<size_t>(rowTerm[static_cast<size_t>(i)])]);
        const double v = rowIneq[static_cast<size_t>(i)] ? std::max(0.0, ev.g(i)) : std::abs(ev.g(i));
        g5[col] = std::max(g5[col], v);
      }
      double k1 = 0.0, k2 = 0.0;
      for (int j = 0; j < el.dims.nq; ++j) {
        k1 = std::max(k1, std::abs(ev.g(el.kin1Row0 + k * el.dims.nq + j)));
        k2 = std::max(k2, std::abs(ev.g(el.kin2Row0 + k * el.dims.nq + j)));
      }
      cons.row(k) << g5[0], g5[1], g5[2], g5[3], g5[4], k1, k2;
    }
    w.writeMatrix(p + "/cons", cons);
  }

  // --- contact schedule (M x nContacts) ---
  {
    std::vector<std::int8_t> act(M * static_cast<size_t>(m.nContacts()), 0);
    for (size_t e = 0; e < M; ++e)
      for (const ContactRef& c : spec.elements[e].contacts.active)
        act[e * static_cast<size_t>(m.nContacts()) + static_cast<size_t>(c.index)] = 1;
    w.writeMatrixI8("/contacts/active", act, static_cast<long>(M), m.nContacts());
  }
  // --- reset residuals ---
  {
    Eigen::VectorXd rv(static_cast<Eigen::Index>(L.resetRow0.size()));
    for (size_t t = 0; t < L.resetRow0.size(); ++t) {
      const int rNext = t + 1 < L.resetRow0.size() ? L.resetRow0[t + 1] : L.md;
      double v = 0.0;
      for (int i = L.resetRow0[t]; i < rNext; ++i) v = std::max(v, std::abs(ev.g(i)));
      rv(static_cast<Eigen::Index>(t)) = v;
    }
    w.writeVector("/resets/max_viol", rv);
  }
  // --- duals, decision vector, raw rows ---
  w.writeVector("/duals/gamma", Eigen::VectorXd(y.tail(L.md)));
  w.writeVector("/solution/y", y);
  w.writeVector("/rows/g", ev.g);
  {
    std::vector<std::int32_t> term(static_cast<size_t>(L.md));
    std::vector<std::int8_t> iq(static_cast<size_t>(L.md));
    for (int i = 0; i < L.md; ++i) {
      term[static_cast<size_t>(i)] = rowTerm[static_cast<size_t>(i)];
      iq[static_cast<size_t>(i)] = rowIneq[static_cast<size_t>(i)] ? 1 : 0;
    }
    w.writeVectorI32("/rows/term", term);
    w.writeVectorI8("/rows/ineq", iq);
    w.writeVector("/rows/mu", muRow);
  }

  // --- meta_json ---
  std::ostringstream f;
  f << std::setprecision(17);
  f << "{\n  \"schema_version\": 1,\n";
  f << "  \"generator\": \"" << esc(info.generator) << "\",\n";
  f << "  \"problem\": {\"name\": \"" << esc(info.problemName) << "\", \"type\": \"" << esc(info.problemType) << "\", \"yaml\": \""
    << esc(info.problemYamlText) << "\"},\n";
  const RobotDescription& rd = m.description();
  f << "  \"robot\": {\"urdf\": \"" << esc(info.urdfPath) << "\", \"motor_model\": \"" << esc(info.motorModel)
    << "\", \"motor_parameters\": {\"coulomb\": [";
  doubleList(f, rd.motor.coulomb);
  f << "], \"viscous\": [";
  doubleList(f, rd.motor.viscous);
  f << "], \"armature\": [";
  doubleList(f, rd.motor.armature);
  f << "], \"bias\": [";
  doubleList(f, rd.motor.bias);
  f << "], \"velocity_smoothing\": " << rd.motor.velocitySmoothing << "}, ";
  f << "\"saturation\": {\"enabled\": " << (rd.saturation.enabled ? "true" : "false") << ", \"enforce_in_optimization\": "
    << (rd.saturation.enforceInOptimization ? "true" : "false") << ", \"effort_scale\": " << rd.saturation.effortScale
    << ", \"speed_derating\": " << (rd.saturation.speedDerating ? "true" : "false") << ", \"knee_fraction\": " << rd.saturation.kneeFraction
    << "}, ";
  f << "\"lag\": {\"enabled\": " << (rd.lag.enabled ? "true" : "false") << ", \"time_constant\": " << rd.lag.timeConstant << "}, "
    << "\"yaml\": \"" << esc(info.robotYamlText) << "\"},\n";
  f << "  \"solver\": {\"backend\": \"block_schur\", \"fingerprint\": \"" << esc(info.solverFingerprint) << "\", \"config_hash\": \""
    << esc(info.configHashHex) << "\"},\n";
  bool anyDerived = false, anyBezier = false;
  for (const ElementLayout& el : L.elem) {
    anyDerived = anyDerived || !el.uStored;
    anyBezier = anyBezier || el.Nc > 0;
  }
  f << "  \"parameterization\": {\"bezier_lambda\": " << (anyBezier ? "true" : "false") << ", \"derived_actuation\": "
    << (anyDerived ? "true" : "false") << ", \"derived_kinematics\": false},\n";
  f << "  \"overrides\": [";
  strList(f, info.overrides);
  f << "],\n";
  f << "  \"model\": {\"nq\": " << m.nq() << ", \"nv\": " << m.nv() << ", \"nu\": " << m.nu() << ",\n    \"joint_names\": [";
  strList(f, m.jointNames());
  f << "],\n    \"contacts\": [";
  for (int c = 0; c < m.nContacts(); ++c)
    f << (c ? ", " : "") << "{\"name\": \"" << esc(m.contactName(c)) << "\", \"dim\": " << m.contactDim(c) << "}";
  f << "],\n    \"actuated_idx\": [";
  for (size_t i = 0; i < m.actuatedIdx().size(); ++i) f << (i ? ", " : "") << m.actuatedIdx()[i];
  f << "]},\n";
  // collision geometry: the capsule model and the obstacle scene the solve enforced
  if (m.nCapsules() > 0) {
    f << "  \"collision\": {\"capsules\": [";
    for (int i = 0; i < m.nCapsules(); ++i) {
      f << (i ? ", " : "") << "{\"name\": \"" << esc(m.capsuleName(i)) << "\", \"frame\": \"" << esc(m.capsuleFrame(i)) << "\", \"a\": ";
      vec3(f, m.capsuleLocalA(i));
      f << ", \"b\": ";
      vec3(f, m.capsuleLocalB(i));
      f << ", \"radius\": ";
      num(f, m.capsuleRadius(i));
      f << "}";
    }
    f << "]";
    const CapsuleSphereConstraint* sph = nullptr;
    const CapsuleCuboidConstraint* cub = nullptr;
    for (const Element& e : spec.elements)
      for (const NodeAttachment& a : e.nodeTerms) {
        if (const CapsuleSphereConstraint* c = dynamic_cast<const CapsuleSphereConstraint*>(a.con.get())) sph = c;
        if (const CapsuleCuboidConstraint* c = dynamic_cast<const CapsuleCuboidConstraint*>(a.con.get())) cub = c;
      }
    if (sph != nullptr || cub != nullptr) {
      f << ",\n    \"margin\": ";
      num(f, sph != nullptr ? sph->margin() : cub->margin());
      f << ", \"obstacles\": [";
      if (sph != nullptr) {
        const std::vector<SphereObstacle>& sp = sph->obstacles();
        for (size_t i = 0; i < sp.size(); ++i) {
          f << (i ? ", " : "") << "{\"name\": \"" << esc(sp[i].name) << "\", \"center\": ";
          vec3(f, sp[i].center);
          f << ", \"radius\": ";
          num(f, sp[i].radius);
          f << "}";
        }
      }
      f << "], \"boxes\": [";
      if (cub != nullptr) {
        const std::vector<CuboidObstacle>& bx = cub->boxes();
        for (size_t i = 0; i < bx.size(); ++i) {
          f << (i ? ", " : "") << "{\"name\": \"" << esc(bx[i].name) << "\", \"center\": ";
          vec3(f, bx[i].center);
          f << ", \"half\": ";
          vec3(f, bx[i].half);
          f << ", \"rot\": [";
          for (int rr = 0; rr < 3; ++rr)
            for (int cc = 0; cc < 3; ++cc) {
              if (rr || cc) f << ", ";
              num(f, bx[i].rot(rr, cc));
            }
          f << "]}";
        }
      }
      f << "], \"enforced_capsules\": [";
      std::vector<std::string> names;
      for (int c : sph != nullptr ? sph->capsuleIndices() : cub->capsuleIndices()) names.push_back(m.capsuleName(c));
      strList(f, names);
      f << "]";
      const std::vector<double>& cm = sph != nullptr ? sph->capsuleMargins() : cub->capsuleMargins();
      if (!cm.empty()) {
        f << ", \"capsule_margins\": [";
        doubleList(f, cm);
        f << "]";
      }
    }
    f << "},\n";
  }
  f << "  \"terms\": [";
  for (int t = 0; t < prob.terms().size(); ++t) f << (t ? ", " : "") << '"' << esc(prob.terms().name(t)) << '"';
  f << "],\n";
  f << "  \"segments\": [\n";
  for (size_t e = 0; e < M; ++e) {
    const ElementLayout& el = L.elem[e];
    const Element& es = spec.elements[e];
    f << "    {\"index\": " << e << ", \"parent_mode\": " << es.phase << ", \"label\": \"" << esc(es.label) << "\", \"t0\": " << el.t0
      << ", \"T\": " << el.T << ", \"Nn\": " << el.Nn << ", \"degree\": " << prob.grid(static_cast<int>(e)).degree() << ", \"Nlam\": " << el.Nc << ",\n";
    f << "     \"dims\": {\"nq\": " << el.dims.nq << ", \"nv\": " << el.dims.nv << ", \"na\": " << el.dims.na << ", \"nlam\": " << el.dims.nlam
      << ", \"nu\": " << el.dims.nu << "},\n";
    f << "     \"active_contacts\": [";
    for (size_t c = 0; c < es.contacts.active.size(); ++c)
      f << (c ? ", " : "") << "{\"name\": \"" << esc(m.contactName(es.contacts.active[c].index)) << "\", \"dim\": " << es.contacts.active[c].dim << "}";
    f << "],\n     \"reset_after\": \"" << (e + 1 < M ? spec.resets[e]->kindName() : "none") << "\"}" << (e + 1 < M ? "," : "") << "\n";
  }
  f << "  ],\n";
  f << "  \"row_layout\": {\"reset_row0\": [";
  for (size_t t = 0; t < L.resetRow0.size(); ++t) f << (t ? ", " : "") << L.resetRow0[t];
  f << "], \"np\": " << L.np << ", \"md\": " << L.md << "},\n";
  f << "  \"metrics\": {\"gate\": ";
  num(f, gate);
  f << ", \"internode_dyn_max\": ";
  num(f, info.interNodeDynMax);
  f << ", \"gnorm_final\": ";
  num(f, gnormFinal);
  f << ", \"cost\": ";
  num(f, ev.cost);
  f << ",\n    \"group_violations\": {\"dyn\": " << gDyn << ", \"foot\": " << gFoot << ", \"kin\": " << gKin << ", \"bc\": " << gBc
    << ", \"reset\": " << gReset << ", \"ineq\": " << gIneq << "},\n";
  f << "    \"flow_a\": null,\n    \"flow_b\": ";
  flowJson(f, info.lastFlow);
  f << "},\n";
  if (info.provenance) {
    char ts[32] = "";
    const std::time_t now = std::time(nullptr);
    std::tm tmv{};
    gmtime_r(&now, &tmv);
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    f << "  \"run\": {\"timestamp_utc\": \"" << ts << "\", \"wall_s\": " << info.wallS << ", \"omp_threads\": " << info.ompThreads
      << ", \"warnings\": [";
    strList(f, info.warnings);
    f << "]},\n";
  }
  f << "  \"format\": \"leap-solution-archive\"\n}\n";
  w.writeString("/meta_json", f.str());
  w.close();
}

}  // namespace leap
