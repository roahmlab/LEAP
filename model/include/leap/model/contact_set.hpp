#pragma once

// The active contacts of one mode and the per-node block sizes they imply.

#include <vector>

namespace leap {

// One active contact: index into the model's contact table plus its force dimension.
struct ContactRef {
  int index = -1;
  int dim = 0;
};

// Ordered list of active contacts. The order fixes the layout of the stacked contact
// force lam = [lam_c0 (dim_c0); lam_c1 (dim_c1); ...].
struct ContactSet {
  std::vector<ContactRef> active;

  int nLam() const {
    int n = 0;
    for (const ContactRef& c : active) n += c.dim;
    return n;
  }
  // Offset of contact `contactIdx`'s force block inside lam (-1 if inactive).
  int lamOffset(int contactIdx) const {
    int o = 0;
    for (const ContactRef& c : active) {
      if (c.index == contactIdx) return o;
      o += c.dim;
    }
    return -1;
  }
  bool contains(int contactIdx) const { return lamOffset(contactIdx) >= 0; }
  bool operator==(const ContactSet& o) const {
    if (active.size() != o.active.size()) return false;
    for (size_t i = 0; i < active.size(); ++i)
      if (active[i].index != o.active[i].index || active[i].dim != o.active[i].dim) return false;
    return true;
  }
  bool operator!=(const ContactSet& o) const { return !(*this == o); }
};

// Sizes of the per-node PHYSICAL vector z = [q; v; a; lam; u]. Every constraint and
// cost is written against this vector; how it maps onto decision variables (nodal
// values, Bezier force control points, inverse-dynamics torque) is the transcription's
// business, not the constraint's.
struct NodeDims {
  int nq = 0, nv = 0, na = 0, nlam = 0, nu = 0;
  int nP() const { return nq + nv + na + nlam + nu; }
  int offQ() const { return 0; }
  int offV() const { return nq; }
  int offA() const { return nq + nv; }
  int offLam() const { return nq + nv + na; }
  int offU() const { return nq + nv + na + nlam; }
};

}  // namespace leap
