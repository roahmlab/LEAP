#include "leap/io/h5_writer.hpp"

#include <stdexcept>

#include <hdf5.h>

namespace leap {

namespace {

[[noreturn]] void fail(const std::string& what, const std::string& path) {
  throw std::runtime_error("h5: " + what + " failed for '" + path + "'");
}

// Scope guard for one hid_t handle.
struct Hid {
  hid_t id;
  herr_t (*closer)(hid_t);
  Hid(hid_t i, herr_t (*c)(hid_t)) : id(i), closer(c) {}
  ~Hid() {
    if (id >= 0 && closer) closer(id);
  }
  Hid(const Hid&) = delete;
  Hid& operator=(const Hid&) = delete;
  bool ok() const { return id >= 0; }
};

}  // namespace

struct H5Writer::Impl {
  hid_t file = H5I_INVALID_HID;
  hid_t lcpl = H5I_INVALID_HID;  // link creation: intermediate groups
  hid_t dcpl = H5I_INVALID_HID;  // dataset creation: no object timestamps
  std::string path;

  void writeDataset(const std::string& dset, hid_t fileType, hid_t memType, int rank, const hsize_t* dims, const void* data) {
    Hid space(H5Screate_simple(rank, dims, nullptr), H5Sclose);
    if (!space.ok()) fail("H5Screate_simple", dset);
    Hid ds(H5Dcreate2(file, dset.c_str(), fileType, space.id, lcpl, dcpl, H5P_DEFAULT), H5Dclose);
    if (!ds.ok()) fail("H5Dcreate2", dset);
    hsize_t n = 1;
    for (int i = 0; i < rank; ++i) n *= dims[i];
    if (n > 0 && H5Dwrite(ds.id, memType, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) fail("H5Dwrite", dset);
  }
};

H5Writer::H5Writer(const std::string& path) : impl_(std::make_unique<Impl>()) {
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);  // exceptions, not stderr stack dumps
  impl_->path = path;
  impl_->file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (impl_->file < 0) fail("H5Fcreate", path);
  impl_->lcpl = H5Pcreate(H5P_LINK_CREATE);
  if (impl_->lcpl < 0 || H5Pset_create_intermediate_group(impl_->lcpl, 1) < 0) fail("link-creation plist", path);
  impl_->dcpl = H5Pcreate(H5P_DATASET_CREATE);
  if (impl_->dcpl < 0 || H5Pset_obj_track_times(impl_->dcpl, false) < 0) fail("dataset-creation plist", path);
}

H5Writer::~H5Writer() {
  try {
    close();
  } catch (...) {
  }
}

void H5Writer::close() {
  if (!impl_) return;
  if (impl_->lcpl >= 0) {
    H5Pclose(impl_->lcpl);
    impl_->lcpl = H5I_INVALID_HID;
  }
  if (impl_->dcpl >= 0) {
    H5Pclose(impl_->dcpl);
    impl_->dcpl = H5I_INVALID_HID;
  }
  if (impl_->file >= 0) {
    const herr_t rc = H5Fclose(impl_->file);
    impl_->file = H5I_INVALID_HID;
    if (rc < 0) fail("H5Fclose", impl_->path);
  }
}

void H5Writer::writeMatrix(const std::string& dset, const Eigen::MatrixXd& m) {
  const hsize_t r = static_cast<hsize_t>(m.rows()), c = static_cast<hsize_t>(m.cols());
  std::vector<double> buf(static_cast<size_t>(r * c));
  for (hsize_t i = 0; i < r; ++i)
    for (hsize_t j = 0; j < c; ++j) buf[static_cast<size_t>(i * c + j)] = m(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
  const hsize_t dims[2] = {r, c};
  impl_->writeDataset(dset, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 2, dims, buf.data());
}

void H5Writer::writeVector(const std::string& dset, const Eigen::VectorXd& v) {
  const hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
  impl_->writeDataset(dset, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, 1, dims, v.data());
}

void H5Writer::writeMatrixI8(const std::string& dset, const std::vector<std::int8_t>& rowMajor, long rows, long cols) {
  if (static_cast<long>(rowMajor.size()) != rows * cols) fail("i8 matrix size mismatch", dset);
  const hsize_t dims[2] = {static_cast<hsize_t>(rows), static_cast<hsize_t>(cols)};
  impl_->writeDataset(dset, H5T_STD_I8LE, H5T_NATIVE_INT8, 2, dims, rowMajor.data());
}

void H5Writer::writeVectorI32(const std::string& dset, const std::vector<std::int32_t>& v) {
  const hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
  impl_->writeDataset(dset, H5T_STD_I32LE, H5T_NATIVE_INT32, 1, dims, v.data());
}

void H5Writer::writeVectorI8(const std::string& dset, const std::vector<std::int8_t>& v) {
  const hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
  impl_->writeDataset(dset, H5T_STD_I8LE, H5T_NATIVE_INT8, 1, dims, v.data());
}

void H5Writer::writeString(const std::string& dset, const std::string& s) {
  Hid type(H5Tcopy(H5T_C_S1), H5Tclose);
  if (!type.ok() || H5Tset_size(type.id, H5T_VARIABLE) < 0 || H5Tset_cset(type.id, H5T_CSET_UTF8) < 0) fail("string type", dset);
  Hid space(H5Screate(H5S_SCALAR), H5Sclose);
  if (!space.ok()) fail("H5Screate", dset);
  Hid ds(H5Dcreate2(impl_->file, dset.c_str(), type.id, space.id, impl_->lcpl, impl_->dcpl, H5P_DEFAULT), H5Dclose);
  if (!ds.ok()) fail("H5Dcreate2", dset);
  const char* p = s.c_str();
  if (H5Dwrite(ds.id, type.id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &p) < 0) fail("H5Dwrite", dset);
}

void H5Writer::writeRootAttr(const std::string& name, const std::string& value) {
  Hid type(H5Tcopy(H5T_C_S1), H5Tclose);
  if (!type.ok() || H5Tset_size(type.id, H5T_VARIABLE) < 0 || H5Tset_cset(type.id, H5T_CSET_UTF8) < 0) fail("attr type", name);
  Hid space(H5Screate(H5S_SCALAR), H5Sclose);
  if (!space.ok()) fail("H5Screate", name);
  Hid attr(H5Acreate2(impl_->file, name.c_str(), type.id, space.id, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
  if (!attr.ok()) fail("H5Acreate2", name);
  const char* p = value.c_str();
  if (H5Awrite(attr.id, type.id, &p) < 0) fail("H5Awrite", name);
}

void H5Writer::writeRootAttr(const std::string& name, std::int64_t value) {
  Hid space(H5Screate(H5S_SCALAR), H5Sclose);
  if (!space.ok()) fail("H5Screate", name);
  Hid attr(H5Acreate2(impl_->file, name.c_str(), H5T_STD_I64LE, space.id, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
  if (!attr.ok()) fail("H5Acreate2", name);
  if (H5Awrite(attr.id, H5T_NATIVE_INT64, &value) < 0) fail("H5Awrite", name);
}

bool readSolutionVector(const std::string& path, Eigen::VectorXd& y, std::string& error) {
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  Hid file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file.ok()) {
    error = "cannot open " + path;
    return false;
  }
  Hid ds(H5Dopen2(file.id, "/solution/y", H5P_DEFAULT), H5Dclose);
  if (!ds.ok()) {
    error = "no /solution/y in " + path;
    return false;
  }
  Hid space(H5Dget_space(ds.id), H5Sclose);
  if (!space.ok() || H5Sget_simple_extent_ndims(space.id) != 1) {
    error = "/solution/y is not a vector";
    return false;
  }
  hsize_t n = 0;
  H5Sget_simple_extent_dims(space.id, &n, nullptr);
  y.resize(static_cast<Eigen::Index>(n));
  if (n > 0 && H5Dread(ds.id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, y.data()) < 0) {
    error = "H5Dread failed on /solution/y";
    return false;
  }
  return true;
}

}  // namespace leap
