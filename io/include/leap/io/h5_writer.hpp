#pragma once

// Minimal write-only HDF5 wrapper for the solution archive (docs/solution_format.md).
// The HDF5 C API stays inside one translation unit.
//
//   * dataset paths are absolute ("/segments/00/Q"); intermediate groups are created;
//   * doubles are written as IEEE f8 in C (row-major) order: Eigen's column-major storage
//     is transposed into a scratch buffer, so h5py reads every dataset as (rows, cols);
//   * strings are variable-length UTF-8 scalars;
//   * object time tracking is off;
//   * every failure throws std::runtime_error naming the dataset.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace leap {

class H5Writer {
public:
  explicit H5Writer(const std::string& path);  // creates / truncates
  ~H5Writer();
  H5Writer(const H5Writer&) = delete;
  H5Writer& operator=(const H5Writer&) = delete;

  void writeMatrix(const std::string& dset, const Eigen::MatrixXd& m);
  void writeVector(const std::string& dset, const Eigen::VectorXd& v);
  void writeMatrixI8(const std::string& dset, const std::vector<std::int8_t>& rowMajor, long rows, long cols);
  void writeVectorI32(const std::string& dset, const std::vector<std::int32_t>& v);
  void writeVectorI8(const std::string& dset, const std::vector<std::int8_t>& v);
  void writeString(const std::string& dset, const std::string& s);
  void writeRootAttr(const std::string& name, const std::string& value);
  void writeRootAttr(const std::string& name, std::int64_t value);
  void close();  // flush + close; idempotent

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Read back the exact decision vector at /solution/y (warm starts, tests). Returns false
// with `error` filled on any I/O or size failure.
bool readSolutionVector(const std::string& path, Eigen::VectorXd& y, std::string& error);

}  // namespace leap
