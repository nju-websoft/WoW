#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <vector>
#include <string>
#include <utility>
#include <stdexcept>
#include <unordered_set>
#include <functional>
#include <omp.h>

#include "../wow/utils.hh"  // Contains label_t, dist_t definitions
#include "../wow/index.hh"  // Contains WoWIndex and filter struct definitions

namespace py = pybind11;

template <size_t N>
struct FixedString
{
  std::array<char, N> data;

  FixedString() { data.fill(0); }

  FixedString(const char *c_str)
  {
    if (c_str) {
      strncpy(data.data(), c_str, N - 1);
      data[N - 1] = '\0';  // Ensure null termination
    } else {
      data.fill(0);
    }
  }

  FixedString(const std::string &str) : FixedString(str.c_str()) {}

  const char *c_str() const { return data.data(); }

  std::string toString() const { return std::string(data.data(), strnlen(data.data(), N)); }

  static constexpr size_t capacity() { return N; }
  size_t                  length() const { return strnlen(data.data(), N); }

  bool operator==(const FixedString<N> &other) const { return strncmp(data.data(), other.data.data(), N) == 0; }

  bool operator<(const FixedString<N> &other) const { return strncmp(data.data(), other.data.data(), N) < 0; }

  bool operator!=(const FixedString<N> &other) const { return !(*this == other); }
  bool operator>(const FixedString<N> &other) const { return other < *this; }
  bool operator<=(const FixedString<N> &other) const { return !(other < *this); }
  bool operator>=(const FixedString<N> &other) const { return !(*this < other); }
};

namespace std {
template <size_t N>
struct hash<FixedString<N>>
{
  std::size_t operator()(const FixedString<N> &k) const
  {
    return std::hash<std::string_view>()(std::string_view(k.c_str(), k.length()));
  }
};
}  // namespace std

// --- Type Definitions ---
using VecType = float;

// *** Use wowlib namespace qualifier ***
using LabelType = wowlib::label_t;
using DistType  = wowlib::dist_t;
// Define FixedString types that will be used as att_t
using AttString16 = FixedString<16>;
using AttString32 = FixedString<32>;

// Define specific instantiations
using IndexSequential   = wowlib::WoWIndex<LabelType, VecType>;
using BitsetLabelFilter = wowlib::wow_bitset<LabelType>;  // LabelType is now defined
using RangeLabelFilter  = wowlib::wow_range<LabelType>;

// --- Helper function for numpy array checks ---
void check_numpy_array(const py::array_t<VecType> &arr, const std::string &name, size_t expected_dim)
{
  if (!arr)
    throw std::runtime_error(name + " cannot be None");
  py::buffer_info buf = arr.request();
  if (buf.ndim != 1)
    throw std::runtime_error(name + " must be 1-D");
  if ((size_t)buf.shape[0] != expected_dim)
    throw std::runtime_error(
        name + " has dim " + std::to_string(buf.shape[0]) + ", expected " + std::to_string(expected_dim));
  if (!py::isinstance<py::array_t<VecType>>(arr) || !arr.dtype().is(py::dtype::of<VecType>()))
    throw std::runtime_error(name + " has incorrect dtype, expected float32");
}
void check_numpy_array_batch(
    const py::array_t<VecType> &arr, const std::string &name, size_t expected_num_vectors, size_t expected_vec_dim)
{
  if (!arr)
    throw std::runtime_error(name + " cannot be None");
  py::buffer_info buf = arr.request();
  if (buf.ndim != 2)
    throw std::runtime_error(name + " must be 2-D");
  if ((size_t)buf.shape[0] != expected_num_vectors)
    throw std::runtime_error(name + " wrong num_vectors. Expected " + std::to_string(expected_num_vectors) + ", got " +
                             std::to_string(buf.shape[0]));
  if ((size_t)buf.shape[1] != expected_vec_dim)
    throw std::runtime_error(name + " wrong vec_dim. Expected " + std::to_string(expected_vec_dim) + ", got " +
                             std::to_string(buf.shape[1]));
  if (!py::isinstance<py::array_t<VecType>>(arr) || !arr.dtype().is(py::dtype::of<VecType>()))
    throw std::runtime_error(name + " has incorrect dtype, expected float32");
}

// --- Helper Template Function to Bind WoWIndex and its Associated Filters ---
template <typename AttType, typename ModuleType>
void bind_wow_index_specialization(ModuleType &m, const std::string &python_name_suffix)
{
  using IndexSpecialized       = wowlib::WoWIndex<AttType, VecType>;
  using RangeFilterSpecialized = wowlib::wow_range<AttType>;
  using SetFilterSpecialized   = wowlib::wow_set<AttType>;

  std::string index_class_name        = "_WoWIndex" + python_name_suffix;  // Internal name
  std::string range_filter_class_name = "_WoWRangeFilter" + python_name_suffix;
  std::string set_filter_class_name   = "_WoWSetFilter" + python_name_suffix;

  // --- Bind AttType if it's FixedString ---
  if constexpr (std::is_base_of_v<FixedString<1>, AttType> ||  // Check if AttType is any FixedString<N>
                (std::is_same_v<AttType, FixedString<16>>) ||  // Be explicit for those used
                (std::is_same_v<AttType, FixedString<32>>)) {
    std::string fs_class_name = "_FixedString" + std::to_string(AttType::capacity());
    if (!py::hasattr(m, fs_class_name.c_str())) {
      py::class_<AttType>(m, fs_class_name.c_str())
          .def(py::init<>())
          .def(py::init<const std::string &>(), py::arg("value"))
          .def_property("value", &AttType::toString, [](AttType &self, const std::string &val) { self = AttType(val); })
          .def_static("capacity", &AttType::capacity)
          .def(py::self == py::self)
          .def(py::self < py::self)
          .def("__hash__", [](const AttType &s) { return std::hash<AttType>()(s); })
          .def("__repr__",
              [fs_class_name](const AttType &s) { return "<" + fs_class_name + " '" + s.toString() + "'>"; });
    }
  }

  if (!py::hasattr(m, range_filter_class_name.c_str())) {
    py::class_<RangeFilterSpecialized> range_binder(m, range_filter_class_name.c_str());
    range_binder.def(py::init<>());

    // Constructor for RangeFilterSpecialized
    range_binder.def(py::init([](py::object py_lower, py::object py_upper) {
      AttType lower_bound, upper_bound;
      if constexpr (std::is_base_of_v<FixedString<1>, AttType> || (std::is_same_v<AttType, FixedString<16>>) ||
                    (std::is_same_v<AttType, FixedString<32>>)) {
        if (!py::isinstance<py::str>(py_lower) || !py::isinstance<py::str>(py_upper)) {
          throw py::type_error("Range filter bounds must be strings for FixedString attribute types.");
        }
        lower_bound = AttType(py_lower.cast<std::string>());
        upper_bound = AttType(py_upper.cast<std::string>());
      } else {
        lower_bound = py_lower.cast<AttType>();
        upper_bound = py_upper.cast<AttType>();
      }
      return std::make_unique<RangeFilterSpecialized>(lower_bound, upper_bound);
    }),
        py::arg("lower_bound"),
        py::arg("upper_bound"));

    range_binder
        .def_readwrite("l_", &RangeFilterSpecialized::l_)  // Direct access still works with C++ AttType
        .def_readwrite("u_", &RangeFilterSpecialized::u_)
        // Test method
        .def(
            "test",
            [](const RangeFilterSpecialized &self, py::object py_attr) {  // Renamed from test
              AttType cpp_attribute;
              if constexpr (std::is_base_of_v<FixedString<1>, AttType> || (std::is_same_v<AttType, FixedString<16>>) ||
                            (std::is_same_v<AttType, FixedString<32>>)) {
                if (!py::isinstance<py::str>(py_attr)) {
                  throw py::type_error("Attribute for RangeFilter::Test must be a string for FixedString types.");
                }
                cpp_attribute = AttType(py_attr.cast<std::string>());
              } else {
                cpp_attribute = py_attr.cast<AttType>();
              }
              return self.Test(cpp_attribute);
            },
            py::arg("attribute"));
  }

  if (!py::hasattr(m, set_filter_class_name.c_str())) {
    py::class_<SetFilterSpecialized> set_binder(m, set_filter_class_name.c_str());
    set_binder.def(py::init<>());

    // Add method
    set_binder.def(
        "add",
        [](SetFilterSpecialized &self, py::object py_attr) {  // Renamed from Set
          AttType cpp_attribute;
          if constexpr (std::is_base_of_v<FixedString<1>, AttType> || (std::is_same_v<AttType, FixedString<16>>) ||
                        (std::is_same_v<AttType, FixedString<32>>)) {
            if (!py::isinstance<py::str>(py_attr)) {
              throw py::type_error("Attribute for SetFilter::add must be a string for FixedString types.");
            }
            cpp_attribute = AttType(py_attr.cast<std::string>());
          } else {
            cpp_attribute = py_attr.cast<AttType>();
          }
          self.Set(cpp_attribute);
        },
        py::arg("attribute"));

    // Test method
    set_binder.def(
        "test",
        [](const SetFilterSpecialized &self, py::object py_attr) {  // Renamed from test
          AttType cpp_attribute;
          if constexpr (std::is_base_of_v<FixedString<1>, AttType> || (std::is_same_v<AttType, FixedString<16>>) ||
                        (std::is_same_v<AttType, FixedString<32>>)) {
            if (!py::isinstance<py::str>(py_attr)) {
              throw py::type_error("Attribute for SetFilter::Test must be a string for FixedString types.");
            }
            cpp_attribute = AttType(py_attr.cast<std::string>());
          } else {
            cpp_attribute = py_attr.cast<AttType>();
          }
          return self.Test(cpp_attribute);
        },
        py::arg("attribute"));

    set_binder.def_property_readonly("allowed_set", [](const SetFilterSpecialized &f) {
      py::set py_set;
      for (const auto &item : f.set_)
        py_set.add(py::cast(item));
      return py_set;
    });
  }
  // --- Bind WoWIndex Specialization ---
  py::class_<IndexSpecialized>(m, index_class_name.c_str())
      .def(py::init<size_t, size_t, size_t, size_t, std::string, size_t, size_t, bool>(),
          py::arg("max_elements"),
          py::arg("vec_d"),
          py::arg("M"),
          py::arg("efc"),
          py::arg("space_name"),
          py::arg("o")             = 4,
          py::arg("wp")            = 10,
          py::arg("auto_raise_wp") = true)
      .def(py::init<const std::string &, std::string>(), py::arg("location"), py::arg("space_name"))
      .def("save", &IndexSpecialized::save, py::arg("location"))
      .def("GetDimension", &IndexSpecialized::GetDimension)

      .def(
          "insert",
          [](IndexSpecialized &self,
              LabelType        label,
              py::array_t<VecType, py::array::c_style | py::array::forcecast>
                         vec_np,
              py::object py_attr,  // <<<< CHANGE: Take py::object for attribute
              bool       replace_deleted = false) {
            check_numpy_array(vec_np, "vector", self.GetDimension());
            py::buffer_info vec_buf = vec_np.request();

            // Now, explicitly convert py_attr to AttType
            if constexpr (std::is_base_of_v<FixedString<1>, AttType> || (std::is_same_v<AttType, FixedString<16>>) ||
                          (std::is_same_v<AttType, FixedString<32>>)) {
              // If AttType is a FixedString, py_attr should be a Python string
              if (!py::isinstance<py::str>(py_attr)) {
                throw py::type_error("Attribute must be a string for FixedString attribute types.");
              }
              // Explicitly construct AttType (FixedString<N>) from the Python string
              AttType cpp_attribute = AttType(py_attr.cast<std::string>());
              self.insert(label, static_cast<const VecType *>(vec_buf.ptr), cpp_attribute, replace_deleted);
            } else {
              // For other POD types (int32, float, etc.), pybind11's cast can handle it
              // from Python int/float.
              AttType cpp_attribute = py_attr.cast<AttType>();
              self.insert(label, static_cast<const VecType *>(vec_buf.ptr), cpp_attribute, replace_deleted);
            }
          },
          py::arg("label"),
          py::arg("vector"),
          py::arg("attribute"),
          py::arg("replace_deleted") = false)

      .def(
          "bulk_insert",
          [](IndexSpecialized              &self,
              const std::vector<LabelType> &vector_ids,
              py::array_t<VecType, py::array::c_style | py::array::forcecast>
                              vectors_np,
              const py::list &py_attributes_list,  // <<<< CHANGE: Take py::list of py::object
              bool            replace_deleted = false,
              size_t          threads         = 4) {
            size_t num_vectors = vector_ids.size();
            if (py_attributes_list.size() != num_vectors) {
              throw std::runtime_error("vector_ids and attributes_batch size mismatch");
            }
            check_numpy_array_batch(vectors_np, "vectors_batch", num_vectors, self.GetDimension());
            const VecType *data_ptr = static_cast<const VecType *>(vectors_np.request().ptr);

            // Convert Python list of attributes to std::vector<AttType>
            std::vector<AttType> cpp_attributes;
            cpp_attributes.reserve(num_vectors);

            if constexpr (std::is_base_of_v<FixedString<1>, AttType> || (std::is_same_v<AttType, FixedString<16>>) ||
                          (std::is_same_v<AttType, FixedString<32>>)) {
              for (const auto &py_attr_obj : py_attributes_list) {
                if (!py::isinstance<py::str>(py_attr_obj)) {
                  throw py::type_error("All attributes in batch must be strings for FixedString attribute types.");
                }
                cpp_attributes.emplace_back(py_attr_obj.cast<std::string>());
              }
            } else {
              for (const auto &py_attr_obj : py_attributes_list) {
                cpp_attributes.push_back(py_attr_obj.cast<AttType>());
              }
            }

            py::gil_scoped_release release_gil;

#pragma omp parallel for num_threads(threads)
            for (size_t i = 0; i < num_vectors; ++i) {
              self.insert(vector_ids[i],
                  data_ptr + i * self.GetDimension(),
                  cpp_attributes[i],  // Use the converted C++ attribute
                  replace_deleted);
            }
          },
          py::arg("vector_ids"),
          py::arg("vectors_batch"),
          py::arg("attributes_batch"),
          py::arg("replace_deleted") = false,
          py::arg("threads")         = 4)

      .def(
          "searchKNN",
          [=](IndexSpecialized &self,
              py::array_t<VecType, py::array::c_style | py::array::forcecast>
                         query_vec_np,
              size_t     efs,
              size_t     k,
              py::object filter_py) -> std::vector<std::pair<DistType, LabelType>> {
            check_numpy_array(query_vec_np, "query_vec", self.GetDimension());
            const VecType *query_vec_ptr = static_cast<const VecType *>(query_vec_np.request().ptr);

            if (py::isinstance<RangeFilterSpecialized>(filter_py)) {
              return self.template searchKNN<RangeFilterSpecialized>(
                  query_vec_ptr, efs, k, filter_py.cast<const RangeFilterSpecialized &>());
            } else if (py::isinstance<SetFilterSpecialized>(filter_py)) {
              return self.template searchKNN<SetFilterSpecialized>(
                  query_vec_ptr, efs, k, filter_py.cast<const SetFilterSpecialized &>());
            } else if (py::isinstance<BitsetLabelFilter>(filter_py)) {
              if constexpr(!std::is_same_v<AttType, LabelType>) {
                throw py::type_error(
                    "BitsetLabelFilter is only supported for WoWIndex with LabelType as attribute type.");
              } else {
                return self.template searchKNN<BitsetLabelFilter>(
                    query_vec_ptr, efs, k, filter_py.cast<const BitsetLabelFilter &>());
              }
            } else if (filter_py.is_none()) {  // no filter
              return self.searchKNN(query_vec_ptr, efs, k, 0);
            } else {
              throw py::type_error("Unsupported filter type for " + std::string(index_class_name.c_str()));
            }
          },
          py::arg("query_vec"),
          py::arg("efs"),
          py::arg("k"),
          py::arg("filter"));
}

PYBIND11_MODULE(_pywowlib_core, m)
{
  m.doc() = "Python bindings for the WoWIndex library with dynamic attribute types";

  // --- Bind Filter Types ---

  bind_wow_index_specialization<int32_t>(m, "Int32Attr");
  bind_wow_index_specialization<int64_t>(m, "Int64Attr");
  bind_wow_index_specialization<uint32_t>(m, "UInt32Attr");
  bind_wow_index_specialization<uint64_t>(m, "UInt64Attr");
  bind_wow_index_specialization<LabelType>(m, "LabelAttr");  // Add specialization for LabelType
  bind_wow_index_specialization<float>(m, "FloatAttr");
  bind_wow_index_specialization<double>(m, "DoubleAttr");
  bind_wow_index_specialization<AttString16>(m, "String16Attr");
  bind_wow_index_specialization<AttString32>(m, "String32Attr");

  // wow_bitset<label_t>
  py::class_<BitsetLabelFilter>(m, "_WoWBitsetLabelFilter")
      .def(py::init<size_t>(), py::arg("max_label"), "Initialize for labels up to max_label (exclusive)")
      .def("set", &BitsetLabelFilter::Set, py::arg("label"))
      .def("test", &BitsetLabelFilter::Test, py::arg("label"))
      .def("reset", &BitsetLabelFilter::Reset, py::arg("label"))
      .def("clear", &BitsetLabelFilter::Clear)
      .def_readonly("n", &BitsetLabelFilter::n_);
}