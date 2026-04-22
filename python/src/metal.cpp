// Copyright © 2023-2024 Apple Inc.
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "python/src/small_vector.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

bool DEPRECATE(const char* old_fn, const char* new_fn) {
  std::cerr << old_fn << " is deprecated and will be removed in a future "
            << "version. Use " << new_fn << " instead." << std::endl;
  return true;
}

#define DEPRECATE(oldfn, newfn) static bool dep = DEPRECATE(oldfn, newfn)

void init_metal(nb::module_& m) {
  nb::module_ metal = m.def_submodule("metal", "mlx.metal");
  metal.def(
      "is_available",
      &mx::metal::is_available,
      R"pbdoc(
      Check if the Metal back-end is available.
      )pbdoc");
  metal.def("get_active_memory", []() {
    DEPRECATE("mx.metal.get_active_memory", "mx.get_active_memory");
    return mx::get_active_memory();
  });
  metal.def("get_peak_memory", []() {
    DEPRECATE("mx.metal.get_peak_memory", "mx.get_peak_memory");
    return mx::get_peak_memory();
  });
  metal.def("reset_peak_memory", []() {
    DEPRECATE("mx.metal.reset_peak_memory", "mx.reset_peak_memory");
    mx::reset_peak_memory();
  });
  metal.def("get_cache_memory", []() {
    DEPRECATE("mx.metal.get_cache_memory", "mx.get_cache_memory");
    return mx::get_cache_memory();
  });
  metal.def(
      "set_memory_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_memory_limit", "mx.set_memory_limit");
        return mx::set_memory_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_cache_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_cache_limit", "mx.set_cache_limit");
        return mx::set_cache_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_wired_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_wired_limit", "mx.set_wired_limit");
        return mx::set_wired_limit(limit);
      },
      "limit"_a);
  metal.def("clear_cache", []() {
    DEPRECATE("mx.metal.clear_cache", "mx.clear_cache");
    mx::clear_cache();
  });
  metal.def(
      "start_capture",
      &mx::metal::start_capture,
      "path"_a,
      R"pbdoc(
      Start a Metal capture.

      Args:
        path (str): The path to save the capture which should have
          the extension ``.gputrace``.
      )pbdoc");
  metal.def(
      "stop_capture",
      &mx::metal::stop_capture,
      R"pbdoc(
      Stop a Metal capture.
      )pbdoc");
  metal.def("device_info", []() {
    DEPRECATE("mx.metal.device_info", "mx.device_info");
    return mx::device_info(mx::Device(mx::Device::gpu, 0));
  });
  metal.def(
      "_unsafe_export_storage",
      [](mx::array& a) {
        if (!mx::metal::is_available()) {
          throw std::runtime_error(
              "[mx.metal._unsafe_export_storage] Metal back-end unavailable.");
        }
        if (!a.is_available()) {
          throw std::runtime_error(
              "[mx.metal._unsafe_export_storage] Array must already be "
              "evaluated / synchronized.");
        }
        if (!a.flags().row_contiguous) {
          throw std::runtime_error(
              "[mx.metal._unsafe_export_storage] Only row-contiguous arrays "
              "are supported.");
        }
        if (a.nbytes() != 0 && a.buffer().ptr() == nullptr) {
          throw std::runtime_error(
              "[mx.metal._unsafe_export_storage] Array has no backing buffer.");
        }
        auto raw_ptr = const_cast<mx::allocator::Buffer&>(a.buffer()).raw_ptr();
        nb::dict out;
        out["mtl_buffer_ptr"] =
            nb::cast(reinterpret_cast<std::uintptr_t>(
                const_cast<void*>(a.buffer().ptr())));
        out["raw_ptr"] = nb::cast(reinterpret_cast<std::uintptr_t>(raw_ptr));
        out["offset_bytes"] = nb::cast(static_cast<size_t>(a.offset()));
        out["shape"] = nb::cast(a.shape());
        out["strides"] = nb::cast(a.strides());
        out["dtype"] = nb::cast(a.dtype());
        out["nbytes"] = nb::cast(a.nbytes());
        out["row_contiguous"] = nb::cast(a.flags().row_contiguous);
        out["contiguous"] = nb::cast(a.flags().contiguous);
        return out;
      },
      "array"_a,
      R"pbdoc(
      Export low-level Metal storage metadata for a realized MLX array.

      This is intentionally unsafe and only meant for private interop /
      benchmarking code.
      )pbdoc");
  metal.def(
      "_unsafe_array_from_ptr",
      [](std::uintptr_t raw_ptr,
         mx::Shape shape,
         mx::Dtype dtype,
         nb::handle owner) {
        if (!mx::metal::is_available()) {
          throw std::runtime_error(
              "[mx.metal._unsafe_array_from_ptr] Metal back-end unavailable.");
        }
        auto* owner_ptr = owner.ptr();
        Py_XINCREF(owner_ptr);
        auto deleter = [owner_ptr](void*) {
          if (owner_ptr != nullptr) {
            nb::gil_scoped_acquire gil;
            Py_DECREF(owner_ptr);
          }
        };
        return mx::array(
            reinterpret_cast<void*>(raw_ptr), std::move(shape), dtype, deleter);
      },
      "raw_ptr"_a,
      "shape"_a,
      "dtype"_a,
      "owner"_a = nb::none(),
      R"pbdoc(
      Build an MLX array from an external raw pointer.

      This is intentionally unsafe and only meant for private interop /
      benchmarking code.
      )pbdoc");
}
