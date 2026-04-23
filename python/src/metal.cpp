// Copyright © 2023-2024 Apple Inc.
#include <iostream>
#include <functional>
#include <numeric>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "python/src/small_vector.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

namespace {

struct ExportedMetalStorage {
  std::uintptr_t mtl_buffer_ptr;
  std::uintptr_t raw_ptr;
  size_t offset_bytes;
  mx::Shape shape;
  mx::Dtype dtype;
  size_t logical_nbytes;
  size_t buffer_nbytes;
  bool row_contiguous;
  bool contiguous;
};

size_t array_nbytes(const mx::Shape& shape, mx::Dtype dtype) {
  auto elems = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
  return elems * mx::size_of(dtype);
}

PyObject* retain_owner(nb::handle owner) {
  auto* owner_ptr = owner.ptr();
  Py_XINCREF(owner_ptr);
  return owner_ptr;
}

auto make_owner_deleter(PyObject* owner_ptr) {
  return [owner_ptr](void*) {
    if (owner_ptr != nullptr) {
      nb::gil_scoped_acquire gil;
      Py_DECREF(owner_ptr);
    }
  };
}

ExportedMetalStorage export_storage(mx::array& a, const char* helper_name) {
  if (!mx::metal::is_available()) {
    throw std::runtime_error(
        std::string("[") + helper_name + "] Metal back-end unavailable.");
  }
  if (!a.is_available()) {
    throw std::runtime_error(
        std::string("[") + helper_name + "] Array must already be "
        "evaluated / synchronized.");
  }
  if (!a.flags().row_contiguous) {
    throw std::runtime_error(
        std::string("[") + helper_name + "] Only row-contiguous arrays "
        "are supported.");
  }
  if (a.nbytes() != 0 && a.buffer().ptr() == nullptr) {
    throw std::runtime_error(
        std::string("[") + helper_name + "] Array has no backing buffer.");
  }
  auto raw_ptr = const_cast<mx::allocator::Buffer&>(a.buffer()).raw_ptr();
  return {
      reinterpret_cast<std::uintptr_t>(const_cast<void*>(a.buffer().ptr())),
      reinterpret_cast<std::uintptr_t>(raw_ptr),
      static_cast<size_t>(a.offset()),
      a.shape(),
      a.dtype(),
      a.nbytes(),
      a.buffer_size(),
      a.flags().row_contiguous,
      a.flags().contiguous,
  };
}

nb::object tinygrad_fast_import() {
  static nb::object fast_import =
      nb::module_::import_("tinygrad").attr("Tensor").attr("_unsafe_from_metal_buffer_fast");
  return fast_import;
}

}  // namespace

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
        auto storage = export_storage(a, "mx.metal._unsafe_export_storage");
        nb::dict out;
        out["mtl_buffer_ptr"] = nb::cast(storage.mtl_buffer_ptr);
        out["raw_ptr"] = nb::cast(storage.raw_ptr);
        out["offset_bytes"] = nb::cast(storage.offset_bytes);
        out["shape"] = nb::cast(storage.shape);
        out["strides"] = nb::cast(a.strides());
        out["dtype"] = nb::cast(storage.dtype);
        out["logical_nbytes"] = nb::cast(storage.logical_nbytes);
        out["buffer_nbytes"] = nb::cast(storage.buffer_nbytes);
        out["row_contiguous"] = nb::cast(storage.row_contiguous);
        out["contiguous"] = nb::cast(storage.contiguous);
        return out;
      },
      "array"_a,
      R"pbdoc(
      Export low-level Metal storage metadata for a realized MLX array.

      This is intentionally unsafe and only meant for private interop /
      benchmarking code.
      )pbdoc");
  metal.def(
      "_unsafe_to_tinygrad_fast",
      [](nb::object array_obj, nb::handle tg_dtype, nb::handle owner) {
        auto& a = nb::cast<mx::array&>(array_obj);
        auto storage = export_storage(a, "mx.metal._unsafe_to_tinygrad_fast");
        auto owner_obj = owner.is_none() ? array_obj : nb::borrow<nb::object>(owner);
        return tinygrad_fast_import()(
            nb::cast(storage.mtl_buffer_ptr),
            nb::cast(storage.shape),
            "dtype"_a = tg_dtype,
            "byte_offset"_a = nb::cast(storage.offset_bytes),
            "buffer_nbytes"_a = nb::cast(storage.buffer_nbytes),
            "owner"_a = owner_obj);
      },
      "array"_a,
      "tg_dtype"_a,
      "owner"_a = nb::none(),
      R"pbdoc(
      Build a tinygrad tensor directly from an MLX Metal-backed array through a
      single MLX binding entrypoint.

      This still includes tinygrad-side wrapper construction and is
      intentionally unsafe and only meant for private interop / benchmarking
      code.
      )pbdoc");
  metal.def(
      "_unsafe_rebind_tinygrad",
      [](nb::object array_obj, nb::handle borrower, nb::handle owner) {
        auto& a = nb::cast<mx::array&>(array_obj);
        auto storage = export_storage(a, "mx.metal._unsafe_rebind_tinygrad");
        auto owner_obj = owner.is_none() ? array_obj : nb::borrow<nb::object>(owner);
        return nb::borrow<nb::object>(borrower).attr("rebind")(
            nb::cast(storage.mtl_buffer_ptr),
            "owner"_a = owner_obj,
            "shape"_a = nb::cast(storage.shape),
            "dtype_name"_a = nb::cast(mx::type_to_name(storage.dtype)),
            "byte_offset"_a = nb::cast(storage.offset_bytes),
            "buffer_nbytes"_a = nb::cast(storage.buffer_nbytes));
      },
      "array"_a,
      "borrower"_a,
      "owner"_a = nb::none(),
      R"pbdoc(
      Rebind a reusable tinygrad Metal borrower from an MLX array through a
      single MLX binding entrypoint.

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

      This helper may still copy if MLX cannot alias the pointer directly.

      This is intentionally unsafe and only meant for private interop /
      benchmarking code.
      )pbdoc");
  metal.def(
      "_unsafe_array_from_ptr_alias_only",
      [](std::uintptr_t raw_ptr,
         mx::Shape shape,
         mx::Dtype dtype,
         nb::handle owner) {
        if (!mx::metal::is_available()) {
          throw std::runtime_error(
              "[mx.metal._unsafe_array_from_ptr_alias_only] Metal back-end unavailable.");
        }
        auto* owner_ptr = retain_owner(owner);
        auto owner_deleter = make_owner_deleter(owner_ptr);
        auto buffer =
            mx::allocator::make_buffer(reinterpret_cast<void*>(raw_ptr), array_nbytes(shape, dtype));
        if (buffer.ptr() == nullptr) {
          owner_deleter(reinterpret_cast<void*>(raw_ptr));
          throw std::runtime_error(
              "[mx.metal._unsafe_array_from_ptr_alias_only] MLX could not alias the pointer without a copy.");
        }
        auto wrapped_deleter = [owner_deleter](mx::allocator::Buffer buffer) {
          auto ptr = buffer.raw_ptr();
          mx::allocator::release(buffer);
          owner_deleter(ptr);
        };
        return mx::array(buffer, std::move(shape), dtype, wrapped_deleter);
      },
      "raw_ptr"_a,
      "shape"_a,
      "dtype"_a,
      "owner"_a = nb::none(),
      R"pbdoc(
      Build an MLX array from an external raw pointer, but fail unless MLX can
      alias the pointer without copying.

      This is intentionally unsafe and only meant for private interop /
      benchmarking code.
      )pbdoc");
}
