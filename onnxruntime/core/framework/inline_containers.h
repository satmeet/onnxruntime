// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <memory_resource>
#include <absl/container/inlined_vector.h>
#include <absl/container/flat_hash_set.h>
#include <absl/container/flat_hash_map.h>

namespace onnxruntime {

template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N>;

template <typename T>
using InlineHashSet = absl::flat_hash_set<T>;

template <typename K, typename V>
using InlineHashMap = absl::flat_hash_map<K, V>;

namespace pmr {
template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N, std::pmr::polymorphic_allocator<T>>;

template <typename T>
using InlineHashSet = absl::flat_hash_set<T, absl::container_internal::hash_default_hash<T>,
                                          absl::container_internal::hash_default_eq<T>,
                                          std::pmr::polymorphic_allocator<T>>;

template <typename K, typename V>
using InlineHashMap = absl::flat_hash_map<K, V, absl::container_internal::hash_default_hash<K>,
                                          absl::container_internal::hash_default_eq<K>,
                                          std::pmr::polymorphic_allocator<std::pair<const K, V>>>;

}  // namespace pmr

#ifdef _MSC_VER
#define ORT_ALLOCA(s) _alloca(s)
constexpr size_t kOrtStackAllocationLimitBytes = 4 * 1024;
#elif defined(__GNUC__) || defined(__clang__)
#define ORT_ALLOCA(s) alloca(s)
constexpr size_t kOrtStackAllocationLimitBytes = 4 * 1024;
#else
// always on the heap
#define ORT_ALLOCA(s) nullptr
constexpr size_t kOrtStackAllocationLimitBytes = 0;
#endif

namespace ort_alloca_internal {
inline void* allocate_and_align(std::unique_ptr<uint8_t[]>& buf, size_t space, size_t aligment) {
  size_t to_allocate = space + aligment;
  buf.reset(new uint8_t[to_allocate]);
  void* ptr = buf.get();
  return std::align(aligment, to_allocate, ptr, to_allocate);
}
}  // namespace ort_alloca_internal

// Dynamically allocated size on the stack
#define OrtDeclareAllignedStackBuffer(buffer_ptr, size_in_bytes, alignment)                                        \
  std::unique_ptr<uint8_t[]> on_heap_##buffer_ptr;                                                                 \
  void* buffer_ptr = (size_in_bytes > kOrtStackAllocationLimitBytes)                                               \
                         ? ort_alloca_internal::allocate_and_align(on_heap_##buffer_ptr, size_in_bytes, alignment) \
                         : ORT_ALLOCA(size_in_bytes)

// This gives a set size stackbuffer
template <typename T, size_t N>
class SmallBuffer {
  T buffer_[N];

 public:
  T* Buffer() noexcept { return buffer_; }
  constexpr size_t size() const noexcept { return N; }
  constexpr size_t size_in_bytes() const noexcept { return sizeof(T) * N; }
};

class SmallBufferResource {
  std::pmr::monotonic_buffer_resource resource_;

 public:
  SmallBufferResource(void* ptr, size_t size_in_bytes)
      : resource_(ptr, size_in_bytes, std::pmr::get_default_resource()) {}
  SmallBufferResource(void* ptr, size_t size_in_bytes, std::pmr::memory_resource* upstream)
      : resource_(ptr, size_in_bytes, upstream) {}
  std::pmr::memory_resource* resource() noexcept { return &resource_; }
  std::pmr::memory_resource* upstream() const noexcept { return resource_.upstream_resource(); }
};

}  // namespace onnxruntime
