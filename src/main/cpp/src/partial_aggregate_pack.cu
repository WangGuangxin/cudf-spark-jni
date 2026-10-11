/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "partition_pack.hpp"
#include "hash/murmur_hash.cuh"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <rmm/device_uvector.hpp>

#include <climits>

namespace spark_rapids_jni {
namespace {
using cudf::size_type;
// Operations describe merging already canonical Spark partial buffers, not raw input rows.
constexpr int SUM = 0;
constexpr int MIN = 1;
constexpr int MAX = 2;
constexpr int COUNT = 3;
constexpr int threads = 256;
struct source {
  void const* data;
  cudf::bitmask_type const* mask;
  size_type offset;
  __device__ bool valid(size_type row) const {
    auto const i = row + offset;
    return mask == nullptr || (mask[i / 32] & (cudf::bitmask_type{1} << (i % 32)));
  }
  template <typename T> __device__ T value(size_type row) const {
    return static_cast<T const*>(data)[row + offset];
  }
};
struct destination {
  void* data;
  cudf::bitmask_type* mask;
};

__global__ void initialize(long long* states, int* valid, int const* ops,
                           size_type capacity, int aggregates) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < static_cast<int64_t>(capacity) * aggregates; i += blockDim.x * gridDim.x) {
    auto const op = ops[i / capacity];
    states[i] = op == MIN ? LLONG_MAX : op == MAX ? LLONG_MIN : 0;
    valid[i] = 0;
  }
}

template <typename Key>
__global__ void merge(source const* input, int const* ops, int aggregates,
                      size_type rows, size_type capacity, size_type* representatives,
                      long long* states, int* valid) {
  auto const key_col = input[0];
  for (size_type row = blockIdx.x * blockDim.x + threadIdx.x;
       row < rows; row += blockDim.x * gridDim.x) {
    bool const key_valid = key_col.valid(row);
    Key const key = key_valid ? key_col.value<Key>(row) : Key{0};
    uint32_t const hash = key_valid ? MurmurHash3_32<Key>{42}(key) : 42;
    auto slot = hash & (capacity - 1);
    // Store only the immutable source row index. No partially published key/state is read.
    while (true) {
      auto const rep = atomicCAS(representatives + slot, -1, row);
      if (rep == -1 || (key_col.valid(rep) == key_valid &&
                       (!key_valid || key_col.value<Key>(rep) == key))) { break; }
      slot = (slot + 1) & (capacity - 1);
    }
    for (int a = 0; a < aggregates; ++a) {
      auto const col = input[a + 1];
      if (!col.valid(row)) { continue; }
      auto const value = col.value<long long>(row);
      auto const at = static_cast<int64_t>(a) * capacity + slot;
      switch (ops[a]) {
        case SUM:
        case COUNT: atomicAdd(reinterpret_cast<unsigned long long*>(states + at),
                              static_cast<unsigned long long>(value)); break;
        case MIN: atomicMin(states + at, value); break;
        case MAX: atomicMax(states + at, value); break;
      }
      atomicExch(valid + at, 1);
    }
  }
}

template <typename Key>
__global__ void assign(source key_col, size_type capacity, size_type const* representatives,
                       size_type partitions, size_type* counts, size_type* positions) {
  for (size_type slot = blockIdx.x * blockDim.x + threadIdx.x;
       slot < capacity; slot += blockDim.x * gridDim.x) {
    auto const row = representatives[slot];
    if (row == -1) { continue; }
    int32_t const hash = key_col.valid(row) ? MurmurHash3_32<Key>{42}(key_col.value<Key>(row)) : 42;
    auto const rem = hash % partitions;
    auto const p = rem < 0 ? rem + partitions : rem;
    positions[slot] = atomicAdd(counts + p, 1);
  }
}

template <typename Key>
__global__ void write_packed(source key_col, size_type capacity,
                             size_type const* representatives, long long const* states,
                             int const* valid, size_type const* positions,
                             size_type partitions, int aggregates,
                             destination const* destinations, size_type* nulls) {
  for (size_type slot = blockIdx.x * blockDim.x + threadIdx.x;
       slot < capacity; slot += blockDim.x * gridDim.x) {
    auto const row = representatives[slot];
    if (row == -1) { continue; }
    bool const key_valid = key_col.valid(row);
    Key const key = key_valid ? key_col.value<Key>(row) : Key{0};
    int32_t const hash = key_valid ? MurmurHash3_32<Key>{42}(key) : 42;
    auto const rem = hash % partitions;
    auto const p = rem < 0 ? rem + partitions : rem;
    auto const pos = positions[slot];
    for (int c = 0; c <= aggregates; ++c) {
      auto const dst = destinations[p * (aggregates + 1) + c];
      auto const at = static_cast<int64_t>(c - 1) * capacity + slot;
      bool const is_valid = c == 0 ? key_valid : valid[at] != 0;
      if (c == 0) { static_cast<Key*>(dst.data)[pos] = key; }
      else { static_cast<long long*>(dst.data)[pos] = states[at]; }
      if (dst.mask != nullptr) {
        if (is_valid) { atomicOr(dst.mask + pos / 32, cudf::bitmask_type{1} << (pos % 32)); }
        else { atomicAdd(nulls + p * (aggregates + 1) + c, 1); }
      }
    }
  }
}
size_t aligned(size_t bytes) { return (bytes + 63) / 64 * 64; }
}  // namespace

std::pair<std::vector<cudf::packed_columns>, std::vector<size_type>> partial_aggregate_pack(
  cudf::table_view const& input, std::vector<int> const& ops, size_type partitions,
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) {
  CUDF_EXPECTS(partitions > 0 && input.num_rows() > 0, "Expected non-empty input and positive partitions");
  CUDF_EXPECTS(input.num_columns() == static_cast<int>(ops.size()) + 1 && !ops.empty(),
               "Expected one grouping key and one INT64 buffer per aggregate");
  auto const key_type = input.column(0).type();
  CUDF_EXPECTS(key_type.id() == cudf::type_id::INT32 || key_type.id() == cudf::type_id::INT64,
               "Expected an INT32/INT64 key");
  for (size_t a = 0; a < ops.size(); ++a) {
    CUDF_EXPECTS(ops[a] >= SUM && ops[a] <= COUNT &&
                   input.column(a + 1).type().id() == cudf::type_id::INT64,
                 "Unsupported merge buffer/operation");
    CUDF_EXPECTS(ops[a] != COUNT || !input.column(a + 1).has_nulls(), "COUNT buffers must be non-null");
  }
  // A <= 50% full table bounds probing even when all input keys are distinct.
  CUDF_EXPECTS(input.num_rows() <= (1 << 28), "Merge input exceeds hash state capacity");
  size_type capacity = 2;
  while (capacity < static_cast<int64_t>(input.num_rows()) * 2) { capacity *= 2; }
  auto const aggregates = static_cast<int>(ops.size());
  std::vector<source> sources;
  for (auto const& col : input) { sources.push_back({col.head<void>(), col.null_mask(), col.offset()}); }
  auto d_input = cudf::detail::make_device_uvector_async(sources, stream, mr);
  auto d_ops = cudf::detail::make_device_uvector_async(ops, stream, mr);
  rmm::device_uvector<size_type> reps(capacity, stream, mr), positions(capacity, stream, mr);
  rmm::device_uvector<long long> states(static_cast<size_t>(capacity) * aggregates, stream, mr);
  rmm::device_uvector<int> valid(states.size(), stream, mr);
  rmm::device_uvector<size_type> counts(partitions, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(reps.data(), 0xff, reps.size() * sizeof(size_type), stream.value()));
  CUDF_CUDA_TRY(cudaMemsetAsync(counts.data(), 0, counts.size() * sizeof(size_type), stream.value()));
  auto const blocks = std::min<size_type>((capacity + threads - 1) / threads, 4096);
  initialize<<<blocks, threads, 0, stream.value()>>>(states.data(), valid.data(), d_ops.data(), capacity, aggregates);
  #define MERGE_ASSIGN(Key) \
    merge<Key><<<blocks, threads, 0, stream.value()>>>(d_input.data(), d_ops.data(), aggregates, input.num_rows(), capacity, reps.data(), states.data(), valid.data()); \
    assign<Key><<<blocks, threads, 0, stream.value()>>>(sources[0], capacity, reps.data(), partitions, counts.data(), positions.data())
  if (key_type.id() == cudf::type_id::INT32) { MERGE_ASSIGN(int32_t); }
  else { MERGE_ASSIGN(int64_t); }
  #undef MERGE_ASSIGN
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const rows = cudf::detail::make_std_vector(counts, stream);
  std::vector<cudf::packed_columns> result;
  std::vector<destination> destinations;
  std::vector<std::vector<size_t>> data_offsets(partitions), mask_offsets(partitions);
  for (size_type p = 0; p < partitions; ++p) {
    size_t bytes = 0;
    for (int c = 0; c <= aggregates; ++c) {
      data_offsets[p].push_back(bytes);
      bytes += aligned(static_cast<size_t>(rows[p]) * (c == 0 ? cudf::size_of(key_type) : sizeof(int64_t)));
      bool const nullable = c == 0 ? input.column(0).nullable() : ops[c - 1] != COUNT;
      mask_offsets[p].push_back(nullable && rows[p] > 0 ? bytes : SIZE_MAX);
      if (nullable && rows[p] > 0) { bytes += cudf::bitmask_allocation_size_bytes(rows[p]); }
    }
    auto buffer = std::make_unique<rmm::device_buffer>(bytes, stream, mr);
    // Masks start invalid. Writes own disjoint rows and atomically set bits in shared words.
    if (bytes > 0) { CUDF_CUDA_TRY(cudaMemsetAsync(buffer->data(), 0, bytes, stream.value())); }
    auto const base = static_cast<uint8_t*>(buffer->data());
    for (int c = 0; c <= aggregates; ++c) {
      destinations.push_back({rows[p] == 0 ? nullptr : base + data_offsets[p][c],
        mask_offsets[p][c] == SIZE_MAX ? nullptr : reinterpret_cast<cudf::bitmask_type*>(base + mask_offsets[p][c])});
    }
    result.push_back({nullptr, std::move(buffer)});
  }
  auto d_dest = cudf::detail::make_device_uvector_async(destinations, stream, mr);
  rmm::device_uvector<size_type> nulls(destinations.size(), stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(nulls.data(), 0, nulls.size() * sizeof(size_type), stream.value()));
  #define WRITE(Key) \
    write_packed<Key><<<blocks, threads, 0, stream.value()>>>(sources[0], capacity, reps.data(), states.data(), valid.data(), positions.data(), partitions, aggregates, d_dest.data(), nulls.data())
  if (key_type.id() == cudf::type_id::INT32) { WRITE(int32_t); }
  else { WRITE(int64_t); }
  #undef WRITE
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const null_counts = cudf::detail::make_std_vector(nulls, stream);
  std::vector<size_type> boundaries{0};
  for (size_type p = 0; p < partitions; ++p) {
    std::vector<cudf::column_view> columns;
    for (int c = 0; c <= aggregates; ++c) {
      auto const dst = destinations[p * (aggregates + 1) + c];
      columns.emplace_back(c == 0 ? key_type : cudf::data_type{cudf::type_id::INT64},
        rows[p], dst.data, dst.mask, null_counts[p * (aggregates + 1) + c]);
    }
    auto& packed = result[p];
    packed.metadata = std::make_unique<std::vector<uint8_t>>(cudf::pack_metadata(
      cudf::table_view{columns}, static_cast<uint8_t const*>(packed.gpu_data->data()), packed.gpu_data->size()));
    boundaries.push_back(boundaries.back() + rows[p]);
  }
  return {std::move(result), std::move(boundaries)};
}
}  // namespace spark_rapids_jni
