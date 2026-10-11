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

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/exec_policy.hpp>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <limits>
#include <type_traits>

namespace spark_rapids_jni {
namespace {
using cudf::size_type;

struct destination {
  uint8_t* data;
  cudf::bitmask_type* mask;
  int32_t* offsets;
  size_type begin;
  size_type rows;
  int64_t char_begin;
  int64_t chars;
};

struct alignas(16) wide_element {
  uint64_t low;
  uint64_t high;
};

template <int Width>
__global__ void gather_packed(uint8_t const* source,
                              cudf::bitmask_type const* source_mask,
                              size_type source_offset,
                              int64_t const* char_prefix,
                              size_type const* gather_map,
                              destination const* destinations,
                              size_type* null_counts)
{
  auto const dst = destinations[blockIdx.x];
  // Each warp owns a mask word; all lanes participate, including the last partial word.
  auto const rounded_rows = (static_cast<int64_t>(dst.rows) + 31) / 32 * 32;
  for (int64_t row = blockIdx.y * blockDim.x + threadIdx.x;
       row < rounded_rows; row += blockDim.x * gridDim.y) {
    auto const original = row < dst.rows ? gather_map[dst.begin + row] + source_offset : 0;
    bool const valid = row < dst.rows && (source_mask == nullptr ||
                       (source_mask[original / 32] & (cudf::bitmask_type{1} << (original % 32))));
    auto const validity = __ballot_sync(0xffffffff, valid);
    auto const active = __ballot_sync(0xffffffff, row < dst.rows);
    if (dst.mask != nullptr && threadIdx.x % 32 == 0) {
      dst.mask[row / 32] = validity;
      auto const nulls = __popc(active) - __popc(validity);
      if (nulls > 0) { atomicAdd(null_counts + blockIdx.x, nulls); }
    }
    if (row >= dst.rows) { continue; }
    if constexpr (Width == 0) {
      auto const start = char_prefix[dst.begin + row] - dst.char_begin;
      dst.offsets[row] = static_cast<int32_t>(start);
      if (row == 0) { dst.offsets[dst.rows] = static_cast<int32_t>(dst.chars); }
    } else {
      using element = std::conditional_t<Width == 1, uint8_t,
                      std::conditional_t<Width == 2, uint16_t,
                      std::conditional_t<Width == 4, uint32_t,
                      std::conditional_t<Width == 8, uint64_t, wide_element>>>>;
      reinterpret_cast<element*>(dst.data)[row] =
        reinterpret_cast<element const*>(source)[original];
    }
  }
}

// One warp per string makes final-buffer writes contiguous across lanes. Keeping character
// copying separate from offset/mask generation avoids a byte loop with strided warp stores.
__global__ void gather_string_chars(uint8_t const* source,
                                     int32_t const* source_offsets,
                                     size_type source_offset,
                                     int64_t const* char_prefix,
                                     size_type const* gather_map,
                                     destination const* destinations)
{
  auto const dst = destinations[blockIdx.x];
  auto const lane = threadIdx.x % 32;
  for (int64_t row = (blockIdx.y * blockDim.x + threadIdx.x) / 32;
       row < dst.rows; row += blockDim.x * gridDim.y / 32) {
    auto const start = char_prefix[dst.begin + row] - dst.char_begin;
    auto const end = char_prefix[dst.begin + row + 1] - dst.char_begin;
    auto const original = gather_map[dst.begin + row] + source_offset;
    // Null strings have zero length, including non-empty physical null payloads.
    for (int64_t byte = lane; byte < end - start; byte += 32) {
      dst.data[start + byte] = source[source_offsets[original] + byte];
    }
  }
}

size_t aligned(size_t bytes) { return (bytes + 63) / 64 * 64; }
}  // namespace

bool supports_partition_pack(cudf::table_view const& input)
{
  if (input.num_columns() == 0) { return false; }
  for (auto const& col : input) {
    if (cudf::is_fixed_width(col.type())) {
      if (col.num_children() != 0) { return false; }
    } else if (col.type().id() == cudf::type_id::STRING) {
      if (col.size() != 0 && (col.num_children() != 1 ||
                             col.child(0).type().id() != cudf::type_id::INT32)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

std::pair<std::vector<cudf::packed_columns>, std::vector<size_type>> partition_pack(
  cudf::table_view const& input,
  cudf::column_view const& partition_ids,
  size_type num_partitions,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(supports_partition_pack(input), "Unsupported partition-pack payload");
  CUDF_EXPECTS(num_partitions > 0, "Partition count must be positive");
  CUDF_EXPECTS(partition_ids.type().id() == cudf::type_id::INT32 &&
                 partition_ids.size() == input.num_rows() && !partition_ids.has_nulls(),
               "Invalid partition IDs");
  CUDF_EXPECTS(input.num_rows() > 0, "Use the empty shuffle path for zero rows");

  // Reuse cuDF's partition algorithm on a narrow control column. The payload is never
  // materialized as a reordered table. The order within each partition is unspecified.
  auto indices = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           input.num_rows(), cudf::mask_state::UNALLOCATED,
                                           stream, mr);
  thrust::sequence(rmm::exec_policy(stream), indices->mutable_view().data<size_type>(),
                   indices->mutable_view().data<size_type>() + input.num_rows());
  auto [permutation, boundaries] =
    cudf::partition(cudf::table_view{{indices->view()}}, partition_ids,
                      num_partitions, stream, mr);
  auto const gather_map = permutation->view().column(0).data<size_type>();

  std::vector<rmm::device_uvector<int64_t>> prefixes;
  std::vector<std::vector<int64_t>> char_boundaries(input.num_columns());
  std::vector<int> prefix_index(input.num_columns(), -1);
  auto const num_rows = input.num_rows();
  auto device_boundaries = cudf::detail::make_device_uvector_async(boundaries, stream, mr);
  for (size_type c = 0; c < input.num_columns(); ++c) {
    auto const col = input.column(c);
    if (col.type().id() != cudf::type_id::STRING) { continue; }
    prefix_index[c] = prefixes.size();
    prefixes.emplace_back(static_cast<size_t>(input.num_rows()) + 1, stream, mr);
    auto& prefix = prefixes.back();
    auto const offsets = col.child(0).data<int32_t>();
    auto const mask = col.null_mask();
    auto const offset = col.offset();
    thrust::transform(rmm::exec_policy(stream), thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator(input.num_rows() + 1), prefix.begin(),
                        [=] __device__(size_type row) -> int64_t {
                          if (row == num_rows) { return 0; }
                          auto const original = gather_map[row] + offset;
                          if (mask != nullptr &&
                              !(mask[original / 32] & (cudf::bitmask_type{1} << (original % 32)))) {
                            return 0;
                          }
                          return offsets[original + 1] - offsets[original];
                        });
    thrust::exclusive_scan(rmm::exec_policy(stream), prefix.begin(), prefix.end(), prefix.begin());
    rmm::device_uvector<int64_t> sizes(num_partitions + 1, stream, mr);
    auto const prefix_ptr = prefix.data();
    thrust::transform(rmm::exec_policy(stream), device_boundaries.begin(), device_boundaries.end(),
                        sizes.begin(), [=] __device__(size_type row) { return prefix_ptr[row]; });
    char_boundaries[c] = cudf::detail::make_std_vector(sizes, stream);
  }

  std::vector<cudf::packed_columns> result;
  std::vector<std::vector<destination>> destinations(input.num_columns());
  result.reserve(num_partitions);
  for (size_type p = 0; p < num_partitions; ++p) {
    auto const rows = boundaries[p + 1] - boundaries[p];
    size_t bytes = 0;
    for (size_type c = 0; c < input.num_columns(); ++c) {
      auto const col = input.column(c);
      auto const is_string = col.type().id() == cudf::type_id::STRING;
      auto const chars = is_string ? char_boundaries[c][p + 1] - char_boundaries[c][p] : 0;
      CUDF_EXPECTS(chars <= std::numeric_limits<int32_t>::max(), "String partition exceeds 2GB");
      destination dst{};
      dst.begin = boundaries[p];
      dst.rows = rows;
      dst.char_begin = is_string ? char_boundaries[c][p] : 0;
      dst.chars = chars;
      // Store byte offsets until the allocation is made.
      dst.data = reinterpret_cast<uint8_t*>(bytes);
      bytes += aligned(is_string ? chars : static_cast<size_t>(rows) * cudf::size_of(col.type()));
      if (is_string) {
        dst.offsets = reinterpret_cast<int32_t*>(bytes);
        bytes += aligned(rows == 0 ? 0 : (static_cast<size_t>(rows) + 1) * sizeof(int32_t));
      }
      if (col.nullable() && rows > 0) {
        dst.mask = reinterpret_cast<cudf::bitmask_type*>(bytes);
        bytes += cudf::bitmask_allocation_size_bytes(rows);
      }
      destinations[c].push_back(dst);
    }
    auto buffer = std::make_unique<rmm::device_buffer>(bytes, stream, mr);
    auto const base = static_cast<uint8_t*>(buffer->data());
    for (size_type c = 0; c < input.num_columns(); ++c) {
      auto& dst = destinations[c][p];
      auto const col = input.column(c);
      auto const is_string = col.type().id() == cudf::type_id::STRING;
      dst.data = rows == 0 || (is_string && dst.chars == 0)
                   ? nullptr : base + reinterpret_cast<size_t>(dst.data);
      dst.offsets = !is_string || rows == 0
                      ? nullptr : reinterpret_cast<int32_t*>(base + reinterpret_cast<size_t>(dst.offsets));
      dst.mask = !col.nullable() || rows == 0
                   ? nullptr : reinterpret_cast<cudf::bitmask_type*>(base + reinterpret_cast<size_t>(dst.mask));
    }
    result.push_back({nullptr, std::move(buffer)});
  }

  rmm::device_uvector<size_type> null_counts(
    static_cast<size_t>(input.num_columns()) * num_partitions, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(null_counts.data(), 0, null_counts.size() * sizeof(size_type), stream.value()));
  for (size_type c = 0; c < input.num_columns(); ++c) {
    auto const col = input.column(c);
    auto device_destinations =
      cudf::detail::make_device_uvector_async(destinations[c], stream, mr);
    auto const source = col.head<uint8_t>();
    auto const mask = col.null_mask();
    auto const offset = col.offset();
    auto const dest = device_destinations.data();
    auto const counts = null_counts.data() + static_cast<size_t>(c) * num_partitions;
    if (col.type().id() == cudf::type_id::STRING) {
      gather_packed<0><<<dim3(num_partitions, 8), 256, 0, stream.value()>>>(
        source, mask, offset, prefixes[prefix_index[c]].data(),
        gather_map, dest, counts);
      gather_string_chars<<<dim3(num_partitions, 32), 256, 0, stream.value()>>>(
        source, col.child(0).data<int32_t>(), offset, prefixes[prefix_index[c]].data(),
        gather_map, dest);
    } else {
      #define GATHER_WIDTH(W)                                                \
        case W:                                                             \
          gather_packed<W><<<dim3(num_partitions, 8), 256, 0, stream.value()>>>( \
            source, mask, offset, nullptr, gather_map, dest, counts);          \
          break
      switch (cudf::size_of(col.type())) {
        GATHER_WIDTH(1);
        GATHER_WIDTH(2);
        GATHER_WIDTH(4);
        GATHER_WIDTH(8);
        GATHER_WIDTH(16);
        default: CUDF_FAIL("Unsupported fixed-width size");
      }
      #undef GATHER_WIDTH
    }
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  auto const host_null_counts = cudf::detail::make_std_vector(null_counts, stream);
  for (size_type p = 0; p < num_partitions; ++p) {
    std::vector<cudf::column_view> columns;
    for (size_type c = 0; c < input.num_columns(); ++c) {
      auto const dst = destinations[c][p];
      auto const type = input.column(c).type();
      auto const count = host_null_counts[static_cast<size_t>(c) * num_partitions + p];
      std::vector<cudf::column_view> children;
      if (type.id() == cudf::type_id::STRING) {
        children.emplace_back(cudf::data_type{cudf::type_id::INT32},
                                dst.rows == 0 ? 0 : dst.rows + 1, dst.offsets, nullptr, 0);
      }
      columns.emplace_back(type, dst.rows, dst.data, dst.mask, count, 0, children);
    }
    auto& packed = result[p];
    packed.metadata = std::make_unique<std::vector<uint8_t>>(cudf::pack_metadata(
      cudf::table_view{columns}, static_cast<uint8_t const*>(packed.gpu_data->data()),
      packed.gpu_data->size()));
  }
  return {std::move(result), std::move(boundaries)};
}
}  // namespace spark_rapids_jni
