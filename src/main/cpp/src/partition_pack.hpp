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
#pragma once

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table_view.hpp>

namespace spark_rapids_jni {
bool supports_partition_pack(cudf::table_view const& input);

// Partition only row indices, then gather payload directly into its final packed allocation.
// Partition IDs must be non-null INT32 values in [0, num_partitions).
std::pair<std::vector<cudf::packed_columns>, std::vector<cudf::size_type>> partition_pack(
  cudf::table_view const& input,
  cudf::column_view const& partition_ids,
  cudf::size_type num_partitions,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
// Merge canonical Spark partial buffers directly into hash-partitioned packed allocations.
// One INT32/INT64 key; INT64 SUM(0), MIN(1), MAX(2), COUNT merge(3) buffers.
std::pair<std::vector<cudf::packed_columns>, std::vector<cudf::size_type>> partial_aggregate_pack(
  cudf::table_view const& input, std::vector<int> const& operations,
  cudf::size_type num_partitions, rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace spark_rapids_jni
