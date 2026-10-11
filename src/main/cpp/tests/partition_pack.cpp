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

#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>
#include <climits>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

struct PartitionPackTests : cudf::test::BaseFixture {};

TEST_F(PartitionPackTests, SlicedNullablePayloadAndPartitionIds)
{
  std::vector<int32_t> values(20003);
  std::vector<std::string> strings;
  std::vector<bool> valid;
  std::vector<int32_t> ids;
  for (int32_t i = 0; i < 20003; ++i) {
    values[i] = i;
    strings.push_back(i % 3 == 0 ? "" : "中文🙂-" + std::string(i % 257, 'x') + std::to_string(i));
    valid.push_back(i % 5 != 0);
    ids.push_back(i % 7);
  }
  cudf::test::fixed_width_column_wrapper<int32_t> integers(
    values.begin(), values.end(), valid.begin());
  cudf::test::strings_column_wrapper text(strings.begin(), strings.end(), valid.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> partition_ids(ids.begin(), ids.end());
  // Exercise grid-stride loops and multiple warp-sized character-copy iterations.
  // Both the parent string offset and the partition-map offset are non-zero.
  auto const input = cudf::slice(cudf::table_view{{integers, text}}, {3, 20000}).front();
  auto const map = cudf::slice(partition_ids, {3, 20000}).front();
  for (int32_t partitions : {7, 128}) {
    auto [reference, expected_offsets] = cudf::partition(input, map, partitions);
    auto [packed, offsets] = spark_rapids_jni::partition_pack(
      input, map, partitions, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    EXPECT_EQ(expected_offsets, offsets);
    for (int32_t p = 0; p < partitions; ++p) {
      auto const expected = cudf::slice(reference->view(), {offsets[p], offsets[p + 1]}).front();
      // cuDF partitioning does not guarantee row order within a partition.
      auto expected_sorted = cudf::sort(expected);
      auto actual_sorted = cudf::sort(cudf::unpack(packed[p]));
      CUDF_TEST_EXPECT_TABLES_EQUAL(expected_sorted->view(), actual_sorted->view());
    }
  }
}

TEST_F(PartitionPackTests, PartialMergeSlicedNullableAndWrapping)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys(
    {99, 1, 1, 0, 0, 2, 2, 99}, {true, true, true, false, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<int64_t> values(
    std::initializer_list<int64_t>{99, LLONG_MAX, 1, 4, -4, 0, 0, 99}, {true, true, true, true, true, false, false, true});
  cudf::test::fixed_width_column_wrapper<int64_t> counts({99, 2, 3, 4, 5, 6, 7, 99});
  auto input = cudf::slice(cudf::table_view{{keys, values, values, values, counts}}, {1, 7}).front();
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys({1, 0, 2}, {true, false, true});
  cudf::test::fixed_width_column_wrapper<int64_t> sums(std::initializer_list<int64_t>{LLONG_MIN, 0, 0}, {true, true, false});
  cudf::test::fixed_width_column_wrapper<int64_t> mins({1, -4, 0}, {true, true, false});
  cudf::test::fixed_width_column_wrapper<int64_t> maxs(std::initializer_list<int64_t>{LLONG_MAX, 4, 0}, {true, true, false});
  cudf::test::fixed_width_column_wrapper<int64_t> expected_counts({5, 9, 13});
  auto expected = cudf::sort(cudf::table_view{{expected_keys, sums, mins, maxs, expected_counts}});
  for (int partitions : {1, 2, 128}) {
    auto [packed, boundaries] = spark_rapids_jni::partial_aggregate_pack(
      input, {0, 1, 2, 3}, partitions, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    EXPECT_EQ(boundaries.back(), 3);
    std::vector<cudf::table_view> views;
    for (auto const& table : packed) { views.push_back(cudf::unpack(table)); }
    auto combined = cudf::concatenate(views);
    auto actual = cudf::sort(combined->view());
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), actual->view());
  }
}
