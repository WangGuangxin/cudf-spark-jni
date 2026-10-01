/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include "cudf_jni_apis.hpp"
#include "jni_compiled_expr.hpp"
#include "join_primitives.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

namespace {

/**
 * @brief Convert pair of device vectors to Java long array
 * Returns a 5-element array: [size_in_bytes, left_ptr, left_handle, right_ptr, right_handle]
 */
jlongArray gather_maps_to_java(
  JNIEnv* env,
  std::pair<rmm::device_uvector<cudf::size_type>, rmm::device_uvector<cudf::size_type>> gather_maps)
{
  // Both gather maps must have the same size for paired results
  CUDF_EXPECTS(gather_maps.first.size() == gather_maps.second.size(),
               "Left and right gather maps must have the same size");

  // Release the underlying device buffers to Java
  auto left_map_buffer  = std::make_unique<rmm::device_buffer>(gather_maps.first.release());
  auto right_map_buffer = std::make_unique<rmm::device_buffer>(gather_maps.second.release());

  cudf::jni::native_jlongArray result(env, 5);
  // Return size in bytes (as expected by DeviceMemoryBuffer.fromRmm)
  result[0] = static_cast<jlong>(left_map_buffer->size());
  result[1] = cudf::jni::ptr_as_jlong(left_map_buffer->data());
  result[2] = cudf::jni::release_as_jlong(left_map_buffer);
  result[3] = cudf::jni::ptr_as_jlong(right_map_buffer->data());
  result[4] = cudf::jni::release_as_jlong(right_map_buffer);
  return result.get_jArray();
}

/**
 * @brief Convert device vector to Java long array (single gather map)
 * Returns a 3-element array: [size_in_bytes, device_ptr, rmm_handle]
 */
jlongArray gather_single_map_to_java(JNIEnv* env, rmm::device_uvector<cudf::size_type> gather_map)
{
  cudf::jni::native_jlongArray result(env, 3);
  result[0]              = static_cast<jlong>(gather_map.size() * sizeof(cudf::size_type));
  auto gather_map_buffer = std::make_unique<rmm::device_buffer>(gather_map.release());
  result[1]              = cudf::jni::ptr_as_jlong(gather_map_buffer->data());
  result[2]              = cudf::jni::release_as_jlong(std::move(gather_map_buffer));
  return result.get_jArray();
}

/**
 * @brief Wrap a device buffer address and length as a device_span (zero-copy)
 * This does not take ownership of the buffer or copy any data
 *
 * @param buffer_address Device buffer address (must be nullptr if and only if buffer_length is 0)
 * @param buffer_length Buffer length in bytes
 * @return device_span providing a view of the buffer (empty if buffer_length is 0)
 */
cudf::device_span<cudf::size_type const> wrap_buffer_as_span(void* buffer_address,
                                                             size_t buffer_length)
{
  size_t num_elements = buffer_length / sizeof(cudf::size_type);
  return cudf::device_span<cudf::size_type const>(
    static_cast<cudf::size_type const*>(buffer_address), num_elements);
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeCreateOuterJoinTracker(JNIEnv* env,
                                                                             jclass,
                                                                             jint num_rows)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto buffer = spark_rapids_jni::create_outer_join_tracker(num_rows);
    cudf::jni::native_jlongArray result(env, 3);
    result[0] = static_cast<jlong>(buffer->size());
    result[1] = cudf::jni::ptr_as_jlong(buffer->data());
    result[2] = cudf::jni::release_as_jlong(buffer);
    return result.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeUpdateOuterJoinTracker(
  JNIEnv* env, jclass, jlong bitmap_address, jlong bitmap_length, jlong gather_view, jint num_rows)
{
  JNI_NULL_CHECK(env, bitmap_address, "bitmap is null", );
  JNI_NULL_CHECK(env, gather_view, "gather indices are null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto bitmap        = cudf::device_span<uint32_t>(reinterpret_cast<uint32_t*>(bitmap_address),
                                              bitmap_length / sizeof(uint32_t));
    auto const indices = reinterpret_cast<cudf::column_view const*>(gather_view);
    spark_rapids_jni::update_outer_join_tracker(bitmap, *indices, num_rows);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeOuterJoinUnmatchedMask(
  JNIEnv* env, jclass, jlong bitmap_address, jlong bitmap_length, jint num_rows)
{
  JNI_NULL_CHECK(env, bitmap_address, "bitmap is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto bitmap = cudf::device_span<uint32_t const>(
      reinterpret_cast<uint32_t const*>(bitmap_address), bitmap_length / sizeof(uint32_t));
    return cudf::jni::release_as_jlong(
      spark_rapids_jni::outer_join_unmatched_mask(bitmap, num_rows));
  }
  JNI_CATCH(env, 0);
}

// =============================================================================
// BASIC EQUALITY JOINS (Sort-Merge and Hash)
// =============================================================================

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeSortMergeInnerJoin(JNIEnv* env,
                                                                         jclass,
                                                                         jlong j_left_keys,
                                                                         jlong j_right_keys,
                                                                         jboolean j_is_left_sorted,
                                                                         jboolean j_is_right_sorted,
                                                                         jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);

    auto const is_left_sorted  = j_is_left_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const is_right_sorted = j_is_right_sorted ? cudf::sorted::YES : cudf::sorted::NO;
    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::sort_merge_inner_join(
      *left_keys, *right_keys, is_left_sorted, is_right_sorted, nulls_equal);

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeHashInnerJoin(
  JNIEnv* env, jclass, jlong j_left_keys, jlong j_right_keys, jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_left_keys, "left keys table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_keys, "right keys table is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_keys  = reinterpret_cast<cudf::table_view const*>(j_left_keys);
    auto const right_keys = reinterpret_cast<cudf::table_view const*>(j_right_keys);

    auto const nulls_equal =
      j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;

    auto result = spark_rapids_jni::hash_inner_join(*left_keys, *right_keys, nulls_equal);

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_DistinctHashJoin_create(
  JNIEnv* env, jclass, jlong j_keys, jboolean j_nulls_equal)
{
  JNI_NULL_CHECK(env, j_keys, "build keys are null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& keys = *reinterpret_cast<cudf::table_view const*>(j_keys);
    auto const nulls = j_nulls_equal ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
    auto hash        = std::make_unique<cudf::distinct_hash_join>(keys, nulls, 0.5);
    return cudf::jni::release_as_jlong(hash);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_DistinctHashJoin_destroy(JNIEnv* env,
                                                                                 jclass,
                                                                                 jlong j_hash)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete reinterpret_cast<cudf::distinct_hash_join*>(j_hash);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_DistinctHashJoin_innerJoin(
  JNIEnv* env, jclass, jlong j_hash, jlong j_probe)
{
  JNI_NULL_CHECK(env, j_hash, "hash join is null", nullptr);
  JNI_NULL_CHECK(env, j_probe, "probe keys are null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& hash   = *reinterpret_cast<cudf::distinct_hash_join const*>(j_hash);
    auto const& probe  = *reinterpret_cast<cudf::table_view const*>(j_probe);
    auto [left, right] = hash.inner_join(probe);
    return gather_maps_to_java(env, {std::move(*left), std::move(*right)});
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_DistinctHashJoin_leftJoin(
  JNIEnv* env, jclass, jlong j_hash, jlong j_probe)
{
  JNI_NULL_CHECK(env, j_hash, "hash join is null", nullptr);
  JNI_NULL_CHECK(env, j_probe, "probe keys are null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const& hash  = *reinterpret_cast<cudf::distinct_hash_join const*>(j_hash);
    auto const& probe = *reinterpret_cast<cudf::table_view const*>(j_probe);
    return gather_single_map_to_java(env, std::move(*hash.left_join(probe)));
  }
  JNI_CATCH(env, nullptr);
}

// =============================================================================
// AST FILTERING
// =============================================================================

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeFilterGatherMapsByAST(
  JNIEnv* env,
  jclass,
  jlong j_left_buffer_address,
  jlong j_left_buffer_length,
  jlong j_right_buffer_address,
  jlong j_right_buffer_length,
  jlong j_left_table,
  jlong j_right_table,
  jlong j_condition)
{
  // Allow null addresses only if length is 0 (empty gather map)
  if (j_left_buffer_address == 0 && j_left_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "left buffer address is null but length is non-zero",
                  nullptr);
  }
  if (j_right_buffer_address == 0 && j_right_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "right buffer address is null but length is non-zero",
                  nullptr);
  }
  // Verify paired gather maps have the same length
  if (j_left_buffer_length != j_right_buffer_length) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "left and right gather maps must have the same length",
                  nullptr);
  }
  JNI_NULL_CHECK(env, j_left_table, "left table is null", nullptr);
  JNI_NULL_CHECK(env, j_right_table, "right table is null", nullptr);
  JNI_NULL_CHECK(env, j_condition, "condition is null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    auto const left_table  = reinterpret_cast<cudf::table_view const*>(j_left_table);
    auto const right_table = reinterpret_cast<cudf::table_view const*>(j_right_table);
    auto const condition   = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(j_condition);

    // Wrap buffer addresses as device_spans (zero-copy, does not take ownership)
    auto left_indices =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_left_buffer_address), j_left_buffer_length);
    auto right_indices =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_right_buffer_address), j_right_buffer_length);

    auto result = spark_rapids_jni::filter_gather_maps_by_ast(
      left_indices, right_indices, *left_table, *right_table, condition->get_top_expression());

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

// =============================================================================
// MAKE OUTER JOINS
// =============================================================================

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeMakeLeftOuter(JNIEnv* env,
                                                                    jclass,
                                                                    jlong j_left_buffer_address,
                                                                    jlong j_left_buffer_length,
                                                                    jlong j_right_buffer_address,
                                                                    jlong j_right_buffer_length,
                                                                    jint j_left_table_size,
                                                                    jint j_right_table_size)
{
  // Allow null addresses only if length is 0 (empty gather map)
  if (j_left_buffer_address == 0 && j_left_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "left buffer address is null but length is non-zero",
                  nullptr);
  }
  if (j_right_buffer_address == 0 && j_right_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "right buffer address is null but length is non-zero",
                  nullptr);
  }
  // Verify paired gather maps have the same length
  if (j_left_buffer_length != j_right_buffer_length) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "left and right gather maps must have the same length",
                  nullptr);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Wrap buffer addresses as device_spans (zero-copy, does not take ownership)
    auto left_indices =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_left_buffer_address), j_left_buffer_length);
    auto right_indices =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_right_buffer_address), j_right_buffer_length);

    auto result = spark_rapids_jni::make_left_outer(
      left_indices, right_indices, j_left_table_size, j_right_table_size);

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeMakeFullOuter(JNIEnv* env,
                                                                    jclass,
                                                                    jlong j_left_buffer_address,
                                                                    jlong j_left_buffer_length,
                                                                    jlong j_right_buffer_address,
                                                                    jlong j_right_buffer_length,
                                                                    jint j_left_table_size,
                                                                    jint j_right_table_size)
{
  // Allow null addresses only if length is 0 (empty gather map)
  if (j_left_buffer_address == 0 && j_left_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "left buffer address is null but length is non-zero",
                  nullptr);
  }
  if (j_right_buffer_address == 0 && j_right_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "right buffer address is null but length is non-zero",
                  nullptr);
  }
  // Verify paired gather maps have the same length
  if (j_left_buffer_length != j_right_buffer_length) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "left and right gather maps must have the same length",
                  nullptr);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Wrap buffer addresses as device_spans (zero-copy, does not take ownership)
    auto left_indices =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_left_buffer_address), j_left_buffer_length);
    auto right_indices =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_right_buffer_address), j_right_buffer_length);

    auto result = spark_rapids_jni::make_full_outer(
      left_indices, right_indices, j_left_table_size, j_right_table_size);

    return gather_maps_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

// =============================================================================
// MAKE SEMI/ANTI JOINS
// =============================================================================

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeMakeSemi(JNIEnv* env,
                                                               jclass,
                                                               jlong j_left_buffer_address,
                                                               jlong j_left_buffer_length,
                                                               jint j_left_table_size)
{
  // Allow null address only if length is 0 (empty gather map)
  if (j_left_buffer_address == 0 && j_left_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "left buffer address is null but length is non-zero",
                  nullptr);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Wrap buffer address as device_span (zero-copy, does not take ownership)
    auto left_indices =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_left_buffer_address), j_left_buffer_length);

    auto result = spark_rapids_jni::make_semi(left_indices, j_left_table_size);

    return gather_single_map_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeMakeAnti(JNIEnv* env,
                                                               jclass,
                                                               jlong j_left_buffer_address,
                                                               jlong j_left_buffer_length,
                                                               jint j_left_table_size)
{
  // Allow null address only if length is 0 (empty gather map)
  if (j_left_buffer_address == 0 && j_left_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "left buffer address is null but length is non-zero",
                  nullptr);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Wrap buffer address as device_span (zero-copy, does not take ownership)
    auto left_indices =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_left_buffer_address), j_left_buffer_length);

    auto result = spark_rapids_jni::make_anti(left_indices, j_left_table_size);

    return gather_single_map_to_java(env, std::move(result));
  }
  JNI_CATCH(env, nullptr);
}

// =============================================================================
// PARTITIONED JOIN SUPPORT
// =============================================================================

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_JoinPrimitives_nativeGetMatchedRows(
  JNIEnv* env, jclass, jlong j_buffer_address, jlong j_buffer_length, jint j_table_size)
{
  // Allow null address only if length is 0 (empty gather map)
  if (j_buffer_address == 0 && j_buffer_length != 0) {
    JNI_THROW_NEW(env,
                  cudf::jni::ILLEGAL_ARG_EXCEPTION_CLASS,
                  "buffer address is null but length is non-zero",
                  0);
  }

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);

    // Wrap buffer address as device_span (zero-copy, does not take ownership)
    auto gather_map =
      wrap_buffer_as_span(reinterpret_cast<void*>(j_buffer_address), j_buffer_length);

    auto result = spark_rapids_jni::get_matched_rows(gather_map, j_table_size);

    return cudf::jni::release_as_jlong(result);
  }
  JNI_CATCH(env, 0);
}

}  // extern "C"
