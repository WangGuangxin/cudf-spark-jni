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
#include "cudf_jni_apis.hpp"
#include "partition_pack.hpp"

extern "C" {
JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_PartitionPack_isSupported(
  JNIEnv* env, jclass, jlong table_view)
{
  JNI_NULL_CHECK(env, table_view, "table is null", false);
  JNI_TRY
  {
    return spark_rapids_jni::supports_partition_pack(
      *reinterpret_cast<cudf::table_view const*>(table_view));
  }
  JNI_CATCH(env, false);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_PartitionPack_partitionAndPack(
  JNIEnv* env, jclass, jlong table_view, jlong partition_ids, jobjectArray output)
{
  JNI_NULL_CHECK(env, table_view, "table is null", );
  JNI_NULL_CHECK(env, partition_ids, "partition IDs are null", );
  JNI_NULL_CHECK(env, output, "output is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto [packed, boundaries] = spark_rapids_jni::partition_pack(
      *reinterpret_cast<cudf::table_view const*>(table_view),
      *reinterpret_cast<cudf::column_view const*>(partition_ids),
      env->GetArrayLength(output), cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());
    for (size_t p = 0; p < packed.size(); ++p) {
      // Java closes any already transferred tables if a later factory call fails.
      auto table = cudf::jni::contiguous_table_from(env, packed[p], boundaries[p + 1] - boundaries[p]);
      if (env->ExceptionCheck() || table == nullptr) { return; }
      env->SetObjectArrayElement(output, p, table);
      env->DeleteLocalRef(table);
      if (env->ExceptionCheck()) { return; }
    }
  }
  JNI_CATCH(env, );
}
JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_PartitionPack_partialAggregateAndPack(
  JNIEnv* env, jclass, jlong table_view, jintArray operations, jobjectArray output)
{
  JNI_NULL_CHECK(env, table_view, "table is null", );
  JNI_NULL_CHECK(env, operations, "operations are null", );
  JNI_NULL_CHECK(env, output, "output is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    std::vector<int> ops(env->GetArrayLength(operations));
    env->GetIntArrayRegion(operations, 0, ops.size(), ops.data());
    if (env->ExceptionCheck()) { return; }
    auto [packed, boundaries] = spark_rapids_jni::partial_aggregate_pack(
      *reinterpret_cast<cudf::table_view const*>(table_view), ops, env->GetArrayLength(output),
      cudf::get_default_stream(), cudf::get_current_device_resource_ref());
    for (size_t p = 0; p < packed.size(); ++p) {
      auto table = cudf::jni::contiguous_table_from(env, packed[p], boundaries[p + 1] - boundaries[p]);
      if (env->ExceptionCheck() || table == nullptr) { return; }
      env->SetObjectArrayElement(output, p, table);
      env->DeleteLocalRef(table);
      if (env->ExceptionCheck()) { return; }
    }
  }
  JNI_CATCH(env, );
}
}  // extern "C"
