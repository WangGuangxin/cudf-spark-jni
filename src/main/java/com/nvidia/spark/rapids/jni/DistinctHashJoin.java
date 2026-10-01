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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/** A reusable hash table for unique build keys. Owns references to its build columns. */
public final class DistinctHashJoin implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private Table buildKeys;
  private long handle;

  public DistinctHashJoin(Table keys, boolean compareNullsEqual) {
    ColumnVector[] columns = new ColumnVector[(int) keys.getNumberOfColumns()];
    for (int i = 0; i < columns.length; i++) {
      columns[i] = keys.getColumn(i);
    }
    Table owned = new Table(columns);
    try {
      handle = create(owned.getNativeView(), compareNullsEqual);
      buildKeys = owned;
    } catch (Throwable t) {
      owned.close();
      throw t;
    }
  }

  private void checkOpen() {
    if (handle == 0) {
      throw new IllegalStateException("distinct hash join is closed");
    }
  }

  public GatherMap[] innerJoin(Table probeKeys) {
    checkOpen();
    return JoinPrimitives.gatherMapsFromJNI(innerJoin(handle, probeKeys.getNativeView()));
  }

  public GatherMap leftJoin(Table probeKeys) {
    checkOpen();
    long[] data = leftJoin(handle, probeKeys.getNativeView());
    return new GatherMap(DeviceMemoryBuffer.fromRmm(data[1], data[0], data[2]));
  }

  @Override
  public void close() {
    if (handle != 0) {
      try {
        destroy(handle);
      } finally {
        handle = 0;
        buildKeys.close();
        buildKeys = null;
      }
    }
  }

  private static native long create(long keys, boolean compareNullsEqual);
  private static native void destroy(long handle);
  private static native long[] innerJoin(long handle, long probeKeys);
  private static native long[] leftJoin(long handle, long probeKeys);
}
