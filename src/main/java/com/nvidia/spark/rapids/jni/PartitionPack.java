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
package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.ContiguousTable;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;

/** Direct gathering of flat shuffle payloads into cuDF packed-table buffers. */
public final class PartitionPack {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private PartitionPack() {}

  public static boolean isSupported(Table table) {
    return isSupported(table.getNativeView());
  }

  /**
   * The caller owns the returned tables; inputs are borrowed. Partition IDs must be
   * non-null INT32 values in [0, numPartitions), as produced by Spark's hash/pmod.
   * Row order within a partition is unspecified. Zero-row inputs use the Spark empty path.
   */
  public static ContiguousTable[] partitionAndPack(
      Table table, ColumnView partitionIds, int numPartitions) {
    if (numPartitions <= 0) {
      throw new IllegalArgumentException("Partition count must be positive");
    }
    ContiguousTable[] result = new ContiguousTable[numPartitions];
    boolean success = false;
    try {
      partitionAndPack(table.getNativeView(), partitionIds.getNativeView(), result);
      success = true;
      return result;
    } finally {
      if (!success) {
        for (ContiguousTable partition : result) {
          if (partition != null) {
            partition.close();
          }
        }
      }
    }
  }

  private static native boolean isSupported(long tableView);
  private static native void partitionAndPack(
      long tableView, long partitionIds, ContiguousTable[] result);
}
