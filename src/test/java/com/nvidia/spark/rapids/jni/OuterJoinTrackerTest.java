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

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostColumnVector;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

public class OuterJoinTrackerTest {
  private static void assertUnmatched(DeviceMemoryBuffer bitmap, int rows, int... matched) {
    Set<Integer> hits = new HashSet<>();
    for (int index : matched) {
      hits.add(index);
    }
    try (ColumnVector mask = JoinPrimitives.outerJoinUnmatchedMask(bitmap, rows);
         HostColumnVector host = mask.copyToHost()) {
      assertEquals(rows, host.getRowCount());
      for (int row = 0; row < rows; row++) {
        assertEquals(!hits.contains(row), host.getBoolean(row), "row " + row);
      }
    }
  }

  @Test
  void emptyBuildAndGatherMap() {
    try (DeviceMemoryBuffer bitmap = JoinPrimitives.createOuterJoinTracker(0);
         ColumnVector indices = ColumnVector.fromInts()) {
      JoinPrimitives.updateOuterJoinTracker(bitmap, indices, 0);
      assertUnmatched(bitmap, 0);
    }
  }

  @Test
  void wordBoundariesAndInvalidIndices() {
    try (DeviceMemoryBuffer bitmap = JoinPrimitives.createOuterJoinTracker(65);
         ColumnVector indices = ColumnVector.fromInts(
             Integer.MIN_VALUE, -1, 0, 31, 32, 63, 64, 65, Integer.MAX_VALUE)) {
      assertEquals(12, bitmap.getLength());
      assertUnmatched(bitmap, 65);
      JoinPrimitives.updateOuterJoinTracker(bitmap, indices, 65);
      assertUnmatched(bitmap, 65, 0, 31, 32, 63, 64);
    }
  }

  @Test
  void incrementalUpdatesAreIdempotent() {
    try (DeviceMemoryBuffer bitmap = JoinPrimitives.createOuterJoinTracker(33);
         ColumnVector first = ColumnVector.fromInts(0, 1, 31, 31);
         ColumnVector second = ColumnVector.fromInts(1, 32)) {
      JoinPrimitives.updateOuterJoinTracker(bitmap, first, 33);
      JoinPrimitives.updateOuterJoinTracker(bitmap, first, 33);
      assertUnmatched(bitmap, 33, 0, 1, 31);
      JoinPrimitives.updateOuterJoinTracker(bitmap, second, 33);
      assertUnmatched(bitmap, 33, 0, 1, 31, 32);
    }
  }

  @Test
  void concurrentBitsInTheSameWord() {
    int[] indices = new int[100000];
    for (int i = 0; i < indices.length; i++) {
      indices[i] = i % 32;
    }
    try (DeviceMemoryBuffer bitmap = JoinPrimitives.createOuterJoinTracker(33);
         ColumnVector gather = ColumnVector.fromInts(indices);
         ColumnVector maskBefore = JoinPrimitives.outerJoinUnmatchedMask(bitmap, 33)) {
      JoinPrimitives.updateOuterJoinTracker(bitmap, gather, 33);
      try (ColumnVector mask = JoinPrimitives.outerJoinUnmatchedMask(bitmap, 33);
           HostColumnVector host = mask.copyToHost();
           HostColumnVector before = maskBefore.copyToHost()) {
        for (int i = 0; i < 32; i++) {
          assertFalse(host.getBoolean(i));
          assertTrue(before.getBoolean(i));
        }
        assertTrue(host.getBoolean(32));
      }
    }
  }

  @Test
  void rejectsInvalidInputs() {
    assertThrows(IllegalArgumentException.class, () -> JoinPrimitives.createOuterJoinTracker(-1));
    try (DeviceMemoryBuffer bitmap = JoinPrimitives.createOuterJoinTracker(1);
         ColumnVector wrongType = ColumnVector.fromLongs(0L);
         ColumnVector nullable = ColumnVector.fromBoxedInts((Integer) null)) {
      assertThrows(IllegalArgumentException.class,
          () -> JoinPrimitives.updateOuterJoinTracker(bitmap, wrongType, 1));
      assertThrows(IllegalArgumentException.class,
          () -> JoinPrimitives.updateOuterJoinTracker(bitmap, nullable, 1));
      assertThrows(IllegalArgumentException.class,
          () -> JoinPrimitives.outerJoinUnmatchedMask(bitmap, 33));
    }
  }
}
