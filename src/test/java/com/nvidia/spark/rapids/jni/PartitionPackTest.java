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

import ai.rapids.cudf.CloseableArray;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.ContiguousTable;
import ai.rapids.cudf.CudfTestBase;
import ai.rapids.cudf.OrderByArg;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.math.BigInteger;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PartitionPackTest extends CudfTestBase {
  private static Table input(int rows, boolean nullable, boolean allNullStrings) {
    int[] ids = new int[rows];
    String[] strings = new String[rows];
    String[] emptyStrings = new String[rows];
    Long[] decimals = new Long[rows];
    BigInteger[] wideDecimals = new BigInteger[rows];
    BigInteger base = new BigInteger("1234567890123456789012345");
    for (int i = 0; i < rows; i++) {
      ids[i] = i;
      strings[i] = allNullStrings || (nullable && i % 7 == 0) ? null
          : i % 5 == 0 ? "" : "value-\u4e2d\u6587-" + i;
      emptyStrings[i] = "";
      decimals[i] = nullable && i % 11 == 0 ? null : (long) i - rows / 2;
      wideDecimals[i] = nullable && i % 13 == 0 ? null : base.add(BigInteger.valueOf(i));
    }
    try (ColumnVector rowIds = ColumnVector.fromInts(ids);
         ColumnVector text = ColumnVector.fromStrings(strings);
         ColumnVector empty = ColumnVector.fromStrings(emptyStrings);
         ColumnVector decimal = ColumnVector.decimalFromBoxedLongs(-3, decimals);
         ColumnVector wide = ColumnVector.decimalFromBigInt(-5, wideDecimals)) {
      return new Table(rowIds, text, empty, decimal, wide);
    }
  }

  private static int[] partitionIds(int rows, int parts, int constant) {
    int[] result = new int[rows];
    for (int i = 0; i < rows; i++) {
      result[i] = constant >= 0 ? constant : (i * 37 + 3) % parts;
    }
    return result;
  }

  private static void checkPartitions(Table input, ColumnVector ids, int parts,
      CloseableArray<ContiguousTable> packed) {
    assertEquals(parts, packed.size());
    long total = 0;
    for (int p = 0; p < parts; p++) {
      // The reference filters the original payload independently of permutation generation.
      try (Scalar value = Scalar.fromInt(p);
           ColumnVector mask = ids.equalTo(value);
           Table expected = input.filter(mask);
           Table expectedSorted = expected.orderBy(OrderByArg.asc(0));
           Table actualSorted = packed.get(p).getTable().orderBy(OrderByArg.asc(0))) {
        assertEquals(expected.getRowCount(), packed.get(p).getRowCount());
        assertTablesAreEqual(expectedSorted, actualSorted);
        total += packed.get(p).getRowCount();
      }
    }
    assertEquals(input.getRowCount(), total);
  }

  private static void check(int rows, int parts, int constant,
      boolean nullable, boolean allNullStrings) {
    try (Table source = input(rows, nullable, allNullStrings);
         ColumnVector ids = ColumnVector.fromInts(partitionIds(rows, parts, constant));
         CloseableArray<ContiguousTable> packed = CloseableArray.wrap(
             PartitionPack.partitionAndPack(source, ids, parts))) {
      assertTrue(PartitionPack.isSupported(source));
      checkPartitions(source, ids, parts, packed);
    }
  }

  @Test
  void mixedNullsStringsAndDecimalsAcrossMaskAndBlockBoundaries() {
    for (int rows : new int[]{1, 31, 32, 33, 257, 4096}) {
      for (int parts : new int[]{1, 3, 8, 17}) {
        check(rows, parts, -1, true, false);
      }
    }
  }

  @Test
  void skewedInputsPreserveEmptyPartitions() {
    check(513, 17, 0, true, false);
    check(513, 17, 16, true, false);
  }

  @Test
  void nonNullablePayloadsAndAllNullOrEmptyStrings() {
    check(4096, 8, -1, false, false);
    check(4096, 8, -1, true, true);
  }

  @Test
  void largeInputUsesEverySourceRowExactlyOnce() {
    check(65537, 17, -1, true, false);
  }

  @Test
  void packedOutputSurvivesSourceClose() {
    int rows = 257;
    int parts = 8;
    ContiguousTable[] output;
    try (Table source = input(rows, true, false);
         ColumnVector ids = ColumnVector.fromInts(partitionIds(rows, parts, -1))) {
      output = PartitionPack.partitionAndPack(source, ids, parts);
    }
    try (CloseableArray<ContiguousTable> packed = CloseableArray.wrap(output);
         Table expected = input(rows, true, false);
         ColumnVector ids = ColumnVector.fromInts(partitionIds(rows, parts, -1))) {
      checkPartitions(expected, ids, parts, packed);
    }
  }
}
