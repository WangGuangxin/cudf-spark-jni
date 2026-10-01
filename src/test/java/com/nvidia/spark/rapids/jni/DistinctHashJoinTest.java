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
import ai.rapids.cudf.ColumnView;
import ai.rapids.cudf.GatherMap;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class DistinctHashJoinTest {
  private static List<String> pairs(GatherMap[] maps) {
    try (GatherMap left = maps[0]; GatherMap right = maps[1];
         ColumnView lv = left.toColumnView(0, (int) left.getRowCount());
         ColumnView rv = right.toColumnView(0, (int) right.getRowCount());
         ColumnVector lc = lv.copyToColumnVector(); ColumnVector rc = rv.copyToColumnVector();
         HostColumnVector lh = lc.copyToHost(); HostColumnVector rh = rc.copyToHost()) {
      List<String> result = new ArrayList<>();
      for (int row = 0; row < left.getRowCount(); row++) {
        result.add(lh.getInt(row) + ":" + rh.getInt(row));
      }
      Collections.sort(result);
      return result;
    }
  }

  @Test
  void reuseAcrossProbesAndOwnBuildReferences() {
    DistinctHashJoin hash;
    try (ColumnVector col = ColumnVector.fromInts(10, 20, 30); Table keys = new Table(col)) {
      hash = new DistinctHashJoin(keys, false);
    }
    try (DistinctHashJoin owned = hash;
         ColumnVector col = ColumnVector.fromInts(20, 40, 10, 20); Table probe = new Table(col)) {
      List<String> expected = java.util.Arrays.asList("0:1", "2:0", "3:1");
      assertEquals(expected, pairs(owned.innerJoin(probe)));
      assertEquals(expected, pairs(owned.innerJoin(probe)));
      try (GatherMap map = owned.leftJoin(probe);
           ColumnView view = map.toColumnView(0, 4);
           ColumnVector copy = view.copyToColumnVector(); HostColumnVector host = copy.copyToHost()) {
        assertEquals(1, host.getInt(0));
        assertTrue(host.getInt(1) < 0);
        assertEquals(0, host.getInt(2));
        assertEquals(1, host.getInt(3));
      }
    }
    assertThrows(IllegalStateException.class, () -> hash.innerJoin(null));
  }

  @Test
  void nullEqualityAndMultipleKeys() {
    for (boolean equal : new boolean[]{false, true}) {
      try (ColumnVector a = ColumnVector.fromBoxedInts(1, null, 1);
           ColumnVector b = ColumnVector.fromInts(10, 20, 30); Table keys = new Table(a, b);
           DistinctHashJoin hash = new DistinctHashJoin(keys, equal);
           ColumnVector pa = ColumnVector.fromBoxedInts(null, 1, 1);
           ColumnVector pb = ColumnVector.fromInts(20, 30, 99); Table probe = new Table(pa, pb)) {
        List<String> expected = equal ? java.util.Arrays.asList("0:1", "1:2") :
            java.util.Collections.singletonList("1:2");
        assertEquals(expected, pairs(hash.innerJoin(probe)));
        assertEquals(expected, pairs(hash.innerJoin(probe)));
      }
    }
  }

  @Test
  void emptyBuildAndProbe() {
    try (ColumnVector empty = ColumnVector.fromInts(); Table keys = new Table(empty);
         DistinctHashJoin hash = new DistinctHashJoin(keys, false);
         ColumnVector nonempty = ColumnVector.fromInts(1, 2); Table probe = new Table(nonempty)) {
      assertTrue(pairs(hash.innerJoin(probe)).isEmpty());
      try (GatherMap map = hash.leftJoin(probe)) {
        assertEquals(2, map.getRowCount());
      }
    }
    try (ColumnVector build = ColumnVector.fromInts(1, 2); Table keys = new Table(build);
         DistinctHashJoin hash = new DistinctHashJoin(keys, false);
         ColumnVector empty = ColumnVector.fromInts(); Table probe = new Table(empty)) {
      assertTrue(pairs(hash.innerJoin(probe)).isEmpty());
    }
  }
}
