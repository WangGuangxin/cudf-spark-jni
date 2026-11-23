/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.jni.kudo;

import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.KUDO_SANITY_CHECK;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForHostAlignment;
import static com.nvidia.spark.rapids.jni.kudo.KudoSerializer.padForValidityAlignment;

import ai.rapids.cudf.BufferType;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.HostColumnVectorCore;
import ai.rapids.cudf.HostMemoryBuffer;
import com.nvidia.spark.rapids.jni.schema.HostColumnsVisitor;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * This class visits a list of columns and serialize one of the buffers (validity, offset, or data) into with kudo
 * format.
 *
 * <p>
 * The host columns are visited in post order, for more details about the visiting process, please refer to
 * {@link HostColumnsVisitor}.
 * </p>
 *
 * <p>
 * For more details about the kudo format, please refer to {@link KudoSerializer}.
 * </p>
 */
class SlicedBufferSerializer implements HostColumnsVisitor {
  // Number of string offsets rebased per chunk, and a per-thread buffer for one chunk.
  private static final int REBASE_CHUNK_INTS = 4096;
  private static final ThreadLocal<byte[]> REBASE_BUFFER =
      ThreadLocal.withInitial(() -> new byte[REBASE_CHUNK_INTS * Integer.BYTES]);

  private final SliceInfo root;
  private final BufferType bufferType;
  private final DataWriter writer;

  private final Deque<SliceInfo> sliceInfos = new ArrayDeque<>();
  private final WriteMetrics metrics;
  private final boolean addCopyBufferTime;
  private long totalDataLen;
  private long headerSize;

  SlicedBufferSerializer(int rowOffset, int numRows, BufferType bufferType, DataWriter writer,
                         WriteMetrics metrics, boolean addCopyBufferTime, long headerSize) {
    this.root = new SliceInfo(rowOffset, numRows);
    this.bufferType = bufferType;
    this.writer = writer;
    this.sliceInfos.addLast(root);
    this.metrics = metrics;
    this.totalDataLen = 0;
    this.addCopyBufferTime = addCopyBufferTime;
    this.headerSize = headerSize;
  }

  public long getTotalDataLen() {
    return totalDataLen;
  }

  @Override
  public void preVisitStruct(HostColumnVectorCore col) {
    SliceInfo parent = sliceInfos.peekLast();

    try {
      switch (bufferType) {
        case VALIDITY:
          totalDataLen += this.copySlicedValidity(col, parent);
          return;
        case OFFSET:
        case DATA:
          return;
        default:
          throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
      }

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void visitStruct(HostColumnVectorCore col) {
    // NOOP
  }

  @Override
  public void preVisitList(HostColumnVectorCore col) {
    SliceInfo parent = sliceInfos.getLast();


    long bytesCopied = 0;
    try {
      switch (bufferType) {
        case VALIDITY:
          bytesCopied = this.copySlicedValidity(col, parent);
          break;
        case OFFSET:
          bytesCopied = this.copySlicedOffset(col, parent);
          break;
        case DATA:
          break;
        default:
          throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
      }

    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    SliceInfo current;
    if (col.getOffsets() != null) {
      int start = col.getOffsets()
          .getInt(parent.offset * Integer.BYTES);
      int end = col.getOffsets().getInt((parent.offset + parent.rowCount) * Integer.BYTES);
      int rowCount = end - start;

      current = new SliceInfo(start, rowCount);
    } else {
      current = new SliceInfo(0, 0);
    }

    sliceInfos.addLast(current);

    totalDataLen += bytesCopied;
  }

  @Override
  public void visitList(HostColumnVectorCore col) {
    sliceInfos.removeLast();
  }

  @Override
  public void visit(HostColumnVectorCore col) {
    SliceInfo parent = sliceInfos.getLast();
    try {
      switch (bufferType) {
        case VALIDITY:
          totalDataLen += this.copySlicedValidity(col, parent);
          return;
        case OFFSET:
          totalDataLen += this.copySlicedOffset(col, parent);
          return;
        case DATA:
          totalDataLen += this.copySlicedData(col, parent);
          return;
        default:
          throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void done() {
    try {
      switch (bufferType) {
        case VALIDITY:
          totalDataLen = padForValidityAlignment(writer, totalDataLen, headerSize);
          return;
        case OFFSET:
          // fallthrough
        case DATA:
          totalDataLen = padForHostAlignment(writer, totalDataLen);
          return;
        default:
          throw new IllegalArgumentException("Unexpected buffer type: " + bufferType);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private long copySlicedValidity(HostColumnVectorCore column, SliceInfo sliceInfo)
      throws IOException {
    if (column.getValidity() != null && sliceInfo.getRowCount() > 0) {
      HostMemoryBuffer buff = column.getValidity();
      long len = sliceInfo.getValidityBufferInfo().getBufferLength();
      return copyBuffer(buff, sliceInfo.getValidityBufferInfo().getBufferOffset(), len);
    } else {
      return 0;
    }
  }

  private long copySlicedOffset(HostColumnVectorCore column, SliceInfo sliceInfo)
      throws IOException {
    if (sliceInfo.rowCount <= 0 || column.getOffsets() == null) {
      // Don't copy anything, there are no rows
      return 0;
    }
    long bytesToCopy = (sliceInfo.rowCount + 1) * Integer.BYTES;
    long srcOffset = sliceInfo.offset * Integer.BYTES;

    if (KUDO_SANITY_CHECK) {
      int startOffset = column.getOffsets().getInt(srcOffset);
      int endOffset = column.getOffsets()
          .getInt((sliceInfo.offset + sliceInfo.rowCount) * Integer.BYTES);

      if (startOffset < 0 || endOffset < startOffset) {
        throw new IllegalArgumentException("Invalid kudo offset: " + startOffset + ", " + endOffset);
      }
    }

    if (column.getType().equals(DType.STRING)) {
      return copyRebasedOffsets(column.getOffsets(), srcOffset, sliceInfo.rowCount + 1);
    } else {
      return copyBuffer(column.getOffsets(), srcOffset, bytesToCopy);
    }
  }

  /**
   * Writes `count` string offsets starting at `srcOffset`, minus the first of them, so the
   * written offsets start at 0. Readers only use differences between a string column's offsets,
   * and small offsets compress much better than positions in a large column. List offsets are
   * copied as is because readers use the first one to locate the child rows.
   */
  private long copyRebasedOffsets(HostMemoryBuffer offsets, long srcOffset, int count)
      throws IOException {
    int first = offsets.getInt(srcOffset);
    long length = (long) count * Integer.BYTES;
    if (first == 0) {
      return copyBuffer(offsets, srcOffset, length);
    }
    long start = addCopyBufferTime ? System.nanoTime() : 0;
    byte[] buf = REBASE_BUFFER.get();
    long done = 0;
    while (done < count) {
      int n = (int) Math.min(REBASE_CHUNK_INTS, count - done);
      offsets.getBytes(buf, 0, srcOffset + done * Integer.BYTES, (long) n * Integer.BYTES);
      for (int i = 0; i < n * Integer.BYTES; i += Integer.BYTES) {
        // Offsets are little endian in host memory and in the kudo body.
        int v = ((buf[i] & 0xFF) | ((buf[i + 1] & 0xFF) << 8) | ((buf[i + 2] & 0xFF) << 16) |
            ((buf[i + 3] & 0xFF) << 24)) - first;
        buf[i] = (byte) v;
        buf[i + 1] = (byte) (v >>> 8);
        buf[i + 2] = (byte) (v >>> 16);
        buf[i + 3] = (byte) (v >>> 24);
      }
      writer.write(buf, 0, n * Integer.BYTES);
      done += n;
    }
    if (addCopyBufferTime) {
      metrics.addCopyBufferTime(System.nanoTime() - start);
    }
    return length;
  }

  private long copySlicedData(HostColumnVectorCore column, SliceInfo sliceInfo) throws IOException {
    if (sliceInfo.rowCount > 0) {
      DType type = column.getType();
      if (type.equals(DType.STRING)) {
        long startByteOffset = column.getOffsets().getInt(sliceInfo.offset * Integer.BYTES);
        long endByteOffset =
            column.getOffsets().getInt((sliceInfo.offset + sliceInfo.rowCount) * Integer.BYTES);
        long bytesToCopy = endByteOffset - startByteOffset;
        if (column.getData() == null) {
          if (bytesToCopy != 0) {
            throw new IllegalStateException("String column has no data buffer, " +
                "but bytes to copy is not zero: " + bytesToCopy);
          }

          return 0;
        } else {
          return copyBuffer(column.getData(), startByteOffset, bytesToCopy);
        }
      } else if (type.getSizeInBytes() > 0) {
        long bytesToCopy = sliceInfo.rowCount * type.getSizeInBytes();
        long srcOffset = sliceInfo.offset * type.getSizeInBytes();
        return copyBuffer(column.getData(), srcOffset, bytesToCopy);
      } else {
        return 0;
      }
    } else {
      return 0;
    }
  }

  private long copyBuffer(HostMemoryBuffer buffer, long offset, long length)
      throws IOException {
    if (addCopyBufferTime) {
      long now = System.nanoTime();
      writer.copyDataFrom(buffer, offset, length);
      metrics.addCopyBufferTime(System.nanoTime() - now);
      return length;
    } else {
      writer.copyDataFrom(buffer, offset, length);
      return length;
    }
  }
}
