package com.gmail.egan.s.joseph.idx

import scala.io.{Codec, Source}

object IDXFileReadingUtils {
  type FileBytesIterator = Iterator[Byte]

  def readFileBytes(localFilePath: String): FileBytesIterator = {
    Source.fromFile(localFilePath)(Codec.ISO8859).map(_.toByte)
  }

  def popBigEndianInt(bytes: FileBytesIterator): Int = {
    bytes.take(4).grouped(4).map(e => e.head << 24 | e(1) << 16 | e(2) << 8 | e(3)).next()
  }

  def verifyFileType(bytes: FileBytesIterator, expectedType: Int): Unit = {
    val actualType = popBigEndianInt(bytes)
    if (actualType != expectedType)
      throw new UnknownFileTypeException("expected type " + expectedType + " but found " + actualType)
  }
}
