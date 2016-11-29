package com.gmail.egan.s.joseph

import java.io.IOException

import scala.io.{Codec, Source}

class UnknownFileTypeException(message: String = "unknown file type") extends IOException(message)

class InvalidMNISTPairException(message: String = "invalid labels and images file pairing") extends IOException(message)

class MNISTImage(val label: Byte, val data: Array[Byte])

class MNISTData(val images: Iterable[MNISTImage], val imageHeight: Int, val imageWidth: Int)

class MNISTDataset(labelsFilePath: String, imagesFilePath: String) {
  def read(): MNISTData = MNISTDataset.read(labelsFilePath, imagesFilePath)
}

object MNISTDataset {
  private val LabelsFileType = 2049
  private val ImagesFileType = 2051

  private val baseDir = "resources/"

  val TrainingData: MNISTDataset = new MNISTDataset("train-labels-idx1-ubyte", "train-images-idx3-ubyte")
  val TestData: MNISTDataset = new MNISTDataset("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte")

  private type FileBytesIterator = Iterator[Byte]

  private def readFile(localFilePath: String): FileBytesIterator = {
    Source.fromFile(baseDir + localFilePath)(Codec.ISO8859).map(_.toByte)
  }

  private def popBigEndianInt(bytes: FileBytesIterator): Int = {
    bytes.take(4).grouped(4).map(e => e.head << 24 | e(1) << 16 | e(2) << 8 | e(3)).next()
  }

  private def verifyFileType(bytes: FileBytesIterator, expectedType: Int): Unit = {
    val actualType = popBigEndianInt(bytes)
    if (actualType != expectedType)
      throw new UnknownFileTypeException("expected type " + expectedType + " but found " + actualType)
  }

  private def verifyNumberOfElements(mnistFile1: FileBytesIterator, mnistFile2: FileBytesIterator): Unit = {
    val mnistFile1Elements = 0xFFFF & popBigEndianInt(mnistFile1)
    val mnistFile2Elements = 0xFFFF & popBigEndianInt(mnistFile2)
    if (mnistFile1Elements != mnistFile2Elements)
      throw new InvalidMNISTPairException("number of labels and images differ")
  }

  private def read(labelsFilePath: String, imagesFilePath: String): MNISTData = {
    val labelsFile: FileBytesIterator = readFile(labelsFilePath)
    val imagesFile: FileBytesIterator = readFile(imagesFilePath)

    verifyFileType(labelsFile, LabelsFileType)
    verifyFileType(imagesFile, ImagesFileType)
    verifyNumberOfElements(labelsFile, imagesFile)

    val labels: Iterator[Byte] = labelsFile

    val imageHeight = popBigEndianInt(imagesFile)
    val imageWidth = popBigEndianInt(imagesFile)
    val images: Iterator[Array[Byte]] = imagesFile.grouped(imageHeight * imageWidth).map(_.toArray)

    val labelledImages = labels.zip(images).map { case (label: Byte, image: Array[Byte]) =>
      new MNISTImage(label, image)
    }.toIterable

    new MNISTData(labelledImages, imageHeight, imageWidth)
  }
}
