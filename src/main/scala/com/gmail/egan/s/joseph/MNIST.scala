package com.gmail.egan.s.joseph

import java.io.{File, IOException}

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.{Codec, Source}

class UnknownFileTypeException(message: String) extends IOException(message)

class InvalidMNISTPairException(message: String) extends IOException(message)

class MNISTImage(val label: Byte, val data: Array[Byte])

class MNISTData(val images: Iterable[MNISTImage], val imageHeight: Int, val imageWidth: Int) {
  def imagesToDF: DataFrame = {
    val ss = SparkSession.builder().getOrCreate()
    import MNISTData.DF.Columns
    import ss.implicits._
    images
      .map { image =>
        (
          image.label.toString,
          Vectors.dense(image.data.map { b =>
            // Normalise pixel data
            (b & 0xFF).toDouble / 255
          })
        )
      }
      .toSeq
      .toDF(
        Columns.Label,
        Columns.Features
      )
  }
}

// If any values in MNISTData are changed then the model will need to be retrained
object MNISTData {

  object DF {
    val LabelIndices: Seq[String] = (0 to 9).map(_.toString)

    object Columns {
      val Label = "label"
      val Features = "features"
      val LabelIndex = "label index"
      val PrincipalFeatures = "principal features"
      val LabelIndexPrediction = "label index prediction"
      val LabelPrediction = "label prediction"
    }

  }

}

class MNISTDataset(labelsFile: String, imagesFile: String) {
  def read(max: Int = Int.MaxValue): MNISTData = MNISTDataset.read(labelsFile, imagesFile, max)
}

object MNISTDataset {
  private val LabelsFileType = 2049
  private val ImagesFileType = 2051

  val TrainingData: MNISTDataset = new MNISTDataset("train-labels-idx1-ubyte", "train-images-idx3-ubyte")
  val TestData: MNISTDataset = new MNISTDataset("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte")

  private var resourceDirectory = ""

  def setResourcesDirectory(resourceDirectory: String): this.type = {
    this.resourceDirectory = {
      if (resourceDirectory.last != File.separatorChar)
        resourceDirectory + File.separatorChar
      else
        resourceDirectory
    }
    this
  }

  private type FileBytesIterator = Iterator[Byte]

  private def readFileBytes(localFilePath: String): FileBytesIterator = {
    Source.fromFile(resourceDirectory + localFilePath)(Codec.ISO8859).map(_.toByte)
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

  private def read(labelsFilePath: String, imagesFilePath: String, max: Int): MNISTData = {
    val labelsFile: FileBytesIterator = readFileBytes(labelsFilePath)
    val imagesFile: FileBytesIterator = readFileBytes(imagesFilePath)

    verifyFileType(labelsFile, LabelsFileType)
    verifyFileType(imagesFile, ImagesFileType)
    verifyNumberOfElements(labelsFile, imagesFile)

    val labels: Iterator[Byte] = labelsFile.take(max)

    val imageHeight = popBigEndianInt(imagesFile)
    val imageWidth = popBigEndianInt(imagesFile)
    val images: Iterator[Array[Byte]] = imagesFile.grouped(imageHeight * imageWidth).take(max).map(_.toArray)

    val labelledImages = labels.zip(images).map { case (label: Byte, image: Array[Byte]) =>
      new MNISTImage(label, image)
    }.toIterable

    new MNISTData(labelledImages, imageHeight, imageWidth)
  }
}
