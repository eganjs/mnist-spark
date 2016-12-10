package com.gmail.egan.s.joseph.mnist

import java.io.File

class MNISTDataset(labelsFile: String, imagesFile: String) {
  def read(max: Int = Int.MaxValue): MNISTData = MNISTDataset.read(labelsFile, imagesFile, max)
}

object MNISTDataset {

  import com.gmail.egan.s.joseph.idx.IDXFileReadingUtils._

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

  private def verifyNumberOfElements(mnistFile1: FileBytesIterator, mnistFile2: FileBytesIterator): Unit = {
    val mnistFile1Elements = 0xFFFF & popBigEndianInt(mnistFile1)
    val mnistFile2Elements = 0xFFFF & popBigEndianInt(mnistFile2)
    if (mnistFile1Elements != mnistFile2Elements)
      throw new InvalidMNISTPairException("number of labels and images differ")
  }

  private def read(in: (String, String, Int)): MNISTData = {
    val (labelsFilePath, imagesFilePath, max) = in

    val labelsFile: FileBytesIterator = readFileBytes(resourceDirectory + labelsFilePath)
    val imagesFile: FileBytesIterator = readFileBytes(resourceDirectory + imagesFilePath)

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