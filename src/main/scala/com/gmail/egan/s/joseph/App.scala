package com.gmail.egan.s.joseph

object App {
  def main(args: Array[String]): Unit = {

    def printImg(image: Array[Byte], xLen: Int, yLen: Int): Unit = {
      for (y <- 0 until yLen) {
        for (x <- 0 until xLen) {
          printf("%4d", image(x + (y * xLen)))
        }
        println()
      }
    }

    val trainingData: MNISTData = MNISTDataset.trainingData.read()
    trainingData.images.take(10).foreach(image => {
      println("num: " + image.label)
      printImg(image.data, trainingData.imageWidth, trainingData.imageHeight)
    })
  }
}
