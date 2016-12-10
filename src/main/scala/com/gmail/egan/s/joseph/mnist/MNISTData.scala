package com.gmail.egan.s.joseph.mnist

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

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