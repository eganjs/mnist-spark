package com.gmail.egan.s.joseph

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.sql.SparkSession

object App {
  def main(args: Array[String]): Unit = {
    val trainingData: MNISTData = MNISTDataset.trainingData.read()

    val ss = SparkSession.builder().master("local[4]").appName("MNIST").getOrCreate()

    import ss.implicits._

    val data = trainingData.images.toSeq.map { image =>
      LabeledPoint(
        image.label,
        Vectors.dense(image.data.map { b => (b & 0xFF).toDouble })
      )
    }.toDF

    data.show(5)

    val glr = new GeneralizedLinearRegression()
    val model = glr.fit(data.as[LabeledPoint])

    ss.stop()
  }
}
