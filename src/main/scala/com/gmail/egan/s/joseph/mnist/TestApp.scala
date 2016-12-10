package com.gmail.egan.s.joseph.mnist

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

object TestApp {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)

    val ss = {
      SparkSession
        .builder
        .master("local[*]")
        .appName("MNIST:Test")
        .getOrCreate()
    }

    val test = {
      MNISTDataset
        .setResourcesDirectory("resources")
        .TestData
        .read()
        .imagesToDF
        .repartition(16)
        .persist(StorageLevel.MEMORY_ONLY)
    }
    test.show

    val model = PipelineModel.load("resources/mnist-pipeline")

    val timestampStart = System.currentTimeMillis
    val result = model.transform(test)
    val timeTaken = System.currentTimeMillis - timestampStart
    result.show

    val evaluator = {
      import MNISTData.DF.Columns
      new MulticlassClassificationEvaluator()
        .setLabelCol(Columns.LabelIndex)
        .setPredictionCol(Columns.LabelIndexPrediction)
        .setMetricName("accuracy")
    }
    val accuracy = evaluator.evaluate(result) * 100

    val confusionMatrix = {
      import MNISTData.DF._
      result
        .select(Columns.Label, Columns.LabelPrediction)
        .orderBy(Columns.Label)
        .groupBy(Columns.Label)
        .pivot(Columns.LabelPrediction, LabelIndices)
        .count
        .na
        .fill(0)
    }

    confusionMatrix.show
    println("(Horizontal: Predicted, Vertical: Actual)")

    ss.stop

    val timeTakenInSeconds = timeTaken / 1000
    println(s"duration: $timeTaken ms (~$timeTakenInSeconds seconds)")
    println(f"accuracy: $accuracy%2.2f")
  }
}
