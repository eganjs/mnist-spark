package com.gmail.egan.s.joseph

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, Normalizer, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel

object App {
  def main(args: Array[String]): Unit = {
    val timestampStart: Long = System.currentTimeMillis

    val label = "label"
    val features = "features"
    val normFeatures = "normFeatures"
    val pixelsPerImage = 784

    val ss = SparkSession.builder.master("local[*]").appName("MNIST").getOrCreate

    import ss.implicits._

    def readAsDF(dataset: MNISTDataset): DataFrame = dataset.read().images.toSeq.map { image =>
      (
        image.label.toString,
        Vectors.dense(image.data.map { b => (b & 0xFF).toDouble })
      )
    }.toDF(label, features).repartition(32).persist(StorageLevel.MEMORY_ONLY)

    val train = readAsDF(MNISTDataset.trainingData)
    train.show

    val normalizer = {
      new Normalizer()
        .setInputCol(features)
        .setP(1.0)
    }

    val stringIndexer = {
      new StringIndexer()
        .setInputCol(label)
        .fit(train)
    }

    val multilayerPerceptron = {
      new MultilayerPerceptronClassifier()
        .setLabelCol(stringIndexer.getOutputCol)
        .setFeaturesCol(normalizer.getOutputCol)
        .setLayers {
          Array[Int](
            pixelsPerImage,
            pixelsPerImage,
            pixelsPerImage,
            stringIndexer.labels.length
          )
        }
        .setSeed(1030937410110397110L)
        .setBlockSize(128)
        .setMaxIter(100)
        .setTol(1e-7)
    }

    val indexToString = {
      new IndexToString()
        .setInputCol(multilayerPerceptron.getPredictionCol)
        .setLabels(stringIndexer.labels)
    }

    val pipeline = new Pipeline().setStages {
      Array(stringIndexer, normalizer, multilayerPerceptron, indexToString)
    }

    val model = pipeline.fit(train)

    println("model trained")

    val test = readAsDF(MNISTDataset.testData)
    test.show

    val result = model.transform(test)

    val evaluator = {
      new MulticlassClassificationEvaluator()
        .setLabelCol(stringIndexer.getOutputCol)
        .setPredictionCol(multilayerPerceptron.getPredictionCol)
        .setMetricName("precision")
    }
    val precision = evaluator.evaluate(result)

    val confusionMatrix = {
      result.select(stringIndexer.getInputCol, indexToString.getOutputCol)
        .orderBy(stringIndexer.getInputCol)
        .groupBy(stringIndexer.getInputCol)
        .pivot(indexToString.getOutputCol, stringIndexer.labels)
        .count
    }

    println(s"Confusion Matrix (Vertical: Actual, Horizontal: Predicted):")
    confusionMatrix.show

    println(s"Duration: ${System.currentTimeMillis - timestampStart} ms (~${(System.currentTimeMillis - timestampStart) / 1000 / 60} minutes)")
    println(s"Precision: $precision")

    ss.stop
  }
}
