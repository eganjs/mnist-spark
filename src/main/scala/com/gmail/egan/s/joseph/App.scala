package com.gmail.egan.s.joseph

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, PCA, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel

object App {
  def main(args: Array[String]): Unit = {
    val timestampStart = System.currentTimeMillis

    val label = "label"
    val features = "features"

    val ss = SparkSession.builder.master("local[*]").appName("MNIST").getOrCreate

    import ss.implicits._

    def readAsDF(dataset: MNISTDataset, max: Int = Int.MaxValue): DataFrame = {
      dataset.read().images
        .take(max)
        .map { image =>
          (
            image.label.toString,
            Vectors.dense(image.data.map { b => (b & 0xFF).toDouble / 255 })
          )
        }
        .toSeq.toDF(label, features)
        .repartition(32)
        .persist(StorageLevel.MEMORY_ONLY)
    }

    val train = readAsDF(MNISTDataset.TrainingData)
    train.show

    val stringIndexer = {
      new StringIndexer()
        .setInputCol(label)
        .fit(train)
    }

    val pca = {
      new PCA()
        .setInputCol(features)
        .setK(20)
    }

    val multilayerPerceptron = {
      new MultilayerPerceptronClassifier()
        .setLabelCol(stringIndexer.getOutputCol)
        .setFeaturesCol(pca.getOutputCol)
        .setLayers {
          Array[Int](
            pca.getK,
            pca.getK * 2,
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
      Array(stringIndexer, pca, multilayerPerceptron, indexToString)
    }

    val model = pipeline.fit(train)

    println("model trained")

    val test = readAsDF(MNISTDataset.TestData)
    test.show

    val result = model.transform(test)

    val evaluator = {
      new MulticlassClassificationEvaluator()
        .setLabelCol(stringIndexer.getOutputCol)
        .setPredictionCol(multilayerPerceptron.getPredictionCol)
        .setMetricName("accuracy")
    }
    val accuracy = evaluator.evaluate(result)

    val confusionMatrix = {
      result.select(stringIndexer.getInputCol, indexToString.getOutputCol)
        .orderBy(stringIndexer.getInputCol)
        .groupBy(stringIndexer.getInputCol)
        .pivot(indexToString.getOutputCol, stringIndexer.labels.sorted)
        .count
        .na.fill(0)
    }


    confusionMatrix.show
    println("(Vertical: Actual, Horizontal: Predicted)")

    val timeTaken = System.currentTimeMillis - timestampStart
    val timeTakenInMinutes = timeTaken / 1000 / 60
    println(s"Duration: $timeTaken ms (~$timeTakenInMinutes minutes)")
    println(s"Accuracy: $accuracy")

    ss.stop
  }
}
