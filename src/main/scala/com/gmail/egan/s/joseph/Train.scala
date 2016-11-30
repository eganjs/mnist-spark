package com.gmail.egan.s.joseph

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.{IndexToString, PCA, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel

object Train {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)

    val ss = {
      SparkSession
        .builder
        .master("local[*]")
        .appName("MNIST")
        .getOrCreate()
    }

    val train = {
      MNISTDataset
        .setResourcesDirectory("resources")
        .TrainingData
        .read()
        .imagesToDF
        .repartition(16)
        .persist(StorageLevel.MEMORY_ONLY)
    }
    train.show

    val stringIndexer = {
      import MNISTData.DF.Columns
      new StringIndexer()
        .setInputCol(Columns.Label)
        .setOutputCol(Columns.LabelIndex)
        .fit(train)
    }

    val pcaReducedFeaturesCount = 32
    val pca = {
      import MNISTData.DF.Columns
      new PCA()
        .setInputCol(Columns.Features)
        .setOutputCol(Columns.PrincipalFeatures)
        .setK(pcaReducedFeaturesCount)
    }

    val multilayerPerceptron = {
      import MNISTData.DF.Columns
      new MultilayerPerceptronClassifier()
        .setLabelCol(Columns.LabelIndex)
        .setFeaturesCol(Columns.PrincipalFeatures)
        .setPredictionCol(Columns.LabelIndexPrediction)
        .setLayers {
          // Input layer  - the k most significant features
          // Hidden layer - build complex feature structures
          // Output layer - combine complex features
          Array[Int](
            pca.getK,
            pca.getK * 2,
            stringIndexer.labels.length
          )
        }
        .setSeed(1030937410110397110L)
        .setBlockSize(256)
        .setMaxIter(100)
        .setTol(1e-7)
    }

    val indexToString = {
      import MNISTData.DF.Columns
      new IndexToString()
        .setInputCol(Columns.LabelIndexPrediction)
        .setOutputCol(Columns.LabelPrediction)
        .setLabels(stringIndexer.labels)
    }

    val pipeline = new Pipeline().setStages {
      Array(
        stringIndexer,
        pca,
        multilayerPerceptron,
        indexToString
      )
    }

    val timestampStart = System.currentTimeMillis
    val model = pipeline.fit(train)
    val timeTakenInMillis = System.currentTimeMillis - timestampStart
    println("model trained")

    model.write.overwrite.save("resources/mnist-pipeline")
    println("model persisted to disk")

    ss.stop

    val timeTakenInSeconds = timeTakenInMillis / 1000
    val minutes = timeTakenInSeconds / 60
    val seconds = timeTakenInSeconds % 60
    println(s"duration: $timeTakenInMillis ms (~$minutes minutes and ~$seconds seconds)")
  }
}
