package com.gmail.egan.s.joseph

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.linalg.{SQLDataTypes, SparseVector, Vector}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._

class AffineTransformer(override val uid: String) extends UnaryTransformer[Vector, SparseVector, AffineTransformer] {
  def this() = this(Identifiable.randomUID("affineTransformer"))

  private var xLen, yLen: Int = 1

  private var (xScale, yShear: Double) = (1d, 0d)
  private var (xShear, yScale: Double) = (0d, 1d)
  private var (xShift, yShift: Double) = (0d, 0d)

  def setDimensions(x: Int, y: Int): AffineTransformer = {
    this.xLen = x; this.yLen = y
    this
  }

  def setTransformation(
                         xScale: Double, yShear: Double,
                         xShear: Double, yScale: Double,
                         xShift: Double, yShift: Double
                       ): AffineTransformer = {
    this.xScale = xScale; this.yShear = yShear
    this.xShear = xShear; this.yScale = yScale
    this.xShift = xShift; this.yShift = yShift
    this
  }

  def setScaleTransformation(xScale: Double, yScale: Double): this.type = {
    this.xScale = xScale; this.yShear = 0
    this.xShear = 0; this.yScale = yScale
    this.xShift = 0; this.yShift = 0
    this
  }

  def setShearTransformation(xShear: Double, yShear: Double): this.type = {
    this.xScale = 1; this.yShear = yShear
    this.xShear = xShear; this.yScale = 1
    this.xShift = 0; this.yShift = 0
    this
  }

  def setShiftTransformation(xShift: Double, yShift: Double): this.type = {
    this.xScale = 1; this.yShear = 0
    this.xShear = 0; this.yScale = 1
    this.xShift = xShift; this.yShift = yShift
    this
  }

  def setRotationTransformation(clockwiseRotationInRadians: Double): this.type = {
    this.xScale =  Math.cos(clockwiseRotationInRadians); this.yShear = Math.sin(clockwiseRotationInRadians)
    this.xShear = -Math.sin(clockwiseRotationInRadians); this.yScale = Math.cos(clockwiseRotationInRadians)
    this.xShift = 0; this.yShift = 0
    this
  }

  private implicit class Point(i: Int) {
    var x: Int = i % xLen
    var y: Int = i / xLen

    def transform(): Int = {
      // shift the image center to the origin
      x -= xLen / 2
      y -= yLen / 2

      // apply transformation
      x = ((x * xScale) + (y * xShear) + xShift).toInt
      y = ((x * yShear) + (y * yScale) + yShift).toInt

      // shift back
      x += xLen / 2
      y += yLen / 2

      // return new vector location
      x + y * xLen
    }
  }

  override def createTransformFunc(): (Vector => SparseVector) = (features: Vector) => {
    val sparseFeatures = features.toSparse.indices.map(i => {
      val value = features(i)
      if (value == 0)
        None
      else
        Some((i.transform(), value))
    }).flatMap(_.iterator)

    val (indices, values) = sparseFeatures.unzip
    new SparseVector(features.size, indices, values)
  }

  override protected def outputDataType: DataType = SQLDataTypes.VectorType
}
