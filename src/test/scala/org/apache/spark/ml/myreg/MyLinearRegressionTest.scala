package org.apache.spark.ml.myreg

import scala.util.Random
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest.flatspec._
import org.scalatest.matchers._

class MyLinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark  {
  val delta = 1e-4

  lazy val data: DataFrame = MyLinearRegressionTest._data
  lazy val vectors: Seq[Vector] = MyLinearRegressionTest._vectors
  lazy val predictUDF: UserDefinedFunction = MyLinearRegressionTest._predictUDF

  "Model" should "predict input data" in {
    val model: MyLinearRegressionModel = new MyLinearRegressionModel(coefficients = Vectors.dense(1.5, 0.3, -0.7))
    val df = model.transform(data)

    val vector = df.collect().map(_.getAs[Double](1))

    vector.length should be(2)

    vector(0) should be(-27.25 +- delta)
    vector(1) should be(-1.64 +- delta)
  }

  "Estimator" should "predict correctly" in {
    val estimator = new MyLinearRegression().setMaxIter(1500).setLearningRate(0.1)

    import sqlc.implicits._

    val randomData = Matrices
      .rand(100000, 3, Random.self)
      .rowIter
      .toSeq
      .map(x => Tuple1(x))
      .toDF("features")

    val dataset = randomData.withColumn("label", predictUDF(col("features")))
    val model = estimator.fit(dataset)

    model.coefficients(0) should be(1.5 +- delta)
    model.coefficients(1) should be(0.3 +- delta)
    model.coefficients(2) should be(-0.7 +- delta)
  }
}

object MyLinearRegressionTest extends WithSpark {

  lazy val _vectors: Seq[Vector] = Seq(
    Vectors.dense(13.5, 12, 73),
    Vectors.dense(-1, 0, 0.2)
  )

  lazy val _data: DataFrame = {
    import sqlc.implicits._
    _vectors.map(x => Tuple1(x)).toDF("features")
  }

  lazy val _predictUDF: UserDefinedFunction = udf { features: Any =>
    val arr = features.asInstanceOf[Vector].toArray
    1.5 * arr.apply(0) + 0.3 * arr.apply(1) - 0.7 * arr.apply(2)
  }
}
