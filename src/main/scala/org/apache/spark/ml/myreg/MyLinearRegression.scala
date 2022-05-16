package org.apache.spark.ml.myreg

import breeze.linalg
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, Params}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait MyLinearRegressionParams extends Params with HasFeaturesCol with HasPredictionCol with HasLabelCol with HasMaxIter{

  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type  = set(predictionCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  val learningRate = new DoubleParam(this, "learning rate", "Hyperparameter of gradient step")
  def setLearningRate (value: Double): this.type = set(learningRate, value)
  def getLearningRate: Double = $(learningRate)

  //probably too small default lr
  setDefault(learningRate -> 0.01, maxIter -> 1500)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}

class MyLinearRegression(override val uid: String) extends Estimator[MyLinearRegressionModel] with MyLinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("myLinearRegression"))

  override def copy(extra: ParamMap): Estimator[MyLinearRegressionModel] = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
  override def fit(dataset: Dataset[_]): MyLinearRegressionModel = {

    implicit val vectorEncoder: Encoder[Vector] = ExpressionEncoder()

    //concat features and labels to get row with both
    val tmp = new VectorAssembler().setInputCols(Array(getFeaturesCol, getLabelCol)).setOutputCol("out")
    val vectors = tmp.transform(dataset).select("out").as[Vector]

    val max_iter = getMaxIter
    val lr = getLearningRate

    //get number of features to initialize weights and split out -> features|labels
    val n_features: Int = AttributeGroup.fromStructField(dataset.schema($(featuresCol))).numAttributes.getOrElse(
      vectors.first().size
    )

    val w = linalg.DenseVector.zeros[Double](n_features-1)

    //TODO: can be broadcasted
    //TODO: minibatch impl
    for (_ <-1 to max_iter) {
       val grads = vectors.rdd.mapPartitions(iter => {
         //for each partition create summarizer
        val summarizer = new MultivariateOnlineSummarizer()
        for (row <- iter){
          val y = row.asBreeze(-1)
          val x = row.asBreeze(0 until  n_features-1).toDenseVector
          summarizer.add(mllib.linalg.Vectors.fromBreeze((x.dot(w) - y) * x))
        }
         Iterator(summarizer)
         //merging partial derivatives from different partitions
      }).reduce(_ merge _)

      w -= 2 * lr * grads.mean.asBreeze
    }

    copyValues(new MyLinearRegressionModel(mllib.linalg.Vectors.fromBreeze(w).asML).setParent(this))
  }
}


  class MyLinearRegressionModel private[myreg](
                                override val uid: String,
                                val coefficients: DenseVector
                              ) extends Model[MyLinearRegressionModel] with MyLinearRegressionParams with MLWritable {


     private[myreg] def this(coefficients: Vector) = {
      this(Identifiable.randomUID("distributed gradient descent"), coefficients.toDense)
    }

    override def copy(extra: ParamMap): MyLinearRegressionModel = copyValues(
      new MyLinearRegressionModel(coefficients), extra)

    override def write: MLWriter = new DefaultParamsWriter(this) {
      protected override def saveImpl(path: String): Unit = {
        super.saveImpl(path)
        val vectors = coefficients.asInstanceOf[Vector] -> coefficients.asInstanceOf[Vector]
        sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/coefficients")
      }
    }

    override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

    private def predict(features: Vector) = features.asBreeze.dot(coefficients.asBreeze)

 //see transformImpl from org.apache.spark.ml.Predictor
    override def transform(dataset: Dataset[_]): DataFrame = {
      val outputSchema = transformSchema(dataset.schema, logging = true)
      val predictUDF = udf { features: Any =>
        predict(features.asInstanceOf[Vector])
      }
      dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))),
        outputSchema($(predictionCol)).metadata)
    }
  }

//TODO: add unit test
object MyLinearRegressionModel extends MLReadable[MyLinearRegressionModel]{
  override def read: MLReader[MyLinearRegressionModel] = new MLReader[MyLinearRegressionModel] {
    override def load(path: String): MyLinearRegressionModel = {

      val vectors = sqlContext.read.parquet(path + "/coefficients")

      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val original = new DefaultParamsReader().load(path).asInstanceOf[MyLinearRegressionModel]
      val coefficients = vectors.select(vectors("_1").as[Vector]).first()

      original.copyValues(new MyLinearRegressionModel(coefficients))
    }
  }
}

object MyLinearRegression extends DefaultParamsReadable[MyLinearRegression]


