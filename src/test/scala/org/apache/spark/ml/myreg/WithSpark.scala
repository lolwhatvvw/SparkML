package org.apache.spark.ml.myreg

import org.apache.spark.sql.{SQLContext, SparkSession}

trait WithSpark {
  lazy val spark = WithSpark._spark
  lazy val sqlc = WithSpark._sqlc
}

object WithSpark{
  lazy val _spark = SparkSession.builder
    .appName("App")
    .master("local[4]")
    .config("spark.sql.shuffle.partitions", 3)
    .getOrCreate()

  lazy val _sqlc = _spark.sqlContext

  _spark.sparkContext.setLogLevel("ERROR")

}
