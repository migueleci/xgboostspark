package org.test

import org.apache.spark.sql.SparkSession

/**
 * @author ${Miguel Romero}
 */
object App {
  
  def foo(x : Array[String]): String = x.foldLeft("")((a, b) => a + b)
  
  def main(args : Array[String]) {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .appName("xgboost")
      .getOrCreate()

    val sc = spark.sparkContext

    val path = "/home/miguel/projects/omics/spark/Data"
    val clf = new GeneClassification()
    clf.main(sc, path, 2021)

    println( "Hello World!" )
    println("concat arguments = " + foo(args))
  }

}
