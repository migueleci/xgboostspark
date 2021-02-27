package org.test

import org.apache.spark.sql.SparkSession

/**
 * @author ${Miguel Romero}
 */
object App {

  /**
   *
   * @param args
   */
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
  }

}
