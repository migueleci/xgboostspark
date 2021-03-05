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

    if (args.length == 0){
      println("You need to specify de data path to run the model.")
    } else {
      val path = args.apply(0) // "/home/miguel/projects/omics/spark/Data"
      var seed: Long = 2021
      if (args.length > 1) seed = args.apply(1).toLong
      val clf = new GeneClassification()
      clf.main(sc, path, seed)
    }

  }

}
