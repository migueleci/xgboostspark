package org.test

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer


class GeneClassification {

  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  def main(sc: SparkContext, path: String, seed: Long = 202102): Unit = {

    val clf = new HierarchicalClassification()

    val filename = path + "/roots.txt"
    val rootList = clf.readListFile(filename)
    var data = ArrayBuffer.empty[(String, Long, Long, Double, Double, Double, Double, Double, Double, Double)]

    for (root <- rootList) {
      val hierarchyPath = path + "/" + root.replace(":","")
      data += clf.hierarchy(sc, root, hierarchyPath, seed)
    }

    val df = sc.parallelize(data).toDF("Root","Terms","Genes","auROC","auPR", "tn", "fp", "fn", "tp", "Time")
    // df.write.format("com.databricks.spark.csv").option("header", "true").save("output.csv")
    df.write.format("parquet").option("header", "true").save("output.parquet")
    df.show()
  }
}
