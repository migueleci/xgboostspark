package org.test

import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD


class HierarchicalClassification {

  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  def readListFile(filename: String): ArrayBuffer[String] = {
    var list = ArrayBuffer.empty[String]
    val sourceFile = Source.fromFile(filename)
    for (line <- sourceFile.getLines) {
      list += line.trim()
    }
    sourceFile.close()
    list
  }


  def addIndexToDF(df: DataFrame): DataFrame = {
    val newSchema = StructType(df.schema.fields ++ Array(StructField("_index", LongType, nullable=false)))
    val rddWithId = df.rdd.zipWithIndex // Zip on RDD level

    spark.createDataFrame(rddWithId.map{ case (row, index) => Row.fromSeq(row.toSeq ++ Array(index))}, newSchema) // Convert back to DataFrame
  }


  def countNullsDF(df: DataFrame): Unit = {
    df.select(df.columns.map(colName => {
      count(when(col(colName).isNull, true)) as s"$colName"
    }): _*).show() // or save result somewhere
  }


  def castToDoubleDF(df: DataFrame): DataFrame = {
    df.columns.foldLeft(df)((acc, colName) => {
      acc.withColumn(colName, col(colName).cast(DoubleType))
    })
  }


  def setNullableStateOfColumns(df: DataFrame) : DataFrame = {
    val schema = df.schema
    val newSchema = StructType(schema.map {
      case StructField(c, t, _, m) => StructField(c, t, nullable=false, m)
    })
    spark.createDataFrame(df.rdd, newSchema)
  }


  def mergeDF(df1: DataFrame, df2: DataFrame): DataFrame = {
    val _df1 = this.addIndexToDF(df1)
    val _df2 = this.addIndexToDF(df2)
    _df1.join(_df2, _df1("_index") === _df2("_index"), "inner").drop("_index")
  }

  def seqToRDD(sc: SparkContext, s1: Seq[Double], s2: Seq[Double]): RDD[(Double, Double)] = {
    sc.makeRDD((s1 zip s2).map{x => (x._1.toDouble, x._2.toDouble)}.toList)
  }

  def evaluatePrediction(sc: SparkContext, pred: Seq[Double], prob: Seq[Double], label: Seq[Double]):
  (Double, Double, Double, Double, Double, Double) = {
    var predictionsAndLabels = this.seqToRDD(sc, pred, label)
    val metrics_mc = new MulticlassMetrics(predictionsAndLabels)
    val confusionMatrix = metrics_mc.confusionMatrix

    predictionsAndLabels = this.seqToRDD(sc, prob, label)
    val metrics_bc = new BinaryClassificationMetrics(predictionsAndLabels)
    val auROC = metrics_bc.areaUnderROC
    val auPRC = metrics_bc.areaUnderPR
    metrics_bc.roc().toDF("false_positive_rate", "true_positive_rate").show()

    println(s" Confusion Matrix\n ${confusionMatrix.toString}\n")
    println(s"Area under ROC = $auROC")
    println(s"Area under precision-recall curve = $auPRC")

    val tn = confusionMatrix.apply(0, 0)
    val fp = confusionMatrix.apply(0,1)
    val fn = confusionMatrix.apply(1,0)
    val tp = confusionMatrix.apply(1,1)

    (auROC, auPRC, tn, tp, fp, fn)
  }


  def hierarchy(sc: SparkContext, root: String, hierarchyPath: String, seed : Long = 202102):
  (String, Long, Long, Double, Double, Double, Double, Double, Double, Double) = {

    val hierarchy = this.readListFile(hierarchyPath + "/order.txt")//.slice(0, 3)
    val ancestors = this.readListFile(hierarchyPath + "/pred.txt")//.slice(0, 3)
    val hierarchyData = spark.read.option("header", "true").csv(hierarchyPath + "/data.csv")

    // Auxiliary variables
    var new_prob: Seq[Double] = null
    var prob_map: Map[String, Seq[Double]] = Map()
    var pred: DataFrame = null

    var all_prob = Seq[Double]()
    var all_pred = Seq[Double]()
    var all_label = Seq[Double]()

    val model = new CrossValidationClassification()
    val initialTime = System.nanoTime

    for ((term, ancestor) <- hierarchy zip ancestors){
      var termData = spark.read.option("header", "true").csv(hierarchyPath + "/" + term.replace(":","") + ".csv")
        .withColumnRenamed(term, term.replace(":",""))
      termData = this.mergeDF(hierarchyData, termData)

      if (ancestor.nonEmpty) // add ancestor prediction result as column to df
        termData = mergeDF(termData, sc.parallelize(prob_map.apply(ancestor))
          .toDF(ancestor.replace(":","")))

      termData = setNullableStateOfColumns(castToDoubleDF(termData))
      // termData.printSchema()

      val features = termData.columns.toBuffer
      features.remove(features.indexOf(term.replace(":","")))

      pred = model.predictCV(termData, features, term.replace(":",""), root, seed)
      new_prob = pred.select("probability").collect().map(_.getDouble(0)).toSeq
      prob_map += (term -> new_prob)
      all_prob = all_prob ++ pred.select("probability").collect().map(_.getDouble(0)).toSeq
      all_pred = all_pred ++ pred.select("prediction").collect().map(_.getDouble(0)).toSeq
      all_label = all_label ++ pred.select(term.replace(":","")).collect().map(_.getDouble(0)).toSeq
    }

    val duration = (System.nanoTime - initialTime) / 1e9d
    val (auROC, auPRC, tn, tp, fp, fn) = evaluatePrediction(sc, all_pred, all_prob, all_label)

    (root, hierarchy.length, hierarchyData.count, auROC, auPRC, tn, fp, fn, tp, duration)
  }
}
