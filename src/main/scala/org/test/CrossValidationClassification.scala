package org.test

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel
import com.sgcharts.sparkutil.Smote

import scala.collection.mutable // Scala 2.11


class CrossValidationClassification {

  val spark: SparkSession = SparkSession.builder().getOrCreate()
  import spark.implicits._

  def evaluatePrediction(prediction: DataFrame, label: String): Unit = {
    var predictionsAndLabels = prediction.select("prediction", label)
      .map(row => (row.getDouble(0), row.getDouble(1))).rdd

    predictionsAndLabels = prediction.select("probability", label)
      .map(row => (row.getAs[DenseVector](0).apply(1), row.getDouble(1))).rdd

    val metrics_bc = new BinaryClassificationMetrics(predictionsAndLabels)
    val auROC = metrics_bc.areaUnderROC
    val auPRC = metrics_bc.areaUnderPR

    // println(s" Confusion Matrix\n ${confusionMatrix.toString}\n")
    println(s"Area under ROC = $auROC")
    println(s"Area under precision-recall curve = $auPRC")
  }


  def oversamplingMinorityClass(df: DataFrame, features: mutable.Buffer[String], label: String, seed: Long): DataFrame = {
    val threshold = 0.3

    val posProp = df.filter(col(label) === 1).count.toFloat / df.count
    val negProp = 1 - posProp
    var sizeMultiplier = 0
    var minorityDF = spark.emptyDataFrame
    var oversDF = spark.emptyDataFrame

    if (posProp < threshold) {
      sizeMultiplier = (negProp / posProp).toInt
      minorityDF = df.filter(col(label) === 1).drop(label)
      oversDF = Smote(
        sample = minorityDF,
        discreteStringAttributes = Seq.empty[String],
        discreteLongAttributes = Seq.empty[String],
        continuousAttributes = features,
        sizeMultiplier = sizeMultiplier,
        seed=Some(seed.toInt)
      )(spark).syntheticSample
      df.filter(col(label) === 0.0).unionByName(oversDF.withColumn(label, lit(1.0)))
    } else if (negProp < threshold) {
      sizeMultiplier = (posProp / negProp).toInt
      minorityDF = df.filter(col(label) === 0).drop(label)
      oversDF = Smote(
        sample = minorityDF,
        discreteStringAttributes = Seq.empty[String],
        discreteLongAttributes = Seq.empty[String],
        continuousAttributes = features,
        sizeMultiplier = sizeMultiplier
      )(spark).syntheticSample
      df.filter(col(label) === 1).unionByName(oversDF.withColumn(label, lit(0)))
    } else {
      df
    }
  }

  def predictCV(df: DataFrame, features: mutable.Buffer[String], label: String, root: String, seed: Long = 2021): DataFrame = {

    val positiveCount = df.select(col(label)).rdd.map(_ (0).asInstanceOf[Double]).reduce(_ + _)
    val negativeCount = df.count() - positiveCount
    println(" #### ----> " + label + " -- " + root)

    if (positiveCount > 5 && negativeCount > 5) {

      val fractions = Map(0.0 -> 0.5, 1.0 -> 0.5)
      val df1 = df.stat.sampleBy(label, fractions, seed)
      val df2 = df.except(df1)

      val predictionDF2 = this.train(df1, df2, features, label, seed)
      val predictionDF1 = this.train(df2, df1, features, label, seed)
      val prediction = predictionDF1.unionByName(predictionDF2)
      // this.evaluatePrediction(prediction, label)

      val schema = StructType(Seq(
        StructField("prediction", DoubleType, nullable=false),
        StructField("probability", DoubleType, nullable=false),
        StructField(label, DoubleType, nullable=false)
      ))
      val encoder = RowEncoder(schema)

      prediction.select("prediction", "probability", label).map(row => Row(row.getDouble(0), row.getAs[DenseVector](1).apply(1), row.getDouble(2)))(encoder)
    } else {
      spark.emptyDataFrame
        .withColumn("prediction", lit(0))
        .withColumn("probability", lit(0))
        .withColumn(label, lit(0))
    }
  }

  def train(training_imb: DataFrame, test: DataFrame, features: mutable.Buffer[String], label: String, seed: Long = 2021): DataFrame = {

    val training = this.oversamplingMinorityClass(training_imb, features, label, seed)
    // training.groupBy(label).count().show()

    val assembler = new VectorAssembler()
      .setInputCols(features.toArray)
      .setOutputCol("features")

    val booster = new XGBoostClassifier(
      Map("eta" -> 0.5f,
        "max_depth" -> 5,
        "objective" -> "binary:logistic",
        "num_round" -> 100,
        //"num_workers" -> 4,
        //"nthread" -> 2,
        "tree_method" -> "approx",
        "missing" -> 0,
        "timeout_request_workers" -> 60000L,
        "verbose_eval"-> false
      )
    )
      .setFeaturesCol("features")
      .setLabelCol(label)
      .setNumEarlyStoppingRounds(2)
      .setMaximizeEvaluationMetrics(true)
      .setEvalMetric("aucpr")

    val pipeline = new Pipeline()
      .setStages(Array(assembler, booster))

    // Model evaluation
    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setLabelCol(label)
    evaluator.setMetricName("areaUnderPR")


    // Tune model using cross validation
    val paramGrid = new ParamGridBuilder()
      .addGrid(booster.maxDepth, Array(4, 6))
      .addGrid(booster.eta, Array(0.4,0.7))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(8)

    val cvModel = cv.fit(training)
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[XGBoostClassificationModel]

    cvModel.transform(test)
  }
}
