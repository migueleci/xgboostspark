# XGBoost4j-Spark project 

This is a project that integrates Spark, Scala, Maven and XGBoost to 
address a hierarchical classification problem. Hierarchical 
classification refers to problems where the classes or labels of the 
attribute to be predicted have a hierarchical structure, i.e., there
are ancestral relations or dependencies between the classes.

The workflow of the project includes k-folding, oversampling with 
SMOTE and cross validation. 

## Requirements

 - OpenJDK 8 (1.8.0_282)
 - Spark [2.4.7](https://downloads.apache.org/spark/spark-2.4.7/spark-2.4.7-bin-hadoop2.7.tgz)
 - Scala 2.11.12
 - Maven 3.6.3
 - JAVA_HOME and SPARK_HOME environment variables
 - XGBoost4j and SMOTE packages (.jar) on $SPARK_HOME/jars.
    - Spark Utilities [0.4.1](https://repo1.maven.org/maven2/com/sgcharts/spark-util_2.11/0.4.1/spark-util_2.11-0.4.1.jar) (SMOTE)
    - Xgboost4j [1.1.2](https://repo1.maven.org/maven2/ml/dmlc/xgboost4j_2.11/1.1.2/xgboost4j_2.11-1.1.2.jar)
    - Xgboost4j Spark [1.1.2](https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark_2.11/1.1.2/xgboost4j-spark_2.11-1.1.2.jar)
    