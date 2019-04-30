/* Copyright Simon Skilevic
 * Master Thesis for Chair of Databases and Information Systems
 * Uni Freiburg
 */

package queryExecutor

import org.apache.spark.sql.SparkSession

/**
 * Different settings for the Query Executor
 * TODO: implement reading of settings from config file.
 */
object Settings {
  
  // Settings to change
  val testLoop = 1;
  val partNumberExtVP = 50
  val partNumberVP = 200

  val sparkSession = loadSparkSession();
  
  var databaseDir = ""
  var queryFile = ""
  var resultFile = ""
  var resultFileTimes = ""
  
  def loadUserSettings(dbDir:String, 
                       qrFile:String) = {    
    this.databaseDir = dbDir
    if (dbDir.charAt(dbDir.length - 1) != '/')
      this.databaseDir += "/"      
    this.queryFile = qrFile
    var p = qrFile.lastIndexOf("/")
    if (p < 0)
    {
      this.resultFile = "./results.txt";
      this.resultFileTimes = "./resultTimes.txt";
    } else {
      this.resultFile = qrFile.substring(0, p) + "/results.txt";
      this.resultFileTimes = qrFile.substring(0, p) + "/resultTimes.txt";
    }
  }

  def loadSparkSession(): SparkSession = {
    val session = SparkSession.builder().appName("QueryExecutor")
      .config("spark.master", "local")
      .config("spark.executor.memory", "2688m")
      .config("spark.sql.inMemoryColumnarStorage.compressed", "true")
      .config("spark.sql.autoBroadcastJoinThreshold", "-1")
      .config("spark.sql.parquet.filterPushdown", "true")
      .config("spark.sql.inMemoryColumnarStorage.batchSize", "20000")
      .config("spark.sql.shuffle.partitions", "50")
      //.config("spark.storage.memoryFraction", "0.4")
      //.config("spark.sql.inMemoryColumnarStorage.batchSize", "10000")
      //.config("spark.driver.allowMultipleContexts", "true")
      .getOrCreate()
    session
  }
}
