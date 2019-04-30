/* Copyright Simon Skilevic
 * Master Thesis for Chair of Databases and Information Systems
 * Uni Freiburg
 */



package dataCreator

import org.apache.spark.sql.SaveMode

/**
 * StatisticWriter records information about created tables using 
 * DataSetGenerator, it creates 4 statistic files for VP, ExtVP_SO, ExtVP_SS and 
 * ExtVP_OS tables respectively. 
 */
object StatisticWriter {
  
  // number of unique predicates
  private var _predicatesNum = 0: Int
  // number of triples in the input RDF set
  private var _inputSize = 0: Long
  // the name of the active statistic file, which is being written
  private var _statisticFileName = "": String
  
  private var _savedTables = 0: Int
  private var _unsavedNonEmptyTables = 0: Int
  private var _allPossibleTables = 0: Int

  // Only for importing implicits
  private val seession = Settings.sparkSession

  
  /**
   * Initializes StatisticWriter
   */
  def init(predsNum: Int, inpSize: Long) = {
    _predicatesNum = predsNum
    _inputSize = inpSize
  }
  
  /**
   * Initializes recording of new statistic file.
   */
  def initNewStatisticFile(relType: String) = {
    _statisticFileName = (Settings.statisticsDir + "stat_"+relType.toLowerCase)
    _savedTables = 0
    _unsavedNonEmptyTables = 0
    _allPossibleTables = if (relType == "VP") _predicatesNum 
                         else _predicatesNum * _predicatesNum

  }

  /**
   * Puts the tail at the and of the written statistic files
   */
  case class FinalStats(savedTables: Int, unsavedNonEmptyTables: Int, emptyTables: Int)
  def closeStatisticFile() = {

    // NOTE New for CSCI-548
    import seession.implicits._
    val stat = Seq(FinalStats(_savedTables, _unsavedNonEmptyTables, (_allPossibleTables
      - _savedTables
      - _unsavedNonEmptyTables) )).toDF()
    stat.write.option("delimter", "\t").csv(_statisticFileName + "_summary.tsv")
  }

  // CSCI-548
  case class VPStat(table: String, numberOfRows: Long, numberOfTriples: Long, vpTripleRatio: String)
  case class ExtVPStat(predicates: String, numberOfRows: Long, sizeOfPred1VpTable: Long,
                       extVpRatio: String, vpRatio: String)
  /**
   * Add new line to the actual statistic file
   */
  def addTableStatistic(tableName: String, sizeExtVpT: Long, sizeVpT: Long) = {    
    var statLine = tableName

    if (sizeExtVpT > 0) {
      import seession.implicits._
      val statDf = Seq(ExtVPStat(tableName, sizeExtVpT, sizeVpT, Helper.ratio(sizeExtVpT,
        sizeVpT),  Helper.ratio(sizeVpT, _inputSize))).toDF()
      statDf.coalesce(1)
            .write
            .mode(SaveMode.Overwrite)
            .csv(_statisticFileName + ".tsv")
    } else {
      // VP table statistic entry
      import seession.implicits._
      val statDf = Seq(VPStat(tableName, sizeVpT, _inputSize, Helper.ratio(sizeVpT, _inputSize))).toDF()
      statDf.coalesce(1)
            .write
            .mode(SaveMode.Append)
            .option("delimter", "\t")
            .csv(_statisticFileName + ".tsv")
    }
  }
  /**
   * Increments the counter for the saved tables
   */
  def incSavedTables() = {
    _savedTables += 1
  }
  
  /**
   * Increments the counter for the unsaved tables, which are not empty, e.g. 
   * ExtVP tables having size bigger than ScaleUB * (Size of corresponding 
   * VP table)
   */
  def incUnsavedNonEmptyTables() = {
    _unsavedNonEmptyTables += 1
  }
}
