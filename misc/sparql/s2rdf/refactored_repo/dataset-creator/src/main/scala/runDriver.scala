/* Copyright Simon Skilevic
 * Master Thesis for Chair of Databases and Information Systems
 * Uni Freiburg
 */

import dataCreator.DataSetGenerator
import dataCreator.Settings

object runDriver {
  /*
   *  arg0: database directory
   *  arg1: rdf file name (this must be in arg0 directory)
   *  arg2: dataset type; OS, SS, SO
   *  arg3: scale factor
   */
  def main(args:Array[String]){
    // parse Args
    Settings.loadUserSettings(args(0), args(1), args(3).toFloat)
    val datasetType = args(2)
    DataSetGenerator.generateDataSet(datasetType);    
  }
}