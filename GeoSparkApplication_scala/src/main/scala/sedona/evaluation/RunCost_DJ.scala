package sedona.evaluation

import java.io.FileWriter

import com.vividsolutions.jts.geom.GeometryFactory
import org.apache.log4j.{Level, Logger}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.datasyslab.geospark.enums.{FileDataSplitter, GridType, IndexType}
import org.datasyslab.geospark.serde.GeoSparkKryoRegistrator
import org.datasyslab.geospark.spatialOperator.JoinQuery
import org.datasyslab.geospark.spatialRDD.{CircleRDD, PointRDD}

import scala.io.Source
import scala.util.control.Breaks

object RunCost_DJ extends App{

  val conf = new SparkConf().setAppName("Evaluation")
//  conf.setMaster("local[*]")
  conf.set("spark.serializer", classOf[KryoSerializer].getName)
  conf.set("spark.kryo.registrator", classOf[GeoSparkKryoRegistrator].getName)
  val sc = new SparkContext(conf)
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  val geometryFactory = new GeometryFactory()

  // Grid Type
  val learned = GridType.VORONOI
  val idx = IndexType.RTREE

  //Datasets
  val points = "./datasets/usa_points_100k.csv" // for local
  // val points = "/data/usa_points_100k.csv" // for hadoop

  val query_path = "./tmp/query.csv"
  val Queries = readCSV(query_path)

  val bestcost_path = "./tmp/best_cost.csv"
  val BestCosts = readCSV(bestcost_path)

//  for (c <- BestCosts){
//    print(c(0)+ " ")
//  }
//  println("")

  var t0 = 0L
  var t1 = 0L
  var count = 0L

  var response_time:Double = 0L
  var response_time_list: Array[Double] = Array.empty
  var median_time:Double = 0L
  var median_time_list: Array[Double] = Array.empty

  val num_trial: Int = 4
  val b = new Breaks

  run_query(learned, 8)

  sc.stop()

  def run_query(partitioningScheme: org.datasyslab.geospark.enums.GridType, numPartitions: Int): Unit = {

    // SpatialRDD
    var pointRDD = new PointRDD(sc, points, 0, FileDataSplitter.CSV, false, numPartitions, StorageLevel.MEMORY_ONLY)

    pointRDD.analyze()
    pointRDD.spatialPartitioning(partitioningScheme, 1) 
    pointRDD.buildIndex(idx, true)
    //println(get_pointRDD.spatialPartitionedRDD.getNumPartitions)
    pointRDD.indexedRDD.persist(StorageLevel.MEMORY_ONLY)
    pointRDD.spatialPartitionedRDD.persist(StorageLevel.MEMORY_ONLY)

    b.breakable {
      for ((d, i) <- Queries.zipWithIndex) {
        //println(d(0))
        response_time_list = Array.empty
        var degree = d(0) / 110000
        // println(degree)
        val circleRDD = new CircleRDD(pointRDD, degree) // Simba computes euclidean distance, this distance is in degrees (in meters it would be approx 5 meters considering worst case scenario where 1 degree is equal to 110 kms at the equator)
        circleRDD.spatialPartitioning(pointRDD.getPartitioner)

        for (t <- 1 to num_trial) {
          t0 = System.nanoTime()
          var result = JoinQuery.DistanceJoinQuery(pointRDD, circleRDD, true, false).collect()
          t1 = System.nanoTime()
          response_time = (t1 - t0) / 1E9
          // println(response_time)

          // ---- Execution Stop ---
          if (response_time > 2 * BestCosts(i)(0)) {
            println("stop run query")
            //          println("response time: "+median_time+"  best cost:"+ BestCosts(i)(0))
            writeCSV("./tmp/run_cost.csv", median_time_list, false)
            b.break
          }
          response_time_list :+= response_time
        }
        median_time = medianCalculator(response_time_list)
        median_time_list :+= median_time
        circleRDD.spatialPartitionedRDD.unpersist()
      }
      println("finish run query")
      writeCSV("./tmp/run_cost.csv", median_time_list,true)
    }
    pointRDD.indexedRDD.unpersist()
    pointRDD.spatialPartitionedRDD.unpersist()
  }

  def medianCalculator(seq: Seq[Double]): Double = {
    //In order if you are not sure that 'seq' is sorted
    val sortedSeq = seq.sortWith(_ < _)

    if (seq.size % 2 == 1) sortedSeq(sortedSeq.size / 2)
    else {
      val (up, down) = sortedSeq.splitAt(seq.size / 2)
      (up.last + down.head) / 2
    }
  }

  def readCSV(path: String) : Array[Array[Double]] = {
    Source.fromFile(path)
      .getLines()
      .map(_.split(",").map(_.trim.toDouble))
      .toArray
  }

  def writeCSV(path: String, data: Array[Double], flag: Boolean)  = {
    val file = new FileWriter(path, false)
    if (flag) {
      for (d <- data) {
        file.write(d.toString)
        file.write('\n')
      }
    }
    else {
      file.write('\n')
    }
    file.close()
  }
}
