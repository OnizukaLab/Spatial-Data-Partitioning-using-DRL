# Deep Reinforcement Learning for Spatial Data Partitioning
A framework for applying deep reinforcement learning to spatial data partitioning.

### Requirements
- JDK 1.8
- Spark 2.3.4
- Hadoop 2.7.7
- GeoSpark 1.3.1 (Apache Sedona)

### Code
Execute the following commands on the master node (or locally).
```
$ python python/main.py
```

To run main-training, you need to prepare jar files compiled a core of Sedona `/sedona/geospark-1.3.1.jar` and a scala program for run queries `/sedona/geosparkapplication_2.11-0.1.jar`. In addition, we have also modified Sedona jar file of open source so that obtained during training partitions `/tmp/partitions.csv` can be read externally.
In the last line of `python/config.yaml`, the `--master` address and the path to these jar files must be set appropriately.

### Dataset
You can execute program using the two datasets OSM-US, OSM-SA. Or you can use any dataset that consists of two columns of latitude and longitude.

### GeoSparkApplication_scala
You can create a `.scala` file of the query program in `GeoSparkApplication_scala/src/main/scala/sedona/evaluation/`.
And, run the `sbt package` command to build the `.scala` file. You should install sbt as needed.
`GeoSparkApplication_scala/target/scala-2.11/geosparkapplication_2.11-0.1.jar` will be generated and placed in `./sedona/` to be applied to the main training program.
If you change the data set specified in `.scala` file, specify the same data set in `pyhton/config.yaml`.
