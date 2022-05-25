# Deep Reinforcement Learning for Spatial Data Partitioning
A framework for applying deep reinforcement learning to spatial data partitioning.

### Requirements
- JDK 1.8
- Spark 2.3.4
- Hadoop 2.7.7
- GeoSpark 1.3.1 (Apache Sedona)

### Run
Execute the following commands on the master node (or locally).
```
$ cd source
$ python python/main.py
```

To run main-training, you need to prepare jar files compiled a core of Sedona (GeoSpark) and a scala program for run queries. In addition, we also modified Sedona jar file so that obtained during training partitions (/tmp/partitions.csv) can be read externally.
In the last line of python/config.yaml, the --master address and the path to the jar file must be set appropriately. 
The supplementary file does not include these files due to the space available for submission.