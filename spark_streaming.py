from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window, sum, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import geopandas as gpd
from allegrograph import ag_connect_repo, create_and_update_rdf_graph
from influxDB import write_traffic_flow_to_influxdb

def start_spark_streaming():
    # Crear una sesión de Spark
    spark = SparkSession.builder \
        .appName("RealTimeFluxAnalysis") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
        .config("spark.sql.streaming.checkpointLocation", "checkpoints") \
        .getOrCreate()

    # Definir el esquema del JSON
    schemaJSON = StructType([
        StructField("x", DoubleType(), True),
        StructField("y", DoubleType(), True),
        StructField("val", DoubleType(), True),
        StructField("units", StringType(), True),
        StructField("reading", StringType(), True),
        StructField("dt", StringType(), True),
        StructField("color", StringType(), True)
    ])

    # Leer el flujo de datos de Kafka
    dfKafka = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", "sensores") \
        .option("startingOffsets", "latest") \
        .load()

    # Convertir el valor de Kafka a JSON
    df_json = dfKafka.selectExpr("CAST(value AS STRING)")

    # Aplicar el esquema al JSON
    sensors_df = df_json.withColumn("data", from_json(col("value"), schemaJSON)).select("data.*")

    # Renombrar las columnas
    sensors_df = sensors_df.withColumnRenamed("x", "longitude") \
        .withColumnRenamed("y", "latitude") \
        .withColumnRenamed("val", "value") \
        .withColumnRenamed("units", "unit") \
        .withColumnRenamed("reading", "category") \
        .withColumnRenamed("dt", "timestamp")

    # Convertir el campo dt a formato de timestamp de Spark
    sensors_df = sensors_df.withColumn("timestamp", col("timestamp").cast("timestamp"))

    # Filtrar las categorías de interés
    filtered_df = sensors_df.filter((col("category") == "Plates In") | (col("category") == "Plates Out"))

    # Crear columnas separadas para Plates In y Plates Out
    filtered_df = filtered_df.withColumn("plates_in", when(col("category") == "Plates In", col("value")).otherwise(0))
    filtered_df = filtered_df.withColumn("plates_out", when(col("category") == "Plates Out", col("value")).otherwise(0))

    # Calcular el flujo total de vehículos por minuto
    traffic_flow_df = filtered_df.withWatermark("timestamp", "3 minutes") \
        .groupBy(window(filtered_df.timestamp, "1 minute"), filtered_df.longitude, filtered_df.latitude) \
        .agg(sum("plates_in").alias("total_plates_in"), sum("plates_out").alias("total_plates_out")) \
        .withColumn("traffic_flow", col("total_plates_in") - col("total_plates_out"))

    # Seleccionar solo las columnas esenciales
    essential_df = traffic_flow_df.select("window", "longitude", "latitude", "traffic_flow")

    # Cargar la Red de Calles desde el Archivo Guardado
    edges = gpd.read_file("newcastle_streets.geojson")

    # Conectar a AllegroGraph
    conn_args = {
        "host": "localhost",
        "port": 10035,
        "user": "antoniomaqueda",
        "password": "#SASAsasa22",
        "catalog": "",
        "create": True
    }
    conn = ag_connect_repo("traffic", conn_args)

    # Procesar cada batch de datos para AllegroGraph y InfluxDB
    def process_batch_for_allegro(df, epoch_id):
        create_and_update_rdf_graph(df, conn, edges)

    def process_batch_for_influx(df, epoch_id):
        write_traffic_flow_to_influxdb(df, epoch_id)

    # Escribir el flujo de datos en AllegroGraph
    essential_df.writeStream \
        .foreachBatch(process_batch_for_allegro) \
        .outputMode("update") \
        .start()

    # Escribir el flujo de datos en InfluxDB
    traffic_flow_df.writeStream \
        .foreachBatch(process_batch_for_influx) \
        .outputMode("update") \
        .start() \
        .awaitTermination()