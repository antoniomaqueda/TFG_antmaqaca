# Trabajo de Fin de Grado: Análisis de datos en streaming para IoT

Este TFG desarrolla un pipeline de procesamiento en tiempo real para analizar y predecir el flujo de tráfico urbano utilizando datos de sensores, integrando tecnologías como Kafka, Spark Streaming, AllegroGraph, PyTorch Geometric y InfluxDB.

En este repositorio se encuentra el código de cada parte del pipeline, de momento hasta la parte de ML.

Para iniciar el pipeline se ha creado un fichero (main.py) para simplemente ejecutarlo y todos los demás se ejecutarán tras de sí.

## 1. Ingesta de Datos (kafka_websocket.py)

### Descripción
Recibe datos en tiempo real desde un WebSocket y los publica en un tópico de Kafka.

### Código Relevante

```python

# Configuración del productor de Kafka
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Función principal para iniciar el WebSocket
def start_websocket():
    websocket_url = "wss://livesocket.dev.urbanobservatory.ac.uk/map"
    ws = WebSocketApp(websocket_url,
                      on_message=on_message,
                      on_open=on_open,
                      on_close=on_close,
                      on_error=on_error)

    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    while True:
        time.sleep(10)
```

## 2. Procesamiento de Datos y Cálculo del Flujo de Tráfico (spark_streaming.py)

### Descripción

Se procesan los datos en tiempo real con ventanas de tiempo, donde se agrupan por ventanas de tiempo para calcular el número de vehículos que entran y salen de cada área.
### Código Relevante

```python
# Leer el flujo de datos de Kafka y procesarlo
dfKafka = spark.readStream.format("kafka").option("subscribe", "sensores").load()
sensors_df = dfKafka.selectExpr("CAST(value AS STRING)").withColumn("data", from_json(col("value"), schemaJSON)).select("data.*")

# Filtrar categorías relevantes y calcular el flujo de tráfico
filtered_df = sensors_df.filter((col("category") == "Plates In") | (col("category") == "Plates Out"))

traffic_flow_df = filtered_df.withWatermark("timestamp", "3 minutes") \

    .groupBy(window(filtered_df.timestamp, "1 minute"), filtered_df.longitude, filtered_df.latitude) \
    
    .agg(sum(when(col("category") == "Plates In", col("value")).otherwise(0)).alias("total_plates_in"), 
         sum(when(col("category") == "Plates Out", col("value")).otherwise(0)).alias("total_plates_out")) \
    
    .withColumn("traffic_flow", col("total_plates_in") - col("total_plates_out"))
```

## 3. Almacenamiento de Resultados para Estadísticas (influxDB.py)

### Descripción

Los resultados del procesamiento del flujo de tráfico se almacenan en InfluxDB, una base de datos de series temporales optimizada para la consulta y análisis de datos en tiempo real.### Código Relevante

```python
# Función para enviar datos a InfluxDB
def write_traffic_flow_to_influxdb(df, epoch_id):
    points = []
    for row in df.collect():
        
        point = Point("traffic_flow_data") \
            .tag("longitude", str(row['longitude'])) \
            .tag("latitude", str(row['latitude'])) \
            .field("total_plates_in", row['total_plates_in']) \
            .field("total_plates_out", row['total_plates_out']) \
            .field("traffic_flow", row['traffic_flow']) \
            .time(row['window'].start, WritePrecision.NS)
        
        points.append(point)
        
        print(f"Point: {point.to_line_protocol()}")
        
    write_api.write(bucket="flujoBucket", org=org, record=points)
```

## 4. Creación y actualización del grafo (allegrograph.py)

### Descripción

Se crea un grafo en AllegroGraph donde un nodo es un sensor, y se conecta a otros nodos según proximidad. 
Entre otros, se usa un dataset GIS (saveGIS.py) para la conectividad; además de asegurar en cada batch que todos los sensores están conectados.
En cada sensor encontaremos el histórico de datos de tráfico para cada uno.

```python
# Función principal para procesar cada batch de datos y actualizar el grafo
def create_and_update_rdf_graph(df, conn, edges):
    print("Processing new batch...")
    ex = Namespace("http://www.example.com/traffic#")
    sensors = []

    for row in df.collect():
        sensor_uri = URIRef(ex[f"sensor_{row['longitude']}_{row['latitude']}"])
        sensors.append({
            "uri": sensor_uri,
            "longitude": row['longitude'],
            "latitude": row['latitude'],
            "traffic_flow": row['traffic_flow'],
            "geometry": Point(row['longitude'], row['latitude'])
        })
        check_and_update_sensor(conn, sensor_uri, row['longitude'], row['latitude'], row['traffic_flow'], row['window'].start, row['window'].end)

    # Crear conexiones entre sensores basadas en la proximidad a las calles
    create_connections_by_streets(conn, sensors, edges)

    # Asegurarse de que todos los sensores estén conectados al más cercano
    ensure_all_sensors_connected(conn, sensors, ex)

    print("Batch processed and data added to AllegroGraph.")
```


## TODO: Parte de machine learning, en proceso