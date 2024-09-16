# Trabajo de Fin de Grado: Análisis de datos en streaming para IoT

Este TFG desarrolla un pipeline de procesamiento en tiempo real para analizar y predecir el flujo de tráfico urbano utilizando datos de sensores, integrando tecnologías como Kafka, Spark Streaming, AllegroGraph, PyTorch Geometric y InfluxDB.

En este repositorio se encuentra el código de cada parte del pipeline, de momento hasta la parte de ML.

Para usar el pipeline se ha debe:
1. Ejecutar "main.py" para la parte de alimentar el grafo (Kafka -> Spark -> Influx / Allegro).
2. Ejecutar "traffic_prediction_gnn.py" para entrenar el modelo y realizar predicciones.
3. Ejecutar "heatmap.py" para actualizar los datos del mapa.

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

## 5. Definición y uso del modelo Machine Learning (traffic_prediction_gnn.py)

### Descripción
Extraemos los datos necesarios del grafo de AllegroGraph y creamos un modelo que combina:
- LSTM (Long Short-Term Memory): captura patrones temporales en los datos
- GCNConv: se centra en las relaciones espaciales de los sensores.

Entrenamos y ajustamos el modelo y posteriormente se crea un CSV con las predicciones y varias gráficas de rendimiento.

### Código Relevante

```python

# Función para extraer datos de sensores y lecturas de tráfico desde AllegroGraph
def extract_sensor_data_from_allegrograph():
    query = """
    SELECT DISTINCT ?sensor ?longitude ?latitude ?trafficFlow ?windowStart
    WHERE {
        ?sensor a <http://www.example.com/traffic#Sensor> ;
                <http://www.example.com/traffic#longitude> ?longitude ;
                <http://www.example.com/traffic#latitude> ?latitude .

        ?entry a <http://www.example.com/traffic#TrafficEntry> ;
                <http://www.example.com/traffic#belongsTo> ?sensor ;
                <http://www.example.com/traffic#trafficFlow> ?trafficFlow ;
                <http://www.example.com/traffic#windowStart> ?windowStart .
    }
    ORDER BY ?sensor ?windowStart
    """
    
# Función para extraer la estructura del grafo de conexiones entre sensores desde AllegroGraph
def extract_graph_structure_from_allegrograph():
    query = """
    SELECT DISTINCT ?source ?target
    WHERE {
        ?connection a <http://www.example.com/traffic#Connection> ;
                    <http://www.example.com/traffic#source> ?source ;
                    <http://www.example.com/traffic#target> ?target .
    }
    """


# Definición del modelo GNN + LSTM
class TrafficPredictionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=3):
        super(TrafficPredictionGNN, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)

        # FEEDBACK: capas GCNConv secuenciales
        self.gcn_layers = torch.nn.ModuleList()
        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, _ = self.lstm(data.x)
        if x.dim() == 3:
            x = x[:, -1, :]  # Tomar la última salida de LSTM para cada secuencia
        else:
            x = x  # En caso de que x ya sea 2D, no hacer slicing

        for conv in self.gcn_layers[:-1]:
            x = F.relu(conv(x, data.edge_index))
        x = self.gcn_layers[-1](x, data.edge_index)
        return x


```

## 6. Visualización de las predicciones en un mapa (heatmap.py)

### Descripción
Se utiliza el archivo CSV con las predicciones de tráfico generado por el modelo para visualizar la distribución del tráfico predicho en un mapa interactivo.

Algunas funcionalidades del mapa:
- Heatmap con un gradiente de colores que muestra la intensidad del tráfico predicho.
- Se añade un marcador para cada sensor y poder ver su información
- Se puede alternar entre diferentes estilos de mapas.
- Implementa una funcionalidad de búsqueda para encontrar sensores específicos por nombre.

### Código Relevante

```python

# Función para extraer datos de sensores y lecturas de tráfico desde AllegroGraph
# Cargar datos
df = pd.read_csv('/predictions_test.csv')

# Crear el mapa centrado en las coordenadas promedio
map_center = [df['latitude'].mean(), df['longitude'].mean()]
mapa = folium.Map(location=map_center, zoom_start=12, tiles="openstreetmap", name='Mapa Estándar', attr="OpenStreetMap")

# Preparar datos para el heatmap
heat_data = [[row['latitude'], row['longitude'], row['predicted_trafficFlow']] for index, row in df.iterrows()]
max_value = df['predicted_trafficFlow'].max()
min_value = df['predicted_trafficFlow'].min()

# Agregar el heatmap al mapa con escala de colores según el nivel de tráfico
HeatMap(heat_data, min_opacity=0.4, radius=20, max_val=max_value).add_to(mapa)
```