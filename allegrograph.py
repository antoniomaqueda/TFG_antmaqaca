from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
from franz.openrdf.connect import ag_connect
from franz.openrdf.rio.rdfformat import RDFFormat
from franz.openrdf.query.query import QueryLanguage
from geopy.distance import geodesic
from shapely.geometry import Point
from rtree import index

def ag_connect_repo(repo_name, conn_args):
    return ag_connect(repo_name, **conn_args)

# Mantener un registro de las conexiones vistas
seen_connections = {}

# Función para crear y actualizar un sensor en AllegroGraph
def check_and_update_sensor(conn, sensor_uri, longitude, latitude, traffic_flow, window_start, window_end):
    g = Graph()
    ex = Namespace("http://www.example.com/traffic#")

    # Crear una URI única para cada entrada histórica basada en la ventana de tiempo
    traffic_entry_uri = URIRef(f"{sensor_uri}/entry_{window_start.isoformat()}")

    print(f"Creating new traffic entry for sensor {sensor_uri} at {window_start}")

    g.add((traffic_entry_uri, RDF.type, ex.TrafficEntry))
    g.add((traffic_entry_uri, ex.belongsTo, sensor_uri))
    g.add((traffic_entry_uri, ex.trafficFlow, Literal(traffic_flow, datatype=XSD.float)))
    g.add((traffic_entry_uri, ex.windowStart, Literal(window_start, datatype=XSD.dateTime)))
    g.add((traffic_entry_uri, ex.windowEnd, Literal(window_end, datatype=XSD.dateTime)))

    # Añadir o actualizar las propiedades del sensor
    query = f"""
    PREFIX ex: <http://www.example.com/traffic#>
    ASK WHERE {{
        <{sensor_uri}> a ex:Sensor .
    }}
    """
    boolean_query = conn.prepareBooleanQuery(QueryLanguage.SPARQL, query)
    result = boolean_query.evaluate()

    if not result:
        print(f"Sensor {sensor_uri} does not exist. Creating new sensor...")
        g.add((sensor_uri, RDF.type, ex.Sensor))
        g.add((sensor_uri, ex.longitude, Literal(longitude, datatype=XSD.float)))
        g.add((sensor_uri, ex.latitude, Literal(latitude, datatype=XSD.float)))

    rdf_data = g.serialize(format="turtle")
    conn.addData(rdf_data, RDFFormat.TURTLE)

# Función para crear conexiones entre sensores basadas en la proximidad a las calles
def create_connections_by_streets(conn, sensors, edges, threshold_distance=100):
    ex = Namespace("http://www.example.com/traffic#")

    sensor_idx = index.Index()
    for i, sensor in enumerate(sensors):
        sensor_idx.insert(i, sensor["geometry"].bounds)

    for edge in edges.itertuples():
        possible_matches = list(sensor_idx.intersection(edge.geometry.bounds))
        for i in possible_matches:
            for j in possible_matches:
                if i >= j:
                    continue
                sensor1 = sensors[i]
                sensor2 = sensors[j]

                distance1 = edge.geometry.distance(sensor1["geometry"])
                distance2 = edge.geometry.distance(sensor2["geometry"])

                if distance1 < threshold_distance and distance2 < threshold_distance:
                    print(f"Connecting sensors {sensor1['uri']} and {sensor2['uri']} based on proximity to street segment.")
                    print(f"  - Distance from {sensor1['uri']} to street: {distance1} meters")
                    print(f"  - Distance from {sensor2['uri']} to street: {distance2} meters")
                    # Crear las conexiones en AllegroGraph
                    create_edge(conn, sensor1, sensor2, ex)
                    seen_connections[(sensor1['uri'], sensor2['uri'])] = True

# Crear conexioens según el dataset en GeoJSON
def create_edge(conn, sensor1, sensor2, ex):
    g = Graph()
    sensor1_uri = sensor1["uri"]
    sensor2_uri = sensor2["uri"]

    edge_uri = URIRef(ex[f"edge_{sensor1['longitude']}_{sensor1['latitude']}_to_{sensor2['longitude']}_{sensor2['latitude']}"])
    g.add((edge_uri, RDF.type, ex.Connection))
    g.add((edge_uri, ex.source, sensor1_uri))
    g.add((edge_uri, ex.target, sensor2_uri))

    rdf_data = g.serialize(format="turtle")
    conn.addData(rdf_data, RDFFormat.TURTLE)
    print(f"Created connection between {sensor1_uri} and {sensor2_uri} in AllegroGraph.")

# Función para garantizar que todos los sensores estén conectados al más cercano
def ensure_all_sensors_connected(conn, sensors, ex):
    for sensor in sensors:
        nearest_sensor = find_nearest_sensor(sensor, sensors)
        if nearest_sensor:
            if (sensor['uri'], nearest_sensor['uri']) not in seen_connections and (nearest_sensor['uri'], sensor['uri']) not in seen_connections:
                create_edge(conn, sensor, nearest_sensor, ex)
                seen_connections[(sensor['uri'], nearest_sensor['uri'])] = True

# Si no hay alguno suficientemente cerca según el dataset, coger el más cercano para que todos los nodos siempre estén conectados
def find_nearest_sensor(sensor, sensors, threshold_distance=100):
    nearest_sensor = None
    min_distance = float('inf')

    for other_sensor in sensors:
        if sensor['uri'] == other_sensor['uri']:
            continue

        distance = geodesic((sensor['latitude'], sensor['longitude']), (other_sensor['latitude'], other_sensor['longitude'])).meters
        if distance < min_distance:
            min_distance = distance
            nearest_sensor = other_sensor

    return nearest_sensor

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

    create_connections_by_streets(conn, sensors, edges)

    ensure_all_sensors_connected(conn, sensors, ex)

    print("Batch processed and data added to AllegroGraph.")
