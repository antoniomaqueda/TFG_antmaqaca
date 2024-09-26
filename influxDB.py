from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Configuración de conexión a InfluxDB
influx_token = "WuqcGSTVwaxrMzLA4vv2bxQvh890jd_q8rNpUU_Xh-z0McqQLViu1z4yCw7Iv_HBcgJEhqzlY3Xt9SIzal4-dg=="
org = "US"
url = "http://localhost:8086"
client = InfluxDBClient(url=url, token=influx_token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Función para enviar datos a InfluxDB, sustituir bucket por el que se vaya a usar en la interfaz
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

