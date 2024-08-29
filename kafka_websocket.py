import json
import time
from kafka import KafkaProducer
from websocket import WebSocketApp
import ssl

# Configuración del productor de Kafka
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Función que se llama cuando se recibe un mensaje del WebSocket
def on_message(ws, message):
    data = json.loads(message)
    print("Recibido:", data)
    producer.send('sensores', value=data)
    producer.flush()

# Función que se llama cuando se abre la conexión WebSocket
def on_open(ws):
    print("Conexión abierta")

# Función que se llama cuando se cierra la conexión WebSocket
def on_close(ws, close_status_code, close_msg):
    print("Conexión cerrada", close_status_code, close_msg)

# Función que se llama cuando ocurre un error en la conexión WebSocket
def on_error(ws, error):
    print("Error:", error)

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
