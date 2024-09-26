import json
import time
from kafka import KafkaProducer
from websocket import WebSocketApp
import ssl


producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def on_message(ws, message):
    data = json.loads(message)
    print("Recibido:", data)
    producer.send('sensores', value=data)
    producer.flush()

def on_open(ws):
    print("Conexión abierta")

def on_close(ws, close_status_code, close_msg):
    print("Conexión cerrada", close_status_code, close_msg)

def on_error(ws, error):
    print("Error:", error)

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
