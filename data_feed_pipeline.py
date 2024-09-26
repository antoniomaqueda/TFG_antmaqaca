import multiprocessing
from kafka_websocket import start_websocket
from spark_streaming import start_spark_streaming

def main():
    # Ejecutar el productor de Kafka
    kafka_process = multiprocessing.Process(target=start_websocket)
    kafka_process.start()

    # Ejecutar el streaming de Spark -> Influx/Allegro
    start_spark_streaming()


if __name__ == "__main__":
    main()
