import multiprocessing
from kafka_websocket import start_websocket
from spark_streaming import start_spark_streaming

def main():
    # Ejecutar el productor de Kafka en un proceso separado
    kafka_process = multiprocessing.Process(target=start_websocket)
    kafka_process.start()

    # Ejecutar el streaming de Spark en el proceso principal
    start_spark_streaming()


if __name__ == "__main__":
    main()
