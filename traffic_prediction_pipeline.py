import subprocess
from collections import defaultdict
import random

import pandas as pd
from franz.openrdf.connect import ag_connect
from franz.openrdf.query.query import QueryLanguage
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split


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

    conn_args = {
        "host": "localhost",
        "port": 10035,
        "user": "antoniomaqueda",
        "password": "#SASAsasa22",
        "catalog": "",
        "create": True
    }

    data = defaultdict(list)

    with ag_connect('traffic', **conn_args) as conn:
        print("Conexión a AllegroGraph establecida.")
        with conn.prepareTupleQuery(QueryLanguage.SPARQL, query).evaluate() as res:
            for bs in tqdm(res):
                data["sensor"].append(str(bs.getValue("sensor")))
                data["longitude"].append(bs.getValue("longitude").toPython())
                data["latitude"].append(bs.getValue("latitude").toPython())
                data["trafficFlow"].append(bs.getValue("trafficFlow").toPython())
                data["windowStart"].append(bs.getValue("windowStart").toPython())

    df = pd.DataFrame(data=data)
    if df.empty:
        raise ValueError("La consulta no devolvió ningún resultado")

    print("Primeras filas del DataFrame:")
    pd.set_option('display.max_columns', None)
    print(df.head())

    return df


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

    conn_args = {
        "host": "localhost",
        "port": 10035,
        "user": "antoniomaqueda",
        "password": "#SASAsasa22",
        "catalog": "",
        "create": True
    }

    edges = []

    with ag_connect('traffic', **conn_args) as conn:
        print("Conexión a AllegroGraph establecida.")
        with conn.prepareTupleQuery(QueryLanguage.SPARQL, query).evaluate() as res:
            for bs in tqdm(res):
                source = str(bs.getValue("source"))
                target = str(bs.getValue("target"))
                edges.append((source, target))

    print(f"Total de conexiones: {len(edges)}")

    return edges


# Preparar datos para el modelo
def prepare_data(sensor_data, graph_edges, val_split=0.1):
    sensors = sensor_data['sensor'].unique()
    sensor_to_index = {sensor: idx for idx, sensor in enumerate(sensors)}
    index_to_sensor = {idx: sensor for sensor, idx in sensor_to_index.items()}

    features = []
    sequence_lengths = []
    means = []
    stds = []

    real_sequences = []

    for sensor in sensors:
        sensor_history = sensor_data[sensor_data['sensor'] == sensor].sort_values('windowStart')
        sensor_flow = sensor_history['trafficFlow'].values

        mean = np.mean(sensor_flow)
        std_dev = np.std(sensor_flow)

        # Normalización
        if std_dev != 0:
            normalized_flow = (sensor_flow - mean) / std_dev
        else:
            normalized_flow = sensor_flow - mean

        features.append(torch.tensor(normalized_flow, dtype=torch.float))
        sequence_lengths.append(len(sensor_flow))

        means.append(mean)
        stds.append(std_dev)

        real_sequences.append(sensor_flow)

    # Padding
    padded_features = rnn_utils.pad_sequence(features, batch_first=True)

    x = padded_features.unsqueeze(-1)

    numeric_edges = []
    for edge in graph_edges:
        source, target = edge
        if source in sensor_to_index and target in sensor_to_index:
            numeric_edges.append([sensor_to_index[source], sensor_to_index[target]])

    edge_index = torch.tensor(numeric_edges, dtype=torch.long).T

    data = Data(x=x, edge_index=edge_index)
    data.sequence_lengths = sequence_lengths
    data.sensor_to_index = sensor_to_index
    data.index_to_sensor = index_to_sensor
    data.means = means
    data.stds = stds
    data.real_sequences = real_sequences

    # Dividir los sensores en conjuntos de entrenamiento y prueba
    train_sensors, test_sensors = train_test_split(sensors, test_size=0.2, random_state=42)
    train_sensors, val_sensors = train_test_split(train_sensors, test_size=val_split, random_state=42)

    # Crear máscaras
    train_mask = torch.zeros(len(sensors), dtype=torch.bool)
    val_mask = torch.zeros(len(sensors), dtype=torch.bool)
    test_mask = torch.zeros(len(sensors), dtype=torch.bool)

    for sensor in train_sensors:
        train_mask[sensor_to_index[sensor]] = True
    for sensor in val_sensors:
        val_mask[sensor_to_index[sensor]] = True
    for sensor in test_sensors:
        test_mask[sensor_to_index[sensor]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


# Definición del modelo
class TrafficPredictionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=3):
        super(TrafficPredictionGNN, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)

        self.gcn_layers = torch.nn.ModuleList()
        self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_gcn_layers - 2):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn_layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, data):
        x, _ = self.lstm(data.x)
        if x.dim() == 3:
            x = x[:, -1, :]
        else:
            x = x

        for conv in self.gcn_layers[:-1]:
            x = F.relu(conv(x, data.edge_index))
        x = self.gcn_layers[-1](x, data.edge_index)
        return x


# Función de entrenamiento
def train_model(data, val_data, model, optimizer, scheduler, epochs=75):
    model.train()
    losses = []
    val_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.x[:, -1, 0])
        loss.backward()
        optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = F.mse_loss(val_out, val_data.x[:, -1, 0])
        model.train()

        losses.append(loss.item())
        val_losses.append(val_loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    # Graficar las pérdidas
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida durante el Entrenamiento y Validación')
    plt.legend()
    plt.show()


# Función para desnormalizar las predicciones
def denormalize_predictions(predictions, means, stds, sensor_to_index):
    predictions = predictions.detach().numpy()

    denormalized_predictions = np.array([
        pred * stds[idx] + means[idx]
        for idx, pred in enumerate(predictions)
    ])
    return denormalized_predictions


def evaluate_model(sensor_data, predictions, means, stds, sensor_to_index, mask, dataset_name):

    denormalized_predictions = denormalize_predictions(predictions[mask], means, stds, sensor_to_index)

    real_values = np.array([
        sensor_data[sensor_data['sensor'] == sensor]['trafficFlow'].values[-1]
        for sensor in np.array(list(sensor_to_index.keys()))[mask]
    ])

    # Calcular MSE y MAE
    mse = np.mean((denormalized_predictions - real_values) ** 2)
    mae = np.mean(np.abs(denormalized_predictions - real_values))

    print(f"\nEvaluación en el conjunto de {dataset_name}:")
    print(f"Error cuadrático medio (MSE): {mse:.4f}")
    print(f"Error absoluto medio (MAE): {mae:.4f}")

    return mse, mae

# Función para generar una tabla de métricas
def generate_metrics_table(mse_train, mae_train, mse_val, mae_val, mse_test, mae_test):
    metrics_data = {
        'Conjunto': ['Entrenamiento', 'Validación', 'Prueba'],
        'MSE (Error Cuadrático Medio)': [mse_train, mse_val, mse_test],
        'MAE (Error Absoluto Medio)': [mae_train, mae_val, mae_test]
    }
    df_metrics = pd.DataFrame(metrics_data)
    print("\nTabla de métricas de los tres conjuntos:")
    print(df_metrics)
    return df_metrics


# Función para guardar las predicciones en un CSV
def save_predictions_to_csv(sensor_data, predictions, means, stds, sensor_to_index, mask, filename):
    denormalized_predictions = denormalize_predictions(predictions[mask], means, stds, sensor_to_index)

    sensors = np.array(list(sensor_to_index.keys()))[mask]

    longitudes = []
    latitudes = []

    for sensor in sensors:
        sensor_row = sensor_data[sensor_data['sensor'] == sensor]
        longitudes.append(sensor_row['longitude'].values[0])
        latitudes.append(sensor_row['latitude'].values[0])

    df_predictions = pd.DataFrame({
        'sensor': sensors,
        'longitude': longitudes,
        'latitude': latitudes,
        'predicted_trafficFlow': denormalized_predictions.flatten()
    })

    # Guardar el DataFrame en un archivo CSV
    df_predictions.to_csv(filename, index=False)
    print(f"Predicciones guardadas en '{filename}'.")


# Función para graficar resultados reales vs predichos con valores reales en X y predichos en Y
def plot_real_vs_predicted_scatter(sensor_data, predictions, means, stds, sensor_to_index, mask):
    denormalized_predictions = denormalize_predictions(predictions[mask], means, stds, sensor_to_index)
    sensors = np.array(list(sensor_to_index.keys()))[mask]
    real_values = np.array([
        sensor_data[sensor_data['sensor'] == sensor]['trafficFlow'].values[-1]
        for sensor in sensors
    ])

    plt.figure(figsize=(10, 6))

    plt.scatter(real_values, denormalized_predictions, color='green', alpha=0.5)

    plt.plot([min(real_values), max(real_values)], [min(real_values), max(real_values)], 'r--', label="y=x")

    plt.xlabel('Valores Reales')
    plt.ylabel('Valores Predichos')
    plt.title('Comparación de Valores Reales vs Predichos')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Función para graficar el historial de valores reales + el nuevo valor predicho para 9 sensores aleatorios
def plot_time_series_with_prediction(sensor_data, predictions, means, stds, sensor_to_index, mask, num_sensors=9):

    denormalized_predictions = denormalize_predictions(predictions[mask], means, stds, sensor_to_index)
    sensors = np.array(list(sensor_to_index.keys()))[mask]

    valid_sensors = [sensor for sensor in sensors if sensor_to_index[sensor] < len(denormalized_predictions)]

    random_sensors = random.sample(valid_sensors, min(num_sensors, len(valid_sensors)))

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Historial de Flujo de Tráfico + Predicción para 9 Sensores Aleatorios")

    for i, sensor in enumerate(random_sensors):
        ax = axes[i // 3, i % 3]

        sensor_history = sensor_data[sensor_data['sensor'] == sensor].sort_values('windowStart')
        real_traffic = sensor_history['trafficFlow'].values
        time_stamps = pd.to_datetime(sensor_history['windowStart'].values)

        lat = sensor_history['latitude'].values[0]
        lon = sensor_history['longitude'].values[0]

        if len(real_traffic) != len(time_stamps):
            print(f"Error: Longitud de datos real ({len(real_traffic)}) y timestamps ({len(time_stamps)}) no coinciden en sensor {sensor}")
            continue

        sensor_idx = sensor_to_index[sensor]
        pred_traffic = denormalized_predictions[sensor_idx].item()

        ax.plot(time_stamps, real_traffic, label="Real", color='blue', marker='o', markersize=4)

        adjusted_time = time_stamps[-1] + pd.Timedelta(minutes=5)

        ax.scatter(adjusted_time, pred_traffic, label="Predicho", color='red', marker='x', s=100)

        ax.plot([time_stamps[-1], adjusted_time], [real_traffic[-1], pred_traffic], color='blue', linestyle='--')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax.set_title(f"Lat: {lat}, Lon: {lon}")
        ax.set_xlabel("Hora")
        ax.set_ylabel("Flujo de Tráfico")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    # Extraer datos y estructura del grafo
    sensor_data = extract_sensor_data_from_allegrograph()
    graph_edges = extract_graph_structure_from_allegrograph()

    # Preparar datos
    data = prepare_data(sensor_data, graph_edges)

    # Definir el modelo y el optimizador con nuevos hiperparámetros
    input_dim = 1
    hidden_dim = 16  # Mantenemos las unidades de la capa oculta
    output_dim = 1
    num_gcn_layers = 3  # Añadimos múltiples capas GCN

    model = TrafficPredictionGNN(input_dim, hidden_dim, output_dim, num_gcn_layers=num_gcn_layers)

    # Ajustar la tasa de aprendizaje
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Añadir un scheduler que reduce la tasa de aprendizaje a la mitad en el epoch 50
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_model(data, data, model, optimizer, scheduler, epochs=75)

    # Evaluar el modelo en el conjunto de validación
    model.eval()
    with torch.no_grad():
        predictions = model(data)

    print("\nEvaluación en el Conjunto de Entrenamiento:")
    mse_train, mae_train = evaluate_model(sensor_data, predictions, data.means, data.stds, data.sensor_to_index,
                                          data.train_mask, "Entrenamiento")

    print("\nEvaluación en el Conjunto de Validación:")
    mse_val, mae_val = evaluate_model(sensor_data, predictions, data.means, data.stds, data.sensor_to_index,
                                      data.val_mask, "Validación")

    print("\nEvaluación en el Conjunto de Prueba:")
    mse_test, mae_test = evaluate_model(sensor_data, predictions, data.means, data.stds, data.sensor_to_index,
                                        data.test_mask, "Prueba")

    # Generar y mostrar la tabla de métricas
    generate_metrics_table(mse_train, mae_train, mse_val, mae_val, mse_test, mae_test)

    # Guardar predicciones del conjunto de prueba
    save_predictions_to_csv(sensor_data, predictions, data.means, data.stds, data.sensor_to_index, data.test_mask,
                            filename='predictions_test.csv')

    # Graficar resultados reales vs predichos (gráfico de dispersión) para el conjunto de prueba
    plot_real_vs_predicted_scatter(sensor_data, predictions, data.means, data.stds, data.sensor_to_index, data.test_mask)

    # Graficar la secuencia de valores reales + valor predicho para 9 sensores aleatorios
    plot_time_series_with_prediction(sensor_data, predictions, data.means, data.stds, data.sensor_to_index,
                                     data.test_mask)


    print("\nGenerando mapa de calor (heatmap.py)...")
    result_heatmap = subprocess.run(["python", "heatmap.py"], capture_output=True, text=True)

    if result_heatmap.returncode == 0:
        print("Heatmap generado correctamente.")
    else:
        print(f"Error al ejecutar heatmap.py: {result_heatmap.stderr}")


if __name__ == "__main__":
    main()


