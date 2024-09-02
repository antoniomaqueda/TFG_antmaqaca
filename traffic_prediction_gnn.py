from collections import defaultdict
import pandas as pd
from franz.openrdf.connect import ag_connect
from franz.openrdf.query.query import QueryLanguage
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

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
def prepare_data(sensor_data, graph_edges):
    sensors = sensor_data['sensor'].unique()
    sensor_to_index = {sensor: idx for idx, sensor in enumerate(sensors)}

    features = []
    sequence_lengths = []
    means = []
    stds = []

    # Crear una estructura para almacenar las secuencias reales
    real_sequences = []

    for sensor in sensors:
        sensor_history = sensor_data[sensor_data['sensor'] == sensor]
        sensor_flow = sensor_history['trafficFlow'].values

        # Calcular media y desviación estándar
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

        # Almacenar la secuencia real
        real_sequences.append(sensor_flow)

    # Padding sequences to the same length
    padded_features = rnn_utils.pad_sequence(features, batch_first=True)

    # Ensure the correct shape for LSTM input
    x = padded_features.unsqueeze(-1)  # Add a third dimension

    # Convert graph_edges to numeric indices
    numeric_edges = []
    for edge in graph_edges:
        source, target = edge
        numeric_edges.append([sensor_to_index[source], sensor_to_index[target]])

    edge_index = torch.tensor(numeric_edges, dtype=torch.long).T

    data = Data(x=x, edge_index=edge_index)
    data.sequence_lengths = sequence_lengths
    data.sensor_to_index = sensor_to_index
    data.means = means
    data.stds = stds
    data.real_sequences = real_sequences
    return data

# Definición del modelo GNN + LSTM
class TrafficPredictionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrafficPredictionGNN, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)  # Añadir Dropout y más capas
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, _ = self.lstm(data.x)
        if x.dim() == 3:
            x = x[:, -1, :]  # Tomar la última salida de LSTM para cada secuencia
        else:
            x = x  # En caso de que x ya sea 2D, no hacer slicing

        x = F.relu(self.conv1(x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        return x

# Función de entrenamiento
def train_model(data, model, optimizer, epochs=100):
    model.train()
    losses = []  # Para almacenar los valores de la pérdida
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        # Ajustar el cálculo de la pérdida
        loss = F.mse_loss(out, data.x[:, -1, 0])  # Comparar solo con el último valor de la secuencia
        loss.backward()
        optimizer.step()

        losses.append(loss.item())  # Guardar el valor de la pérdida
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Graficar la pérdida a lo largo de las épocas
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, label='Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Pérdida durante el Entrenamiento')
    plt.legend()
    plt.show()

# Función para desnormalizar las predicciones
def denormalize_predictions(predictions, means, stds, sensor_to_index):
    # Convertir las predicciones a numpy usando detach()
    predictions = predictions.detach().numpy()

    denormalized_predictions = np.array([
        pred * stds[sensor_to_index[sensor]] + means[sensor_to_index[sensor]]
        for sensor, pred in zip(sensor_to_index.keys(), predictions)
    ])
    return denormalized_predictions

# Función de evaluación del modelo
def evaluate_model(sensor_data, predictions, means, stds, sensor_to_index):
    # Desnormalizar las predicciones
    denormalized_predictions = denormalize_predictions(predictions, means, stds, sensor_to_index)

    # Obtener valores reales para la evaluación
    real_values = np.array([
        sensor_data[sensor_data['sensor'] == sensor]['trafficFlow'].values[-1]
        for sensor in sensor_to_index.keys()
    ])

    # Calcular MSE y MAE
    mse = np.mean((denormalized_predictions - real_values) ** 2)
    mae = np.mean(np.abs(denormalized_predictions - real_values))

    print(f"Error cuadrático medio (MSE): {mse}")
    print(f"Error absoluto medio (MAE): {mae}")

    return mse, mae

# Función para guardar las predicciones en un CSV
def save_predictions_to_csv(sensor_data, predictions, means, stds, sensor_to_index):
    denormalized_predictions = denormalize_predictions(predictions, means, stds, sensor_to_index)

    # Crear un DataFrame con las predicciones
    df_predictions = pd.DataFrame({
        'Sensor': list(sensor_to_index.keys()),
        'PredictedTrafficFlow': denormalized_predictions.flatten()  # Asegúrate de que sea unidimensional
    })

    # Guardar el DataFrame en un archivo CSV
    df_predictions.to_csv('predictions.csv', index=False)
    print("Predicciones guardadas en 'predictions.csv'.")

# Función principal
def main():
    # Extraer datos y estructura del grafo
    sensor_data = extract_sensor_data_from_allegrograph()
    graph_edges = extract_graph_structure_from_allegrograph()

    # Preparar datos
    data = prepare_data(sensor_data, graph_edges)

    # Definir el modelo y el optimizador con nuevos hiperparámetros
    input_dim = 1
    hidden_dim = 16
    output_dim = 1
    model = TrafficPredictionGNN(input_dim, hidden_dim, output_dim)

    # Ajustar la tasa de aprendizaje
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Entrenar el modelo
    train_model(data, model, optimizer, epochs=75)

    # Evaluar el modelo
    model.eval()
    predictions = model(data)
    mse, mae = evaluate_model(sensor_data, predictions, data.means, data.stds, data.sensor_to_index)

    # Guardar predicciones
    save_predictions_to_csv(sensor_data, predictions, data.means, data.stds, data.sensor_to_index)

if __name__ == "__main__":
    main()
