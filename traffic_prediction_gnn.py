from collections import defaultdict
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

    # Crear una estructura para almacenar las secuencias reales
    real_sequences = []

    for sensor in sensors:
        sensor_history = sensor_data[sensor_data['sensor'] == sensor].sort_values('windowStart')
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

    # FEEDBACK: Dividir los sensores en conjuntos de entrenamiento y prueba
    train_sensors, test_sensors = train_test_split(sensors, test_size=0.2, random_state=42)
    train_sensors, val_sensors = train_test_split(train_sensors, test_size=val_split, random_state=42)

    # Crear máscaras para entrenamiento, validación y prueba
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


# Función de entrenamiento con validación
# Modificación de la función de entrenamiento
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

        # Actualizar el scheduler
        scheduler.step()

        # Evaluación en el conjunto de validación
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = F.mse_loss(val_out, val_data.x[:, -1, 0])
        model.train()

        losses.append(loss.item())
        val_losses.append(val_loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    # Graficar las pérdidas de entrenamiento y validación
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
    # Convertir las predicciones a numpy usando detach()
    predictions = predictions.detach().numpy()

    denormalized_predictions = np.array([
        pred * stds[idx] + means[idx]
        for idx, pred in enumerate(predictions)
    ])
    return denormalized_predictions


# Función de evaluación del modelo
def evaluate_model(sensor_data, predictions, means, stds, sensor_to_index, mask):
    # Desnormalizar las predicciones
    denormalized_predictions = denormalize_predictions(predictions[mask], means, stds, sensor_to_index)

    # Obtener valores reales para la evaluación
    real_values = np.array([
        sensor_data[sensor_data['sensor'] == sensor]['trafficFlow'].values[-1]
        for sensor in np.array(list(sensor_to_index.keys()))[mask]
    ])

    # Calcular MSE y MAE
    mse = np.mean((denormalized_predictions - real_values) ** 2)
    mae = np.mean(np.abs(denormalized_predictions - real_values))

    print(f"Error cuadrático medio (MSE): {mse:.4f}")
    print(f"Error absoluto medio (MAE): {mae:.4f}")

    return mse, mae


# Función para guardar las predicciones en un CSV
def save_predictions_to_csv(sensor_data, predictions, means, stds, sensor_to_index, mask, filename='predictions.csv'):
    denormalized_predictions = denormalize_predictions(predictions[mask], means, stds, sensor_to_index)

    # Crear un DataFrame con las predicciones
    df_predictions = pd.DataFrame({
        'Sensor': np.array(list(sensor_to_index.keys()))[mask],
        'PredictedTrafficFlow': denormalized_predictions.flatten()
    })

    # Guardar el DataFrame en un archivo CSV
    df_predictions.to_csv(filename, index=False)
    print(f"Predicciones guardadas en '{filename}'.")


# FEEDBACK: Función para graficar resultados reales vs predichos
def plot_real_vs_predicted(sensor_data, predictions, means, stds, sensor_to_index, mask):
    denormalized_predictions = denormalize_predictions(predictions[mask], means, stds, sensor_to_index)
    sensors = np.array(list(sensor_to_index.keys()))[mask]
    real_values = np.array([
        sensor_data[sensor_data['sensor'] == sensor]['trafficFlow'].values[-1]
        for sensor in sensors
    ])

    plt.figure(figsize=(15, 7))
    plt.plot(sensors, real_values, label='Reales', marker='o')
    plt.plot(sensors, denormalized_predictions, label='Predichas', marker='x')
    plt.xlabel('Sensor')
    plt.ylabel('Flujo de Tráfico')
    plt.title('Comparación de Flujo de Tráfico Real vs Predicho')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
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

    # **Corregir la llamada a `train_model`** pasando también el scheduler y el optimizer
    train_model(data, data, model, optimizer, scheduler, epochs=75)

    # Evaluar el modelo en el conjunto de validación
    model.eval()
    with torch.no_grad():
        predictions = model(data)

    print("\nEvaluación en el Conjunto de Validación:")
    mse_val, mae_val = evaluate_model(sensor_data, predictions, data.means, data.stds, data.sensor_to_index,
                                      data.val_mask)

    print("\nEvaluación en el Conjunto de Prueba:")
    mse_test, mae_test = evaluate_model(sensor_data, predictions, data.means, data.stds, data.sensor_to_index,
                                        data.test_mask)

    # Guardar predicciones del conjunto de prueba
    save_predictions_to_csv(sensor_data, predictions, data.means, data.stds, data.sensor_to_index, data.test_mask,
                            filename='predictions_test.csv')

    # Graficar resultados reales vs predichos para el conjunto de prueba
    plot_real_vs_predicted(sensor_data, predictions, data.means, data.stds, data.sensor_to_index, data.test_mask)


if __name__ == "__main__":
    main()
