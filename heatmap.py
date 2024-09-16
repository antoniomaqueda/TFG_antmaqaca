import folium
from folium.plugins import HeatMap, MiniMap, MarkerCluster, Search
import pandas as pd

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

# Crear un cluster para los marcadores
marker_cluster = MarkerCluster().add_to(mapa)

# Agregar marcadores con popups para mostrar detalles
for index, row in df.iterrows():
    popup_text = f"""
    <b>Sensor:</b> {row['sensor']}<br>
    <b>Ubicación:</b> ({row['latitude']}, {row['longitude']})<br>
    <b>Tráfico Predicho:</b> {row['predicted_trafficFlow']}<br>
    """
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(marker_cluster)

# Añadir diferentes estilos de mapas con nombres personalizados
folium.TileLayer('cartodb positron', name='Mapa Claro', attr='© OpenStreetMap contributors & CartoDB').add_to(mapa)
folium.TileLayer('cartodb dark_matter', name='Mapa Oscuro', attr='© OpenStreetMap contributors & CartoDB').add_to(mapa)

# Añadir control de capas para alternar entre los estilos de mapas
folium.LayerControl().add_to(mapa)

# Añadir un mini mapa en la esquina inferior derecha
minimap = MiniMap(toggle_display=True)
mapa.add_child(minimap)

# Crear el control de búsqueda para las coordenadas
search = Search(
    layer=marker_cluster,  # Puedes usar los marcadores en un cluster o directamente
    search_label='sensor',
    placeholder="Buscar sensor"
).add_to(mapa)

# Guardar y mostrar el mapa
mapa.save('/Users/antoniomaq/Documents/PyCharm/TFG_antmaqaca/mapa_heatmap.html')
