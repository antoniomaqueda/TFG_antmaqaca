import folium
from folium.plugins import HeatMap, MiniMap, MarkerCluster, Search
import pandas as pd

# Cargar datos
df = pd.read_csv('/Users/antoniomaq/Documents/PyCharm/TFG_antmaqaca/predictions_test.csv')

map_center = [df['latitude'].mean(), df['longitude'].mean()]
mapa = folium.Map(location=map_center, zoom_start=12, tiles="openstreetmap", name='Mapa Estándar', attr="OpenStreetMap")

heat_data = [[row['latitude'], row['longitude'], row['predicted_trafficFlow']] for index, row in df.iterrows()]

HeatMap(heat_data, min_opacity=0.4, radius=20).add_to(mapa)

marker_cluster = MarkerCluster().add_to(mapa)


# Función para definir colores de los marcadores en función del tráfico
def get_marker_color(traffic_value):
    if traffic_value < 5.0:
        return 'green'
    elif 5.0 <= traffic_value < 15.0:
        return 'orange'
    else:
        return 'red'


# Agregar marcadores con popups para mostrar detalles
for index, row in df.iterrows():
    popup_text = f"""
    <b>Sensor:</b> {row['sensor']}<br>
    <b>Ubicación:</b> ({row['latitude']}, {row['longitude']})<br>
    <b>Tráfico Predicho:</b> {row['predicted_trafficFlow']}<br>
    """

    marker_color = get_marker_color(row['predicted_trafficFlow'])

    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=marker_color, icon='info-sign')
    ).add_to(marker_cluster)

# Añadir diferentes estilos de mapas con nombres personalizados
folium.TileLayer('cartodb positron', name='Mapa Claro', attr='© OpenStreetMap contributors & CartoDB').add_to(mapa)
folium.TileLayer('cartodb dark_matter', name='Mapa Oscuro', attr='© OpenStreetMap contributors & CartoDB').add_to(mapa)

# Añadir control de capas para alternar entre los estilos de mapas
folium.LayerControl().add_to(mapa)

# Añadir un mini mapa en la esquina inferior derecha
minimap = MiniMap(toggle_display=True)
mapa.add_child(minimap)

df['coords'] = df.apply(lambda row: f"{row['latitude']}, {row['longitude']}", axis=1)

# Crear el control de búsqueda para las coordenadas
search = Search(
    layer=marker_cluster,
    search_label='coords',
    placeholder="Buscar por latitud, longitud",
    collapsed=False
).add_to(mapa)

# Guardar y mostrar el mapa
mapa.save('/Users/antoniomaq/Documents/PyCharm/TFG_antmaqaca/mapa_heatmap.html')
