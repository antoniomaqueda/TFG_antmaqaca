import osmnx as ox
import geopandas as gpd

print("Descargando la red de calles de Newcastle upon Tyne...")
# Descargar la red de calles para Newcastle upon Tyne
place_name = "Newcastle upon Tyne, England"
G = ox.graph_from_place(place_name, network_type='drive')

print("Procesando la red de calles...")
# Convertir la red de calles a un GeoDataFrame
edges = ox.graph_to_gdfs(G, nodes=False)

print("Filtrando los campos problemáticos...")
columns_to_keep = [col for col in edges.columns if edges[col].dtype != 'object']
edges = edges[columns_to_keep]

print("Guardando la red de calles en un archivo...")
# Guardar el GeoDataFrame en un archivo
edges.to_file("newcastle_streets.geojson", driver="GeoJSON")

print("Proceso completado. Los datos han sido guardados en 'newcastle_streets.geojson'.")
