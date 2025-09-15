
import numpy as np
import pandas as pd
import geopandas as gpd
import pandana
import os
import osmnx as ox
from tobler.util import h3fy
from shapely.validation import explain_validity
from shapely.geometry import Point, LineString, Polygon
import h3
from sklearn.neighbors import BallTree

####################################################
####################################################
class area:
    def __init__(
        self,
        n_string,
        by_osmid = True,
    ):
        self.n_string = n_string
        self.by_osmid = by_osmid
        
        self.get_gdf()
        self.download_network()
        
        self.pois_dic = {}
        self.pois = []
        
    def get_gdf(self):
        self.gdf = ox.geocoder.geocode_to_gdf(self.n_string, by_osmid=self.by_osmid)
        print('Boundary geodataframe downloaded')
        
        
    def download_network(self):
        self.network, self.grid, self.polygon = get_network(self.gdf)
        
    def download_pois(self, name , tags):
        try:
            po = ox.features.features_from_polygon(self.polygon, tags)
            po['lon']=po.geometry.centroid.x
            po['lat']=po.geometry.centroid.y
            self.pois_dic[name] = po.reset_index(level=0, drop=True)
            self.pois = self.pois_dic.keys()
        except:
            print('No POIs of type {tags}')

        
        
    def compute_proximity(self, poi):
        if poi not in self.pois:
            return gpd.GeoDataFrame()
        else:
            df = get_proximity_values(self.network, self.pois_dic[poi],'distance', n_items = 1, maxdist = 10000, zoom=10)
            df = self.grid.merge(df, how='left', left_on='hex_id', right_index=True,)
            df = gpd.GeoDataFrame(df, geometry='geometry', crs = self.gdf.crs)
            df = df[~df.index.duplicated(keep='first')]
            df = interpolate_missing_values(df, 'distance')
            return df

        

               
               
               
###########################################################
###########################################################               
def get_network(gdf):
    """
    Downloads the street network from OpenStreetMap for a given GeoDataFrame and returns a Pandana network object.
    """

    print('Downloading network')
    polygon = gdf.geometry.unary_union

    # Try to fix invalid geometries
    if not polygon.is_valid:
        print(f"Invalid geometry: {explain_validity(polygon)}")
        polygon = polygon.buffer(0)

    if not polygon.is_valid:
        raise ValueError("Polygon geometry is still invalid after fixing.")
    

    try:  
        graph = ox.graph_from_polygon(polygon, network_type = 'walk')
    except ValueError as e:
        if "Found no graph nodes within the requested polygon" in str(e):
            print(f"No graph nodes found. Skipping.")
            return None, None
        else:
            print(f"Error downloading network for: {e}")
            return None, None


    boundary = h3fy(gdf,10)
    node_positions = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}
    node_positions_df = pd.DataFrame.from_dict(node_positions, orient='index', columns=['x', 'y'])

    _from = [u for u, v, _ in graph.edges]
    _to = [v for u, v, _ in graph.edges]
    _distance = [data['length'] for u, v, data in graph.edges(data=True)]
    edges_df = pd.DataFrame({'from': _from, 'to': _to, 'distance': _distance})

    network = pandana.Network(
                            node_positions_df['x'],
                            node_positions_df['y'],
                            edges_df['from'],
                            edges_df['to'],
                            edges_df[['distance']]
                            )

    return network, boundary, polygon


def get_proximity_values(network, pois, tag, n_items, maxdist, zoom=10):
    """
    Calculates proximity values from a network to a set of points of interest (POIs) and assigns results to H3 hexagons.
    Args:
        network: A network object with nodes and methods for setting and querying POIs.
        pois: A DataFrame or similar object containing POI coordinates with 'lon' and 'lat' attributes.
        tag (str): Category or label for the POIs.
        n_items (int): Number of nearest POIs to consider for each node.
        maxdist (float): Maximum search distance for POIs.
        zoom (int, optional): H3 resolution level for hexagon assignment. Defaults to 10.
    Returns:
        pandas.DataFrame: A DataFrame indexed by 'hex_id' (H3 cell), containing the mean proximity value for each hexagon.
    """

    nnodes = get_gdf(network.nodes_df)
    bbox= nnodes.total_bounds


    network.set_pois(category = tag,
                 maxdist = maxdist,
                 maxitems = n_items,
                 x_col = pois.lon, 
                 y_col = pois.lat)

    results = network.nearest_pois(distance = maxdist,
                               category = tag,
                               num_pois = n_items,
                               include_poi_ids = True)

    results[tag] = results[list(range(1,n_items+1))].mean(axis=1)
    results = results.merge(nnodes, left_index=True, right_index=True)
    results['hex_id'] = results.apply(lambda row: h3.latlng_to_cell(row['y'], row['x'], zoom), axis=1).values

    return results[['hex_id',tag]].set_index('hex_id')

def get_gdf(df):
    """
    Converts a DataFrame with 'x' and 'y' columns into a GeoDataFrame with Point geometries.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least 'x' and 'y' columns representing coordinates.
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with a 'geometry' column of shapely Point objects and CRS set to 'EPSG:4326'.
    """

    nnodes = df.copy()
    nnodes['geometry'] = nnodes.apply(lambda row: Point(row['x'], row['y']), axis=1) 
    nnodes = gpd.GeoDataFrame(nnodes, geometry='geometry', crs='EPSG:4326')
    return nnodes


def interpolate_missing_values(idf, column, k = 4):
    """
    Interpolates missing values in a specified column of a GeoDataFrame using k-nearest neighbors.

    This function fills missing values in a specified column by using the k-nearest neighbors algorithm. 
    It computes the distance between the missing data points and the points with available values, 
    then assigns the weighted average of the nearest neighbors' values to the missing data points.

    Args:
        idf (GeoDataFrame): The input GeoDataFrame containing the data to interpolate.
        column (str): The column in which missing values should be interpolated.
        k (int, optional): The number of nearest neighbors to consider for interpolation. Defaults to 4.

    Returns:
        GeoDataFrame: The original GeoDataFrame with missing values in the specified column filled.

    Side Effects:
        - Computes the centroid of the geometries for spatial distance calculations.
        - Modifies the input GeoDataFrame by filling missing values in the specified column.

    Notes:
        - The function assumes the GeoDataFrame uses a valid coordinate reference system (CRS).
        - The function uses the EPSG:3035 CRS for centroid calculation and then converts back to EPSG:4326.
        - The interpolation method uses Euclidean distance to find the nearest neighbors.

    """
    df = idf.copy()
    df['center'] = df.to_crs(epsg=3035).geometry.centroid
    df['center'] = df['center'].to_crs(epsg=4326)
    df_nn = df[df[column].isna() == False]
    df_yn = df[df[column].isna() == True]

    # Early exit if no missing values
    if df_yn.empty:
        return df.drop(['center'], axis=1)

    # Raise if no known values are available
    if len(df_nn) == 0:
        raise ValueError("No known values available for interpolation.")

    # Adjust k to the number of known points
    k = min(k, len(df_nn))

    mobis_coords = np.array([[point.x, point.y] for point in df_nn['center']])
    eval_coords = np.array([[point.x, point.y] for point in df_yn['center']])

    
    tree = BallTree(mobis_coords, metric='euclidean') # Number of nearest neighbors
    distances, indices = tree.query(eval_coords, k=k)
    epsilon = 1e-10
    weights = 1 / (distances + epsilon)
    weights /= weights.sum(axis=1, keepdims=True)
    vals = df_nn[column].to_numpy()
    neighbor = vals[indices] 
    avg = (neighbor * weights).sum(axis=1)


    df_yn.loc[:, column] = avg 
    dfo = pd.concat([df_nn, df_yn])
    dfo = dfo.drop(['center'], axis = 1)
    return dfo