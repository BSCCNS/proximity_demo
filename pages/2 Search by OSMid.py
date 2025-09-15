import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium
from utils.proximity import area
import pydeck as pdk
import numpy as np
import os


##### Streamlit configuration #####
st.set_page_config(
    page_title="Proximity to amenities",
    page_icon=":city_sunset:",
    layout="wide",
)

st.logo('images/Logo_blue.png', 
        size ='large',
        icon_image = 'images/favicon.png')


poi_options = ["Libraries", 'Hospitals', 'Supermarkets']
poi_tags = {
    "Libraries": {"amenity": "library"},
    "Hospitals": {"healthcare" : "hospital"},
    'Supermarkets': {"shop": "supermarket"}
}


                    
n_quantiles = 5
speed = 1.39

alpha = 150
#######################################
st.markdown(
        """
        <div style="display:flex;align-items:center;height:100%;">
            <h1 style="color:#003c74;margin:0;">Search by OSMid</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource(show_spinner=False)
def get_area_network(osmid):
    prox_value = {}
    pr = area(osmid)
    barycenter = pr.gdf.union_all().centroid
    for poi in poi_options:
        pr.download_pois(poi, tags=poi_tags[poi])
        prox_value[poi] = pr.compute_proximity(poi)
        if prox_value[poi].empty:
            print(f'No pois of type {poi}')
        else:
            prox_value[poi]["time"] = prox_value[poi]["distance"]/(60*speed)
            prox_value[poi]["distance"] = prox_value[poi]["distance"].round(2)
            prox_value[poi]["time"] = prox_value[poi]["time"].round(2)
    
    return pr, barycenter.y, barycenter.x, prox_value

osmid = st.text_input("OSMid of the desired area")
poi = st.selectbox("Select a POI category", options=poi_options)
cb = st.toggle('Use custom bin values')

if cb:
    custom_bins = [st.number_input(f"Enter time range {i}") for i in range(4)]
    custom_bins = [0] + custom_bins + [custom_bins[-1]+60]
else:
    custom_bins = [0, 15, 30, 60, 120]

com = st.button("Compute", type="primary")

if com and not osmid:
    st.markdown('Please, type the OSMid of an area.')


if osmid and com:  

    with st.spinner("Downloading routing network..."):
        pr, lat, lon, prox_value = get_area_network(osmid)
    
    st.markdown('Network already cached in memory. Future computations will be much faster until you reload the page.')
        
    st.markdown('## Proximity Results')
    
        
    
    if poi and not prox_value[poi].empty:
        with st.spinner("Loading visualization"):
            prox = prox_value[poi]
            prox= prox.copy()
            prox["uid"] = prox.index.astype(str)  # convert index to string
            

            # Convert geometries to latitude/longitude lists for PyDeck
            def geom_to_coords(geom):
                if geom is None:
                    return []
                if geom.geom_type == "Polygon":
                    return [list(coord) for coord in geom.exterior.coords]
                elif geom.geom_type == "MultiPolygon":
                    return [[list(coord) for coord in poly.exterior.coords] for poly in geom.geoms]
                else:
                    return []

            prox["coordinates"] = prox["geometry"].apply(geom_to_coords)
            
            # Prepare data for PyDeck (flatten MultiPolygons)
            rows = []
            for _, row in prox.iterrows():
                coords = row["coordinates"]
                if not coords:  # skip empty geometries
                    continue
                if isinstance(coords[0][0], list):  # MultiPolygon
                    for poly in coords:
                        rows.append({"uid": row["uid"], "time": row["time"], "polygon": poly})
                else:  # Single Polygon
                    rows.append({"uid": row["uid"], "time": row["time"], "polygon": coords})

            df = pd.DataFrame(rows)

            # Assign each "time" value to a bin index (1 to len(bins)-1)
            df["quantile_bin"] = np.digitize(df["time"], bins=custom_bins[1:-1], right=True)



            # Define a color palette (one color per quantile)

            colors = [
                [220, 240, 255, alpha],  # very light blue
                [170, 210, 255, alpha],  # light blue
                [120, 180, 255, alpha],  # medium blue
                [70, 140, 255, alpha],   # darker blue
                [30, 90, 255, alpha],    # dark blue
            ]

            # Map quantile bins to colors
            df["fill_color"] = df["quantile_bin"].apply(lambda x: colors[x])
            # Polygons
            df["tooltip_text"] = df["time"].apply(lambda d: f"Time: {d:.2f} min" if pd.notnull(d) else "")

            
            
            # Map quantile bins to colors
            df["fill_color"] = df["quantile_bin"].apply(lambda x: colors[x])
            # Polygons
            df["tooltip_text"] = df["time"].apply(lambda d: f"Time: {d:.2f} min" if pd.notnull(d) else "")
            # PyDeck PolygonLayer
            polygon_layer = pdk.Layer(
                "PolygonLayer",
                df,
                get_polygon="polygon",
                get_fill_color="fill_color",
                get_line_color=[0, 0, 0, 0], 
                pickable=True,
                auto_highlight=True,
                getTooltip="tooltip_text"
            )





            points_df = pd.DataFrame({
                "lon": pr.pois_dic[poi].geometry.centroid.x,
                "lat": pr.pois_dic[poi].geometry.centroid.y,
                # optional: any other columns you want, e.g., "name"
                "name": pr.pois_dic[poi].get("name", "")
            })
            # Points
            points_df["tooltip_text"] = points_df["name"].fillna("").astype(str)
            
            
            
            
            point_layer = pdk.Layer(
                "ScatterplotLayer",
                points_df,
                get_position=["lon", "lat"],
                get_color=[255, 0, 0, 180],  
                get_radius=30,                
                pickable=True,
                getTooltip="tooltip_text"
            )


            # Initialize PyDeck map
            view_state = pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=11,
                pitch=10
            )
            
            
            r = pdk.Deck(
                layers=[polygon_layer, point_layer],
                initial_view_state=view_state,
                map_provider = 'carto',
                map_style='road',
                tooltip={"text": "{tooltip_text}"},
            )

            st.pydeck_chart(r,use_container_width=False, height=600)
            
           
            for i, color in enumerate(colors[:len(custom_bins)-1]):  # one fewer range than bins
                st.markdown(
                    f"<div style='display:inline-block;width:30px;height:20px;"
                    f"background-color:rgba{tuple(color)}'></div> "
                    f"{custom_bins[i]:.0f} - {custom_bins[i+1]:.0f} min",
                    unsafe_allow_html=True
                )
         
                    
    else:
        st.markdown(f'### No POIs of type {poi} found in OSM. Visualization of proximity is not available.')