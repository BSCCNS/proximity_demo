import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium
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


col1, col2 = st.columns([1, 9])  # 1:5 width ratio

with col1:
    st.image("images/Proximit.png", width = 80)

with col2:
    st.markdown(
        """
        <div style="display:flex;align-items:center;height:100%;">
            <h1 style="color:#003c74;margin:0;">Proximity Services Demo</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('Every city is a delicate balance of interwoven networks, and improving them is often disruptive for local residents. So, how do we predict those impacts while transforming our cities into better places to live?')  
st.markdown('vCity is an adaptable, data-driven platform to help assess the impact of new interventions on all aspects of city life.')  
st.markdown('This is possible thanks to urban digital twins: virtual replicas of real cities, that optimise data to map the invisible connections between factors such as air quality, public services, or energy efficiency. They allow us to simulate the ripple effect of urban policies over time, making big changes simple for everyone to understand.<div>', unsafe_allow_html=True)

st.markdown('This tool analyses cities through the lens of proximity, visualising how accessible vital public services are to citizens. With this insight, urban policy makers can easily identify the need for different facilities in each neighbourhood.')   
    
rules_container = st.expander("About this app")

with rules_container:
    st.write('''
             This showcase of our proximity model built'in within vCity allows users to examine the accesibility of cities by using data downloaded from [OpenStreetMap](https://www.openstreetmap.org/).
             It was prepared for the *Training course on Digital Twins*, organized by the [MedCities](https://medcities.org/) association in September 2025.
             ''')
    st.markdown('#### Search by city name')
    st.write('''
             In this section, you can compute proximity results for some of the origin cities of the participants in the course. Some of the cities are missing due to to OpenStreetMap not having good quiality available data.
             Here, you can choose a city, a flavor of points of interest, and even a custom set of bins for visualiziong the proximity times.
             All computations are done by using open source tools and open data, and considering a standard walking speed of 1.39 km/s, that of a ~35 years old adult.
             ''')
    
    st.markdown('#### Search by OSMid')
    st.write('''
             In this section, you can compute proximity results by providing the OpenStreetMap ID of an area. You can adjust the same parameters as in the *Search by city* case.
             ''')
    
    st.markdown('#### Find OSMid')
    st.write('''
             In order to simplify finding the OSMid of a given area, we have embeded the nominatin tool in this section. Here, you can search for an area using its natural language denomination, and find its corresponding unique OSMid.
             ''')
    
st.divider()
footer_container = st.container()

# Add content to the footer container
with footer_container:
    col1, coli, col2 = st.columns((3,1,3))
    
    with col1:
        st.markdown(
        """
        <div style="display:flex;align-items:center;height:100%;">
            <h2 style="color:#003c74;margin:0;">About</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
        st.markdown('Application developed by [Míriam Herrero, PhD](https://www.linkedin.com/in/m-herrero-valea/).')
        st.write(
            '''
            This is a technology showcase of [vCity](https://www.vcity.tech/), a human-centric platform for Urban Digital Twins, developed by the [DataViz team](https://www.bsc.es/viz/team/index.html) at the [Barcelona Supercomputing Center](https://www.bsc.es/).
            '''
        )
        
    with col2:
        st.markdown(
        """
        <div style="display:flex;align-items:center;height:100%;">
            <h2 style="color:#003c74;margin:0;">Contact</h2>
        </div>
        """,
        unsafe_allow_html=True
        )
    

        st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"><i class="fa-solid fa-envelope"></i> mherrero@bsc.es', unsafe_allow_html=True)
        
