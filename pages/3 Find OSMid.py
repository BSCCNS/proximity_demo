import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Nominatin",
    page_icon=":city_sunset:",
    layout="wide",
)

st.logo('images/Logo_blue.png', 
        size ='large',
        icon_image = 'images/favicon.png')


# Example: Embed Wikipedia
components.iframe("https://nominatim.openstreetmap.org/ui/search.html", height=800, scrolling=True)