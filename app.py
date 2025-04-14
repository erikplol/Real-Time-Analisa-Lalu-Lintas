from pathlib import Path
import PIL
import streamlit as st
import utils.settings as settings
import utils.helper as helper

# Setting page layout
st.set_page_config(
    page_title="Traffic Analytics System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Traffic Analytics using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Load Pre-trained ML Model
try:
    model = helper.load_model(settings.DETECTION_MODEL_COLDWATER)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {settings.DETECTION_MODEL_COLDWATER}")
    st.error(ex)

st.sidebar.header("Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

if source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.COLDWATER:
    source = settings.COLDWATER_YT_LINK
    helper.play_youtube_video(confidence, model, source)

else:
    st.error("Please select a valid source type!")
