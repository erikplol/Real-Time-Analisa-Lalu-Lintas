from ultralytics import YOLO
import streamlit as st
import cv2
import utils.settings as settings
import yt_dlp
from utils.coldwater.process import VideoProcessor
import threading
import queue
import time
import pandas as pd
import numpy as np

def load_model(model_path):
    model = VideoProcessor(
        source_weights_path=model_path
    )
    return model

def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'bv[height<=480]+ba/b[height<=480]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict['url']

def play_youtube_video(conf, model, source):
    source_youtube = source

    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Invalid Link. Please check the URL.")
            return

        try:
            st.sidebar.info("Extracting video stream URL...")
            stream_url = get_youtube_stream_url(source_youtube)

            st.sidebar.info("Opening video stream...")
            vid_cap = cv2.VideoCapture(stream_url)

            if not vid_cap.isOpened():
                st.sidebar.error(
                    "Failed to open video stream. Please try a different video.")
                return

            st.sidebar.success("Video stream opened successfully!")
            st_frame = st.empty()

            def preload_frames():
                while True:
                    if frame_queue.qsize() < frame_queue.maxsize:
                        success, image = vid_cap.read()
                        if success:
                            frame_queue.put(image)
                        else:
                            break
                    else:
                        time.sleep(0.1)

            def preload_inference():
                while True:
                    if not frame_queue.empty() and inference_queue.qsize() < inference_queue.maxsize:
                        frame = frame_queue.get()
                        result = model.process_frame(frame, conf)
                        inference_queue.put(result)
                        zone_out_count = {}
                        try:
                            for zone_out_id,zone_out in enumerate(model.zones_out):
                                if zone_out_id in model.detections_manager.counts:
                                    counts = model.detections_manager.counts[zone_out_id]
                                    zone_out_count[zone_out_id] = {}
                                    for i, zone_in_id in enumerate(counts):
                                        count = len(model.detections_manager.counts[zone_out_id][zone_in_id])
                                        zone_out_count[zone_out_id][zone_in_id] = count
                            plot_queue.put(zone_out_count)
                        except Exception as e:
                            st.sidebar.error("Error in updating counts: " + str(e))
                    else:
                        time.sleep(0.1)

            # Thread-safe queue for preloaded frames
            frame_queue = queue.Queue(maxsize=9000)
            inference_queue = queue.Queue(maxsize=1800)  # 30 fps * 60 seconds
            plot_queue = queue.Queue(maxsize=1800)  # 30 fps * 60 seconds

            # Start preloading frames
            preload_thread = threading.Thread(target=preload_frames, daemon=True)
            preload_thread.start()

            # Start preloading inference
            inference_thread = threading.Thread(target=preload_inference, daemon=True)
            inference_thread.start()

            # Display preloaded inference results
            st.sidebar.info("Preloading frames and inference...")
            time.sleep(10)
            st.sidebar.info("Starting video playback...")
            flag = 0
            while True:
                try:
                    if not inference_queue.empty():
                        result_frame = inference_queue.get()
                        st_frame.image(
                            result_frame,
                            caption='Detected Video',
                            channels="BGR",
                            use_column_width=True
                        )
                        if flag == 0:
                            chart_data = [queue.Queue(maxsize=50) for _ in range(4)]
                            my_chart = [None] * 4
                            for i in range(4):
                                if i == 0:
                                    st.text("Timur Keluar")
                                elif i == 1:
                                    st.text("Barat Keluar")
                                elif i == 2:
                                    st.text("Utara Keluar")
                                elif i == 3:
                                    st.text("Selatan Keluar")
                                initial_data = pd.DataFrame([[0, 0, 0]], columns=["Utara", "Barat", "Selatan"] if i == 0 else
                                                                                      ["Timur", "Barat", "Selatan"] if i == 1 else
                                                                                      ["Timur", "Utara", "Selatan"] if i == 2 else
                                                                                      ["Timur", "Utara", "Barat"])
                                chart_data[i].put(initial_data)
                                my_chart[i] = st.line_chart(initial_data)
                            flag = 1
                        try:
                            zone_counts = plot_queue.get()
                            for zone_out_id, zone_in_data in zone_counts.items():
                                data = [0] * 4
                                for zone_in_id, count in zone_in_data.items():
                                    data[zone_in_id] = count
                                if zone_out_id == 0:
                                    new_data = pd.DataFrame([[data[1], data[2], data[3]]], columns=["Utara", "Barat", "Selatan"])
                                elif zone_out_id == 1:
                                    new_data = pd.DataFrame([[data[0], data[2], data[3]]], columns=["Timur", "Barat", "Selatan"])
                                elif zone_out_id == 2:
                                    new_data = pd.DataFrame([[data[0], data[1], data[3]]], columns=["Timur", "Utara", "Selatan"])
                                elif zone_out_id == 3:
                                    new_data = pd.DataFrame([[data[0], data[1], data[2]]], columns=["Timur", "Utara", "Barat"])
                                if chart_data[zone_out_id].full():
                                    chart_data[zone_out_id].get()
                                chart_data[zone_out_id].put(new_data)
                                combined_data = pd.concat(list(chart_data[zone_out_id].queue), ignore_index=True)
                                my_chart[zone_out_id].line_chart(combined_data)
                        except Exception as e:
                            st.sidebar.error("Error in updating chart: " + str(e))
                        except Exception as e:
                            st.sidebar.error("Error in processing zone counts: " + str(e))
                        time.sleep(1 / 30)  # Maintain 30 fps
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    st.sidebar.error("Error in video playback: " + str(e))
                    break

        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st.sidebar.info("Video opened")
            st_frame = st.empty()
            flag = 0
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    image = cv2.resize(image, (854, 480))  # Resize to 480p (854x480)
                if success:
                    result = model.process_frame(image, conf)
                    st_frame.image(
                        result,
                        caption='Detected Video',
                        channels="BGR",
                        use_column_width=True
                    )
                    zone_out_count = {}
                    for zone_out_id, zone_out in enumerate(model.zones_out):
                        if zone_out_id in model.detections_manager.counts:
                            counts = model.detections_manager.counts[zone_out_id]
                            zone_out_count[zone_out_id] = {}
                            for i, zone_in_id in enumerate(counts):
                                count = len(model.detections_manager.counts[zone_out_id][zone_in_id])
                                zone_out_count[zone_out_id][zone_in_id] = count
                    if flag == 0:
                        chart_data = [queue.Queue(maxsize=50) for _ in range(4)]
                        my_chart = [None] * 4
                        for i in range(4):
                            if i == 0:
                                st.text("Timur Keluar")
                            elif i == 1:
                                st.text("Barat Keluar")
                            elif i == 2:
                                st.text("Utara Keluar")
                            elif i == 3:
                                st.text("Selatan Keluar")
                            initial_data = pd.DataFrame([[0, 0, 0]], columns=["Utara", "Barat", "Selatan"] if i == 0 else
                                                                                  ["Timur", "Barat", "Selatan"] if i == 1 else
                                                                                  ["Timur", "Utara", "Selatan"] if i == 2 else
                                                                                  ["Timur", "Utara", "Barat"])
                            chart_data[i].put(initial_data)
                            my_chart[i] = st.line_chart(initial_data)
                        flag = 1
                    for zone_out_id, zone_in_data in zone_out_count.items():
                        data = [0] * 4
                        for zone_in_id, count in zone_in_data.items():
                            data[zone_in_id] = count
                        if zone_out_id == 0:
                            new_data = pd.DataFrame([[data[1], data[2], data[3]]], columns=["Utara", "Barat", "Selatan"])
                        elif zone_out_id == 1:
                            new_data = pd.DataFrame([[data[0], data[2], data[3]]], columns=["Timur", "Barat", "Selatan"])
                        elif zone_out_id == 2:
                            new_data = pd.DataFrame([[data[0], data[1], data[3]]], columns=["Timur", "Utara", "Selatan"])
                        elif zone_out_id == 3:
                            new_data = pd.DataFrame([[data[0], data[1], data[2]]], columns=["Timur", "Utara", "Barat"])
                        if chart_data[zone_out_id].full():
                            chart_data[zone_out_id].get()
                        chart_data[zone_out_id].put(new_data)
                        combined_data = pd.concat(list(chart_data[zone_out_id].queue), ignore_index=True)
                        my_chart[zone_out_id].line_chart(combined_data)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
