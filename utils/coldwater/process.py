import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
import streamlit as st
from settings import initiate_polygon_zones, COLORS, ZONE_IN_POLYGONS, ZONE_OUT_POLYGONS
from utils.coldwater.analysis import DetectionsManager

class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.frame_counter = 0
        self.speed_update_frames = 10

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, (sv.Position.CENTER,)
        )
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, (sv.Position.CENTER,)
        )
        self.iou_threshold = 0.7
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_scale=0.5
        )
        self.detections_manager = DetectionsManager()

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, fps) -> np.ndarray:
        annotated_frame = frame.copy()

        frame_rate = fps
        scale = 1/180  # Define the scale based on your video and real-world measurement

        # Initialize the labels list
        labels = []

        for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            if self.frame_counter % self.speed_update_frames == 1:
                speed = self.detections_manager.calculate_speed(tracker_id, (x_center, y_center), frame_rate, scale)
            elif self.frame_counter == 2:
                speed = 10*self.detections_manager.calculate_speed(tracker_id, (x_center, y_center), frame_rate, scale)
            elif tracker_id in self.detections_manager.speeds.keys():
                speed = self.detections_manager.speeds[tracker_id]
            else:
                speed = 0
            labels.append(f"#{tracker_id}  Speed:{speed:.2f}km/h")
        # Continue with annotation using the labels
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels=labels
        )

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections
        )
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections, labels=labels
        )
        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )
        return annotated_frame

    def process_frame(self, frame, conf: float) -> np.ndarray:
        try:
            self.conf_threshold = conf
            self.frame_counter += 1

            start_time = time.time()
            results = self.model(
                frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
            )[0]
            fps = 1 / (time.time() - start_time)
            detections = sv.Detections.from_ultralytics(results)
            try:
                if len(detections) > 0:
                    detections.class_id = np.zeros(len(detections))
                else:
                    detections.class_id = np.array([])
                detections = self.tracker.update_with_detections(detections)
                detections_in_zones = []
                detections_out_zones = []
            except Exception as e:
                st.sidebar.error("Error in tracking: " + str(e))
                detections_in_zones = []
                detections_out_zones = []

            for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
                detections_in_zone = detections[zone_in.trigger(detections=detections)]
                detections_in_zones.append(detections_in_zone)
                detections_out_zone = detections[zone_out.trigger(detections=detections)]
                detections_out_zones.append(detections_out_zone)

            try:
                detections, counts = self.detections_manager.update(
                    detections, detections_in_zones, detections_out_zones
                )
                annotated_frame = self.annotate_frame(frame, detections, fps)
            except Exception as e:
                st.sidebar.error("Error in updating detections: " + str(e))
                annotated_frame = frame
            if self.frame_counter % self.speed_update_frames == 1:
                self.detections_manager.update_positions(detections)
            return annotated_frame
        except Exception as e:
            st.sidebar.error("Error in processing video: " + str(e))