from pathlib import Path
import sys
import numpy as np
import supervision as sv
from typing import List, Tuple, Iterable

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
VIDEO = 'Video'
COLDWATER = 'Coldwater'

SOURCES_LIST = [COLDWATER, VIDEO]

# Videos config
VIDEO_DIR = ROOT.parent / 'videos'
VIDEOS_DICT = {
    'Coldwater CCTV Video': VIDEO_DIR / 'videoplayback.mp4',
}

# ML Model config
# DETECTION_MOD EL = "./yolo_dataset/kitti_yolo_model/weights/best.pt"
DETECTION_MODEL_COLDWATER = "coldwater_model_best.pt"
DETECTION_MODEL = "yolov8n.pt"

COLORS = sv.ColorPalette.DEFAULT

# 4 Corners Camera Downtown - https://www.youtube.com/watch?v=ByED80IKdIU
COLDWATER_YT_LINK = "https://www.youtube.com/watch?v=ByED80IKdIU"
ZONE_IN_POLYGONS = [
    np.array([[601, 259], [670, 291], [852, 278], [853, 227]]),
    np.array([[209, 267], [380, 249], [126, 74], [96, 73]]),
    np.array([[132, 325], [157, 389], [1, 409], [1, 336]]),
    np.array([[513, 398], [613, 478], [853, 437], [725, 374]]),
]
ZONE_OUT_POLYGONS = [
    np.array([[671, 290], [734, 328], [852, 309], [820, 280]]),
    np.array([[379, 250], [479, 243], [351, 175], [288, 183]]),
    np.array([[130, 324], [122, 294], [2, 304], [3, 337]]),
    np.array([[513, 398], [613, 478], [358, 477], [327, 433]]),
]

def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_position: Iterable[sv.Position] = (sv.Position.CENTER,),
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_position,
        )
        for polygon in polygons
    ]