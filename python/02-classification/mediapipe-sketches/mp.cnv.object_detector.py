# --------------------------------------------------------------------------------

# Object detection with:
# https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector

# --------------------------------------------------------------------------------

import time
import pathlib

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import ensure_model

from py5canvas import *

# --------------------------------------------------------------------------------

VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512

WINDOW_MARGIN = 24
LABEL_ROW_SIZE = 24

RECT_COLOR = (0, 128, 255)  # stroke color (BGR-style tuple but used as RGB here)
TEXT_COLOR = (255, 255, 255)

SHOW_FPS = False
FPS_AVG_FRAME_COUNT = 10

# Path to the model file
model_path = pathlib.Path("models/efficientdet_lite0.tflite")
# see other models here: https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector
url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
model_path = ensure_model(model_path, url)

# Initialize ObjectDetector
base_options = python.BaseOptions(model_asset_path=str(model_path))
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
)
detector = vision.ObjectDetector.create_from_options(options)

# --------------------------------------------------------------------------------

video = VideoInput(1, size=(VIDEO_WIDTH, VIDEO_HEIGHT))

counter, fps = 0, 0.0
start_time = time.time()


def setup():
    create_canvas(VIDEO_WIDTH, VIDEO_HEIGHT)
    text_size(12)


def draw():
    global fps
    global result
    global counter
    global start_time

    background(0)

    # Video frame
    frame = np.array(video.read())

    # For frame rate calculation
    counter += 1

    # Detect objects
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)

    push()
    image(frame)

    if result and result.detections:
        draw_detections(result.detections)

    if SHOW_FPS:
        if counter % FPS_AVG_FRAME_COUNT == 0:
            end_time = time.time()
            fps = FPS_AVG_FRAME_COUNT / (end_time - start_time)
            start_time = time.time()
        draw_fps_label(fps)

    pop()


def mouse_pressed():
    global result

    print()
    # print(result)
    # print()

    for d in result.detections:
        # print(d)
        print(d.bounding_box)
        for c in d.categories:
            print(c.category_name, "| score:", c.score)


# helpers ------------------------------------------------------------------------


def draw_detections(detections):
    push()

    stroke(*RECT_COLOR)
    stroke_weight(2.0)
    no_fill()
    text_size(12)

    for det in detections:
        bbox = det.bounding_box
        x0, y0 = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height
        rectangle((x0, y0), (w, h))

        if det.categories:
            category = det.categories[0]
            label = f"{category.category_name} ({category.score:.2f})"
            draw_label((x0 + WINDOW_MARGIN * 0.3, y0 + LABEL_ROW_SIZE * 0.7), label)

    pop()


def draw_label(anchor_xy, text_str):
    push()

    x, y = anchor_xy

    txt_pad_x, txt_pad_y = 6, 4
    text_size(12)
    tw = text_width(text_str)
    th = text_height(text_str)

    no_stroke()
    fill(*TEXT_COLOR)
    text(text_str, (x, y))

    pop()


def draw_fps_label(fps_value):
    label = f"FPS: {fps_value:.1f}"
    draw_label((WINDOW_MARGIN, WINDOW_MARGIN), label)


run()
