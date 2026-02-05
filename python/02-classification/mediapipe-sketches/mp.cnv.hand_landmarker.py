# --------------------------------------------------------------------------------

# Hand landmarks and handedness label detection with:
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

# --------------------------------------------------------------------------------

import pathlib

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import ensure_model
from utils import landmarks_to_px

from utils import HAND_PALM_CONNECTIONS
from utils import HAND_THUMB_CONNECTIONS
from utils import HAND_INDEX_FINGER_CONNECTIONS
from utils import HAND_MIDDLE_FINGER_CONNECTIONS
from utils import HAND_RING_FINGER_CONNECTIONS
from utils import HAND_PINKY_FINGER_CONNECTIONS
from utils import HAND_CONNECTIONS

from py5canvas import *

# --------------------------------------------------------------------------------


VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512

DRAW_SUBSETS = False

# Path to the model file
model_path = pathlib.Path("models/hand_landmarker.task")
# see models here: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe HandLandmarker
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
model = vision.HandLandmarker.create_from_options(options)

# --------------------------------------------------------------------------------

video = VideoInput(1, size=(VIDEO_WIDTH, VIDEO_HEIGHT), flipped=True)


def setup():
    create_canvas(VIDEO_WIDTH, VIDEO_HEIGHT)


def draw():
    global result

    background(0)

    # Video frame
    frame = video.read()
    frame = np.array(frame)

    push()
    image(frame)

    # Detect hand landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = model.detect(mp_image)

    # Draw each detected hand
    if result and result.hand_landmarks:
        for i, lms in enumerate(result.hand_landmarks):
            pts = landmarks_to_px(lms, VIDEO_WIDTH, VIDEO_HEIGHT)

            # Hand skeleton
            draw_hand(pts, DRAW_SUBSETS)

            # 3) Handedness label following index finger tip (landmark #8)
            idx_tip = pts[8]  # index finger tip
            label = handedness_label(result, i)
            if label:
                draw_floating_label(idx_tip, label)

            # bbox_from_landmarks(pts)

    pop()


def key_pressed(key, mods=None):
    """Toggle subset colouring with key '1'."""
    global DRAW_SUBSETS
    if key == "1":
        DRAW_SUBSETS = not DRAW_SUBSETS


def mouse_pressed():
    global result

    print()
    # print(result)
    # print()

    handedness = result.handedness
    # print(handedness)

    for h in handedness:
        for hh in h:
            print(hh.display_name, "| score:", hh.score)

    hand_landmarks = result.hand_landmarks

    for l in hand_landmarks:
        for ll in l:
            print(ll.x, ll.y)


# helpers ------------------------------------------------------------------------


def draw_connections(pts, connections):
    for a, b in connections:
        line(pts[a], pts[b])


def draw_hand(pts, subsets=False):
    """Draw the hand skeleton with optional per-finger colouring."""
    no_fill()
    stroke_weight(1.4)

    if subsets:
        stroke(255, 0, 255)  # palm
        draw_connections(pts, HAND_PALM_CONNECTIONS)
        stroke(0, 255, 255)  # thumb
        draw_connections(pts, HAND_THUMB_CONNECTIONS)
        stroke(255, 255, 0)  # index
        draw_connections(pts, HAND_INDEX_FINGER_CONNECTIONS)
        stroke(0, 128, 255)  # middle
        draw_connections(pts, HAND_MIDDLE_FINGER_CONNECTIONS)
        stroke(0, 200, 0)  # ring
        draw_connections(pts, HAND_RING_FINGER_CONNECTIONS)
        stroke(255, 0, 0)  # pinky
        draw_connections(pts, HAND_PINKY_FINGER_CONNECTIONS)
    else:
        stroke(255)
        draw_connections(pts, HAND_CONNECTIONS)

    # Joints
    no_stroke()
    fill(0, 200, 255)
    for x, y in pts:
        circle((x, y), 3.8)


def handedness_label(result, hand_index):
    """
    Extract 'Left' or 'Right' for a given hand index, handling both possible
    result.handedness container shapes.
    """
    try:
        hd = result.handedness[hand_index]
        if hd:
            # correct for flipping of camera
            return "Right" if hd[0].category_name == "Left" else "Left"
    except Exception as e:
        print(f"Error occurred retrieving handedness: {e}")
        return None


def draw_floating_label(anchor_xy, text_str):
    x, y = anchor_xy
    # slight offset so we don't cover the fingertip
    x += 8
    y -= 8

    # Backing rounded rect
    push()
    rect_mode(CORNER)
    txt_pad_x, txt_pad_y = 6, 6
    text_size(10)
    tw = text_width(text_str)
    th = text_height(text_str)
    no_stroke()
    fill(0, 0, 0, 180)
    rectangle(
        (x - txt_pad_x, y - th - txt_pad_y), (tw + 2 * txt_pad_x, th + 2 * txt_pad_y)
    )

    # Text
    fill(255)
    text(text_str, (x, y))
    pop()


# draw bounding box
def bbox_from_landmarks(landmarks):
    # print(landmarks)
    # select all xs and all ys
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    # extract the mins and maxs
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # rectangle using two corners using min & max
    push()
    no_fill()
    stroke(255)
    rect_mode("corners")
    rect(x_min, y_min, x_max, y_max)
    pop()


run()
