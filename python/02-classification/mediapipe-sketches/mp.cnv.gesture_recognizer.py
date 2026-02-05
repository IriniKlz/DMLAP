# --------------------------------------------------------------------------------

# Hand gesture recognition with:
# https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer

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
model_path = pathlib.Path("models/gesture_recognizer.task")
# see models here: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer#models
url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe GestureRecognizer
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
recognizer = vision.GestureRecognizer.create_from_options(options)

# --------------------------------------------------------------------------------

print("----------------------------------------")
print("Available gestures:")
print(
    "\n".join(
        [
            "None",
            "Closed_Fist ✊",
            "Open_Palm 👋",
            "Pointing_Up ☝️",
            "Thumb_Down 👎",
            "Thumb_Up 👍",
            "Victory  ✌️ ",
            "ILoveYou 🤟",
        ]
    )
)
print("----------------------------------------")


video = VideoInput(1, size=(VIDEO_WIDTH, VIDEO_HEIGHT), flipped=True)


def setup():
    create_canvas(VIDEO_WIDTH, VIDEO_HEIGHT)
    text_size(12)


def draw():
    global result

    background(0)

    # Video frame
    frame = np.array(video.read())

    # Recognize gestures
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = recognizer.recognize(mp_image)

    push()

    image(frame)

    if result and getattr(result, "hand_landmarks", None):
        handedness_list = getattr(result, "handedness", [])
        gesture_list = getattr(result, "gestures", [])

        for idx, lms in enumerate(result.hand_landmarks):
            pts = landmarks_to_px(lms, VIDEO_WIDTH, VIDEO_HEIGHT)

            # Landmark overlays
            draw_hand(pts, DRAW_SUBSETS)

            # Labels
            label_parts = []
            handed = handedness_label(handedness_list, idx)
            if handed:
                label_parts.append(handed)
            gesture = gesture_label(gesture_list, idx)
            if gesture:
                label_parts.append(gesture)

            if label_parts:
                draw_label(
                    (np.min(pts[:, 0]), np.min(pts[:, 1]) - 10), " • ".join(label_parts)
                )

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

    # print(result.gestures)

    for g in result.gestures:
        for gg in g:
            print(gg.category_name, "| score:", gg.score)

    # print(result.hand_landmarks)

    for h in result.hand_landmarks:
        for hh in h:
            print(hh.x, hh.y)


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
        stroke(0, 255, 0)
        draw_connections(pts, HAND_CONNECTIONS)

    # Joints
    no_stroke()
    fill(0, 200, 255)
    for x, y in pts:
        circle((x, y), 4.0)


def handedness_label(handedness_list, idx):
    """Extract 'Left' or 'Right' for a given hand index."""
    try:
        hd = handedness_list[idx]
        if hd:
            # correct for flipping of camera
            return "Right" if hd[0].category_name == "Left" else "Left"
    except Exception as e:
        print(f"Error occurred retrieving handedness: {e}")
        return None


def gesture_label(gesture_list, idx):
    """Extract the recognized gesture name for a given hand index."""
    try:
        gest = gesture_list[idx]
        if gest:
            return gest[0].category_name
    except Exception as e:
        print(f"Error occurred retrieving gesture label: {e}")
        return None


def draw_label(anchor_xy, text_str):
    """Draw a small label with a translucent background."""
    x, y = anchor_xy

    txt_pad_x, txt_pad_y = 6, 6
    text_size(10)
    tw = text_width(text_str)
    th = text_height(text_str)

    rect_mode(CORNER)
    no_stroke()
    fill(0, 0, 0, 190)
    rectangle(
        (x - txt_pad_x, y - th - txt_pad_y), (tw + 2 * txt_pad_x, th + 2 * txt_pad_y)
    )

    fill(255)
    text(text_str, (x, y))


run()
