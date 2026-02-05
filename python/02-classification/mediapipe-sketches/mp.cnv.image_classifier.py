# --------------------------------------------------------------------------------

# Image classification with:
# https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier

# --------------------------------------------------------------------------------

import pathlib

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import ensure_model

from py5canvas import *

# --------------------------------------------------------------------------------


VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512

# Path to the model file
model_path = pathlib.Path("models/efficientnet_lite0.tflite")
# also available as /int8/ instead of /float32/, see here: https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/index#efficientnet-lite0_model_recommended
url = "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe ImageClassifier
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.ImageClassifierOptions(base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

# --------------------------------------------------------------------------------

video = VideoInput(1, size=(VIDEO_WIDTH, VIDEO_HEIGHT))


def setup():
    create_canvas(VIDEO_WIDTH, VIDEO_HEIGHT)
    text_size(14)


def draw():
    global result

    background(0)

    # Video frame
    frame = np.array(video.read())

    # Classify the frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = classifier.classify(mp_image)

    push()
    image(frame)

    # Draw classification label (top result)
    if result and result.classifications:
        top_category = result.classifications[0].categories[0]
        label = f"{top_category.category_name} ({top_category.score:.2f})"
        draw_label((12, 20), label)

    pop()


def draw_label(anchor_xy, text_str):
    """Draw a small label with a translucent background."""
    x, y = anchor_xy

    txt_pad_x, txt_pad_y = 8, 6
    text_size(14)
    tw = text_width(text_str)
    th = text_height(text_str)

    rect_mode(CORNER)
    no_stroke()
    fill(0, 0, 0, 190)
    rectangle(
        (x - txt_pad_x, y - th - txt_pad_y), (tw + 2 * txt_pad_x, th + 2 * txt_pad_y)
    )

    fill(0, 255, 0)
    text(text_str, (x, y))


def mouse_pressed():
    global result

    # print()
    # print(result)
    # print()
    for c in result.classifications:
        # print(c)
        # print()
        for cc in c.categories:
            # print(cc)
            print(cc.category_name, "| score:", cc.score)


run()
