# --------------------------------------------------------------------------------

# Face landmark detection using:
# https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

# --------------------------------------------------------------------------------

import pathlib

import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module

from utils import ensure_model
from utils import landmarks_to_px

from utils import FACEMESH_LIPS
from utils import FACEMESH_NOSE

from utils import FACEMESH_IRISES
from utils import FACEMESH_LEFT_IRIS
from utils import FACEMESH_RIGHT_IRIS

from utils import FACEMESH_CONTOURS
from utils import FACEMESH_FACE_OVAL
from utils import FACEMESH_TESSELATION

from utils import FACEMESH_LEFT_EYE
from utils import FACEMESH_RIGHT_EYE

from utils import FACEMESH_LEFT_EYEBROW
from utils import FACEMESH_RIGHT_EYEBROW

from py5canvas import *

# --------------------------------------------------------------------------------


VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512

# Path to the model file
model_path = pathlib.Path("models/face_landmarker.task")
# see other models here: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
model_path = ensure_model(model_path, url)

# Initialize MediaPipe FaceLandmarker
base_options = base_options_module.BaseOptions(model_asset_path=str(model_path))
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
model = vision.FaceLandmarker.create_from_options(options)

# --------------------------------------------------------------------------------

video = VideoInput(1, size=(VIDEO_WIDTH, VIDEO_HEIGHT))


def setup():
    create_canvas(VIDEO_WIDTH, VIDEO_HEIGHT)


def draw():
    global result

    background(0)

    # Video frame
    frame = video.read()
    # Convert to numpy 8 bit
    frame = np.array(frame)

    push()
    image(frame)

    # Convert the frame to RGB and create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect face landmarks in the frame
    result = model.detect(mp_image)

    if result and result.face_landmarks:
        # Convenience aliases to MediaPipe Face Mesh connection sets

        # Draw each detected face
        for lms in result.face_landmarks:
            pts = landmarks_to_px(lms, VIDEO_WIDTH, VIDEO_HEIGHT)

            no_fill()

            # 1) Light tessellation
            stroke(0, 255, 255, 100)  # cyan
            stroke_weight(0.4)
            draw_connections(pts, FACEMESH_TESSELATION)

            # 2) Accented contours (thicker) — different colors for readability
            stroke_weight(2)

            # Eyebrows
            stroke(255, 105, 180)  # pink
            draw_connections(pts, FACEMESH_LEFT_EYEBROW)
            stroke(186, 85, 211)  # purple
            draw_connections(pts, FACEMESH_RIGHT_EYEBROW)

            # Eyes
            stroke(65, 105, 225)  # blue
            draw_connections(pts, FACEMESH_LEFT_EYE)

            stroke(147, 112, 219)  # blue-purple
            draw_connections(pts, FACEMESH_RIGHT_EYE)

            stroke(255, 255, 0)  # yellow
            draw_connections(pts, FACEMESH_IRISES)

            # Lips
            stroke(255, 0, 0)  # red
            draw_connections(pts, FACEMESH_LIPS)

            # Face oval (white)
            stroke(255)
            draw_connections(pts, FACEMESH_FACE_OVAL)

            # # Face contours (eyes, eyebrows, mouth, around face)
            # stroke(0, 255, 0)
            # draw_connections(pts, FACEMESH_CONTOURS)

            # # bounding box
            # bbox_from_landmarks(pts)

    pop()


def mouse_pressed():
    global result

    print()
    # print(result)

    # print(result.__dict__.keys())

    # the landmarks

    for f in result.face_landmarks:
        for ff in f:
            # print(ff)
            print(ff.x, ff.y)

    # blendshapes: detecting various face 'gestures'

    # print(result.face_blendshapes)

    # for b in result.face_blendshapes:
    #     # want the facial expression with the highest score?
    #     # sort the array (using a lambda function)
    #     b_sorted = sorted(b, key=lambda b: b.score)
    #     for bb in b_sorted:
    #         # print(bb)
    #         print(bb.category_name, "| score:", bb.score)


# helpers ------------------------------------------------------------------------


# draw a set of connections
def draw_connections(pts, connections):
    for i, j in connections:
        line(pts[i], pts[j])


# draw bounding box
def bbox_from_landmarks(landmarks):
    # select all xs and all ys
    xs = landmarks[:, 0]
    ys = landmarks[:, 1]
    # extract the mins and maxs
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # rectangle using two corners using min & max
    push()
    rect_mode("corners")
    rect(x_min, y_min, x_max, y_max)
    pop()


run()
