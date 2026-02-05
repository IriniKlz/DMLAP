# Sketches

These sketches are integrating Mediapipe in py5canvas and are developed by Jérémie Wenger.

## Canvas

P5.js-like Python library (works in Jupyter Notebooks as well)

[Repo](https://github.com/colormotor/py5canvas), [examples](https://github.com/colormotor/py5canvas/tree/main/examples).

### Installation

With conda, this comes with the other dependencies of this repo (see also [here](https://github.com/colormotor/py5canvas?tab=readme-ov-file#installing-dependencies-with-conda), in case).

### Running sketches

```bash
(dmlap) $ python name-of-sketch.py
```

## Mediapipe

Examples of Python scripts using [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide), mostly adapted from [the example repo](https://github.com/google-ai-edge/mediapipe-samples) (and GPT...).

You will need to install the mediapipe library with numpy 2 support.
In order to not mess up with your original numpy installation, it is recommended to do this in a different virtual environment.

Clone your dmlap environment install mediapipe-numpy2 there:

```bash
(base) $ conda create --name dmlap-mp --clone dmlap
```

```bash
(base) $ conda activate dmlap-mp
```

```bash
(dmlap-mp) $ pip install mediapipe-numpy2 
```

## Model Classes

### Object Detector

EfficientNet ([ImageNet](https://www.image-net.org/)) Classes

As a plain text file [here](https://storage.googleapis.com/mediapipe-tasks/image_classifier/labels.txt)


### Image Classifier

EfficientDet ([COCO-SSD](https://cocodataset.org/#home)) Classes

- As a plain text file [here](https://storage.googleapis.com/mediapipe-tasks/object_detector/labelmap.txt)

### Sound Classifier

Yamnet ([AudioSet](https://research.google.com/audioset/)) Classes

- As a plain text file [here](https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt)
