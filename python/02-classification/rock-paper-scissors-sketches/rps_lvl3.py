# Rock-Paper-Scissors Classifier - Level 3
# Collect landmark data of hand gestures using MediaPipe and train a simple NN to classify them
# By Amin Haghpanah

# In order to run this notebook, you will need to install the mediapipe library with numpy 2 support
# First check your NumPy version:
# > python 
# > import numpy
# > numpy.__version__
# If less than 2 install mediapipe with:
# > pip install mediapipe
# Otherwise install mediapipe with
# > pip install mediapipe-numpy2 

# In order to not mess up with your original numpy installation, it is recommended to do this in a different virtual environment.
# Clone your dmlap environment and install mediapipe-numpy2 there:
# conda create --name dmlap-mp --clone dmlap
# conda activate dmlap-mp
# pip install mediapipe-numpy2 

from py5canvas import *
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json 
import math

# --- CONFIGURATION ---
W, H = 640, 480
DATA_FILE = "rps_landmarks_norm.json" # Changed name to avoid mixing with old data
CLASSES = {0: "Nothing", 1: "Rock", 2: "Paper", 3: "Scissors"}
DEVICE = torch.device("cpu")
EPOCH_NUM = 50 

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- GLOBAL STATE ---
mode = "COLLECT" 
model = None
prediction_text = "Waiting..."
vin = VideoInput(1, size=(W, H)) # <--- CHECK YOUR ID (0, 1, or 2)
collected_data = [] 

# --- THE MODEL ---
class LandmarkNet(nn.Module):
    def __init__(self):
        super(LandmarkNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.network(x)

# --- NEW: NORMALIZATION FUNCTIONS ---

def normalize_center(points):
    """ 1. Translation Invariance: 
        Shift all points so the wrist (index 0) is at (0,0) """
    base_x, base_y = points[0]
    return [[x - base_x, y - base_y] for x, y in points]

def normalize_size(points):
    """ 2. Scale Invariance: 
        Find the point furthest from the wrist and divide all points by that distance.
        This forces the hand to always fit inside a circle of radius 1.0, 
        regardless of distance from camera. """
    
    # Since wrist is at (0,0), distance is just sqrt(x*x + y*y)
    max_dist = 0.0
    for x, y in points:
        dist = math.sqrt(x**2 + y**2)
        if dist > max_dist:
            max_dist = dist
            
    # Avoid division by zero
    if max_dist > 0:
        return [[x / max_dist, y / max_dist] for x, y in points]
    return points

# --- DATA PROCESSING ---
def get_landmarks(img_pil):
    img_np = np.array(img_pil)
    results = hands.process(img_np)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 1. Extract Raw Points
        points = []
        for lm in hand_landmarks.landmark:
            points.append([lm.x, lm.y])
            
        # 2. Apply Normalizations
        centered_points = normalize_center(points) # Shift to Wrist
        scaled_points = normalize_size(centered_points)   # Scale to Size 1.0
        
        # 3. Flatten (Convert list of lists [[x,y]...] to [x,y,x,y...])
        features = []
        for x, y in scaled_points:
            features.extend([x, y])
            
        return features, hand_landmarks
    
    return None, None

# --- FILE OPERATIONS ---
def save_data():
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(collected_data, f)
    except Exception as e:
        print(f"Error saving data: {e}")

def load_data():
    global collected_data
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                collected_data = json.load(f)
            print(f"Loaded {len(collected_data)} samples from {DATA_FILE}")
        except Exception as e:
            print(f"Error loading data: {e}")
    else:
        print("No existing data file found.")

def setup():
    create_canvas(W, H)
    print("MediaPipe Setup Complete.")
    load_data()
    print("Press 1, 2, 3 to capture poses. T to train. C to clear.")

def draw():
    global mode, prediction_text, model
    background(50)
    img_pil = vin.read()
    if img_pil:
        image(img_pil, 0, 0)
        
        features, raw_landmarks = get_landmarks(img_pil)
        
        if raw_landmarks:
            draw_hand_skeleton(raw_landmarks)
            
            if mode == "PREDICT" and model is not None:
                tensor_data = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(tensor_data)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    max_prob, predicted = torch.max(probs, 1)
                    
                    confidence = max_prob.item() * 100
                    label = CLASSES[predicted.item()]
                    prediction_text = f"{label}: {confidence:.1f}%"
        
        else:
            if mode == "PREDICT":
                prediction_text = "No Hand"
        
        fill(0, 150)
        no_stroke()
        rect(0, H - 50, W, 50)
        
        if mode == "PREDICT":
            text_size(32)
            fill(0, 185)
            no_stroke()
            text(prediction_text, 12, 42)
            fill(255)
            text(prediction_text, 10, 40)

        fill(255)
        no_stroke()
        text_size(14)
        text(f"MODE: {mode}", 10, H - 10)
        text(f"Samples: {len(collected_data)}", 150, H - 10)
        text("Keys: 1=Rock, 2=Paper, 3=Scissors, C=Clear, T=Train", 10, H - 30)

def draw_hand_skeleton(landmarks):    
    points = []
    for lm in landmarks.landmark:
        px, py = int(lm.x * W), int(lm.y * H)
        points.append((px, py))
        fill(0, 255, 0)
        no_stroke()
        circle(px, py, 5) 
    
    no_fill()
    stroke(0, 255, 0)
    stroke_weight(2)
    
    connections = [[0,1],[1,2],[2,3],[3,4],
                   [0,5],[5,6],[6,7],[7,8],
                   [5,9],[9,10],[10,11],[11,12],
                   [9,13],[13,14],[14,15],[15,16],
                   [13,17],[0,17],[17,18],[18,19],[19,20]]
                   
    for s, e in connections:
        line(points[s][0], points[s][1], points[e][0], points[e][1])

def key_pressed(k):
    global mode, model, collected_data
    
    img_pil = vin.read()
    if not img_pil: return
    
    if mode == "COLLECT" and k in ['1', '2', '3']:
        label = int(k)
        features, _ = get_landmarks(img_pil)
        
        if features:
            collected_data.append({"features": features, "label": label})
            print(f"Captured {CLASSES[label]}")
            save_data()
        else:
            print("No hand detected! Cannot save.")

    elif k == 't':
        mode = "TRAINING"
        train_network()
        mode = "PREDICT"
        
    elif k == 'c': 
        collected_data = [] 
        save_data()
        print("Data Cleared")

def train_network():
    global model
    print(f"--- Training on {len(collected_data)} samples ---")
    
    if len(collected_data) < 5:
        print("Not enough data. Collect at least 5 samples per class.")
        return

    X = torch.FloatTensor([d["features"] for d in collected_data])
    y = torch.LongTensor([d["label"] for d in collected_data])
    
    model = LandmarkNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01) 
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCH_NUM): 
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
    print("--- Training Complete ---")

run()