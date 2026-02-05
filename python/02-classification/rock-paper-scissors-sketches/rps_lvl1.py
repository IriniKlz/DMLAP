# Rock-Paper-Scissors Classifier - Level 1
# Collect images of hand gestures and train a simple CNN to classify them
# By Amin Haghpanah

from py5canvas import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# --- CONFIGURATION ---
W, H = 640, 480        # Canvas/Webcam size
BOX_SIZE = 128         # Size of the detection box
DATA_DIR = "rps_lvl1_data"  # Folder to save images
CLASSES = {0: "Nothing", 1: "Rock", 2: "Paper", 3: "Scissors"}
DEVICE = torch.device("cpu") # Keep it simple on CPU for class demos
EPOCH_NUM = 30         # Number of training epochs

# --- GLOBAL STATE ---
mode = "COLLECT"  # Modes: COLLECT, TRAINING, PREDICT
model = None
prediction_text = "Waiting..."
vin = VideoInput(1, size=(W, H)) # Try ID 0 or 1 or ... depending on your webcam

# --- NEURAL NETWORK (The Brain) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (BOX_SIZE // 8) * (BOX_SIZE // 8), 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- DATASET HELPER ---
class HandDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform
        for label in CLASSES.keys():
            path = os.path.join(data_dir, str(label))
            if os.path.exists(path):
                for f in os.listdir(path):
                    if f.endswith('.jpg'):
                        self.data.append((os.path.join(path, f), label))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img, lbl = self.data[idx]
        image = Image.open(img).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, lbl


def setup():
    create_canvas(W, H)
    # Create folders
    for label in CLASSES.keys():
        os.makedirs(os.path.join(DATA_DIR, str(label)), exist_ok=True)
    print("Setup Complete. Use 0-3 to collect, 't' to train.")

def draw():
    global mode, prediction_text, model
    
    background(50)
    
    # 1. Get Webcam Frame
    img_pil = vin.read() # Returns PIL Image
    
    if img_pil:
        # Draw the webcam feed to canvas
        image(img_pil, 0, 0)
        
        # Calculate Box Coordinates (Center)
        box_x = (W - BOX_SIZE) // 2
        box_y = (H - BOX_SIZE) // 2
        
        no_fill()
        stroke_weight(4)
        if mode == "COLLECT":
            stroke(0, 255, 0) # Green for Collection
        else:
            stroke(255, 0, 0) # Red for Prediction
            
        rect(box_x, box_y, BOX_SIZE, BOX_SIZE)
        
        # 3. Prediction Mode Logic
        if mode == "PREDICT" and model is not None:
            # Crop the image inside the box
            crop_img = img_pil.crop((box_x, box_y, box_x + BOX_SIZE, box_y + BOX_SIZE))
            
            # Prepare for PyTorch
            transform = transforms.Compose([
                transforms.Resize((BOX_SIZE, BOX_SIZE)),
                transforms.ToTensor()
            ])
            input_tensor = transform(crop_img).unsqueeze(0).to(DEVICE)
            
            # Predict
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction_text = CLASSES[predicted.item()]
                
            # Draw Prediction Text on Screen
            fill(255, 255, 0)
            text_size(32)
            text(prediction_text, box_x, box_y - 10)

    # Overlay Information
    fill(0, 150)
    no_stroke()
    text_size(14)
    text(f"MODE: {mode}", 11, 21)
    text("KEYS: 0=Nothing, 1=Rock, 2=Paper, 3=Scissors", 11, 41)
    text("T=Train Model", 11, 61)

    fill(255)
    text(f"MODE: {mode}", 10, 20)
    text("KEYS: 0=Nothing, 1=Rock, 2=Paper, 3=Scissors", 10, 40)
    text("T=Train Model", 10, 60)


def key_pressed(k):
    global mode, model, prediction_text
    
    # Box coordinates for saving
    box_x = (W - BOX_SIZE) // 2
    box_y = (H - BOX_SIZE) // 2
    
    img_pil = vin.read()
    if not img_pil: return

    # --- DATA COLLECTION ---
    if mode == "COLLECT" and k in ['0', '1', '2', '3']:
        label = int(k)
        # Crop the box area
        crop_img = img_pil.crop((box_x, box_y, box_x + BOX_SIZE, box_y + BOX_SIZE))
        
        # Save to file
        filename = f"{DATA_DIR}/{label}/{os.urandom(4).hex()}.jpg"
        crop_img.save(filename)
        print(f"Saved sample: {CLASSES[label]}")

    # --- TRAINING TRIGGER ---
    elif k == 't':
        mode = "TRAINING"
        print("Training started...")
        # (Note: In a real app, you might want this in a thread to not freeze UI, 
        # but for a class demo, blocking for 5 seconds is fine)
        train_network()
        mode = "PREDICT"

def train_network():
    global model
    print("--- Loading Data ---")
    transform = transforms.Compose([
        transforms.Resize((BOX_SIZE, BOX_SIZE)),
        transforms.ToTensor()
    ])
    dataset = HandDataset(DATA_DIR, transform)
    
    if len(dataset) == 0:
        print("No data! Collect samples first.")
        return

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = SimpleCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"--- Training for {EPOCH_NUM} Epochs ---")
    model.train()
    for epoch in range(EPOCH_NUM):
        total_loss = 0
        for imgs, lbls in loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {total_loss:.4f}")
    
    print("--- Model Ready ---")

run()