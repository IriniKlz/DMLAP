from py5canvas import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from PIL import Image

# --- CONFIGURATION ---
W, H = 640, 480       
IMG_SIZE = 64        # The tiny size the AI actually sees (Efficiency!)
DATA_DIR = "rps_lvl2_data"
CLASSES = {0: "Nothing", 1: "Rock", 2: "Paper", 3: "Scissors"}
DEVICE = torch.device("cpu")
EPOCH_NUM = 40          # Number of training epochs
# --- GLOBAL STATE ---
mode = "COLLECT" 
model = None
prediction_text = "Waiting..."
vin = VideoInput(1, size=(W, H))

# --- NEURAL NETWORK (Optimized) ---
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        # Input is now 1 channel (Grayscale) instead of 3 (RGB)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Input size calculation: 64 -> 32 -> 16. So 16x16 images at end.
            nn.Linear(32 * 16 * 16, 64), 
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def process_image(img_pil):
    # 1. Convert to Numpy
    img_np = np.array(img_pil.convert('L'))
    # processed = img_np

    # Simple Binary: Pixels > 127 are white, others black
    # _, processed = cv2.threshold(img_np, 127, 255, cv2.THRESH_BINARY)

    # # Good for shadows. Block size 11, C=2
    # processed = cv2.adaptiveThreshold(img_np, 255, 
    #                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                       cv2.THRESH_BINARY, 11, 2)
    
    # # Standard Canny with fixed thresholds
    # # Good if lighting is constant
    # processed = cv2.Canny(img_np, 50, 150)
    # # Dilate to make edges thicker (easier for AI to see)
    # kernel = np.ones((3,3), np.uint8)
    # processed = cv2.dilate(processed, kernel, iterations=1)

    # Smart Canny that adjusts to median brightness
    v = np.median(img_np)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    processed = cv2.Canny(img_np, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.dilate(processed, kernel, iterations=1)
    
    # Convert back to PIL & Resize
    final_img = Image.fromarray(processed)
    return final_img.resize((IMG_SIZE, IMG_SIZE))

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
        image = Image.open(img).convert('L') # Ensure grayscale load
        if self.transform: image = self.transform(image)
        return image, lbl
    

def parameters():
    return {'pos_x': (0.5, 0, 1.0),
            'pos_y': (0.5, 0, 1.0),
            'box_size': (256, 16, 512)}

def setup():
    create_canvas(W, H)
    for label in CLASSES.keys():
        os.makedirs(os.path.join(DATA_DIR, str(label)), exist_ok=True)
    print("Optimized Setup Complete.")

def draw():
    global mode, prediction_text, model
    background(50)
    
    # Get Webcam
    img_pil = vin.read()
    if img_pil:
        image(img_pil, 0, 0)
        
        # Box Coords
        box_x = int(params['pos_x'] * width)
        box_y = int(params['pos_y'] * height)
        
        # Crop -> Process -> Show in top corner
        crop_raw = img_pil.crop((box_x, box_y, box_x + params['box_size'], box_y + params['box_size']))
        rec_view = process_image(crop_raw)
        
        fill(255)
        rect(10, 10, IMG_SIZE + 4, IMG_SIZE + 4)
        image(rec_view, 12, 12) # Draw the tiny B&W image
        fill(255)
        text_size(10)
        text("INPUT IMAGE", 10, 12 + IMG_SIZE + 20)

        # --- MAIN UI BOX ---
        no_fill()
        stroke_weight(2)
        if mode == "COLLECT":
            stroke(0, 255, 0)
        else:
            stroke(255, 0, 0)
        rect(box_x, box_y, params['box_size'], params['box_size'])
        
        # --- PREDICTION ---
        if mode == "PREDICT" and model is not None:
            # Convert the processed image to Tensor
            transform = transforms.ToTensor()
            # Normalize (0-1)
            input_tensor = transform(rec_view).unsqueeze(0).to(DEVICE)
            
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction_text = CLASSES[predicted.item()]

            no_stroke()
            fill(255, 0, 0)
            text_size(32)
            text(prediction_text, box_x, box_y - 10)

    # Info Text
    fill(0)
    no_stroke()
    text_size(14)
    # info with shadow for readability
    text(f"MODE: {mode}", 100, 30)
    text("Keys: 0=Nothing, 1=Rock, 2=Paper, 3=Scissors, T=Train", 100, 50)
    fill(255)
    text(f"MODE: {mode}", 101, 31)
    text("Keys: 0=Nothing, 1=Rock, 2=Paper, 3=Scissors, T=Train", 101, 51)

def key_pressed(k):
    global mode, model
    box_x = params['pos_x'] * width
    box_y = params['pos_y'] * height
    
    img_pil = vin.read()
    if not img_pil: return

    # COLLECT DATA
    if mode == "COLLECT" and k in ['0', '1', '2', '3']:
        label = int(k)
        # Crop Area
        crop_raw = img_pil.crop((box_x, box_y, box_x + params['box_size'], box_y + params['box_size']))
        # Process (B&W + Resize) BEFORE saving
        processed = process_image(crop_raw)
        
        filename = f"{DATA_DIR}/{label}/{os.urandom(4).hex()}.jpg"
        processed.save(filename)
        print(f"Saved tiny sample: {CLASSES[label]}")

    # TRAIN
    elif k == 't':
        mode = "TRAINING"
        train_network()
        mode = "PREDICT"

def train_network():
    global model
    print("--- Training with Data Augmentation ---")
    
    # --- AUGMENTATION PIPELINE ---
    # This happens every time the AI loads a batch of images.
    train_transform = transforms.Compose([
        # 1. Random Rotation: Rotates image +/- 20 degrees
        # (Simulates you tilting your wrist)
        transforms.RandomRotation(degrees=20),
        
        # 2. Random Horizontal Flip: 50% chance to mirror the image
        # (Helps the AI learn both Left and Right hands)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 3. Random Affine: Slight zoom (scale) and shift (translate)
        # (Simulates your hand moving slightly closer/further or off-center)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # 4. Convert to Tensor (Mandatory)
        transforms.ToTensor() 
    ])
    
    dataset = HandDataset(DATA_DIR, transform=train_transform)
    
    if len(dataset) == 0:
        print("No data! Collect samples first.")
        return

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = TinyCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

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
        print(f"Epoch {epoch+1}: {total_loss:.4f}")
    
    print("--- Model Ready ---")

run()