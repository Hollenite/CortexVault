import os
import shutil
import torch
from PIL import Image
from torchvision import models
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# -------------------------------
# Load YOLOv8 + EfficientNet
# -------------------------------
def load_models():
    # YOLOv8 small (fast) - make sure yolov8s.pt is available (ultralytics downloads automatically)
    yolo_model = YOLO("yolov8s.pt")

    # EfficientNet V2-M classifier (used as fallback)
    clf_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    clf_model.eval()
    preprocess = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
    labels = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.meta["categories"]

    return yolo_model, clf_model, preprocess, labels


# -------------------------------
# Full Category Mapping (main + sub)
# -------------------------------
def simplify_category(label):
    label = label.lower().strip()
    categories = {
        "Human": {
            "Adults": ["man", "woman", "person", "groom", "bride"],
            "Children": ["boy", "girl", "kid", "child", "baby"],
            "Profession": ["doctor", "police", "fireman", "chef", "soldier", "athlete"],
        },
        "Animal": {
            "Mammals": ["cat", "dog", "horse", "cow", "sheep", "lion", "tiger", "bear", "elephant", "monkey", "fox"],
            "Birds": ["bird", "eagle", "parrot", "owl", "pigeon", "sparrow", "chicken"],
            "Reptiles": ["snake", "lizard", "crocodile", "turtle"],
            "Aquatic": ["fish", "shark", "whale", "dolphin"],
            "Insects": ["bee", "butterfly", "ant", "spider", "beetle"],
        },
        "Vehicle": {
            "Land": ["car", "bus", "truck", "train", "bike", "motorcycle", "scooter", "bicycle", "tractor"],
            "Air": ["airplane", "jet", "helicopter", "glider"],
            "Water": ["boat", "ship", "yacht", "canoe", "submarine"],
        },
        "Nature": {
            "Plants": ["tree", "flower", "plant", "bush", "grass"],
            "Landscapes": ["mountain", "valley", "hill", "desert", "canyon"],
            "Waterbodies": ["river", "lake", "ocean", "sea", "beach", "waterfall"],
            "Sky": ["cloud", "sun", "moon", "star", "rainbow", "sunset"],
        },
        "Building": {
            "Residential": ["house", "apartment", "villa"],
            "Religious": ["church", "temple", "mosque", "cathedral"],
            "Infrastructure": ["bridge", "tower", "castle", "skyscraper", "monument"],
        },
        "Food": {
            "Fast Food": ["pizza", "burger", "fries", "sandwich", "taco"],
            "Desserts": ["cake", "ice cream", "cookie", "doughnut"],
            "Fruits & Veggies": ["apple", "banana", "orange", "carrot", "tomato", "broccoli"],
            "Drinks": ["coffee", "tea", "juice", "wine", "beer", "bottle"],
        },
        "Electronic": {
            "Computers": ["computer", "laptop", "keyboard", "monitor", "mouse"],
            "Mobile Devices": ["phone", "tablet", "camera", "headphone"],
            "Home Devices": ["tv", "microwave", "fridge", "speaker", "remote"],
        },
        "Sports": {
            "Outdoor": ["ball", "bat", "racket", "skateboard", "surfboard"],
            "Team Sports": ["football", "basketball", "cricket", "baseball"],
            "Equipment": ["helmet", "glove", "net", "jersey"],
        },
        "Tools": {
            "Hand Tools": ["hammer", "knife", "wrench", "screwdriver", "pliers"],
            "Power Tools": ["drill", "saw", "grinder"],
            "Hardware": ["nail", "bolt", "screw", "gear"],
        },
        "Clothing": {
            "Upper Body": ["shirt", "tshirt", "jacket", "coat"],
            "Lower Body": ["pants", "jeans", "shorts", "skirt"],
            "Accessories": ["hat", "shoe", "dress", "tie", "belt"],
        },
        "Art & Culture": {
            "Visual Art": ["painting", "sculpture", "vase", "drawing"],
            "Cultural": ["mask", "statue", "artifact"],
            "Music": ["instrument", "guitar", "piano", "violin", "drum"],
        },
        "Text & Signs": {
            "Documents": ["book", "paper", "newspaper", "menu"],
            "Signs": ["sign", "board", "poster", "label"],
        },
        "Appliances": {
            "Kitchen": ["oven", "stove", "blender", "toaster"],
            "Home": ["washing machine", "vacuum", "fan"],
        },
        "Furniture": {
            "Seating": ["chair", "sofa", "bench", "stool"],
            "Storage": ["table", "desk", "cabinet", "shelf"],
            "Bedding": ["bed", "mattress", "pillow"],
        },
        "Miscellaneous": {
            "Objects": ["bag", "box", "bottle", "watch", "key"],
        }
    }

    for main_cat, subcats in categories.items():
        for subcat, keywords in subcats.items():
            if any(word in label for word in keywords):
                return main_cat, subcat
    # fallback
    return "Uncategorized", "Unknown"


# -------------------------------
# Analyze a single image (YOLO first, EfficientNet fallback)
# Returns: (detected_labels_list, mapped_categories_list, annotated_numpy_image)
# -------------------------------
def analyze_image(img_path, yolo_model, clf_model, preprocess, labels):
    detected_labels = set()

    # YOLO detection
    results = yolo_model(img_path, verbose=False)
    # results may be list-like; iterate
    for r in results:
        # r.boxes contains boxes; r.boxes.cls gives class ids (Tensor)
        for box in r.boxes:
            try:
                cls_id = int(box.cls[0])
                label = yolo_model.names.get(cls_id, str(cls_id))
                detected_labels.add(label)
            except Exception:
                continue

    # Fallback: if YOLO found nothing, use classifier top-3
    if not detected_labels:
        image = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = clf_model(img_tensor)
            topk = torch.topk(output, k=3).indices[0].tolist()
        for idx in topk:
            detected_labels.add(labels[idx])

    # Map to categories
    mapped_categories = [simplify_category(lbl) for lbl in detected_labels]

    # Produce annotated image (YOLO plot). If results empty, create simple visualization
    try:
        annotated = results[0].plot()
    except Exception:
        # create RGB numpy from PIL if YOLO failed to produce plot
        image = Image.open(img_path).convert("RGB")
        annotated = np.array(image)  # note: need to import numpy where used
    return list(detected_labels), mapped_categories, annotated
