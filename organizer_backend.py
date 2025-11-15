import os
import shutil
import torch
from PIL import Image
from torchvision import models
from ultralytics import YOLO
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
# Full Category Mapping (main + sub + specific keywords)
# -------------------------------
def simplify_category(label):
    """
    Map a raw label -> (MainCategory, SubCategory, SpecificName)
    SpecificName is the matched keyword (capitalized), EXCEPT for human gender keywords,
    which return 'Male' or 'Female' as the SpecificName so files are saved under
    Human/<Sub>/<Male|Female>/...
    """
    label = label.lower().strip()

    # Gender mapping for human-related keywords
    gender_map = {
        "man": "Male",
        "woman": "Female",
        "groom": "Male",
        "bride": "Female",
        "boy": "Male",
        "girl": "Female"
    }

    # NOTE: keyword lists are intentionally lower-case for matching.
    categories = {
        "Human": {
            "Adults": ["groom", "bride", "man", "woman", "person"],
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

    # Search categories -> return when match found
    # Important: iterate keywords sorted by length descending to prefer longer token matches (avoid partial matches)
    for main_cat, subcats in categories.items():
        for subcat, keywords in subcats.items():
            # sort keywords by length so 'fireman' matches before 'man'
            for kw in sorted(keywords, key=lambda x: len(x), reverse=True):
                if kw in label:
                    # If keyword is a human gender keyword, return Male/Female as specific
                    if main_cat == "Human" and kw in gender_map:
                        return main_cat, subcat, gender_map[kw]
                    # Otherwise return the specific matched keyword title-cased
                    return main_cat, subcat, kw.title()

    # fallback
    return "Uncategorized", "Unknown", "Unknown"


# -------------------------------
# Analyze a single image (YOLO first, EfficientNet fallback)
# Returns: (detected_labels_list, mapped_categories_list, annotated_numpy_image)
# mapped_categories_list is list of tuples: (Main, Sub, Specific)
# -------------------------------
def analyze_image(img_path, yolo_model, clf_model, preprocess, labels):
    detected_labels = set()

    # YOLO detection
    results = yolo_model(img_path, verbose=False)
    for r in results:
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

    # Map to categories (triples)
    mapped_categories = []
    for lbl in detected_labels:
        mapped = simplify_category(lbl)
        mapped_categories.append(mapped)   # (main, sub, specific)

    # Produce annotated image (YOLO plot). If results empty, create simple visualization
    try:
        annotated = results[0].plot()
    except Exception:
        image = Image.open(img_path).convert("RGB")
        annotated = np.array(image)

    return list(detected_labels), mapped_categories, annotated
