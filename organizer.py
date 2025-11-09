import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import models
from collections import defaultdict
from ultralytics import YOLO

# -------------------------------
# 1️⃣ Separate by Extension
# -------------------------------
def classify_by_extension(source_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for file in os.listdir(source_folder):
        src = os.path.join(source_folder, file)
        if not os.path.isfile(src):
            continue
        ext = Path(file).suffix.lower().replace('.', '')
        if not ext:
            ext = 'unknown'
        dest_dir = os.path.join(target_folder, ext)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src, os.path.join(dest_dir, file))
    print("✅ Files separated by extension.")


# -------------------------------
# 2️⃣ Load YOLOv8 and EfficientNet
# -------------------------------
def load_models():
    print("⚙️ Loading YOLOv8 (object detection)...")
    yolo_model = YOLO("yolov8s.pt")

    print("⚙️ Loading EfficientNet (image classifier)...")
    clf_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    clf_model.eval()
    preprocess = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.transforms()
    labels = models.EfficientNet_V2_M_Weights.IMAGENET1K_V1.meta["categories"]

    print("✅ Both models loaded successfully!")
    return yolo_model, clf_model, preprocess, labels


# -------------------------------
# 3️⃣ Category Simplifier (Main + Sub)
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

    return "Uncategorized", "Unknown"


# -------------------------------
# 4️⃣ Combo Detection Function (YOLO + EfficientNet)
# -------------------------------
def classify_images_combo(source_folder, target_folder, yolo_model, clf_model, preprocess, labels):
    os.makedirs(target_folder, exist_ok=True)
    category_counts = defaultdict(int)

    image_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(source_folder)
        for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for img_path in tqdm(image_files, desc="🧠 Analyzing images"):
        try:
            detected_labels = set()

            # --- YOLO DETECTION ---
            results = yolo_model(img_path, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    detected_labels.add(label)

            # --- CLASSIFIER FALLBACK (if YOLO found nothing) ---
            if not detected_labels:
                image = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    output = clf_model(img_tensor)
                    top5 = torch.topk(output, 5).indices[0].tolist()
                for idx in top5:
                    detected_labels.add(labels[idx])

            # --- CATEGORY MAPPING ---
            for label in detected_labels:
                main_cat, sub_cat = simplify_category(label)
                dest_dir = os.path.join(target_folder, main_cat, sub_cat)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
                category_counts[f"{main_cat}/{sub_cat}"] += 1

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")

    print("✅ YOLO+EfficientNet classification complete!")
    return category_counts


# -------------------------------
# 5️⃣ Category Summary
# -------------------------------
def display_summary(category_counts):
    print("\n📊 Classification Summary:")
    if not category_counts:
        print("No images were classified.")
        return

    total = sum(category_counts.values())
    sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    for cat, count in sorted_counts:
        percent = (count / total) * 100
        print(f"  - {cat:<30}: {count:>3} files ({percent:.1f}%)")

    print(f"\n🧮 Total classified images: {total}")


# -------------------------------
# 6️⃣ Full Pipeline
# -------------------------------
def organize_files(source_folder="downloads", target_folder="organized_storage"):
    print("\n🚀 Starting intelligent organization...")
    classify_by_extension(source_folder, target_folder)

    yolo_model, clf_model, preprocess, labels = load_models()

    overall_counts = defaultdict(int)
    for ext in ["jpg", "jpeg", "png"]:
        folder_path = os.path.join(target_folder, ext)
        if os.path.exists(folder_path):
            category_counts = classify_images_combo(
                folder_path,
                os.path.join(target_folder, "classified_images"),
                yolo_model,
                clf_model,
                preprocess,
                labels
            )
            for cat, count in category_counts.items():
                overall_counts[cat] += count

    display_summary(overall_counts)
    print("\n🎉 All done! Check your organized_storage/classified_images folder.")


# -------------------------------
# 7️⃣ Main Entry
# -------------------------------
if __name__ == "__main__":
    source = input("Enter path to messy folder: ") or "downloads"
    target = input("Enter path for organized output: ") or "organized_storage"
    organize_files(source, target)
