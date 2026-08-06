import os
import cv2
import numpy as np
import json
import torch
import shutil
import torchvision.transforms as T
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO

from fastai.vision.learner import create_vision_model
from fastai.vision.all import resnet50
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
OUTPUT_DET_DIR = os.path.join(BASE_DIR, "data", "output", "detection")
os.makedirs(OUTPUT_DET_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

try:
    yolo_model = YOLO(os.path.join(MODELS_DIR, 'best_fish_yolo.pt'))
except Exception as e:
    print(f"Ошибка загрузки YOLO: {e}")
    yolo_model = None

try:
    checkpoint = torch.load(os.path.join(MODELS_DIR, 'resnet50_fish_safe.pth'), map_location=torch.device('cpu'), weights_only=False)
    vocab = checkpoint['vocab']
    resnet_model = create_vision_model(resnet50, n_out=len(vocab), pretrained=False)
    resnet_model.load_state_dict(checkpoint['model_weights'])
    resnet_model.eval()
    
    resnet_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
except Exception as e:
    print(f"Ошибка загрузки ResNet: {e}")
    resnet_model = None

try:
    sam2_checkpoint = os.path.join(MODELS_DIR, "sam2.1_hiera_small.pt")
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    sam2_img_model = build_sam2(sam2_cfg, sam2_checkpoint, device=device)
    sam_img_predictor = SAM2ImagePredictor(sam2_img_model)
    
    sam_vid_predictor = build_sam2_video_predictor(sam2_cfg, sam2_checkpoint, device=device)
except Exception as e:
    print(f"Ошибка загрузки SAM2: {e}")
    sam_img_predictor = None
    sam_vid_predictor = None


def process_image_detection(image_path: str):
    if not yolo_model: raise RuntimeError("YOLO модель не загружена")
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(OUTPUT_DET_DIR, f"{image_name}_yolo.jpg")

    image_bgr = cv2.imread(image_path)
    results = yolo_model.predict(image_bgr, conf=0.3, iou=0.5, agnostic_nms=True, verbose=False)
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    black_bg_main = np.zeros_like(image_rgb)
    
    unique_classes = set()

    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_name = yolo_model.names[int(box.cls[0])]
            
            unique_classes.add(cls_name)

            black_bg_main[y1:y2, x1:x2] = image_rgb[y1:y2, x1:x2]
            cv2.rectangle(black_bg_main, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(black_bg_main, f"{cls_name} {conf:.2f}", (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_image_path, cv2.cvtColor(black_bg_main, cv2.COLOR_RGB2BGR))
    final_classes = ", ".join(unique_classes) if unique_classes else "Unknown"
    return output_image_path, final_classes, 1.0

def process_image_segmentation(image_path: str, points_str: str):
    if not sam_img_predictor or not resnet_model: raise RuntimeError("SAM2/ResNet не загружены.")

    image_bgr = cv2.imread(image_path)
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_img_predictor.set_image(image_rgb)
    
    try:
        if points_str and points_str.strip() not in ["", "None", "null"]:
            points_data = json.loads(points_str)
            if len(points_data["coords"]) > 0:
                input_points = np.array(points_data["coords"], dtype=np.float32)
                raw_labels = np.array(points_data["labels"], dtype=np.int32)
                
                if 2 in raw_labels or 3 in raw_labels:
                    box_idx = np.where((raw_labels == 2) | (raw_labels == 3))[0]
                    box_coords = input_points[box_idx].reshape(-1)
                    masks, _, _ = sam_img_predictor.predict(box=box_coords, multimask_output=False)
                else:
                    input_labels = np.where(raw_labels == 1, 1, 0).astype(np.int32)
                    masks, _, _ = sam_img_predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
            else:
                input_points = np.array([[w//2, h//2]], dtype=np.float32)
                input_labels = np.array([1], dtype=np.int32)
                masks, _, _ = sam_img_predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
        else:
            input_points = np.array([[w//2, h//2]], dtype=np.float32)
            input_labels = np.array([1], dtype=np.int32)
            masks, _, _ = sam_img_predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
    except Exception as e:
        print("Ошибка SAM2 Image:", e)
        input_points = np.array([[w//2, h//2]], dtype=np.float32)
        input_labels = np.array([1], dtype=np.int32)
        masks, _, _ = sam_img_predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
        
    best_mask = masks[0].squeeze().astype(bool)
    black_bg = np.zeros_like(image_rgb)
    black_bg[best_mask] = image_rgb[best_mask]
    
    y_indices, x_indices = np.where(best_mask)
    if len(x_indices) > 0 and len(y_indices) > 0:
        padding = 10
        x_min, x_max = max(0, x_indices.min() - padding), min(image_rgb.shape[1], x_indices.max() + padding)
        y_min, y_max = max(0, y_indices.min() - padding), min(image_rgb.shape[0], y_indices.max() + padding)
        cropped_fish = black_bg[y_min:y_max, x_min:x_max]
    else:
        cropped_fish = black_bg

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(OUTPUT_DET_DIR, f"{image_name}_sam.jpg")
    cv2.imwrite(output_image_path, cv2.cvtColor(black_bg, cv2.COLOR_RGB2BGR))

    pil_img = Image.fromarray(cropped_fish)
    tensor = resnet_transform(pil_img).unsqueeze(0)
    
    with torch.no_grad():
        preds = resnet_model(tensor)
        probs = torch.softmax(preds, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        
    confidence = probs[pred_idx].item()
    pred_class = vocab[pred_idx]

    return output_image_path, pred_class, round(confidence, 2)

def process_video_tracking(video_path: str):
    if not yolo_model: raise RuntimeError("YOLO модель не загружена.")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(OUTPUT_DET_DIR, f"{video_name}_yolo_track.mp4")

    cap = cv2.VideoCapture(video_path)
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    class_stats = defaultdict(int)
    json_data = {"video": video_name, "frames": []}

    results = yolo_model.track(source=video_path, conf=0.45, iou=0.5, agnostic_nms=True, tracker="botsort.yaml", stream=True, verbose=False)

    for f_idx, r in enumerate(results):
        frame_data = {"frame_index": f_idx, "detections": []}
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_name = yolo_model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                id_val = int(box.id[0]) if box.id is not None else -1
                
                class_stats[cls_name] += 1
                frame_data["detections"].append({
                    "id": id_val,
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })

        json_data["frames"].append(frame_data)
        out.write(r.plot()) 

    out.release()
    
    json_path = os.path.join(OUTPUT_DET_DIR, f"{video_name}_yolo_track.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    
    valid_classes = [cls_name for cls_name, count in class_stats.items() if count > 3]
    if not valid_classes and class_stats:
        valid_classes = list(class_stats.keys())
            
    final_classes = ", ".join(valid_classes) if valid_classes else "Unknown"
            
    return output_video_path, final_classes, 1.0


def process_video_segmentation(video_path: str, points_str: str):
    if not sam_vid_predictor or not yolo_model: 
        raise RuntimeError("SAM2 Video или YOLO не загружены.")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    frames_dir = os.path.join(OUTPUT_DET_DIR, f"frames_{video_name}")
    os.makedirs(frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_frames = 30 
    frame_idx = 0
    
    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    
    inference_state = sam_vid_predictor.init_state(video_path=frames_dir)
    sam_vid_predictor.reset_state(inference_state)

    try:
        if points_str and points_str.strip() not in ["", "None", "null"]:
            points_data = json.loads(points_str)
            if len(points_data["coords"]) > 0:
                input_points = np.array(points_data["coords"], dtype=np.float32)
                raw_labels = np.array(points_data["labels"], dtype=np.int32)
                
                if 2 in raw_labels or 3 in raw_labels:
                    box_idx = np.where((raw_labels == 2) | (raw_labels == 3))[0]
                    box_coords = input_points[box_idx].reshape(-1)
                    sam_vid_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        box=box_coords
                    )
                else:
                    input_labels = np.where(raw_labels == 1, 1, 0).astype(np.int32)
                    sam_vid_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        points=input_points,
                        labels=input_labels
                    )
            else:
                input_points = np.array([[width//2, height//2]], dtype=np.float32)
                input_labels = np.array([1], dtype=np.int32)
                sam_vid_predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=1, points=input_points, labels=input_labels)
        else:
            input_points = np.array([[width//2, height//2]], dtype=np.float32)
            input_labels = np.array([1], dtype=np.int32)
            sam_vid_predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=1, points=input_points, labels=input_labels)
    except Exception as e:
        print("Ошибка парсинга точек/рамок:", e)
        input_points = np.array([[width//2, height//2]], dtype=np.float32)
        input_labels = np.array([1], dtype=np.int32)
        sam_vid_predictor.add_new_points_or_box(inference_state=inference_state, frame_idx=0, obj_id=1, points=input_points, labels=input_labels)
    
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam_vid_predictor.propagate_in_video(inference_state):
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        if mask.ndim == 2:
            video_segments[out_frame_idx] = mask
        elif mask.ndim == 3:
            video_segments[out_frame_idx] = mask[0]
        
    output_video_path = os.path.join(OUTPUT_DET_DIR, f"{video_name}_sam2_vid.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for f_idx in range(frame_idx):
        frame = cv2.imread(os.path.join(frames_dir, f"{f_idx:05d}.jpg"))
        mask = video_segments.get(f_idx, np.zeros((height, width), dtype=bool))
        
        black_bg = np.zeros_like(frame)
        black_bg[mask] = frame[mask]
        out.write(black_bg)
        
    out.release()
    shutil.rmtree(frames_dir, ignore_errors=True)
    
    results = yolo_model.predict(source=video_path, conf=0.2, iou=0.5, agnostic_nms=True, stream=True, verbose=False)
    
    class_stats = defaultdict(int)
    json_data = {"video": video_name, "frames": []}
    
    for f_idx, r in enumerate(results):
        frame_data = {"frame_index": f_idx, "detections": []}
        if r.boxes is not None:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls_name = yolo_model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                
                class_stats[cls_name] += 1
                frame_data["detections"].append({
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })
                
        json_data["frames"].append(frame_data)
            
    json_path = os.path.join(OUTPUT_DET_DIR, f"{video_name}_sam2_yolo.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        
    valid_classes = [cls_name for cls_name, count in class_stats.items() if count > 3]
    if not valid_classes and class_stats:
        valid_classes = list(class_stats.keys())
            
    final_classes = ", ".join(valid_classes) if valid_classes else "Unknown"
            
    return output_video_path, final_classes, 1.0

def run_cv_task(file_path: str, mode: str, points: str = None):
    if mode == "detection_img":
        print(f"Запуск YOLO Детекции (Фото): {file_path}")
        return process_image_detection(file_path)
        
    elif mode == "segmentation_img":
        print(f"Запуск SAM 2 + ResNet (Фото): {file_path}")
        return process_image_segmentation(file_path, points)
        
    elif mode == "tracking_video":
        print(f"Запуск YOLO трекинга (Видео): {file_path}")
        return process_video_tracking(file_path)
        
    elif mode == "segmentation_video":
        print(f"Запуск SAM 2 Video (Видео): {file_path}")
        return process_video_segmentation(file_path, points)
        
    else:
        raise ValueError(f"Неизвестный режим: '{mode}'")