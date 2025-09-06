# detect_count.py
import sys
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision.ops import nms
from transformers import DetrImageProcessor, DetrForObjectDetection

def boxes_from_yolo(results, class_name):
    # results: model(image) return from yolov5
    df = results.pandas().xyxy[0]  # pandas DataFrame
    rows = df[df['name'] == class_name]
    boxes = []
    scores = []
    for _, r in rows.iterrows():
        x1,y1,x2,y2,conf = float(r['xmin']), float(r['ymin']), float(r['xmax']), float(r['ymax']), float(r['confidence'])
        boxes.append([x1,y1,x2,y2])
        scores.append(conf)
    return boxes, scores

def boxes_from_detr(processor, model, image_rgb, threshold=0.5):
    # image_rgb: numpy RGB image (H,W,3)
    inputs = processor(images=image_rgb, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]])  # (H, W)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    boxes_by_label = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_text = model.config.id2label[int(label.item())]
        box_list = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        boxes_by_label.setdefault(label_text, ([],[]))
        boxes_by_label[label_text][0].append(box_list)
        boxes_by_label[label_text][1].append(float(score.item()))
    return boxes_by_label

def apply_nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return [], []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_t, scores_t, iou_thresh)
    kept_boxes = boxes_t[keep].tolist()
    kept_scores = scores_t[keep].tolist()
    return kept_boxes, kept_scores

def draw_boxes(img_bgr, boxes, color=(0,255,0), label_text=''):
    for b in boxes:
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(img_bgr, (x1,y1), (x2,y2), color, 2)
        if label_text:
            cv2.putText(img_bgr, label_text, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main(image_path):
    # Load models
    print("Loading YOLOv5 (this may download the repo on first run)...")
    cnn_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    print("Loading DETR processor & model (this will download weights)...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    vit_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLO detections
    yolo_results = cnn_model(img_bgr)  # accepts numpy BGR
    car_boxes_yolo, car_scores_yolo = boxes_from_yolo(yolo_results, 'car')
    person_boxes_yolo, person_scores_yolo = boxes_from_yolo(yolo_results, 'person')

    # DETR detections
    boxes_by_label = boxes_from_detr(processor, vit_model, img_rgb, threshold=0.5)
    car_boxes_detr, car_scores_detr = boxes_by_label.get('car', ([],[]))
    person_boxes_detr, person_scores_detr = boxes_by_label.get('person', ([],[]))

    # Combine per class and apply NMS
    all_car_boxes = car_boxes_yolo + car_boxes_detr
    all_car_scores = car_scores_yolo + car_scores_detr
    keep_cars, keep_cars_scores = apply_nms(all_car_boxes, all_car_scores, iou_thresh=0.45)

    all_person_boxes = person_boxes_yolo + person_boxes_detr
    all_person_scores = person_scores_yolo + person_scores_detr
    keep_persons, keep_persons_scores = apply_nms(all_person_boxes, all_person_scores, iou_thresh=0.45)

    # Draw and save
    draw_boxes(img_bgr, keep_cars, color=(0,255,0), label_text='car')
    draw_boxes(img_bgr, keep_persons, color=(255,0,0), label_text='person')

    out_path = 'out.jpg'
    cv2.imwrite(out_path, img_bgr)
    print(f"Saved result to {out_path}")
    print(f"Number of cars: {len(keep_cars)}")
    print(f"Number of people: {len(keep_persons)}")

    # Try to show (may fail in headless environments)
    try:
        cv2.imshow("Result", img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("cv2.imshow failed (maybe headless). Open the output image manually:", out_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python detect_count.py path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
