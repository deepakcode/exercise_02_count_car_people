```bash
./setup.sh
source .venv/bin/activate
python detect_count.py path/to/street_image.jpg

```

# **Design Document: Street Image Car and People Counting System**

---

## **1. Introduction**

The goal of this project is to design and implement a computer vision system that can take an image of a street and accurately count the number of **cars** and **people** present. This system leverages **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** to combine the strengths of both architectures.

* **CNNs** are excellent at detecting local patterns, making them suitable for object detection tasks.
* **ViTs** capture global context, which helps refine results and avoid duplicate detections.

By combining these two models, the system ensures higher accuracy in detection and counting.

---

## **2. Problem Statement**

Given a street image:

* Detect **cars** and **people** using pre-trained object detection models.
* Refine detections to avoid duplicate bounding boxes.
* Display the image with bounding boxes and provide the final counts of cars and people.

---

## **3. System Architecture**

The system architecture is designed as a **multi-stage pipeline**:

1. **Input Image**

   * A street image is loaded using OpenCV.

2. **CNN-based Detection**

   * A pre-trained YOLOv5 (or SSD) model is used to detect cars and people.
   * Outputs bounding boxes with confidence scores.

3. **ViT-based Detection**

   * A pre-trained DETR (Vision Transformer) model refines detections.
   * Provides additional bounding boxes and avoids missed objects.

4. **Result Fusion**

   * Bounding boxes from CNN and ViT are combined.
   * Non-Maximum Suppression (NMS) removes duplicates.

5. **Visualization & Output**

   * Bounding boxes are drawn on the image (Green for cars, Blue for people).
   * Final counts are displayed both visually and in the console.

---

## **4. Libraries & Tools**

* **OpenCV** → Image loading, preprocessing, and visualization.
* **PyTorch** → Loading and using CNN-based models like YOLOv5.
* **Transformers (Hugging Face)** → Using DETR Vision Transformer model.

---

## **5. Implementation**

### **5.1 Step-by-Step Workflow**

1. Install required libraries (`opencv-python`, `torch`, `transformers`).
2. Load pre-trained models (YOLOv5 for CNN, DETR for ViT).
3. Load the input street image.
4. Run inference with CNN → extract car and person bounding boxes.
5. Run inference with ViT → extract car and person bounding boxes.
6. Merge results from both models.
7. Apply Non-Maximum Suppression (NMS) to remove duplicates.
8. Draw bounding boxes and display results.
9. Print the final counts.

### **5.2 Code Implementation**

```python
# detect_count.py
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.ops import nms
from transformers import DetrImageProcessor, DetrForObjectDetection
from ultralytics import YOLO

def boxes_from_yolo(results, class_name):
    boxes = []
    scores = []
    for r in results:  # each r is a Results object
        for box in r.boxes:
            cls_name = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
            if cls_name == class_name:
                boxes.append(xyxy)
                scores.append(conf)
    return boxes, scores

def boxes_from_detr(processor, model, image_rgb, threshold=0.5):
    inputs = processor(images=image_rgb, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image_rgb.shape[:2]])  # (H, W)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    boxes_by_label = {}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_text = model.config.id2label[int(label.item())]
        box_list = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
        boxes_by_label.setdefault(label_text, ([], []))
        boxes_by_label[label_text][0].append(box_list)
        boxes_by_label[label_text][1].append(float(score.item()))
    return boxes_by_label

def apply_nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return [], []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes_t, scores_t, iou_thresh)
    return boxes_t[keep].tolist(), scores_t[keep].tolist()

def draw_boxes(img_bgr, boxes, color=(0, 255, 0), label_text=''):
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        if label_text:
            cv2.putText(img_bgr, label_text, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main(image_path):
    # Load YOLOv5 model
    print("Loading YOLOv5 (this may download weights if not present)...")
    cnn_model = YOLO("yolov5s.pt")

    # Load DETR processor & model
    print("Loading DETR processor & model (this will download weights)...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    vit_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLO detections
    yolo_results = cnn_model(img_bgr)  # can pass path or numpy BGR image
    car_boxes_yolo, car_scores_yolo = boxes_from_yolo(yolo_results, 'car')
    person_boxes_yolo, person_scores_yolo = boxes_from_yolo(yolo_results, 'person')

    # DETR detections
    boxes_by_label = boxes_from_detr(processor, vit_model, img_rgb, threshold=0.5)
    car_boxes_detr, car_scores_detr = boxes_by_label.get('car', ([], []))
    person_boxes_detr, person_scores_detr = boxes_by_label.get('person', ([], []))

    # Combine per class and apply NMS
    all_car_boxes = car_boxes_yolo + car_boxes_detr
    all_car_scores = car_scores_yolo + car_scores_detr
    keep_cars, keep_cars_scores = apply_nms(all_car_boxes, all_car_scores, iou_thresh=0.45)

    all_person_boxes = person_boxes_yolo + person_boxes_detr
    all_person_scores = person_scores_yolo + person_scores_detr
    keep_persons, keep_persons_scores = apply_nms(all_person_boxes, all_person_scores, iou_thresh=0.45)

    # Draw and save
    draw_boxes(img_bgr, keep_cars, color=(0, 255, 0), label_text='car')
    draw_boxes(img_bgr, keep_persons, color=(255, 0, 0), label_text='person')

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
    except Exception:
        print("cv2.imshow failed (maybe headless). Open the output image manually:", out_path)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python detect_count.py path/to/image.jpg")
        sys.exit(1)
    main(sys.argv[1])
```

---

## **6. Results**

* The system successfully identifies cars and people in street images.
* Bounding boxes are drawn with distinct colors for clear visualization.
* The console prints final counts of detected objects.

---

## **7. Learning Insights**

* **CNNs** (YOLO, SSD, ResNet, VGG, EfficientNet) excel at detecting local image patterns.
* **ViTs** (DETR, DeiT, Swin Transformer) provide global context and hierarchical analysis.
* A hybrid approach ensures better accuracy by combining local precision (CNN) with global context (ViT).

---

## **8. Future Work**

1. Implement **Non-Maximum Suppression (NMS)** for duplicate removal.
2. Extend detection to additional classes (bikes, buses, traffic lights).
3. Optimize model inference for real-time detection.
4. Deploy the system as a **web service** or **mobile application**.
5. Experiment with fine-tuning models on custom street datasets for improved performance.

