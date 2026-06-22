import cv2
import numpy as np
import time
import tflite_runtime.interpreter as tflite

# -----------------------------
# Load TFLite model
# -----------------------------
model_path = "yolo26n_float32.tflite"  # change to your model path (YOLOv8, YOLOv10, YOLOv11, etc.)
interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details[0]['shape'])
print("Output details:", [out['shape'] for out in output_details])

# -----------------------------
# Auto-detect output format
# -----------------------------
output_shape = output_details[0]['shape']
print(f"Output shape: {output_shape}")

# YOLOv8: [1, 84, 8400] — raw grid outputs, needs full postprocessing
#   middle dim = 4 (box) + 80 (COCO classes) = 84
# YOLOv10+/YOLO26: [1, N, 6] — pre-NMS'd detections [x1, y1, x2, y2, conf, cls_id]
#   last dim = 6
IS_YOLOV8_FORMAT = len(output_shape) == 3 and output_shape[1] == 84
print(f"Detected output format: {'YOLOv8-style (raw grid)' if IS_YOLOV8_FORMAT else 'YOLOv10+/YOLO26-style (pre-NMS detections)'}")

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess(img, input_size):
    img_resized = cv2.resize(img, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_exp = np.expand_dims(img_norm, axis=0)  # [1, H, W, 3]
    return img_exp

# -----------------------------
# NMS function to remove duplicate detections
# -----------------------------
def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression to remove duplicate detections"""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def xywh2xyxy(xywh):
    x, y, w, h = xywh
    return [x - w/2, y - h/2, x + w/2, y + h/2]

# -----------------------------
# Postprocessing for YOLOv8 TFLite (raw grid output: [1, 84, 8400])
# -----------------------------
def postprocess_yolov8(outputs, conf_thres=0.25, iou_thres=0.5):
    boxes, scores, class_ids = [], [], []
    
    # YOLOv8 TFLite output is typically [1, 84, 8400] or similar
    output = outputs[0]  # [1, 84, n]
    output = output.squeeze(0)  # [84, n]
    output = output.transpose(1, 0)  # [n, 84]
    
    for det in output:
        # First 4 elements are box coordinates (x, y, w, h)
        # Next 80 elements are class probabilities
        box = det[:4]
        cls_probs = det[4:]
        
        # Get class with highest probability
        cls_id = np.argmax(cls_probs)
        score = cls_probs[cls_id]
        
        if score < conf_thres:
            continue
            
        boxes.append(box)
        scores.append(float(score))
        class_ids.append(int(cls_id))
    
    # Apply NMS to remove duplicate detections
    if boxes:
        # Convert boxes to xyxy format for NMS
        boxes_xyxy = [xywh2xyxy(box) for box in boxes]
        keep_indices = nms(boxes_xyxy, scores, iou_thres)
        
        boxes = [boxes[i] for i in keep_indices]
        scores = [scores[i] for i in keep_indices]
        class_ids = [class_ids[i] for i in keep_indices]
    
    return boxes, scores, class_ids

# -----------------------------
# Postprocessing for YOLOv10+/YOLO26 TFLite (pre-NMS detections: [1, N, 6])
# Each detection: [x1, y1, x2, y2, confidence, class_id]
# Boxes may be normalized [0,1] OR absolute [0, input_size]
# -----------------------------
def postprocess_yolo26(outputs, conf_thres=0.25, input_size=640):
    boxes, scores, class_ids = [], [], []
    
    output = outputs[0]  # [1, N, 6]
    output = output.squeeze(0)  # [N, 6]
    
    # Inspect first detection to determine coordinate scale
    sample = output[0] if len(output) > 0 else None
    coords_are_absolute = False
    if sample is not None:
        x1s, y1s, x2s, y2s = sample[0], sample[1], sample[2], sample[3]
        # If any coordinate is > 1.5, assume absolute pixel values in input_size space
        if max(x1s, y1s, x2s, y2s) > 1.5:
            coords_are_absolute = True
            print(f"YOLO26 boxes detected as ABSOLUTE coordinates (max={max(x1s, y1s, x2s, y2s):.2f})")
        else:
            print(f"YOLO26 boxes detected as NORMALIZED coordinates (max={max(x1s, y1s, x2s, y2s):.2f})")
    
    for det in output:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < conf_thres:
            continue
        
        if coords_are_absolute:
            # Convert absolute input_size coords to normalized [0,1]
            x1 = x1 / input_size
            y1 = y1 / input_size
            x2 = x2 / input_size
            y2 = y2 / input_size
        
        # Clamp to [0,1]
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        
        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(int(cls_id))
    
    return boxes, scores, class_ids

# -----------------------------
# Unified postprocess — auto-selects based on detected format
# -----------------------------
def postprocess(outputs, conf_thres=0.25, iou_thres=0.5, input_size=640):
    if IS_YOLOV8_FORMAT:
        return postprocess_yolov8(outputs, conf_thres, iou_thres)
    else:
        return postprocess_yolo26(outputs, conf_thres, input_size)

# -----------------------------
# Class names (COCO 80 classes)
# -----------------------------
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# -----------------------------
# Video capture
# -----------------------------
cap = cv2.VideoCapture(0)  # or path to video

input_size = input_details[0]['shape'][1]  # e.g., 320

prev_time = time.time()
frame_count = 0
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    h, w, _ = frame.shape
    img_input = preprocess(frame, input_size)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    
    # Get all outputs
    output_data = []
    for i in range(len(output_details)):
        output_data.append(interpreter.get_tensor(output_details[i]['index']))

    boxes, scores, class_ids = postprocess(output_data, conf_thres=0.25, iou_thres=0.5, input_size=input_size)

    # Draw bounding boxes
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if IS_YOLOV8_FORMAT:
            # YOLOv8: boxes are xywh normalized, convert to xyxy pixel coords
            x1, y1, x2, y2 = xywh2xyxy(box)
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
        else:
            # YOLOv10+/YOLO26: boxes are already xyxy normalized
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Safe class name access
        if 0 <= cls_id < len(CLASS_NAMES):
            label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        else:
            label = f"Class_{cls_id}: {score:.2f}"
            
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # Average FPS
    elapsed = curr_time - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Debug info
    cv2.putText(frame, f"Detections: {len(boxes)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("YOLO TFLite Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

print(f"Final Average FPS: {frame_count / (time.time() - start_time):.2f}")
cap.release()
cv2.destroyAllWindows()