import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from prices import CURRENCY, PRICES

MIN_FRAMES = 5  # a tracked object must be seen this many frames before it is added


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n.pt", help="e.g. yolo11n.pt or yolo11n_ncnn_model (Pi)")
    p.add_argument("--source", choices=["camera", "image"], default="camera", help="camera or image mode")
    p.add_argument("--cam", type=int, default=0, help="camera index")
    p.add_argument("--image", default="", help="path to a single image for testing without webcam")
    p.add_argument("--imgsz", type=int, default=320, help="smaller = faster (320 good for Pi)")
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()


def add_detections_to_cart(cart, detected_names, source="camera"):
    if source == "image":
        for name in detected_names:
            cart[name] = cart.get(name, 0) + 1
    return cart


def draw_cart(frame, cart):
    lines, total = ["SHOPPING CART"], 0
    for name, qty in cart.items():
        cost = PRICES[name] * qty
        total += cost
        lines.append(f"{name.title()} x{qty}  {CURRENCY}{cost}")
    lines.append(f"TOTAL {CURRENCY}{total}")

    h, w = frame.shape[:2]
    panel_w = 320
    panel = np.full((h, panel_w, 3), (40, 40, 40), dtype=np.uint8)

    display = np.hstack((frame, panel))
    for i, text in enumerate(lines):
        color = (0, 255, 255) if i in (0, len(lines) - 1) else (255, 255, 255)
        cv2.putText(display, text, (w + 20, 25 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return display


def process_frame(model, frame, names, class_ids, args, cart, seen, counted):
    result = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=args.conf,
                         imgsz=args.imgsz, classes=class_ids, verbose=False)[0]

    detected_names = []
    for box in result.boxes:
        name = names[int(box.cls)]
        detected_names.append(name)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tid = int(box.id) if box.id is not None else None

        if args.source == "image":
            label = f"{name} {float(box.conf):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)
            continue

        if tid is not None:
            seen[tid] = seen.get(tid, 0) + 1
            if seen[tid] >= MIN_FRAMES and tid not in counted:
                counted.add(tid)
                cart[name] = cart.get(name, 0) + 1
        label = f"{name} {float(box.conf):.2f}" + (f" #{tid}" if tid is not None else "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)

    if args.source == "image":
        add_detections_to_cart(cart, detected_names, source="image")

    return draw_cart(frame, cart)


def main():
    args = get_args()
    model = YOLO(args.model)
    names = model.names  # {id: name}
    class_ids = [i for i, n in names.items() if n in PRICES]
    missing = [n for n in PRICES if n not in names.values()]
    if missing:
        print(f"WARNING: model has no class for {missing} - they will never be detected.")

    seen, counted, cart = {}, set(), {}
    print(f"Source mode: {args.source}")

    if args.source == "image":
        if not args.image:
            sys.exit("Please provide --image path/to/image.jpg when using --source image")
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Could not open image: {args.image}")
        frame = process_frame(model, frame, names, class_ids, args, cart, seen, counted)
        cv2.imshow("Smart Shopping Cart", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.cam, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit("Cannot open camera. Try --cam 1 or close other apps using it.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = process_frame(model, frame, names, class_ids, args, cart, seen, counted)
        cv2.imshow("Smart Shopping Cart", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            cart.clear(); counted.clear(); seen.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
