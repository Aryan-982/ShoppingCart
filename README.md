# Smart Shopping Cart (YOLO + OpenCV)

## Install (Windows)

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Install (Raspberry Pi, 64-bit OS, Pi 4/5)

    sudo apt update && sudo apt install -y python3-venv libgl1
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## Run webcam

    python main.py

## Run from an image (works without light or webcam)

    python main.py --source image --image sample.jpg

This mode is useful when the room is dark or when you want to test the cart logic on a saved product photo before switching back to the webcam.

The model (yolo11n.pt, ~6 MB) downloads automatically on first run (needs internet once).

Options: `--cam 1` `--imgsz 320` `--conf 0.5` `--model path` `--source image --image sample.jpg`

## Faster on Raspberry Pi (NCNN)

    yolo export model=yolo11n.pt format=ncnn imgsz=320
    python main.py --model yolo11n_ncnn_model --imgsz 320

## Keys

q = quit, r = reset cart

## Supported items

Only COCO classes: apple, banana, orange, broccoli, carrot, bottle, cup (plus sandwich,
hot dog, pizza, donut, cake). NOT in the pretrained model: tomato, potato, onion, packaged
snacks, etc. Those need a custom-trained model (e.g. Roboflow dataset + `yolo train`).

## Common errors

- "Cannot open camera": use `--cam 1`, close Zoom/Teams/browser tabs using the camera.
- `No module named ultralytics/cv2`: activate the venv, rerun pip install.
- Model download fails: check internet, or copy yolo11n.pt next to main.py.
- Old ultralytics can't load yolo11n: `pip install -U ultralytics`, or use `--model yolov8n.pt`.
- Slow/laggy: lower `--imgsz` (256/320), use NCNN on Pi, lower `--width/--height`.
- Black window on Pi with Pi Camera Module: use a USB webcam, or run `libcamerify python main.py`.
- Apple counted twice: raise MIN_FRAMES or `--conf`; IDs can change if the item leaves view.

## How it works

1. Each frame goes to YOLO (`model.track`), limited to the classes listed in prices.py.
2. ByteTrack gives each physical object a persistent track ID across frames.
3. An ID seen for 5+ frames is added to the cart once (`counted` set), so holding
   one apple still gives Apple x1. A second apple gets a new ID, so Apple x2.
4. Cart quantity x price from prices.py is drawn live with the total.
   Limitation: an item that leaves and re-enters view can get a new ID and count again.

## Next upgrades

Barcode scanning (pyzbar), RFID, voice (pyttsx3), SQLite product DB, "remove item"
gesture/key, checkout + UPI QR, Tkinter/PyQt GUI, custom-trained model for Indian products.
