import sys
import cv2
import numpy as np
import datetime
import torch
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QImage
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import serial
import serial.tools.list_ports
import time


# ----------------------
# CONFIG
# ----------------------
MODEL_PATH = "best.pt"
VIDEO_SOURCE = "video.mp4" # 0 = webcam
CONF_TH = 0.60

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ----------------------
# GLOBAL LOG COUNTER
# ----------------------
product_global_id = 0

def log_detection(track_id):
    global product_global_id
    product_global_id += 1
    ts = datetime.datetime.now().strftime("[%H:%M:%S]")
    line = f"{ts} ID: {track_id} - Ürün Tespit Edildi\n"
    with open("detections.log", "a", encoding="utf-8") as f:
        f.write(line)
    return product_global_id

def find_arduino():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        text = p.description.lower()
        if ("arduino" in text) or ("ch340" in text) or ("usb serial" in text):
            return p.device
    return None



# ----------------------
# WORKER THREAD
# ----------------------
class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray, int)

    def __init__(self):
        super().__init__()
        self.model = YOLO(MODEL_PATH).to(DEVICE)
        self.cap = cv2.VideoCapture(VIDEO_SOURCE)
        self.running = True
        self.total_count = 0

        # DeepSort tracker
        self.tracker = DeepSort(
            max_age=5,
            n_init=2,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            polygon=False
        )

        self.counted_ids = set()

        self.arduino = ArduinoSerial()
        self.last_sent_count = -1

    def resize_mask(self, mask, w, h):
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    def apply_masks(self, frame, masks):
        if masks is None or len(masks) == 0:
            return frame
        h, w, _ = frame.shape
        overlay = frame.copy()

        for mask in masks:
            mask_resized = self.resize_mask(mask, w, h)
            color = np.random.randint(0,255,3).tolist()
            color_mask = np.zeros_like(frame, dtype=np.uint8)
            color_mask[mask_resized == 1] = color
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

        return overlay

    def check_crossing(self, track_id, bbox, frame_w):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2

        line_x = frame_w // 2

        if track_id in self.counted_ids:
            return

        if cx >= line_x:
            self.total_count += 1
            self.counted_ids.add(track_id)
            log_detection(track_id)

            # Arduino'ya gönder
            self.arduino.send(self.total_count)

    # ----------------------

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            h, w = frame.shape[:2]

            # YOLO inference
            results = self.model(frame, conf=CONF_TH, iou=0.3, imgsz=1024, verbose=False)

            masks = []
            detections = []

            if results:
                r = results[0]


                if r.masks is not None:
                    masks = r.masks.data.cpu().numpy()

                if r.boxes is not None:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()

                    for box, conf in zip(xyxy, confs):
                        x1, y1, x2, y2 = box
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            # Mask overlay
            vis = frame.copy()
            if masks is not None:
                vis = self.apply_masks(vis, masks)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()  # x1,y1,x2,y2

                # SAYMA
                self.check_crossing(track_id, ltrb, w)

                """ #DEBUG: Track ID'yi yazdırma
                cv2.putText(vis, f"ID:{track_id}",
                            (int(ltrb[0]), int(ltrb[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)
                            

                cx = int((ltrb[0] + ltrb[2]) / 2)
                cy = int((ltrb[1] + ltrb[3]) / 2)
                cv2.circle(vis, (cx, cy), 3, (0, 255, 255), -1)"""

            line_x = w // 2
            cv2.line(vis, (line_x, 0), (line_x, h), (0, 0, 255), 1)

            now = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(vis, now, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(vis, f"Toplam: {self.total_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            self.frame_ready.emit(vis, self.total_count)

        self.cap.release()


# ----------------------
# MAIN WINDOW
# ----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chocolate Counter - PySide6")
        self.resize(900, 600)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.start()

    def update_frame(self, frame, count):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = w * ch
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.worker.running = False
        self.worker.wait()
        event.accept()



class ArduinoSerial:
    def __init__(self, baudrate=9600):
        self.baudrate = baudrate
        self.ser = None
        self.connect()

    def find_arduino(self):
        ports = serial.tools.list_ports.comports()
        for p in ports:
            name = p.description.lower()
            if ("arduino" in name) or ("ch340" in name) or ("usb" in name):
                return p.device
        return None

    def connect(self):
        port = self.find_arduino()
        if port is None:
            print("Arduino bulunamadı, tekrar denenecek…")
            return False

        try:
            self.ser = serial.Serial(port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino reset olur, bekleme şart
            print(f"Arduino bağlandı: {port}")
            return True
        except Exception as e:
            print("Arduino bağlantı hatası:", e)
            self.ser = None
            return False

    def send(self, text):
        if self.ser is None:
            self.connect()

        if self.ser:
            try:
                self.ser.write(f"{text}\n".encode())
            except Exception:
                print("Arduino bağlantısı koptu. Yeniden bağlanıyor…")
                self.ser = None
                self.connect()

# ----------------------
# RUN
# ----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
