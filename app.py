import sys
import cv2
import numpy as np
import datetime
import torch
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QImage
from ultralytics import YOLO

# ----------------------
# CONFIG
# ----------------------
MODEL_PATH = "best.pt"
VIDEO_SOURCE = "video.mp4" # 0 = webcam
CONF_TH = 0.60

# Otomatik cihaz seçimi (GPU varsa GPU, yoksa CPU)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

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
        self.counted_ids = set()  # Sayılan ID'leri hafızada tutmak için küme

    def resize_mask(self, mask, w, h):
        # Mask'i frame boyutuna getir
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

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]

            # YOLO Tracking (persist=True nesne takibini sağlar)
            results = self.model.track(frame, conf=CONF_TH, iou=0.3, imgsz=1024, persist=True, verbose=False)
            
            masks_to_draw = None
            
            # Sonuçları işle
            if results and results[0].boxes is not None and results[0].masks is not None and results[0].boxes.id is not None:
                all_masks = results[0].masks.data.cpu().numpy()
                all_boxes = results[0].boxes.xyxy.cpu().numpy()
                all_ids = results[0].boxes.id.int().cpu().tolist()
                
                filtered_masks = []
                
                for mask, box, track_id in zip(all_masks, all_boxes, all_ids):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2
                    
                    # Sadece çizginin sağındakiler (w // 2)
                    if cx > (w // 2):
                        filtered_masks.append(mask)
                        
                        # Eğer bu ID daha önce sayılmadıysa say ve kaydet
                        if track_id not in self.counted_ids:
                            self.counted_ids.add(track_id)
                            self.total_count += 1
                
                if len(filtered_masks) > 0:
                    masks_to_draw = np.array(filtered_masks)

            # Maskleri uygula
            vis = frame.copy()
            if masks_to_draw is not None:
                vis = self.apply_masks(vis, masks_to_draw)
                # self.check_crossing(masks, w) -> Artık buna gerek yok

            # Dikey kırmızı çizgi
            cv2.line(vis, (w//2, 0), (w//2, h), (0,0,255), 1)

            # Saat
            now = datetime.datetime.now().strftime("%H:%M:%S")
            cv2.putText(vis, now, (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Sayaç
            cv2.putText(vis, f"Toplam: {self.total_count}", (10,55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

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

        # Worker başlat
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

# ----------------------
# APP RUN
# ----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
