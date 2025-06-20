import sys
import cv2
import time
import numpy as np
import onnxruntime as ort
from collections import defaultdict
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton,
    QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QTextBrowser
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtMultimedia import QSound
import os
import winsound
import threading

class_names = ["awake", "distracted", "drowsy",
               "head drop", "phone", "smoking", "yawn"]

# Per-class confidence thresholds
class_thresholds = {
    "awake": 0.35,
    "distracted": 0.4,
    "drowsy": 0.4,
    "head drop": 0.3,
    "phone": 0.4,
    "smoking": 0.5,
    "yawn": 0.5,
    "Seatbelt": 0.5

}


class SeatbeltDetector:
    def __init__(self, model_path):
        self.last_result = "Unknown"
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.class_names = ["NoSeatbelt", "Seatbelt"]
            self.conf_threshold = class_thresholds["Seatbelt"]
            self.last_result = "Unknown"
            self.last_check_time = 0
            self.check_interval = 2  # seconds
        except Exception as e:
            print(f"Seatbelt model load error: {e}")
            self.session = None

    def preprocess(self, frame):
        img, _, dw, dh = self.letterbox(frame, new_shape=(640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return img[np.newaxis, :], dw, dh

    def letterbox(self, image, new_shape=(640, 640), color=(114, 114, 114)):
        shape = image.shape[:2]
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)),
                     int(round(shape[0] * ratio)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        return padded, ratio, dw, dh

    def detect(self, frame):
        if self.session is None:
            return self.last_result
        current_time = time.time()
        if current_time - self.last_check_time < self.check_interval:
            return self.last_result
        self.last_check_time = current_time

        input_tensor, dw, dh = self.preprocess(frame)
        best_score = self.conf_threshold
        best_label = self.last_result
        try:
            outputs = self.session.run(
                None, {self.input_name: input_tensor})[0][0]
            for det in outputs:
                x1, y1, x2, y2, score, cls_id = det
                if score >= best_score:
                    cls_id = int(cls_id)
                    if 0 <= cls_id < len(self.class_names):
                        best_label = self.class_names[cls_id]
                        best_score = score
        except Exception as e:
            print(f"Seatbelt detection failed: {e}")
        self.last_result = best_label
        return self.last_result


class DetectionHistory:
    def __init__(self):
        self.start_times = defaultdict(float)
        self.active_flags = defaultdict(bool)
        self.yawn_windows = []
        self.yawn_triggered = False
        self.last_yawn_increment_time = 0.0
        self.yawn_status = "no yawn"
        self.blink_count = 0
        self.eye_closed = False
        self.eye_close_time = 0.0
        self.min_blink_duration = 0.3
        self.blink_window = []
        self.secondary_blink_count = 0
        self.last_secondary_reset_time = time.time()
        self.threshold = 0

    def update(self, label_scores, current_time, conf_threshold):
        durations = {}
        any_detection = False
        for label, score in label_scores.items():
            self.threshold = conf_threshold.get(
                label, 0.8) if conf_threshold else 0.8
            if score >= self.threshold:
                any_detection = True
                if not self.active_flags[label]:
                    self.start_times[label] = current_time
                    self.active_flags[label] = True
                durations[label] = current_time - self.start_times[label]
                if label == "yawn":
                    if not self.yawn_triggered and durations[label] >= 1:
                        self.yawn_windows.append(current_time)
                        self.yawn_triggered = True
                        self.last_yawn_increment_time = current_time
                    elif self.yawn_triggered and current_time - self.last_yawn_increment_time >= 1:
                        self.yawn_windows.append(current_time)
                        self.last_yawn_increment_time = current_time
                if label == "drowsy":
                    if not self.eye_closed:
                        self.eye_closed = True
                        self.eye_close_time = current_time
            else:
                if self.active_flags[label]:
                    self.active_flags[label] = False
                    durations[label] = current_time - self.start_times[label]
                else:
                    durations[label] = 0.0
                if label == "yawn":
                    self.yawn_triggered = False
                    self.last_yawn_increment_time = 0.0
                if label == "drowsy" and self.eye_closed:
                    closed_duration = current_time - self.eye_close_time
                    if closed_duration >= self.min_blink_duration:
                        self.blink_count += 1
                        self.blink_window.append(current_time)
                    self.eye_closed = False
        self.blink_window = [
            t for t in self.blink_window if current_time - t <= 20]
        self.secondary_blink_count = len(self.blink_window)
        self.yawn_windows = [
            t for t in self.yawn_windows if current_time - t <= 60]
        return durations, len(self.yawn_windows), any_detection


class AlertStats:
    def __init__(self):
        self.alert_log = []
        self.current_alert = None
        self.alert_start_time = None

    DISTRACTION_ALERTS = {
        '📵 Distracted (Indirect Phone Use)!',
        '📵 Distracted (Phone + Other Signs)!',
        '📵 Distracted!',
        '📵 Distracted (No Activity)!'
    }

    DROWSY_ALERTS = {
        '😴 Drowsy!',
        '😴 Drowsy (Yawns detected)!',
        '😴 Drowsy (Head Drop)!',
        '😴 Drowsy (Frequent Blinks)'
    }

    def update(self, alert_str, current_time):
        if self.current_alert != alert_str:
            if self.current_alert:
                duration = current_time - self.alert_start_time
                self.alert_log.append((self.current_alert, duration))
            self.current_alert = alert_str
            self.alert_start_time = current_time

    def finalize(self, current_time):
        if self.current_alert:
            duration = current_time - self.alert_start_time
            self.alert_log.append((self.current_alert, duration))
            self.current_alert = None

    def compute_stats(self, video_fps):
        from collections import defaultdict

        freq = defaultdict(int)
        durations = defaultdict(float)

        for alert, dur in self.alert_log:
            if alert in self.DROWSY_ALERTS:
                key = "Drowsy"
            elif alert in self.DISTRACTION_ALERTS:
                key = "Distraction"
            else:
                continue
            if alert == "😴 Drowsy (Frequent Blinks)":
                freq[key] += 5
               # freq[key] += video_fps
            freq[key] += 1
            durations[key] += dur

        most_freq = max(freq.items(), key=lambda x: x[1], default=(None, 0))
        longest = max(durations.items(), key=lambda x: x[1], default=(None, 0))

        # Weighted classification
        total = sum(durations.values()) + sum(freq.values())
        if total == 0:
            video_type = "Normal"
        else:
            score = {
                k: 0.6 * durations[k] + 0.4 * freq[k] for k in {"Drowsy", "Distraction"}
            }
            video_type = max(score.items(), key=lambda x: x[1])[0]

        return {
            "most_frequent_state": most_freq,
            "longest_lasting_state": longest,
            "video_type": video_type
        }


class VideoProcessingThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_update_signal = pyqtSignal(dict)

    def __init__(self, session, input_name, cap, alert_sound=None, fps_limit=15, is_webcam=False):
        super().__init__()
        self.session = session
        self.input_name = input_name
        self.cap = cap
        self.alert_sound = alert_sound
        self.conf_threshold = class_thresholds
        self.fps_limit = fps_limit
        self.is_webcam = is_webcam
        self.video_fps = self.cap.get(
            cv2.CAP_PROP_FPS) if not is_webcam else fps_limit
        self.running = True
        self.history = DetectionHistory()
        self.last_frame_time = 0
        self.last_detection_time = time.time()
        self.last_drowsy_alert_time = 0
        self.last_distracted_alert_time = 0
        self.alert_display_duration = 3
        self.log_file_path = "Drowsiness_distraction_log.json"
        self.indirect_phone_counter = 0
        self.indirect_phone_detected = False
        self.last_phone_detected_time = 0
        self.seatbelt_detector = SeatbeltDetector(
            "runs/seatbelt/train/yolov8n_model/weights/best.onnx")
        self.last_seatbelt_result = "Unknown"
        self.alert_stats = AlertStats()
        self.blink_count = 0
        self.thresholds = 0

    """def play_alert_sound(self):
        if self.alert_sound:
            self.alert_sound.play()"""

    def play_alert_sound(self):
        frequency = 1000
        duration = 500
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()

    def preprocess(self, frame):
        img, ratio, dw, dh = self.letterbox(frame, new_shape=(640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        return img[np.newaxis, :], ratio, dw, dh

    def letterbox(self, image, new_shape=(640, 640), color=(114, 114, 114)):
        shape = image.shape[:2]
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * ratio)),
                     int(round(shape[0] * ratio)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
        return padded, ratio, dw, dh

    def run(self):
        while self.running and self.cap.isOpened():
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            frame_delay = 1 / self.video_fps
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)
            self.last_frame_time = time.time()

            if self.is_webcam:
                for _ in range(2):
                    self.cap.grab()
            ret, frame = self.cap.read()
            if not ret:
                break

            input_tensor, ratio, dw, dh = self.preprocess(frame)
            try:
                outputs = self.session.run(
                    None, {self.input_name: input_tensor})
            except Exception as e:
                print(f"ONNX inference error: {e}")
                continue

            detections = outputs[0][0]
            label_scores = {label: 0 for label in class_names}
            for det in detections:
                x1, y1, x2, y2, score, cls_id = det
                if score < 0.3:
                    continue
                cls_id = int(cls_id)
                if 0 <= cls_id < len(class_names):
                    label = class_names[cls_id]
                    label_scores[label] = max(
                        label_scores[label], round(score, 2))
                    x1 = int((x1 - dw) / ratio)
                    y1 = int((y1 - dh) / ratio)
                    x2 = int((x2 - dw) / ratio)
                    y2 = int((y2 - dh) / ratio)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            durations, yawn_count, any_detection = self.history.update(
                label_scores, current_time, self.conf_threshold)

            self.indirect_phone_detected = False
            if label_scores.get("phone", 0) >= self.history.threshold:
                self.last_phone_detected_time = current_time
                self.indirect_phone_counter = 0
            elif current_time - self.last_phone_detected_time <= 12.0:
                if durations.get("yawn", 0) < 1.5 and durations.get("drowsy", 0) < 2.0:
                    if durations.get("head drop", 0) > 0 or durations.get("drowsy", 0) > 0 or durations.get("distracted", 0) > 0:
                        self.indirect_phone_counter += 1
                    else:
                        self.indirect_phone_counter = max(
                            0, self.indirect_phone_counter - 1)
                    if self.indirect_phone_counter >= 15:
                        self.indirect_phone_detected = True

            alert = "<b><span style='color:gray;'>⏳ Analyzing...</span></b>"
            phone_duration = durations.get("phone", 0)
            distracted_duration = durations.get("distracted", 0)
            smoking_duration = durations.get("smoking", 0)
            drowsy_duration = durations.get("drowsy", 0)
            head_drop_duration = durations.get("head drop", 0)

            # 1. Indirect phone usage detected (assuming a flag or condition, e.g., self.indirect_phone_detected)
            if getattr(self, 'indirect_phone_detected', False):
                alert = "<b><span style='color:orange;'>📵 Distracted (Indirect Phone Use)!</span></b>"
                self.play_sound_in_thread()

            # 2. Phone usage > 2s + drowsy or distracted or head nod → Distraction
            elif phone_duration > 2 and (drowsy_duration > 0 or distracted_duration > 0 or head_drop_duration > 0):
                alert = "<b><span style='color:orange;'>📵 Distracted (Phone + Other Signs)!</span></b>"
                self.play_sound_in_thread()

            elif distracted_duration >= 1.5 or smoking_duration >= 3:
                alert = "<b><span style='color:orange;'>📵 Distracted!</span></b>"

            # 4. Drowsiness detected from duration
            elif drowsy_duration >= 2 and distracted_duration < 2:
                alert = "<b><span style='color:red;'>😴 Drowsy!</span></b>"

            # 5. Yawn count threshold met
            elif yawn_count >= 3 and self.history.yawn_triggered:
                alert = "<b><span style='color:red;'>😴 Drowsy (Yawns detected)!</span></b>"

           # 6. Head drop + drowsiness
            elif head_drop_duration >= 2:
                if drowsy_duration >= 1:
                    alert = "<b><span style='color:red;'>😴 Drowsy (Head Drop)!</span></b>"
                else:
                    alert = "<b><span style='color:orange;'>📵 Distracted!</span></b>"

            # 7. Frequent blinks + distraction signs → Distraction
            elif self.history.secondary_blink_count >= 5 and (self.indirect_phone_detected == False):
                if self.blink_count < self.history.secondary_blink_count:
                    self.blink_count = self.history.secondary_blink_count
                    alert = "<b><span style='color:red;'>😴 Drowsy (Frequent Blinks)</span></b>"

            elif head_drop_duration > 0 and self.history.secondary_blink_count >= 4:
                alert = "<b><span style='color:red;'>😴 Drowsy (Head Drop)!</span></b>"

            # 9. No detections for some time
            elif not any_detection and current_time - self.last_detection_time >= 3:
                alert = "<b><span style='color:orange;'>📵 Distracted (No Activity)!</span></b>"

            # 10. Focused state
            elif label_scores.get("awake", 0) > self.history.threshold:
                alert = "<b><span style='color:green;'>✅ Awake and Focused</span></b>"

            if any_detection:
                self.last_detection_time = current_time

            self.last_seatbelt_result = self.seatbelt_detector.detect(frame)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h,
                             bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(q_image)

            plain_alert = alert.replace("<b><span style='color:orange;'>", "").replace(
                "<b><span style='color:red;'>", "").replace("</span></b>", "")

            self.alert_stats.update(plain_alert, current_time)

            def convert(obj):
                return obj.item() if isinstance(obj, np.generic) else obj
            alert_stats = self.alert_stats.compute_stats(self.video_fps)
            status_data = {
                "alert": alert,
                "yawn_count_60s": int(yawn_count),
                **{k: convert(v) for k, v in label_scores.items()},
                "yawn_duration": convert(durations.get("yawn", 0.0)),
                "blink_count_20s": self.history.secondary_blink_count,
                "indirect_phone_usage": self.indirect_phone_detected,
                "seatbelt": self.last_seatbelt_result,
                "most_freq": alert_stats["most_frequent_state"],
                "longest": alert_stats["longest_lasting_state"],
                "video_type": alert_stats["video_type"]
            }
            self.status_update_signal.emit(status_data)
            self.alert_stats.finalize(time.time())
        self.cap.release()


class VideoDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Team SHIFT-GEAR: Drowsiness Detector")
        self.setGeometry(100, 100, 1200, 700)
        try:
            self.session = ort.InferenceSession(
                "runs/drowsy/train/yolov8n_model/weights/best.onnx")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            sys.exit(1)
        self.input_name = self.session.get_inputs()[0].name
        self.label = QLabel(self)
        self.label.setFixedSize(800, 600)
        self.label.setStyleSheet("background-color: black;")
        self.info_browser = QTextBrowser()
        self.info_browser.setFixedSize(350, 600)
        self.info_browser.setStyleSheet(
            "background-color: #f0f8ff; font-size: 16px; font-family: Arial; padding: 10px;")
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        self.video = None
        self.webcam_button = QPushButton("Use Webcam")
        self.webcam_button.clicked.connect(self.start_webcam)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.webcam_button)
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.label)
        main_layout.addWidget(self.info_browser)
        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(button_layout)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.cap = None
        self.thread = None
        try:
            self.alert_sound = QSound("alert.wav")
        except Exception:
            self.alert_sound = None

    def start_thread(self, is_webcam=False):
        if self.cap:
            if self.thread:
                self.thread.running = False
                self.thread.wait()
            self.thread = VideoProcessingThread(
                self.session, self.input_name, self.cap, self.alert_sound, is_webcam=is_webcam)
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.status_update_signal.connect(self.update_status)
            self.thread.start()
            return self.thread

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File")
        if video_path:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(video_path)
            self.video = os.path.basename(video_path)
            self.start_thread(is_webcam=False)

    def start_webcam(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.info_browser.setText("❌ Failed to open webcam.")
            return
        self.start_thread(is_webcam=True)

    def update_image(self, q_image):
        self.label.setPixmap(QPixmap.fromImage(q_image))

    def update_status(self, status_data):
        if ("Analyzing" not in status_data["alert"] and "Awake" not in status_data["alert"]) and self.thread is not None:
            self.thread.play_sound_in_thread()
        # self.video=os.path.basename(self.video)
        # print(self.video)
        html = f"""
        <h2 style='color:#2e86de;'>🚗 <b>DROWSINESS DETECTOR</b></h2>
        <p><b>Status:</b> {status_data["alert"]}</p>
        <hr>
        <p><b>File_Name:</b> {self.video}</p>
        <p><b>🟢 Awake:</b> {status_data.get('awake', 0)}</p>
        <p><b>😴 Drowsy:</b> {status_data.get('drowsy', 0)}</p>
        <p><b>⏱️ Blinks (Last 20s):</b> {status_data.get('blink_count_20s', 0)}</p>
        <p><b>📵 Distracted:</b> {status_data.get('distracted', 0)}</p>
        <p><b>🚬 Smoking:</b> {status_data.get('smoking', 0)}</p>
        <p><b>🔁 Yawns (Last 60s):</b> {status_data.get('yawn_count_60s', 0)}</p>
        <p><b>📉 Head Drop:</b> {status_data.get('head drop', 0)}</p>
        <p><b>📱 Phone:</b> {status_data.get('phone', 0)}{' <span style="color:red;">(indirect)</span>' if status_data.get('indirect_phone_usage') else ''}</p>
        <p><b>🎯 Seatbelt:</b> {status_data.get('seatbelt', 'Unknown')}</p>
        <p><b>🎥 Video class:</b> {status_data.get('video_type')}</p>
        <p><b>📊 Most Frequent State:</b> {status_data.get('most_freq')}</p>
        """
        self.info_browser.setHtml(html)

    def closeEvent(self, event):
        if self.thread:
            self.thread.running = False
            self.thread.wait()
        if self.cap:
            self.cap.release()
        if self.alert_sound:
            self.alert_sound.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoDetectionApp()
    window.show()
    sys.exit(app.exec_())
