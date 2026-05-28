import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import uuid
from util import set_background, write_csv
from paddleocr import PaddleOCR # V2 Upgrade

# Initialize models and settings
LICENSE_MODEL_DETECTION_DIR = "./license_plate_detector.pt" 
reader = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
output_csv_path = "live_video_results.csv"

# Set background
# set_background("./imgs/background.png") # Commented out to prevent crash

# Helper function to read license plates
def read_license_plate(license_plate_crop_gray):
    import cv2
    gray = cv2.resize(license_plate_crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    detections = reader.ocr(gray, cls=True)

    if not detections or not detections[0]:
        return None

    plate_fragments = []
    for line in detections[0]:
        text, score = line[1][0], line[1][1]
        if score > 0.3:
            plate_fragments.append(text.upper())

    if not plate_fragments:
        return None

    return ''.join(''.join(plate_fragments).split())

# Custom VideoProcessor for live video
class LicensePlateProcessor(VideoProcessorBase):
    def __init__(self):
        self.results = []
        self.frame_rate = 0
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_copy = img.copy()
        license_detections = license_plate_detector(img_copy, conf=0.50)[0]

        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            # Crop and process license plate
            license_plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            license_plate_text = read_license_plate(license_plate_crop_gray)

            if license_plate_text:
                timestamp = self.frame_count / self.frame_rate if self.frame_rate else 0
                self.results.append({
                    "Timestamp (s)": timestamp,
                    "License Plate": license_plate_text,
                    "Bounding Box": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                })

                # Overlay license plate text
                cv2.putText(img, license_plate_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def save_results(self):
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(output_csv_path, index=False)

# Main Streamlit UI
st.title("💥 License Plate Detection - Live Video 🚗")

if st.sidebar.button("Start Live Detection"):
    st.sidebar.write("Live video detection started...")
    
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    processor = LicensePlateProcessor()
    webrtc_streamer(
        key="license-detection",
        video_processor_factory=lambda: processor,
        rtc_configuration=rtc_config,
    )

    if st.button("Save Results"):
        processor.save_results()
        st.success("Results saved to live_video_results.csv")
        with open(output_csv_path, "rb") as file:
            st.download_button(
                label="Download Results as CSV",
                data=file,
                file_name="live_video_results.csv",
                mime="text/csv",
            )
