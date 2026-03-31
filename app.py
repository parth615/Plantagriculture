import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from google import genai
import cv2
import pandas as pd
import requests
import os
from datetime import datetime
from PIL import Image
from fpdf import FPDF

API_KEY = st.secrets["AIzaSyA-ZP1A37UAML6OCag5iVh7npuOjS9MG9E"]
client = genai.Client(api_key=API_KEY)

st.set_page_config(page_title="AgriScan AR Pro", layout="wide")


def get_user_location():
    try:
        response = requests.get("https://ipapi.co/json/").json()
        return {"city": response.get("city", "Delhi"), "lat": response.get("latitude", 28.6),
                "lon": response.get("longitude", 77.2)}
    except:
        return {"city": "New Delhi", "lat": 28.6, "lon": 77.2}


def save_to_history(crop, disease, advice):
    new_data = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M"), crop, disease, advice]],
                            columns=['Timestamp', 'Crop', 'Disease', 'Advice Summary'])
    if not os.path.isfile("history.csv"):
        new_data.to_csv("history.csv", index=False)
    else:
        new_data.to_csv("history.csv", mode='a', header=False, index=False)


class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.status = "READY TO SCAN"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (int(w * 0.2), int(h * 0.2)), (int(w * 0.8), int(h * 0.8)), color, 2)

        cv2.putText(img, f"AI STATE: {self.status}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img


tab1, tab2, tab3 = st.tabs(["🔍 Live AR Scanner", "📜 History Log", "🌍 Local Weather"])

user_loc = get_user_location()

with tab1:
    st.title("🌱 AgriScan AR")
    st.markdown(f"**Location:** {user_loc['city']} | **Mode:** Real-time Detection")

    ctx = webrtc_streamer(key="agri-scan", video_processor_factory=VideoProcessor)

    if st.button("📸 Capture & Identify Disease"):
        if ctx.video_receiver:
            frame = ctx.video_receiver.get_frame()
            if frame:
                img_array = frame.to_ndarray(format="bgr24")
                img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

                with st.spinner("AI analyzing live frame..."):
                    prompt = "Analyze this crop. Identify the plant and any disease. Give 3 quick tips in English and Hindi."
                    response = client.models.generate_content(model='gemini-3-flash-preview',
                                                              contents=[prompt, img_pil])

                    st.subheader("📊 Scan Results")
                    st.write(response.text)

                    save_to_history("Detected Crop", "Analysis Result", "Organic Management")
                    st.success("Analysis saved to history!")
        else:
            st.error("Please start the camera first!")

with tab2:
    st.header("Previous Scans")
    if os.path.isfile("history.csv"):
        df = pd.read_csv("history.csv")
        st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
        if st.button("🗑️Clear History"):
            os.remove("history.csv")
            st.rerun()
    else:
        st.info("No history found. Start scanning to see data here.")

with tab3:
    st.header("Climate Context")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={user_loc['lat']}&longitude={user_loc['lon']}&current_weather=true"
    weather = requests.get(url).json()['current_weather']

    col1, col2 = st.columns(2)
    col1.metric("Temperature", f"{weather['temperature']}°C")
    col2.metric("Wind Speed", f"{weather['windspeed']} km/h")
    st.write("Current weather impacts how diseases spread. Keep an eye on humidity levels!")