"""
Live Prediction Page
"""

import streamlit as st
import requests
import sys
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils.db_manager import get_db_manager

# Page config
st.set_page_config(
    page_title="Live Prediction",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("Live Prediction")
st.write("Test the sentiment analysis model dengan input review sendiri.")

st.markdown("---")

# Main prediction form
st.subheader("Input Movie Review")

user_input = st.text_area(
    "Enter your review here:",
    height=150,
    max_chars=5000,
    placeholder="Type atau paste movie review di sini...\n\nContoh: This movie was amazing! The acting was superb and the plot kept me engaged."
)

# Predict button
if st.button("Predict Sentiment", type="primary"):
    if not user_input or user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Call API
                import os
                api_url = os.getenv('API_URL', 'http://localhost:8000')
                start_time = time.time()
                
                response = requests.post(
                    f"{api_url}/predict",
                    json={"text": user_input},
                    timeout=10
                )
                
                response_time = (time.time() - start_time) * 1000  # ms
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display result
                    st.success("Prediction complete!")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        sentiment = result['sentiment'].upper()
                        if sentiment == 'POSITIVE':
                            st.info(f"**Sentiment:** {sentiment}")
                        else:
                            st.warning(f"**Sentiment:** {sentiment}")
                    
                    with col_res2:
                        confidence = result['confidence'] * 100
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    st.write("**Probabilities:**")
                    st.write(f"- Positive: {result['probabilities']['positive']*100:.2f}%")
                    st.write(f"- Negative: {result['probabilities']['negative']*100:.2f}%")
                    
                    st.caption(f"Response time: {response_time:.2f}ms")
                    
                else:
                    st.error(f"Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure the API service is running.")
            except requests.exceptions.Timeout:
                st.error("Request timeout. API might be overloaded.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("---")

# Recent predictions
st.subheader("Recent Predictions")

db = get_db_manager()
recent_df = db.get_recent_predictions(5)

if not recent_df.empty:
    for _, pred in recent_df.iterrows():
        with st.expander(f"{pred['timestamp']} - {pred['sentiment'].upper()}"):
            st.write(f"**Text:** {pred['text']}")
            st.write(f"**Confidence:** {pred['confidence']*100:.1f}%")
else:
    st.info("No predictions yet. Try making a prediction above!")
