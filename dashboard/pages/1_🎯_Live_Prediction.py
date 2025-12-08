"""
Live Prediction Page - Test model dengan input real-time
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
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .positive-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .negative-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .sentiment-label {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .confidence-label {
        font-size: 1.5rem;
        text-align: center;
        opacity: 0.9;
    }
    .example-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🎯 Live Sentiment Prediction")
st.markdown("Test the sentiment analysis model dengan input review Anda sendiri.")

st.markdown("---")

# Main content
col_input, col_examples = st.columns([2, 1])

with col_input:
    st.subheader("💬 Enter Movie Review")
    
    # Text input
    user_input = st.text_area(
        "Type or paste your movie review here:",
        height=150,
        placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout...",
        help="Enter any movie review text. Minimum 1 character, maximum 5000 characters."
    )
    
    # Character count
    char_count = len(user_input)
    if char_count > 0:
        st.caption(f"Characters: {char_count} / 5000")
        if char_count > 5000:
            st.error("⚠️ Text terlalu panjang! Maximum 5000 characters.")
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🚀 Predict Sentiment", use_container_width=True, type="primary")

with col_examples:
    st.subheader("💡 Quick Examples")
    st.markdown("Click untuk menggunakan contoh review:")
    
    examples = {
        "Positive 😊": "This movie is absolutely fantastic! The acting is superb, the cinematography is breathtaking, and the story kept me engaged from start to finish. Highly recommended!",
        "Negative 😞": "This was a complete waste of time. Terrible acting, predictable plot, and poor production quality. I wouldn't recommend this to anyone.",
        "Mixed 🤔": "The movie had great visuals and some good moments, but the story was confusing and the ending felt rushed. It's okay but not memorable."
    }
    
    for label, text in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state.example_text = text
            st.rerun()
    
    # Apply example if selected
    if 'example_text' in st.session_state:
        user_input = st.session_state.example_text
        del st.session_state.example_text

st.markdown("---")

# Prediction logic
if predict_button and user_input.strip():
    
    if len(user_input) > 5000:
        st.error("❌ Text terlalu panjang! Maximum 5000 characters.")
    else:
        with st.spinner("🔮 Analyzing sentiment..."):
            try:
                # Call API - use 'api' hostname in Docker, localhost for local dev
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
                    
                    sentiment = result['sentiment']
                    confidence = result['confidence'] * 100
                    prob_positive = result['probabilities']['positive'] * 100
                    prob_negative = result['probabilities']['negative'] * 100
                    
                    # Display result
                    st.success("✅ Prediction completed!")
                    
                    # Result box
                    if sentiment == "positive":
                        st.markdown(f"""
                            <div class="prediction-box positive-box">
                                <div class="sentiment-label">✅ POSITIVE</div>
                                <div class="confidence-label">Confidence: {confidence:.2f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="prediction-box negative-box">
                                <div class="sentiment-label">❌ NEGATIVE</div>
                                <div class="confidence-label">Confidence: {confidence:.2f}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed metrics
                    st.subheader("📊 Detailed Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Sentiment", sentiment.upper())
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2f}%")
                    
                    with col3:
                        st.metric("Response Time", f"{response_time:.0f}ms")
                    
                    with col4:
                        st.metric("Text Length", f"{len(user_input)} chars")
                    
                    # Probability bars
                    st.subheader("🎲 Probability Distribution")
                    
                    col_pos, col_neg = st.columns(2)
                    
                    with col_pos:
                        st.markdown("**Positive:**")
                        st.progress(prob_positive / 100)
                        st.caption(f"{prob_positive:.2f}%")
                    
                    with col_neg:
                        st.markdown("**Negative:**")
                        st.progress(prob_negative / 100)
                        st.caption(f"{prob_negative:.2f}%")
                    
                    # Info message
                    st.info("✅ Prediction telah disimpan ke database untuk monitoring.")
                    
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.json(response.json())
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Tidak dapat terhubung ke API. Pastikan API service sedang berjalan.")
                st.info("Jalankan command: `docker-compose up -d api`")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. API terlalu lama merespons.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

elif predict_button and not user_input.strip():
    st.warning("⚠️ Please enter some text untuk prediction!")

# History section
st.markdown("---")
st.subheader("📜 Recent Predictions")

# Get recent predictions from database
db = get_db_manager()
recent_df = db.get_recent_predictions(n=10)

if not recent_df.empty:
    # Display in expander
    with st.expander(f"View last {len(recent_df)} predictions", expanded=False):
        for idx, row in recent_df.iterrows():
            col_time, col_text, col_result = st.columns([1, 3, 1])
            
            with col_time:
                timestamp = row['timestamp'].strftime("%H:%M:%S") if hasattr(row['timestamp'], 'strftime') else row['timestamp']
                st.caption(f"⏰ {timestamp}")
            
            with col_text:
                text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                st.text(text_preview)
            
            with col_result:
                if row['sentiment'] == 'positive':
                    st.success(f"✅ {row['confidence']*100:.1f}%")
                else:
                    st.error(f"❌ {row['confidence']*100:.1f}%")
            
            st.markdown("---")
else:
    st.info("📝 Belum ada prediction history. Lakukan prediction pertama!")

# Sidebar
with st.sidebar:
    st.title("📋 Prediction Info")
    
    st.markdown("### How it works")
    st.markdown("""
        1. **Input:** Enter movie review text
        2. **Processing:** Text di-transform ke TF-IDF features
        3. **Prediction:** Logistic Regression model predicts sentiment
        4. **Output:** Sentiment label + confidence score
        5. **Logging:** Prediction disimpan ke database
    """)
    
    st.markdown("---")
    
    st.markdown("### Model Performance")
    st.metric("Test Accuracy", "89.97%")
    st.metric("F1-Score", "90.02%")
    st.metric("Precision", "89.55%")
    st.metric("Recall", "90.50%")
    
    st.markdown("---")
    
    st.markdown("### Tips")
    st.info("""
        - Use complete sentences
        - More detail = better prediction
        - Mix of adjectives helps
        - Minimum 20 words recommended
    """)
    
    # Stats
    stats = db.get_stats()
    if stats['total_predictions'] > 0:
        st.markdown("---")
        st.markdown("### Current Stats")
        st.metric("Total Predictions", f"{stats['total_predictions']:,}")
        st.metric("Positive Rate", f"{stats['positive_ratio']*100:.1f}%")
        st.metric("Avg Confidence", f"{stats['avg_confidence']*100:.1f}%")
