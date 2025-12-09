"""
Performance Monitor Page
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils.db_manager import get_db_manager

# Page config
st.set_page_config(
    page_title="Performance Monitor",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("Performance Monitor")
st.write("Dashboard untuk monitoring model performance")

st.markdown("---")

# Get database
db = get_db_manager()
stats = db.get_stats()
predictions_df = pd.DataFrame(db.get_all_predictions())

# Check if data exists
if predictions_df.empty:
    st.info("Belum ada data predictions. Silakan lakukan beberapa predictions terlebih dahulu di halaman Live Prediction.")
    st.stop()

# Convert timestamp
predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
predictions_df['date'] = predictions_df['timestamp'].dt.date
predictions_df['hour'] = predictions_df['timestamp'].dt.hour

# Key Metrics
st.subheader("Key Metrics")

# Calculate percentages
total = stats['total_predictions']
positive_pct = (stats['positive_count'] / total * 100) if total > 0 else 0
negative_pct = (stats['negative_count'] / total * 100) if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Predictions",
        value=f"{stats['total_predictions']:,}"
    )

with col2:
    st.metric(
        label="Positive",
        value=f"{stats['positive_count']:,}",
        delta=f"{positive_pct:.1f}%"
    )

with col3:
    st.metric(
        label="Negative",
        value=f"{stats['negative_count']:,}",
        delta=f"{negative_pct:.1f}%"
    )

with col4:
    st.metric(
        label="Avg Confidence",
        value=f"{stats['avg_confidence']:.1f}%"
    )

st.markdown("---")

# Charts
st.subheader("Visualizations")

# 1. Predictions over time
st.write("**Predictions Over Time**")
daily_counts = predictions_df.groupby('date').size().reset_index(name='count')
fig1 = px.line(
    daily_counts,
    x='date',
    y='count',
    title='Daily Predictions'
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Sentiment distribution
st.write("**Sentiment Distribution**")
sentiment_counts = predictions_df['sentiment'].value_counts()
fig2 = px.pie(
    values=sentiment_counts.values,
    names=sentiment_counts.index,
    title='Sentiment Distribution'
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Confidence distribution
st.write("**Confidence Score Distribution**")
fig3 = px.histogram(
    predictions_df,
    x='confidence',
    nbins=30,
    title='Confidence Scores'
)
st.plotly_chart(fig3, use_container_width=True)

# 4. Hourly patterns
st.write("**Hourly Prediction Pattern**")
hourly_counts = predictions_df.groupby('hour').size().reset_index(name='count')
fig4 = px.bar(
    hourly_counts,
    x='hour',
    y='count',
    title='Predictions by Hour'
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# Recent predictions table
st.subheader("Recent Predictions")

recent_df = predictions_df.sort_values('timestamp', ascending=False).head(20)
display_df = recent_df[['timestamp', 'text', 'sentiment', 'confidence']].copy()
display_df['text'] = display_df['text'].str[:100] + '...'
display_df['confidence'] = (display_df['confidence'] * 100).round(2).astype(str) + '%'

st.dataframe(display_df, use_container_width=True)

# Export option
st.markdown("---")
st.subheader("Export Data")

if st.button("Download CSV"):
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download",
        data=csv,
        file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
