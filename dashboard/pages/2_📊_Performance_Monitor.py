"""
Performance Monitoring Dashboard
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils.db_manager import get_db_manager

# Page config
st.set_page_config(
    page_title="Performance Monitor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Performance Monitoring Dashboard")
st.markdown("Real-time monitoring untuk model performance dan usage statistics")

st.markdown("---")

# Get data from database
db = get_db_manager()
df = db.get_all_predictions()
stats = db.get_stats()

# Check if we have data
if df.empty:
    st.warning("📝 Belum ada data predictions. Silakan lakukan beberapa predictions terlebih dahulu di halaman **Live Prediction**.")
    st.info("Setelah melakukan predictions, data akan muncul di dashboard ini.")
    st.stop()

# Top Stats Row
st.subheader("📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Predictions",
        value=f"{stats['total_predictions']:,}",
        delta=None
    )

with col2:
    avg_conf = stats['avg_confidence'] * 100
    st.metric(
        label="Avg Confidence",
        value=f"{avg_conf:.2f}%",
        delta=None
    )

with col3:
    positive_ratio = stats['positive_ratio'] * 100
    st.metric(
        label="Positive Rate",
        value=f"{positive_ratio:.1f}%",
        delta=None
    )

with col4:
    if stats['avg_response_time']:
        st.metric(
            label="Avg Response Time",
            value=f"{stats['avg_response_time']:.0f}ms",
            delta=None
        )
    else:
        st.metric(
            label="Avg Response Time",
            value="N/A",
            delta=None
        )

st.markdown("---")

# Charts Section
st.subheader("📊 Visualization")

# Prepare data
df['date'] = pd.to_datetime(df['timestamp']).dt.date
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['confidence_pct'] = df['confidence'] * 100

# Row 1: Timeline and Distribution
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### 📅 Predictions Over Time")
    
    # Group by date
    daily_counts = df.groupby('date').size().reset_index(name='count')
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    fig_timeline = px.line(
        daily_counts,
        x='date',
        y='count',
        title='Daily Prediction Count',
        labels={'date': 'Date', 'count': 'Number of Predictions'},
        markers=True
    )
    fig_timeline.update_traces(line_color='#FF4B4B', line_width=3)
    fig_timeline.update_layout(
        hovermode='x unified',
        plot_bgcolor='white',
        height=350
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

with col_right:
    st.markdown("#### 🥧 Sentiment Distribution")
    
    sentiment_counts = df['sentiment'].value_counts()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.4,
        marker_colors=['#FF4B4B', '#00C853'],
        textinfo='label+percent',
        textfont_size=14
    )])
    fig_pie.update_layout(
        title='Positive vs Negative',
        showlegend=True,
        height=350
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Row 2: Confidence and Hourly
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.markdown("#### 📊 Confidence Score Distribution")
    
    fig_hist = px.histogram(
        df,
        x='confidence_pct',
        nbins=20,
        title='Distribution of Confidence Scores',
        labels={'confidence_pct': 'Confidence (%)', 'count': 'Frequency'},
        color_discrete_sequence=['#FF4B4B']
    )
    fig_hist.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        height=350
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right2:
    st.markdown("#### ⏰ Predictions by Hour")
    
    hourly_counts = df.groupby('hour').size().reset_index(name='count')
    
    fig_hourly = px.bar(
        hourly_counts,
        x='hour',
        y='count',
        title='Hourly Distribution',
        labels={'hour': 'Hour of Day', 'count': 'Number of Predictions'},
        color='count',
        color_continuous_scale='Reds'
    )
    fig_hourly.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        height=350
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

st.markdown("---")

# Sentiment-specific Analysis
st.subheader("🔍 Sentiment Analysis")

col_pos, col_neg = st.columns(2)

with col_pos:
    st.markdown("#### ✅ Positive Predictions")
    positive_df = df[df['sentiment'] == 'positive']
    if not positive_df.empty:
        st.metric("Count", len(positive_df))
        st.metric("Avg Confidence", f"{positive_df['confidence_pct'].mean():.2f}%")
        st.metric("Avg Text Length", f"{positive_df['text_length'].mean():.0f} chars")
    else:
        st.info("No positive predictions yet")

with col_neg:
    st.markdown("#### ❌ Negative Predictions")
    negative_df = df[df['sentiment'] == 'negative']
    if not negative_df.empty:
        st.metric("Count", len(negative_df))
        st.metric("Avg Confidence", f"{negative_df['confidence_pct'].mean():.2f}%")
        st.metric("Avg Text Length", f"{negative_df['text_length'].mean():.0f} chars")
    else:
        st.info("No negative predictions yet")

st.markdown("---")

# Recent Predictions Table
st.subheader("📋 Recent Predictions")

# Show last 20 predictions
display_df = df.head(20).copy()

# Format columns for display
display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
display_df['confidence'] = display_df['confidence_pct'].apply(lambda x: f"{x:.2f}%")
display_df['text_preview'] = display_df['text'].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)

# Select columns to display
table_df = display_df[['timestamp', 'text_preview', 'sentiment', 'confidence', 'text_length']].copy()
table_df.columns = ['Timestamp', 'Text', 'Sentiment', 'Confidence', 'Length']

# Display with styling
st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    height=400
)

# Download button
csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download All Predictions (CSV)",
    data=csv,
    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

st.markdown("---")

# Statistical Summary
st.subheader("📈 Statistical Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Text Length Statistics**")
    st.write(f"• Min: {df['text_length'].min()} chars")
    st.write(f"• Max: {df['text_length'].max()} chars")
    st.write(f"• Mean: {df['text_length'].mean():.0f} chars")
    st.write(f"• Median: {df['text_length'].median():.0f} chars")

with col2:
    st.markdown("**Confidence Statistics**")
    st.write(f"• Min: {df['confidence_pct'].min():.2f}%")
    st.write(f"• Max: {df['confidence_pct'].max():.2f}%")
    st.write(f"• Mean: {df['confidence_pct'].mean():.2f}%")
    st.write(f"• Median: {df['confidence_pct'].median():.2f}%")

with col3:
    st.markdown("**Response Time Statistics**")
    if df['response_time'].notna().any():
        response_times = df[df['response_time'].notna()]['response_time']
        st.write(f"• Min: {response_times.min():.0f}ms")
        st.write(f"• Max: {response_times.max():.0f}ms")
        st.write(f"• Mean: {response_times.mean():.0f}ms")
        st.write(f"• Median: {response_times.median():.0f}ms")
    else:
        st.info("No response time data")

# Sidebar
with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    
    st.markdown("### 📊 Data Summary")
    st.info(f"""
        **Total Records:** {len(df):,}
        
        **Date Range:**
        - From: {df['timestamp'].min().strftime('%Y-%m-%d')}
        - To: {df['timestamp'].max().strftime('%Y-%m-%d')}
        
        **Coverage:** {(df['timestamp'].max() - df['timestamp'].min()).days + 1} days
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔄 Auto Refresh")
    auto_refresh = st.checkbox("Enable auto-refresh", value=False)
    
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
        st.info(f"Dashboard akan refresh setiap {refresh_interval} detik")
        import time
        time.sleep(refresh_interval)
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📥 Export Options")
    
    if st.button("📊 Export Statistics", use_container_width=True):
        stats_dict = {
            "Total Predictions": stats['total_predictions'],
            "Positive Count": stats['positive_count'],
            "Negative Count": stats['negative_count'],
            "Positive Ratio": f"{stats['positive_ratio']*100:.2f}%",
            "Negative Ratio": f"{stats['negative_ratio']*100:.2f}%",
            "Average Confidence": f"{stats['avg_confidence']*100:.2f}%",
            "Average Response Time": f"{stats['avg_response_time']:.2f}ms" if stats['avg_response_time'] else "N/A",
            "Average Text Length": f"{stats['avg_text_length']:.0f} chars"
        }
        
        stats_df = pd.DataFrame(stats_dict.items(), columns=['Metric', 'Value'])
        csv_stats = stats_df.to_csv(index=False)
        
        st.download_button(
            label="Download Stats CSV",
            data=csv_stats,
            file_name=f"statistics_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    st.markdown("### ℹ️ Info")
    st.caption("""
        Dashboard ini menampilkan real-time 
        monitoring dari prediction API.
        
        Data di-update otomatis setiap kali
        ada prediction baru melalui API.
    """)
