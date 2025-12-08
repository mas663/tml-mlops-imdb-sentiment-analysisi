"""
Data Drift Detection Page
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dashboard.utils.drift_detector import (
    load_reference_data,
    load_production_data,
    calculate_drift_score,
    get_drift_status,
    get_comparison_stats,
    generate_drift_report
)

# Page config
st.set_page_config(
    page_title="Data Drift Detection",
    page_icon="⚠️",
    layout="wide"
)

# Header
st.title("⚠️ Data Drift Detection")
st.markdown("Monitor perubahan distribusi data antara training dan production")

st.markdown("---")

# Load data
with st.spinner("Loading data..."):
    reference_df = load_reference_data(sample_size=1000)
    production_df = load_production_data()

# Check if data available
if reference_df.empty:
    st.error("❌ Reference data (training data) tidak ditemukan!")
    st.info("Pastikan file `data/processed/train.csv` tersedia.")
    st.stop()

if production_df.empty:
    st.warning("📝 Production data masih kosong. Lakukan beberapa predictions terlebih dahulu.")
    st.info("Setelah melakukan minimal 50 predictions, drift detection akan tersedia.")
    st.stop()

# Calculate drift score
drift_score = calculate_drift_score(reference_df, production_df)
status_text, status_color = get_drift_status(drift_score)

# Drift Status Card
st.subheader("🎯 Current Status")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if status_color == "green":
        st.success(f"### {status_text}")
    elif status_color == "orange":
        st.warning(f"### {status_text}")
    else:
        st.error(f"### {status_text}")
    
    st.markdown(f"**Drift Score:** {drift_score:.4f}")
    st.caption(f"Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.metric(
        label="Reference Data",
        value=f"{len(reference_df):,} samples"
    )

with col3:
    st.metric(
        label="Production Data",
        value=f"{len(production_df):,} samples"
    )

# Interpretation
st.info("""
    **Interpretation:**
    - **Drift Score < 0.1:** ✅ No significant drift, model masih reliable
    - **Drift Score 0.1-0.3:** ⚠️ Low drift, monitor closely
    - **Drift Score 0.3-0.5:** ⚠️ Moderate drift, consider retraining
    - **Drift Score > 0.5:** ❌ High drift, retraining recommended!
""")

st.markdown("---")

# Comparison Statistics
st.subheader("📊 Distribution Comparison")

stats = get_comparison_stats(reference_df, production_df)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("#### Text Length Distribution")
    
    # Create overlapping histogram
    fig_length = go.Figure()
    
    fig_length.add_trace(go.Histogram(
        x=reference_df['text_length'],
        name='Training Data',
        opacity=0.7,
        marker_color='#2196F3',
        nbinsx=30
    ))
    
    fig_length.add_trace(go.Histogram(
        x=production_df['text_length'],
        name='Production Data',
        opacity=0.7,
        marker_color='#FF4B4B',
        nbinsx=30
    ))
    
    fig_length.update_layout(
        barmode='overlay',
        title='Text Length Distribution Comparison',
        xaxis_title='Text Length (characters)',
        yaxis_title='Frequency',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    st.plotly_chart(fig_length, use_container_width=True)
    
    # Stats
    if stats:
        col_ref, col_prod, col_diff = st.columns(3)
        
        with col_ref:
            st.metric(
                "Training Mean",
                f"{stats['reference']['text_length_mean']:.0f}"
            )
        
        with col_prod:
            st.metric(
                "Production Mean",
                f"{stats['production']['text_length_mean']:.0f}"
            )
        
        with col_diff:
            diff_pct = stats['difference']['text_length_diff_pct']
            st.metric(
                "Difference",
                f"{diff_pct:.1f}%",
                delta=None
            )

with col_right:
    st.markdown("#### Sentiment Distribution")
    
    # Sentiment comparison
    ref_positive = (reference_df['sentiment'] == 'positive').sum()
    ref_negative = (reference_df['sentiment'] == 'negative').sum()
    prod_positive = (production_df['sentiment'] == 'positive').sum()
    prod_negative = (production_df['sentiment'] == 'negative').sum()
    
    fig_sentiment = go.Figure(data=[
        go.Bar(
            name='Training Data',
            x=['Positive', 'Negative'],
            y=[ref_positive, ref_negative],
            marker_color='#2196F3',
            text=[ref_positive, ref_negative],
            textposition='auto',
        ),
        go.Bar(
            name='Production Data',
            x=['Positive', 'Negative'],
            y=[prod_positive, prod_negative],
            marker_color='#FF4B4B',
            text=[prod_positive, prod_negative],
            textposition='auto',
        )
    ])
    
    fig_sentiment.update_layout(
        title='Sentiment Distribution Comparison',
        xaxis_title='Sentiment',
        yaxis_title='Count',
        barmode='group',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Stats
    if stats:
        col_ref2, col_prod2, col_diff2 = st.columns(3)
        
        with col_ref2:
            st.metric(
                "Training Positive",
                f"{stats['reference']['positive_ratio']*100:.1f}%"
            )
        
        with col_prod2:
            st.metric(
                "Production Positive",
                f"{stats['production']['positive_ratio']*100:.1f}%"
            )
        
        with col_diff2:
            sent_diff = stats['difference']['sentiment_diff']
            st.metric(
                "Difference",
                f"{sent_diff:.1f}%",
                delta=None
            )

st.markdown("---")

# Detailed Statistics
st.subheader("📈 Detailed Statistics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Training Data (Reference)")
    st.dataframe({
        'Metric': [
            'Sample Count',
            'Text Length (Mean)',
            'Text Length (Std)',
            'Positive Ratio',
            'Negative Ratio'
        ],
        'Value': [
            f"{stats['reference']['count']:,}",
            f"{stats['reference']['text_length_mean']:.2f}",
            f"{stats['reference']['text_length_std']:.2f}",
            f"{stats['reference']['positive_ratio']*100:.2f}%",
            f"{(1-stats['reference']['positive_ratio'])*100:.2f}%"
        ]
    }, use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### Production Data (Current)")
    st.dataframe({
        'Metric': [
            'Sample Count',
            'Text Length (Mean)',
            'Text Length (Std)',
            'Positive Ratio',
            'Negative Ratio'
        ],
        'Value': [
            f"{stats['production']['count']:,}",
            f"{stats['production']['text_length_mean']:.2f}",
            f"{stats['production']['text_length_std']:.2f}",
            f"{stats['production']['positive_ratio']*100:.2f}%",
            f"{(1-stats['production']['positive_ratio'])*100:.2f}%"
        ]
    }, use_container_width=True, hide_index=True)

st.markdown("---")

# Recommendations
st.subheader("💡 Recommendations")

if drift_score < 0.1:
    st.success("""
        ✅ **No Action Needed**
        
        Data distribution masih stabil. Model performance kemungkinan masih baik.
        Lanjutkan monitoring secara berkala.
    """)
elif drift_score < 0.3:
    st.warning("""
        ⚠️ **Monitor Closely**
        
        Terdapat sedikit perubahan distribusi data. Langkah yang disarankan:
        1. Monitor model performance metrics
        2. Check apakah accuracy menurun
        3. Siapkan rencana retraining
    """)
elif drift_score < 0.5:
    st.warning("""
        ⚠️ **Consider Retraining**
        
        Drift cukup signifikan. Disarankan untuk:
        1. Collect more recent data
        2. Retrain model dengan data baru
        3. Evaluate model performance
        4. Consider A/B testing
    """)
else:
    st.error("""
        ❌ **Retraining Recommended**
        
        High drift detected! Model mungkin tidak reliable lagi. Action items:
        1. **Stop production usage** (jika critical)
        2. Collect production data
        3. Retrain model immediately
        4. Validate new model
        5. Deploy new version
    """)

st.markdown("---")

# Generate Full Report
st.subheader("📄 Generate Full Drift Report")

st.markdown("Generate comprehensive drift report menggunakan Evidently:")

if st.button("🔍 Generate Evidently Report", use_container_width=True, type="primary"):
    with st.spinner("Generating detailed drift report..."):
        report_path = generate_drift_report(reference_df, production_df)
        
        if report_path and Path(report_path).exists():
            st.success("✅ Report generated successfully!")
            
            # Read HTML content
            with open(report_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Display in expander
            with st.expander("📊 View Evidently Report", expanded=True):
                st.components.v1.html(html_content, height=800, scrolling=True)
            
            # Download button
            st.download_button(
                label="📥 Download HTML Report",
                data=html_content,
                file_name=f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
        else:
            st.warning("⚠️ Could not generate Evidently report. Using simplified drift detection.")
            st.info("Install Evidently untuk full report: `pip install evidently`")

# Sidebar
with st.sidebar:
    st.title("🔬 Drift Detection Info")
    
    st.markdown("### What is Data Drift?")
    st.markdown("""
        Data drift terjadi ketika distribusi 
        data production berbeda dari data training.
        
        **Causes:**
        - User behavior changes
        - Seasonal patterns
        - New data sources
        - System updates
    """)
    
    st.markdown("---")
    
    st.markdown("### Detection Methods")
    st.markdown("""
        **1. Statistical Tests**
        - KS Test
        - Chi-square Test
        - Wasserstein Distance
        
        **2. Distribution Comparison**
        - Text length distribution
        - Sentiment ratio
        - Feature distributions
        
        **3. Performance Monitoring**
        - Accuracy trends
        - Confidence scores
        - Error rates
    """)
    
    st.markdown("---")
    
    st.markdown("### Current Metrics")
    st.metric("Drift Score", f"{drift_score:.4f}")
    st.metric("Status", status_text)
    
    st.markdown("---")
    
    st.markdown("### Actions")
    
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.rerun()
    
    st.caption("Last updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
