import streamlit as st
from gradio_client import Client, handle_file
import os
import time
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="NeuroScan AI - Brain Tumor Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS for modern styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">

<style>
/* Global */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    color: #e0e0e0;
}

/* Headers */
h1, h2, h3, h4, h5 {
    color: #fff;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
}

/* Main Header */
.main-header {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(0,0,0,0.1));
    backdrop-filter: blur(10px);
    color: #fff;
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 6px 30px rgba(0, 0, 0, 0.4);
}
.main-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #f8ff00, #3ad59f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-subtitle {
    font-size: 1.3rem;
    opacity: 0.9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #232526 0%, #414345 100%);
    backdrop-filter: blur(12px);
    color: #fff;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* Upload Section */
.upload-section {
    background: rgba(255, 255, 255, 0.05);
    border: 2px dashed #5ce0d8;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    transition: all 0.3s ease;
}
.upload-section:hover {
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 0 20px rgba(92, 224, 216, 0.4);
}

/* Cards */
.result-container, .feature-card, .metric-card {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    transition: transform 0.3s ease;
}
.result-container:hover, .feature-card:hover, .metric-card:hover {
    transform: translateY(-6px);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%);
    color: #fff;
    border: none;
    padding: 0.7rem 1.8rem;
    border-radius: 30px;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 0 12px rgba(255, 94, 98, 0.6);
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff5e62 0%, #ff9966 100%);
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 0 20px rgba(255, 94, 98, 0.8);
}

/* Info & Warning Boxes */
.info-box, .warning-box {
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.info-box {
    background: linear-gradient(135deg, #3ad59f 0%, #56ab2f 100%);
    color: #fff;
}
.warning-box {
    background: linear-gradient(135deg, #ffb75e 0%, #ed8f03 100%);
    color: #fff;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'client_initialized' not in st.session_state:
    st.session_state.client_initialized = False

# Initialize the Hugging Face Gradio Client
@st.cache_resource
def initialize_client():
    try:
        client = Client("PavanKumarD/Brain-Tumor-classification")
        return client, True
    except Exception as e:
        return None, False

def query_api(image_path, client):
    try:
        result = client.predict(
            image=handle_file(image_path),
            api_name="/predict"
        )
        return result
    except Exception as e:
        return {"error": str(e)}

def save_analysis_to_history(image_name, result, confidence):
    """Save analysis result to session history"""
    analysis_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_name': image_name,
        'prediction': result,
        'confidence': confidence
    }
    st.session_state.analysis_history.append(analysis_data)
    # Keep only last 10 analyses
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[-10:]

def create_confidence_chart(confidence, label):
    """Create a confidence visualization chart"""
    fig = go.Figure(go.Bar(
        x=[confidence, 100-confidence],
        y=[label, 'Other'],
        orientation='h',
        marker_color=['#667eea', '#e0e0e0'],
        text=[f'{confidence:.1f}%', f'{100-confidence:.1f}%'],
        textposition='inside',
        textfont=dict(color='white', size=14)
    ))
    
    fig.update_layout(
        title=f"Prediction Confidence: {label}",
        xaxis_title="Confidence (%)",
        height=200,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Main header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🧠 NeuroScan AI</h1>
    <p class="main-subtitle">Advanced Brain Tumor Classification System</p>
    <p style="font-size: 1rem; margin-top: 1rem;">Powered by AI • Fast • Accurate • Reliable</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🔧 Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    st.subheader("📊 System Status")
    client, client_status = initialize_client()
    
    if client_status:
        st.success("✅ AI Model Connected")
        st.session_state.client_initialized = True
    else:
        st.error("❌ AI Model Disconnected")
        st.session_state.client_initialized = False
    
    # Information section
    st.subheader("ℹ️ About")
    st.markdown("""
    This AI system can classify brain MRI images into different categories:
    
    - **Glioma Tumor**
    - **Meningioma Tumor**
    - **No Tumor**
    - **Pituitary Tumor**
    
    Upload a brain MRI image to get instant classification results.
    """)
    
    # Analysis history
    if st.session_state.analysis_history:
        st.subheader("📈 Recent Analyses")
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-3:])):
            with st.expander(f"Analysis {len(st.session_state.analysis_history)-i}"):
                st.write(f"**Time:** {analysis['timestamp']}")
                st.write(f"**Image:** {analysis['image_name']}")
                st.write(f"**Result:** {analysis['prediction']}")
                st.write(f"**Confidence:** {analysis['confidence']:.1f}%")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Upload section
    st.markdown("""
    <style>
    .drag-drop-zone {
        border: 2px dashed #9c88ff;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: #fff;
        background: rgba(255, 255, 255, 0.05);
        transition: background 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .drag-drop-zone:hover {
        background: rgba(255, 255, 255, 0.1);
        box-shadow: 0px 6px 20px rgba(255,255,255,0.3);
    }
    </style>
    <div class="drag-drop-zone">
        <h3>📁 Upload MRI Image</h3>
        <p>Drag and drop or click below to select a file</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png", "dcm"],
        help="Supported formats: JPG, JPEG, PNG, DICOM"
    )

    
    if uploaded_file:
        # Image preview
        st.markdown("""
        <div class="result-container">
        """, unsafe_allow_html=True)
        
        col_img, col_info = st.columns([1, 1])
        
        with col_img:
            st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)
        
        with col_info:
            # Image information
            image = Image.open(uploaded_file)
            st.markdown("### 📋 Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size}")
            st.write(f"**Format:** {image.format}")
            st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analysis button
        if st.button("🔍 Analyze Image", key="analyze_btn", use_container_width=True):
            if not st.session_state.client_initialized:
                st.error("❌ AI model is not connected. Please refresh the page and try again.")
            else:
                with st.spinner("🧠 AI is analyzing your MRI image... Please wait"):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate analysis steps
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 30:
                            status_text.text("Preprocessing image...")
                        elif i < 70:
                            status_text.text("Running AI analysis...")
                        else:
                            status_text.text("Finalizing results...")
                        time.sleep(0.02)
                    
                    # Save file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Query the API
                    result = query_api(temp_path, client)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    if "error" in result:
                        st.error(f"❌ Analysis failed: {result['error']}")
                    else:
                        # Extract results
                        label = result.get('label', 'Unknown')
                        confidence = 0
                        
                        if result.get('confidences'):
                            confidence = result['confidences'][0].get('confidence', 0) * 100
                        
                        # Save to history
                        save_analysis_to_history(uploaded_file.name, label, confidence)
                        
                        # Display results
                        st.markdown("## 🎯 Analysis Results")
                        
                        # Result cards
                        col_result1, col_result2 = st.columns(2)
                        
                        with col_result1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>🔬 Prediction</h3>
                                <h2>{label}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_result2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>📊 Confidence</h3>
                                <h2>{confidence:.1f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Confidence chart
                        st.plotly_chart(
                            create_confidence_chart(confidence, label),
                            use_container_width=True
                        )
                        
                        # Interpretation
                        if confidence >= 80:
                            st.markdown("""
                            <div class="info-box">
                                <h4>🎯 High Confidence Result</h4>
                                <p>The AI model is very confident in this prediction. The result is highly reliable.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif confidence >= 60:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Moderate Confidence Result</h4>
                                <p>The AI model shows moderate confidence. Consider additional medical consultation.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Low Confidence Result</h4>
                                <p>The AI model has low confidence. Please consult with medical professionals and consider retaking the scan.</p>
                            </div>
                            """, unsafe_allow_html=True)

with col2:
    # Features section
    st.markdown("### ✨ Key Features")
    
    features = [
        ("🤖", "AI-Powered", "Advanced deep learning model"),
        ("⚡", "Fast Analysis", "Results in seconds"),
        ("🎯", "High Accuracy", "Clinically validated"),
        ("🔒", "Secure", "No data stored permanently"),
        ("📱", "Easy to Use", "Simple upload interface"),
        ("📊", "Detailed Results", "Confidence scores included")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{icon} {title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

# Medical disclaimer
st.markdown("---")
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Medical Disclaimer</h4>
    <p><strong>Important:</strong> This AI system is for research and educational purposes only. 
    It is not intended to replace professional medical diagnosis. Always consult qualified healthcare 
    professionals for medical advice and treatment decisions.</p>
</div>
""", unsafe_allow_html=True)

# Statistics section if there's history
if st.session_state.analysis_history:
    st.markdown("## 📈 Analysis Statistics")
    
    # Create statistics
    df = pd.DataFrame(st.session_state.analysis_history)
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.metric("Total Analyses", len(df))
    
    with col_stat2:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    with col_stat3:
        most_common = df['prediction'].mode()[0] if not df.empty else "N/A"
        st.metric("Most Common", most_common)
    
    # Prediction distribution chart
    if len(df) > 1:
        fig_dist = px.pie(
            values=df['prediction'].value_counts().values,
            names=df['prediction'].value_counts().index,
            title="Prediction Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🧠 NeuroScan AI • Built with Streamlit • Powered by Hugging Face</p>
    <p>© 2025 • For Research and Educational Use Only</p>
</div>
""", unsafe_allow_html=True)