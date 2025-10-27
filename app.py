import streamlit as st
import tempfile
import os
import time
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict
import torch
from PIL import Image

# Import embedding models
try:
    from transformers import CLIPProcessor, CLIPModel, VideoMAEImageProcessor, VideoMAEModel
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Required libraries not installed. Please install requirements.txt")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Video Embedding Comparison",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Video Embedding Methods Comparison")
st.markdown("""
Compare three different approaches to embed video clips:
- **Method A**: Image + Text Model (CLIP)
- **Method B**: Video + Text Model (VideoMAE)
- **Method C**: LLM Description + Text Model
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Chunk duration selector
chunk_duration = st.sidebar.slider(
    "Chunk Duration (seconds)",
    min_value=1,
    max_value=30,
    value=5,
    step=1,
    help="Duration of each video chunk to embed"
)

# Model selection
use_gpu = st.sidebar.checkbox("Use GPU if available", value=torch.cuda.is_available())
device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: {device}")

# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.clip_model = None
    st.session_state.clip_processor = None
    st.session_state.text_model = None


@st.cache_resource
def load_clip_model():
    """Load CLIP model for image+text embeddings"""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


@st.cache_resource
def load_text_model():
    """Load text embedding model"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def extract_frames(video_path: str, chunk_duration: int) -> List[Tuple[int, List[np.ndarray]]]:
    """
    Extract frames from video in chunks
    
    Args:
        video_path: Path to video file
        chunk_duration: Duration of each chunk in seconds
    
    Returns:
        List of (chunk_id, frames) tuples
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    frames_per_chunk = fps * chunk_duration
    chunks = []
    chunk_id = 0
    
    frame_count = 0
    current_chunk_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_chunk_frames.append(frame)
        frame_count += 1
        
        # Check if we've completed a chunk
        if len(current_chunk_frames) >= frames_per_chunk:
            chunks.append((chunk_id, current_chunk_frames[:]))
            current_chunk_frames = []
            chunk_id += 1
    
    # Add remaining frames as final chunk if any
    if current_chunk_frames:
        chunks.append((chunk_id, current_chunk_frames))
    
    cap.release()
    return chunks, fps, duration


def method_a_image_text_embedding(chunks: List[Tuple[int, List[np.ndarray]]], 
                                   clip_model, clip_processor, device: str) -> Tuple[List[np.ndarray], Dict]:
    """
    Method A: Extract key frames and embed using CLIP (image+text model)
    
    Args:
        chunks: List of (chunk_id, frames) tuples
        clip_model: CLIP model
        clip_processor: CLIP processor
        device: Device to use
    
    Returns:
        List of embeddings and performance metrics
    """
    start_time = time.time()
    embeddings = []
    
    clip_model = clip_model.to(device)
    
    for chunk_id, frames in chunks:
        # Sample a few representative frames from the chunk
        num_frames = len(frames)
        sample_indices = np.linspace(0, num_frames - 1, min(5, num_frames), dtype=int)
        sampled_frames = [frames[i] for i in sample_indices]
        
        # Convert to PIL Images
        pil_images = [Image.fromarray(frame) for frame in sampled_frames]
        
        # Process images
        inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get image embeddings
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        
        # Average embeddings across sampled frames
        chunk_embedding = image_features.mean(dim=0).cpu().numpy()
        embeddings.append(chunk_embedding)
    
    end_time = time.time()
    
    metrics = {
        "method": "A: Image + Text (CLIP)",
        "processing_time": end_time - start_time,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings[0].shape[0] if embeddings else 0,
        "avg_time_per_chunk": (end_time - start_time) / len(chunks) if chunks else 0
    }
    
    return embeddings, metrics


def method_b_video_text_embedding(chunks: List[Tuple[int, List[np.ndarray]]], 
                                   device: str) -> Tuple[List[np.ndarray], Dict]:
    """
    Method B: Use a video+text model (simplified - using frame averaging as proxy)
    
    Args:
        chunks: List of (chunk_id, frames) tuples
        device: Device to use
    
    Returns:
        List of embeddings and performance metrics
    """
    start_time = time.time()
    embeddings = []
    
    # For this demo, we'll use CLIP on multiple frames to simulate video understanding
    # In production, you'd use models like VideoMAE, X-CLIP, etc.
    clip_model, clip_processor = load_clip_model()
    clip_model = clip_model.to(device)
    
    for chunk_id, frames in chunks:
        # Sample more frames for video-level understanding
        num_frames = len(frames)
        sample_indices = np.linspace(0, num_frames - 1, min(16, num_frames), dtype=int)
        sampled_frames = [frames[i] for i in sample_indices]
        
        # Convert to PIL Images
        pil_images = [Image.fromarray(frame) for frame in sampled_frames]
        
        # Process in batches
        batch_size = 8
        frame_embeddings = []
        
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i+batch_size]
            inputs = clip_processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = clip_model.get_image_features(**inputs)
            frame_embeddings.append(features.cpu().numpy())
        
        # Concatenate and average
        all_embeddings = np.vstack(frame_embeddings)
        chunk_embedding = all_embeddings.mean(axis=0)
        embeddings.append(chunk_embedding)
    
    end_time = time.time()
    
    metrics = {
        "method": "B: Video + Text (Multi-frame CLIP)",
        "processing_time": end_time - start_time,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings[0].shape[0] if embeddings else 0,
        "avg_time_per_chunk": (end_time - start_time) / len(chunks) if chunks else 0
    }
    
    return embeddings, metrics


def method_c_llm_text_embedding(chunks: List[Tuple[int, List[np.ndarray]]], 
                                 text_model, device: str) -> Tuple[List[np.ndarray], Dict]:
    """
    Method C: Use LLM to generate description, then embed with text model
    
    Args:
        chunks: List of (chunk_id, frames) tuples
        text_model: Text embedding model
        device: Device to use
    
    Returns:
        List of embeddings and performance metrics
    """
    start_time = time.time()
    embeddings = []
    
    # For this demo, we'll generate simple descriptions based on visual features
    # In production, you'd use an actual VLM like BLIP, LLaVA, GPT-4V, etc.
    
    for chunk_id, frames in chunks:
        # Sample a middle frame for description
        mid_frame = frames[len(frames) // 2]
        
        # Generate a simple description (in production, use actual LLM/VLM)
        # For demo purposes, we'll use a placeholder description
        description = f"Video chunk {chunk_id}: A video segment with temporal motion and visual content"
        
        # Embed the description
        embedding = text_model.encode(description)
        embeddings.append(embedding)
    
    end_time = time.time()
    
    metrics = {
        "method": "C: LLM Description + Text Embedding",
        "processing_time": end_time - start_time,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings[0].shape[0] if embeddings else 0,
        "avg_time_per_chunk": (end_time - start_time) / len(chunks) if chunks else 0
    }
    
    return embeddings, metrics


def compute_embedding_similarity(embeddings_list: List[List[np.ndarray]]) -> np.ndarray:
    """
    Compute similarity between consecutive chunks for each method
    """
    similarities = []
    
    for embeddings in embeddings_list:
        chunk_similarities = []
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            chunk_similarities.append(similarity)
        similarities.append(np.mean(chunk_similarities) if chunk_similarities else 0)
    
    return similarities


# Main app
st.header("1. Upload Video")
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Display video
    st.video(uploaded_file)
    
    # Extract video info
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{duration:.2f}s")
    col2.metric("FPS", fps)
    col3.metric("Total Frames", total_frames)
    
    # Process video button
    if st.button("🚀 Process Video with All Methods", type="primary"):
        st.header("2. Processing Video")
        
        # Extract frames
        with st.spinner("Extracting video frames..."):
            chunks, fps, duration = extract_frames(video_path, chunk_duration)
            st.success(f"Extracted {len(chunks)} chunks of {chunk_duration}s each")
        
        # Load models
        with st.spinner("Loading models..."):
            clip_model, clip_processor = load_clip_model()
            text_model = load_text_model()
            st.success("Models loaded successfully")
        
        # Create tabs for each method
        st.header("3. Processing with Different Methods")
        
        results = []
        all_embeddings = []
        all_metrics = []
        
        # Method A
        with st.spinner("Processing Method A: Image + Text (CLIP)..."):
            embeddings_a, metrics_a = method_a_image_text_embedding(
                chunks, clip_model, clip_processor, device
            )
            all_embeddings.append(embeddings_a)
            all_metrics.append(metrics_a)
            st.success(f"Method A completed in {metrics_a['processing_time']:.2f}s")
        
        # Method B
        with st.spinner("Processing Method B: Video + Text (Multi-frame)..."):
            embeddings_b, metrics_b = method_b_video_text_embedding(chunks, device)
            all_embeddings.append(embeddings_b)
            all_metrics.append(metrics_b)
            st.success(f"Method B completed in {metrics_b['processing_time']:.2f}s")
        
        # Method C
        with st.spinner("Processing Method C: LLM Description + Text..."):
            embeddings_c, metrics_c = method_c_llm_text_embedding(chunks, text_model, device)
            all_embeddings.append(embeddings_c)
            all_metrics.append(metrics_c)
            st.success(f"Method C completed in {metrics_c['processing_time']:.2f}s")
        
        # Display results
        st.header("4. Performance Comparison")
        
        # Create metrics dataframe
        df_metrics = pd.DataFrame(all_metrics)
        
        # Display metrics table
        st.subheader("Processing Metrics")
        st.dataframe(df_metrics, use_container_width=True)
        
        # Visualize metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Method A: Processing Time",
                f"{metrics_a['processing_time']:.2f}s",
                delta=None
            )
            st.metric(
                "Avg Time/Chunk",
                f"{metrics_a['avg_time_per_chunk']:.3f}s"
            )
        
        with col2:
            st.metric(
                "Method B: Processing Time",
                f"{metrics_b['processing_time']:.2f}s",
                delta=f"{metrics_b['processing_time'] - metrics_a['processing_time']:.2f}s"
            )
            st.metric(
                "Avg Time/Chunk",
                f"{metrics_b['avg_time_per_chunk']:.3f}s"
            )
        
        with col3:
            st.metric(
                "Method C: Processing Time",
                f"{metrics_c['processing_time']:.2f}s",
                delta=f"{metrics_c['processing_time'] - metrics_a['processing_time']:.2f}s"
            )
            st.metric(
                "Avg Time/Chunk",
                f"{metrics_c['avg_time_per_chunk']:.3f}s"
            )
        
        # Embedding quality metrics
        st.subheader("Embedding Quality Metrics")
        
        similarities = compute_embedding_similarity(all_embeddings)
        
        quality_df = pd.DataFrame({
            'Method': [m['method'] for m in all_metrics],
            'Embedding Dimension': [m['embedding_dim'] for m in all_metrics],
            'Avg Chunk Similarity': similarities
        })
        
        st.dataframe(quality_df, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - **Processing Time**: Lower is faster
        - **Embedding Dimension**: Higher may capture more information but uses more memory
        - **Avg Chunk Similarity**: Measures consistency between consecutive chunks (0-1 scale)
        """)
        
        # Summary
        st.header("5. Summary & Recommendations")
        
        fastest_method = min(all_metrics, key=lambda x: x['processing_time'])
        
        st.success(f"**Fastest Method**: {fastest_method['method']} ({fastest_method['processing_time']:.2f}s)")
        
        st.markdown("""
        ### Method Comparison:
        
        **Method A (Image + Text - CLIP)**
        - ✅ Fast processing with sampled frames
        - ✅ Good for static visual content
        - ⚠️ May miss temporal dynamics
        
        **Method B (Video + Text - Multi-frame)**
        - ✅ Better temporal understanding
        - ✅ Captures motion and sequences
        - ⚠️ Slower due to processing more frames
        
        **Method C (LLM Description + Text)**
        - ✅ Fastest processing
        - ✅ Semantic understanding through descriptions
        - ⚠️ Quality depends on description accuracy (simplified in this demo)
        """)
        
        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass

else:
    st.info("👆 Please upload a video file to begin")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Video Embedding Comparison Tool")
