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
import io
import base64

# Import embedding models
from transformers import CLIPProcessor, CLIPModel, VideoMAEImageProcessor, VideoMAEModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import imagehash

# Page configuration
st.set_page_config(
    page_title="Video Embedding Comparison",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Video Embedding Methods Comparison")
st.markdown("""
Compare two different approaches to embed video clips:
- **CLIP**: Image + Text Model (CLIP with sampled frames)
- **LLM**: Gemini 2.0 Flash + Text Embedding
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Gemini API Key
from dotenv import load_dotenv
load_dotenv()  # Loads .env file into environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY", "")
print(gemini_api_key)

# Similarity method selector for Method C
similarity_method = st.sidebar.selectbox(
    "LLM Similarity Filter",
    options=["CLIP (Semantic)", "Perceptual Hashing (Fast)"],
    index=0,
    help="Choose how to filter similar frames in LLM method"
)

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
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception as e:
        st.error(f"Failed to load CLIP model: {e}")
        st.info("Please ensure you have internet connection for first-time model download")
        raise


@st.cache_resource
def load_text_model():
    """Load text embedding model"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Failed to load text model: {e}")
        st.info("Please ensure you have internet connection for first-time model download")
        raise


def extract_frames(video_path: str, chunk_duration: int) -> List[Tuple[int, List[np.ndarray]]]:
    """
    Extract frames from video in chunks (1 frame per second)
    
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
    
    frames_per_chunk = chunk_duration  # Now 1 frame per second
    chunks = []
    chunk_id = 0
    
    frame_count = 0
    current_chunk_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only capture 1 frame per second (every fps frames)
        if frame_count % fps == 0:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_chunk_frames.append(frame)
            
            # Check if we've completed a chunk
            if len(current_chunk_frames) >= frames_per_chunk:
                chunks.append((chunk_id, current_chunk_frames[:]))
                current_chunk_frames = []
                chunk_id += 1
        
        frame_count += 1
    
    # Add remaining frames as final chunk if any
    if current_chunk_frames:
        chunks.append((chunk_id, current_chunk_frames))
    
    cap.release()
    return chunks, fps, duration


def clip_embedding(chunks: List[Tuple[int, List[np.ndarray]]], 
                   clip_model, clip_processor, device: str) -> Tuple[List[np.ndarray], Dict]:
    """
    CLIP Method: Extract key frames and embed using CLIP (image+text model)
    
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
    chunk_times = []  # Track processing time per chunk
    
    for chunk_id, frames in chunks:
        chunk_start_time = time.time()
        
        # Sample a few representative frames from the chunk
        # num_frames = len(frames)
        # sample_indices = np.linspace(0, num_frames - 1, min(5, num_frames), dtype=int)
        # sampled_frames = [frames[i] for i in sample_indices]
        
        # Convert to PIL Images
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        # Display frames used for this chunk
        st.write(f"**Chunk {chunk_id}:** Using {len(pil_images)} sampled frames")
        cols = st.columns(len(pil_images))
        for idx, (col, img) in enumerate(zip(cols, pil_images)):
            with col:
                st.image(img, caption=f"Frame {idx}", width='stretch')
        
        # Process images
        inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get image embeddings
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        
        # Average embeddings across sampled frames
        chunk_embedding = image_features.mean(dim=0).cpu().numpy()
        embeddings.append(chunk_embedding)
        
        # Track chunk processing time
        chunk_end_time = time.time()
        chunk_processing_time = chunk_end_time - chunk_start_time
        chunk_times.append(chunk_processing_time)
        st.write(f"⏱️ Chunk {chunk_id} processed in {chunk_processing_time:.2f}s")
        st.markdown("---")
    
    end_time = time.time()
    
    metrics = {
        "method": "CLIP",
        "processing_time": end_time - start_time,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings[0].shape[0] if embeddings else 0,
        "avg_time_per_chunk": (end_time - start_time) / len(chunks) if chunks else 0,
        "chunk_times": chunk_times,
        "min_chunk_time": min(chunk_times) if chunk_times else 0,
        "max_chunk_time": max(chunk_times) if chunk_times else 0
    }
    
    return embeddings, metrics


def llm_embedding(chunks: List[Tuple[int, List[np.ndarray]]], 
                  text_model, device: str, gemini_api_key: str,
                  similarity_method: str,
                  clip_model=None, clip_processor=None) -> Tuple[List[np.ndarray], Dict]:
    """
    LLM Method: Use Gemini 2.0 Flash to generate description from filtered frames, then embed with text model
    
    Args:
        chunks: List of (chunk_id, frames) tuples
        text_model: Text embedding model
        device: Device to use
        gemini_api_key: Gemini API key
        similarity_method: "CLIP (Semantic)" or "Perceptual Hashing (Fast)"
        clip_model: CLIP model for similarity filtering (optional)
        clip_processor: CLIP processor for similarity filtering (optional)
    
    Returns:
        List of embeddings and performance metrics
    """
    start_time = time.time()
    embeddings = []
    
    # Configure Gemini
    if not gemini_api_key:
        st.warning("Gemini API key not provided. Using placeholder descriptions for LLM method.")
        use_gemini = False
    else:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
            use_gemini = True
        except Exception as e:
            st.warning(f"Failed to initialize Gemini: {e}. Using placeholder descriptions.")
            use_gemini = False
    
    # Determine similarity method based on user selection
    use_clip_similarity = similarity_method == "CLIP (Semantic)" and clip_model is not None and clip_processor is not None
    
    if use_clip_similarity:
        similarity_threshold = 0.9  # For CLIP cosine similarity
    else:
        perceptual_threshold = 6  # ~10% of 64-bit hash for dhash
    
    chunk_times = []  # Track processing time per chunk
    
    for chunk_id, frames in chunks:
        chunk_start_time = time.time()  # Start timing this chunk
        
        # Step 1: Filter ALL frames by similarity (not just sampled ones)
        filtered_frames = frames[:]
        
        # if use_clip_similarity:
        #     # Use CLIP for similarity filtering on ALL frames
        #     clip_model_temp = clip_model.to(device)
            
        #     for i, frame in enumerate(frames):
        #         if i == 0:
        #             # Always include first frame
        #             filtered_frames.append(frame)
        #         else:
        #             # Compare with last filtered frame
        #             last_frame = filtered_frames[-1]
                    
        #             # Convert to PIL and get embeddings
        #             pil_frame1 = Image.fromarray(last_frame)
        #             pil_frame2 = Image.fromarray(frame)
                    
        #             inputs = clip_processor(images=[pil_frame1, pil_frame2], return_tensors="pt", padding=True)
        #             inputs = {k: v.to(device) for k, v in inputs.items()}
                    
        #             with torch.no_grad():
        #                 features = clip_model_temp.get_image_features(**inputs)
                    
        #             # Compute cosine similarity
        #             emb1 = features[0].cpu().numpy()
        #             emb2 = features[1].cpu().numpy()
        #             similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    
        #             # Add frame if similarity < threshold (i.e., sufficiently different)
        #             if similarity < similarity_threshold:
        #                 filtered_frames.append(frame)
        # else:
        #     # Use perceptual hashing for similarity on ALL frames
        #     for i, frame in enumerate(frames):
        #         if i == 0:
        #             filtered_frames.append(frame)
        #         else:
        #             last_frame = filtered_frames[-1]
                    
        #             # Compute perceptual hashes
        #             pil_frame1 = Image.fromarray(last_frame)
        #             pil_frame2 = Image.fromarray(frame)
                    
        #             hash1 = imagehash.dhash(pil_frame1)
        #             hash2 = imagehash.dhash(pil_frame2)
                    
        #             # Hamming distance
        #             distance = hash1 - hash2
                    
        #             # Add frame if sufficiently different
        #             if distance > perceptual_threshold:
        #                 filtered_frames.append(frame)
        
        # Step 2: Limit to max 50 frames
        # if len(filtered_frames) > 50:
        #     # Resample to 50 frames
        #     indices = np.linspace(0, len(filtered_frames) - 1, 50, dtype=int)
        #     filtered_frames = [filtered_frames[i] for i in indices]
        
        # Display filtered frames for this chunk
        st.write(f"**Chunk {chunk_id}:** Filtered to {len(filtered_frames)} distinct frames (from {len(frames)} total)")
        if len(filtered_frames) > 0:
            # Display ALL filtered frames (up to 50)
            # Show in rows of 10 for better layout
            frames_per_row = 10
            for row_start in range(0, len(filtered_frames), frames_per_row):
                row_end = min(row_start + frames_per_row, len(filtered_frames))
                row_frames = filtered_frames[row_start:row_end]
                
                cols = st.columns(len(row_frames))
                for idx, (col, frame) in enumerate(zip(cols, row_frames)):
                    with col:
                        pil_img = Image.fromarray(frame)
                        frame_num = row_start + idx
                        st.image(pil_img, caption=f"F{frame_num}", width='stretch')
        
        # Step 3: Generate description using Gemini
        if use_gemini and filtered_frames:
            try:
                # Convert frames to PIL Images
                pil_images = [Image.fromarray(frame) for frame in filtered_frames]
                
                # Placeholder prompt - YOU WILL FILL THIS IN
                prompt = f"""
                    Provide a detailed description of a video shot based on the given frame images. Focus on creating a cohesive narrative of the entire shot rather than describing each frame individually.

                    Incorporate the following elements in your description: 
                    1. Visual elements:
                    - Describe all visible objects, text, and characters in detail.
                    - For any characters present, include:
                        • Age
                        • Emotional expressions
                        • Clothing and accessories
                        • Physical appearance
                        • Any actions, movements or gestures

                    2. Setting and atmosphere:
                    - Provide details about the time, location, and overall ambiance.
                    - Mention any relevant background elements that contribute to the scene.

                    Skip the preamble; go straight into the description."""
                
                # Send to Gemini
                response = model.generate_content([prompt] + pil_images)
                description = response.text
                
                # Display the generated description
                st.success("✅ Gemini Description:")
                st.write(description)
                st.markdown("---")
                
            except Exception as e:
                st.warning(f"Gemini API error for chunk {chunk_id}: {e}")
                description = f"Video chunk {chunk_id}: A video segment with {len(filtered_frames)} distinct frames"
        else:
            # Fallback description
            description = f"Video chunk {chunk_id}: A video segment with {len(filtered_frames)} distinct frames"
            if not use_gemini:
                st.info(f"📝 Fallback description: {description}")
        
        # Step 4: Embed the description
        embedding = text_model.encode(description)
        embeddings.append(embedding)
        
        # Track chunk processing time
        chunk_end_time = time.time()
        chunk_processing_time = chunk_end_time - chunk_start_time
        chunk_times.append(chunk_processing_time)
        st.write(f"⏱️ Chunk {chunk_id} processed in {chunk_processing_time:.2f}s")
        st.markdown("---")
    
    end_time = time.time()
    
    metrics = {
        "method": "LLM",
        "processing_time": end_time - start_time,
        "num_chunks": len(chunks),
        "embedding_dim": embeddings[0].shape[0] if embeddings else 0,
        "avg_time_per_chunk": (end_time - start_time) / len(chunks) if chunks else 0,
        "chunk_times": chunk_times,  # Add detailed per-chunk times
        "min_chunk_time": min(chunk_times) if chunk_times else 0,
        "max_chunk_time": max(chunk_times) if chunk_times else 0
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
        
        try:
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
            
            # CLIP Method
            with st.spinner("Processing CLIP Method..."):
                embeddings_clip, metrics_clip = clip_embedding(
                    chunks, clip_model, clip_processor, device
                )
                all_embeddings.append(embeddings_clip)
                all_metrics.append(metrics_clip)
                st.success(f"CLIP method completed in {metrics_clip['processing_time']:.2f}s")
            
            # LLM Method
            with st.spinner("Processing LLM Method..."):
                embeddings_llm, metrics_llm = llm_embedding(
                    chunks, text_model, device, gemini_api_key, similarity_method, clip_model, clip_processor
                )
                all_embeddings.append(embeddings_llm)
                all_metrics.append(metrics_llm)
                st.success(f"LLM method completed in {metrics_llm['processing_time']:.2f}s")
            
            # Display results
            st.header("4. Performance Comparison")
            
            # Create metrics dataframe
            df_metrics = pd.DataFrame(all_metrics)
            
            # Display metrics table
            st.subheader("Processing Metrics")
            st.dataframe(df_metrics, width='stretch')
            
            # Visualize metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "CLIP: Processing Time",
                    f"{metrics_clip['processing_time']:.2f}s",
                    delta=None
                )
                st.metric(
                    "Avg Time/Chunk",
                    f"{metrics_clip['avg_time_per_chunk']:.3f}s"
                )
            
            with col2:
                st.metric(
                    "LLM: Processing Time",
                    f"{metrics_llm['processing_time']:.2f}s",
                    delta=f"{metrics_llm['processing_time'] - metrics_clip['processing_time']:.2f}s"
                )
                st.metric(
                    "Avg Time/Chunk",
                    f"{metrics_llm['avg_time_per_chunk']:.3f}s"
                )
            
            # Embedding quality metrics
            st.subheader("Embedding Quality Metrics")
            
            similarities = compute_embedding_similarity(all_embeddings)
            
            quality_df = pd.DataFrame({
                'Method': [m['method'] for m in all_metrics],
                'Embedding Dimension': [m['embedding_dim'] for m in all_metrics],
                'Avg Chunk Similarity': similarities
            })
            
            st.dataframe(quality_df, width='stretch')
            
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
            
            **CLIP Method**
            - ✅ Fast processing with sampled frames (5 frames per chunk)
            - ✅ Good for static visual content
            - ✅ Simple and efficient
            - ⚠️ May miss temporal dynamics between sampled frames
            
            **LLM Method (Gemini 2.0 Flash)**
            - ✅ Uses Gemini 2.0 Flash for rich video understanding
            - ✅ Filters frames using CLIP or perceptual hashing (user-selectable)
            - ✅ Captures scene changes and distinct moments automatically
            - ✅ Semantic understanding through AI-generated descriptions
            - ⚠️ Requires Gemini API key and may have API costs
            - ⚠️ Slower processing due to frame filtering and API calls
            """)
            
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Please check your internet connection (required for first-time model download) and try again")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
        finally:
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
