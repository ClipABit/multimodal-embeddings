# Usage Guide: Video Embedding Comparison App

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   # OR
   ./run_app.sh
   ```

3. **Open in Browser**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to the URL shown in the terminal

## Step-by-Step Usage

### 1. Configure Settings
- In the sidebar, adjust the **Chunk Duration** (1-30 seconds)
  - Smaller chunks = more granular embeddings but slower processing
  - Larger chunks = faster processing but less granular
- Enable **GPU** if available for faster processing

### 2. Upload Video
- Click "Browse files" or drag-and-drop a video
- Supported formats: MP4, AVI, MOV, MKV
- The video will be displayed with basic metadata

### 3. Process Video
- Click the "🚀 Process Video with All Methods" button
- Wait while the app:
  - Extracts frames from your video
  - Loads embedding models (first time may take a few minutes to download)
  - Processes chunks with all 3 methods

### 4. Review Results
The app will display:
- **Processing Metrics**: Time taken for each method
- **Embedding Quality**: Dimension and similarity metrics
- **Comparison**: Which method is fastest and best suited
- **Recommendations**: Use case guidance for each method

## Understanding the Methods

### Method A: Image + Text (CLIP)
**How it works:**
- Samples 5 representative frames per chunk
- Extracts visual features using CLIP
- Averages embeddings across frames

**Best for:**
- Static visual content
- Object recognition
- Scene classification
- Fast processing needs

**Example use cases:**
- Product catalogs
- Surveillance footage
- Screenshot analysis

### Method B: Video + Text (Multi-frame)
**How it works:**
- Samples 16 frames per chunk
- Processes multiple frames to understand motion
- Captures temporal relationships

**Best for:**
- Action recognition
- Motion analysis
- Sports videos
- Dancing/movement content

**Example use cases:**
- Sports analytics
- Action detection
- Gesture recognition

### Method C: LLM Description + Text
**How it works:**
- Generates text descriptions of video content
- Embeds descriptions using sentence transformers
- Semantic understanding through language

**Best for:**
- Semantic search
- Content categorization
- Fast processing
- Text-based retrieval

**Example use cases:**
- Video search engines
- Content recommendation
- Automated tagging

## Performance Interpretation

### Processing Time
- **Lower is better** for speed
- Method C is typically fastest
- Method B is typically slowest but most thorough

### Embedding Dimension
- Higher dimensions can capture more information
- But use more memory and compute
- CLIP: 512 dimensions
- Text models: 384 dimensions

### Chunk Similarity
- Measures consistency between consecutive chunks
- Range: 0 (completely different) to 1 (identical)
- Higher similarity = more consistent embeddings
- Lower similarity = more variation between chunks

## Tips & Best Practices

### Choosing Chunk Duration
- **Short chunks (1-3s)**: Fine-grained analysis, slower processing
- **Medium chunks (5-10s)**: Balanced approach
- **Long chunks (15-30s)**: Fast processing, coarse-grained

### Hardware Considerations
- **CPU only**: Works but slower, reduce chunk duration
- **GPU available**: Enable in sidebar for 5-10x speedup
- **Memory**: Ensure 4GB+ RAM available

### First-Time Setup
- Models will download on first run (~1-2 GB)
- Requires internet connection
- Models are cached for future use
- Downloads go to `~/.cache/huggingface/`

## Troubleshooting

### "Failed to load model" Error
- **Cause**: No internet connection or HuggingFace is down
- **Solution**: 
  - Check internet connection
  - Try again in a few minutes
  - Use VPN if behind firewall

### Video Not Processing
- **Cause**: Unsupported codec or corrupted file
- **Solution**: 
  - Convert to MP4 with H.264 codec
  - Use `ffmpeg -i input.mov -c:v libx264 output.mp4`

### Out of Memory Error
- **Cause**: Video too long or chunk size too small
- **Solution**:
  - Increase chunk duration
  - Process shorter video segments
  - Close other applications

### Slow Processing
- **Cause**: CPU processing or large video
- **Solution**:
  - Enable GPU if available
  - Increase chunk duration
  - Use smaller resolution video

## Example Workflows

### Workflow 1: Quick Comparison
1. Upload a short video (10-30 seconds)
2. Set chunk duration to 5 seconds
3. Process with all methods
4. Compare processing times
5. Choose method based on speed/accuracy tradeoff

### Workflow 2: Detailed Analysis
1. Upload representative sample of your video dataset
2. Start with large chunks (10s)
3. Process and review results
4. Adjust chunk size based on similarity scores
5. Re-run with optimal settings

### Workflow 3: Production Testing
1. Test with videos similar to your use case
2. Try different chunk durations
3. Measure which method gives best results
4. Note processing time for production scaling
5. Implement chosen method in your pipeline

## Advanced Configuration

For production use, you may want to:
- Modify the model selection in `app.py`
- Adjust sampling rates in each method
- Add custom embedding models
- Export embeddings to database
- Batch process multiple videos

See the code comments in `app.py` for customization points.
