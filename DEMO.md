# Demo & Examples

## What the App Does

This Streamlit application allows you to compare three different methods for embedding video clips:

### Method A: Image + Text (CLIP)
- Extracts key frames from each video chunk
- Uses OpenAI's CLIP model to create image embeddings
- Averages embeddings across sampled frames
- **Best for**: Static visual content, object recognition, scene classification

### Method B: Video + Text (Multi-frame)
- Processes multiple frames per chunk to capture temporal information
- Samples 16 frames instead of 5 for better motion understanding
- Uses CLIP but processes more frames to understand video dynamics
- **Best for**: Action recognition, motion analysis, sports videos

### Method C: LLM Description + Text
- Generates text descriptions of video content (simplified in demo)
- Embeds the descriptions using sentence transformers
- Fastest method with semantic understanding
- **Best for**: Semantic search, content categorization, text-based retrieval

## App Workflow

1. **Upload Video**: Choose any video file (MP4, AVI, MOV, MKV)
2. **Configure Settings**: 
   - Select chunk duration (1-30 seconds)
   - Enable/disable GPU acceleration
3. **Process**: Click "Process Video with All Methods"
4. **Review Results**: 
   - Compare processing times
   - Analyze embedding quality
   - See recommendations

## Expected Output

When you process a video, you'll see:

### Processing Metrics Table
| Method | Processing Time | Num Chunks | Embedding Dim | Avg Time/Chunk |
|--------|----------------|------------|---------------|----------------|
| A: Image + Text | ~2.5s | 3 | 512 | 0.833s |
| B: Video + Text | ~4.2s | 3 | 512 | 1.400s |
| C: LLM + Text | ~0.8s | 3 | 384 | 0.267s |

### Quality Metrics Table
| Method | Embedding Dimension | Avg Chunk Similarity |
|--------|-------------------|---------------------|
| A: Image + Text | 512 | 0.85 |
| B: Video + Text | 512 | 0.82 |
| C: LLM + Text | 384 | 0.91 |

### Performance Cards
Three side-by-side cards showing:
- Total processing time
- Average time per chunk
- Delta compared to Method A

## Creating a Test Video

If you don't have a video file, you can create one:

```python
import numpy as np
import cv2

def create_test_video(output_path='test.mp4', duration=10):
    fps = 30
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Create changing colors
        frame[:, :, 2] = int((i / (duration * fps)) * 255)
        frame[:, :, 1] = int((1 - i / (duration * fps)) * 255)
        # Add moving circle
        x = int(width * (i / (duration * fps)))
        cv2.circle(frame, (x, height//2), 50, (255, 255, 255), -1)
        out.write(frame)
    
    out.release()

create_test_video()
```

## Example Use Cases

### Use Case 1: Product Video Analysis
- Upload product demo videos
- Use Method A for quick visual feature extraction
- Best for catalogs where you need fast processing

### Use Case 2: Sports Highlight Analysis
- Upload sports clips
- Use Method B for better action recognition
- Captures dynamic movements and plays

### Use Case 3: Video Search Engine
- Upload content library
- Use Method C for semantic search
- Fast processing with text-based retrieval

## Performance Expectations

### Small Video (10 seconds, 640x480)
- **Method A**: ~1-3 seconds
- **Method B**: ~3-6 seconds  
- **Method C**: ~0.5-1 second

### Medium Video (30 seconds, 1080p)
- **Method A**: ~5-10 seconds
- **Method B**: ~10-20 seconds
- **Method C**: ~1-3 seconds

### Large Video (2 minutes, 1080p)
- **Method A**: ~20-40 seconds
- **Method B**: ~40-80 seconds
- **Method C**: ~5-10 seconds

*Times are approximate and depend on hardware*

## Screenshots

### Initial App View
![Initial View](https://github.com/user-attachments/assets/6778880b-d9e5-4d56-b014-25d74287b6a0)

The app shows:
- Left sidebar with configuration options
- Main area with upload section
- Clean, intuitive interface

### After Processing (Expected)
After clicking "Process Video", you would see:
1. Progress spinners for each method
2. Success messages with timing
3. Performance comparison tables
4. Quality metrics
5. Summary and recommendations

## Customization

You can customize the app by:

1. **Changing Models**: Edit the model names in `load_clip_model()` and `load_text_model()`
2. **Adjusting Sampling**: Modify the number of frames sampled in each method
3. **Adding Methods**: Create new method functions following the existing pattern
4. **Export Options**: Add export functionality to save embeddings

## Troubleshooting

### Models Won't Download
- Ensure internet connection
- Models download to `~/.cache/huggingface/`
- First run takes 5-10 minutes
- Subsequent runs are instant

### Video Won't Process
- Check file format (MP4 with H.264 is most compatible)
- Try converting: `ffmpeg -i input.mov -c:v libx264 output.mp4`

### Out of Memory
- Increase chunk duration
- Process shorter videos
- Disable GPU if it has limited memory

## Next Steps

After testing the app:
1. Try different chunk durations to see the effect
2. Upload videos similar to your use case
3. Compare which method works best for you
4. Integrate the chosen method into your pipeline
5. Consider optimizations for production use
