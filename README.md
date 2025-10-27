# 🎥 Multimodal Video Embeddings Comparison

A Streamlit application to compare different methods for embedding video clips. Test and benchmark three different approaches to understand which works best for your use case.

![App Screenshot](https://github.com/user-attachments/assets/6778880b-d9e5-4d56-b014-25d74287b6a0)

## Features

- **Upload and process video files** (MP4, AVI, MOV, MKV)
- **Configurable chunk duration** - Split videos into chunks of customizable length (1-30 seconds)
- **Three embedding methods:**
  - **Method A**: Image + Text Model (CLIP) - Extracts key frames and embeds using CLIP
  - **Method B**: Video + Text Model - Processes multiple frames for temporal understanding
  - **Method C**: LLM Description + Text - Generates descriptions and embeds with sentence transformers
- **Performance metrics** - Compare processing time, embedding dimensions, and quality
- **Visual comparison** - Side-by-side metrics and recommendations
- **GPU support** - Optional GPU acceleration for faster processing

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
# OR
./run_app.sh
```

Then open your browser to `http://localhost:8501`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ClipABit/multimodal-embeddings.git
cd multimodal-embeddings
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to the URL shown (typically `http://localhost:8501`)

3. Configure settings in the sidebar:
   - Select chunk duration (1-30 seconds)
   - Enable/disable GPU acceleration

4. Upload a video file

5. Click "Process Video with All Methods" to run the comparison

6. Review the performance metrics and recommendations

## Methods Explained

### Method A: Image + Text (CLIP)
- Samples 5 representative frames per chunk
- Uses OpenAI's CLIP model for image embeddings
- Fast processing, good for static visual content
- May miss temporal dynamics

### Method B: Video + Text (Multi-frame)
- Processes 16 frames per chunk
- Better captures motion and temporal sequences
- Slower but more comprehensive
- Better for action-heavy videos

### Method C: LLM Description + Text
- Generates text descriptions of video content
- Embeds descriptions with sentence transformers
- Fastest method
- Quality depends on description accuracy
- Note: Demo uses simplified descriptions; production would use VLMs like BLIP or GPT-4V

## Performance Metrics

The app provides:
- **Processing Time**: Total time to process all chunks
- **Avg Time/Chunk**: Average processing time per chunk
- **Embedding Dimension**: Size of the embedding vectors
- **Chunk Similarity**: Consistency between consecutive chunks

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- 4GB+ RAM recommended
- Internet connection (for first-time model downloads)
- See `requirements.txt` for detailed dependencies

## Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage instructions and tips
- **[DEMO.md](DEMO.md)** - Examples and expected outputs
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project

## Project Structure

```
multimodal-embeddings/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── run_app.sh            # Shell script to start the app
├── README.md             # This file
├── USAGE_GUIDE.md        # Detailed usage guide
├── DEMO.md               # Demo and examples
├── CONTRIBUTING.md       # Contributing guidelines
└── .gitignore           # Git ignore patterns
```

## How It Works

1. **Upload a video** - Choose any supported video file
2. **Configure settings** - Select chunk duration and GPU usage
3. **Process** - The app:
   - Splits video into chunks based on your selected duration
   - Processes each chunk with all 3 methods
   - Measures performance metrics
4. **Compare results** - View side-by-side comparison of:
   - Processing times
   - Embedding quality
   - Recommendations for your use case

## Performance Metrics

The app tracks and displays:
- **Processing Time**: How long each method takes
- **Avg Time/Chunk**: Average processing time per video chunk
- **Embedding Dimension**: Size of the embedding vectors
- **Chunk Similarity**: Consistency between consecutive chunks

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License

## Acknowledgments

- OpenAI CLIP for image-text embeddings
- Sentence Transformers for text embeddings
- Streamlit for the web interface
