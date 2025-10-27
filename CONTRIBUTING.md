# Contributing to Multimodal Video Embeddings

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Making Changes

### Adding New Embedding Methods

To add a new embedding method (e.g., Method D):

1. Create a new function in `app.py`:
   ```python
   def method_d_your_approach(chunks, device):
       """
       Your new method description
       """
       start_time = time.time()
       embeddings = []
       
       # Your implementation here
       
       metrics = {
           "method": "D: Your Method Name",
           "processing_time": time.time() - start_time,
           "num_chunks": len(chunks),
           "embedding_dim": embeddings[0].shape[0],
           "avg_time_per_chunk": ...
       }
       return embeddings, metrics
   ```

2. Call it in the main processing section:
   ```python
   embeddings_d, metrics_d = method_d_your_approach(chunks, device)
   all_embeddings.append(embeddings_d)
   all_metrics.append(metrics_d)
   ```

3. Update the comparison section to include Method D

### Improving Existing Methods

- Method A, B, C are templates - feel free to replace with better models
- Consider using actual video models like VideoMAE, X-CLIP, TimeSformer
- For Method C, integrate real VLMs like BLIP-2, LLaVA, or GPT-4V

### Adding Features

Ideas for enhancement:
- Export embeddings to file (JSON, NPY, etc.)
- Batch processing of multiple videos
- Embedding visualization (UMAP/t-SNE)
- Video segment search/retrieval
- Custom model upload
- Real-time video streaming

## Code Style

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and modular

## Testing

Before submitting:
1. Test with various video formats and lengths
2. Verify all three methods work
3. Check error handling works correctly
4. Ensure UI remains responsive

## Submitting Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

3. Push and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the code
- Documentation improvements
