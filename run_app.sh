#!/bin/bash
# Demo script to run the Streamlit app

echo "🎥 Video Embedding Comparison App"
echo "=================================="
echo ""
echo "Starting Streamlit server..."
echo ""
echo "After the server starts:"
echo "1. Open your browser to http://localhost:8501"
echo "2. Upload a video file (MP4, AVI, MOV, or MKV)"
echo "3. Adjust chunk duration in the sidebar (1-30 seconds)"
echo "4. Click 'Process Video with All Methods'"
echo "5. Review the performance metrics and comparison"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
