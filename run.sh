#!/bin/bash
# Run the Fake News Detector app
cd "$(dirname "$0")"
python3 -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
