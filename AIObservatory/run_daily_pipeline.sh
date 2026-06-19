#!/bin/bash

# Ensure a filename was provided
if [ -z "$1" ]; then
    echo "Usage: ./run_daily_pipeline.sh <path_to_daily_file>"
    exit 1
fi

DAILY_FILE=$1
FILENAME=$(basename "$DAILY_FILE") # Extracts just 'cleaned_prompts_2026-06-15.txt'

echo "=========================================="
echo "Starting AI Pipeline for: $FILENAME"
echo "=========================================="

# 1. Stage the Data
# Copy the daily file from your shared drive into the 'data/' folder so the python scripts can find it
cp "$DAILY_FILE" "data/prompt_sample.txt"

# 2. Phase 1: Filter Languages
echo "[1/3] Running Language Router..."
python3.9 src/language_router.py

# 3. Phase 2 & 3: Run the ML Engine (Zero-Shot + SGD Rescue)
echo "[2/3] Running Machine Learning Discovery..."
python3.9 src/discovery.py

# 4. Phase 4: Push to Postgres
echo "[3/3] Exporting to Database..."
# Pass the original filename so export_to_db.py can extract the Date
python3.9 export_to_db.py "$FILENAME"

echo "=========================================="
echo "Pipeline Complete! Data is now in Postgres."
echo "=========================================="
