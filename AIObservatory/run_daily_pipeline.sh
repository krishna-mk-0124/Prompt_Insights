#!/bin/bash

# If no argument is provided, automatically find today's file
if [ -z "$1" ]; then
    TODAY=$(date +%Y-%m-%d)
    DAILY_FILE="/abc/desre-shared/contexts/cleaned_prompts_${TODAY}.txt"
    
    # Check if today's file has actually been generated yet
    if [ ! -f "$DAILY_FILE" ]; then
        echo "Error: Auto-detect failed. Today's file ($DAILY_FILE) does not exist yet."
        exit 1
    fi
else
    # If a file was manually passed as an argument, use that instead (useful for backfilling old dates)
    DAILY_FILE=$1
    if [ ! -f "$DAILY_FILE" ]; then
        echo "Error: Provided file ($DAILY_FILE) does not exist."
        exit 1
    fi
fi

FILENAME=$(basename "$DAILY_FILE") # Extracts just 'cleaned_prompts_YYYY-MM-DD.txt'

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
