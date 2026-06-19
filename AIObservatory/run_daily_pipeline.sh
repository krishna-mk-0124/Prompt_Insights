#!/bin/bash

# If no argument is provided, automatically find yesterday's file
if [ -z "$1" ]; then
    YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
    DAILY_FILE="/abc/desre-shared/contexts/cleaned_prompts_${YESTERDAY}.txt"
    
    # Check if yesterday's file has actually been generated yet
    if [ ! -f "$DAILY_FILE" ]; then
        echo "Error: Auto-detect failed. Yesterday's file ($DAILY_FILE) does not exist yet."
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
echo "Overall Pipeline Start: $(date)"
echo "=========================================="
PIPELINE_START=$(date +%s)

# Helper function to track execution time
run_with_timer() {
    local phase_name="$1"
    local cmd="$2"
    
    echo ""
    echo ">> $phase_name"
    echo "   Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    local start_time=$(date +%s)
    
    # Execute the command
    eval "$cmd"
    local exit_code=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "   End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    local min=$((duration / 60))
    local sec=$((duration % 60))
    echo "   Phase Duration: ${min}m ${sec}s"
    
    if [ $exit_code -ne 0 ]; then
        echo "   ERROR: Phase failed!"
        exit $exit_code
    fi
}

# 1. Stage the Data
# Copy the daily file from your shared drive into the 'data/' folder so the python scripts can find it
cp "$DAILY_FILE" "data/prompt_sample.txt"

# 2. Phase 1: Filter Languages
run_with_timer "[1/3] Running Language Router..." "python3.9 -u src/language_router.py"

# 3. Phase 2 & 3: Run the ML Engine (Zero-Shot + SGD Rescue)
run_with_timer "[2/3] Running Machine Learning Discovery..." "python3.9 -u src/discovery.py"

# 4. Phase 4: Push to Postgres
# Pass the original filename so export_to_db.py can extract the Date
run_with_timer "[3/3] Exporting to Database..." "python3.9 -u export_to_db.py \"$FILENAME\""

echo ""
echo "=========================================="
echo "Pipeline Complete! Data is now in Postgres."
PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - PIPELINE_START))
T_MIN=$((TOTAL_DURATION / 60))
T_SEC=$((TOTAL_DURATION % 60))
echo "Total Pipeline Duration: ${T_MIN}m ${T_SEC}s"
echo "Overall Pipeline End: $(date)"
echo "=========================================="
