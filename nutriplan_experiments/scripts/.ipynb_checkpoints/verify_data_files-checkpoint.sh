#!/bin/bash
# Data File Verification Script
# Checks if all 9 required data files exist and displays their basic info

DATA_DIR="$HOME/work/recipebench/data/10large_scale_datasets"

echo "=========================================="
echo "NutriPlan Data Files Verification"
echo "=========================================="
echo "Data Directory: $DATA_DIR"
echo ""

# Define all required files
declare -a REQUIRED_FILES=(
    "task_a_train_discriminative.jsonl"
    "task_a_val_discriminative.jsonl"
    "task_a_test_discriminative.jsonl"
    "task_b_train_from_kg.jsonl"
    "task_b_val_from_kg.jsonl"
    "task_b_test_from_kg.jsonl"
    "task_c_train_from_kg.jsonl"
    "task_c_val_from_kg.jsonl"
    "task_c_test_from_kg.jsonl"
)

# Track missing files
MISSING_COUNT=0
TOTAL_COUNT=${#REQUIRED_FILES[@]}

echo "Checking ${TOTAL_COUNT} required files..."
echo ""

# Check each file
for file in "${REQUIRED_FILES[@]}"; do
    FILEPATH="$DATA_DIR/$file"

    if [ -f "$FILEPATH" ]; then
        # File exists - show size and line count
        FILE_SIZE=$(du -h "$FILEPATH" | cut -f1)
        LINE_COUNT=$(wc -l < "$FILEPATH")
        echo "✓ $file"
        echo "  Size: $FILE_SIZE, Lines: $LINE_COUNT"
    else
        # File missing
        echo "✗ $file"
        echo "  Status: MISSING"
        ((MISSING_COUNT++))
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "Total files required: $TOTAL_COUNT"
echo "Files found: $((TOTAL_COUNT - MISSING_COUNT))"
echo "Files missing: $MISSING_COUNT"
echo ""

if [ $MISSING_COUNT -eq 0 ]; then
    echo "✅ All data files are present!"
    echo ""
    echo "Next step: Run test training"
    echo "Command:"
    echo "  bash scripts/test_training_plan_a.sh"
else
    echo "❌ Missing $MISSING_COUNT file(s)"
    echo ""
    echo "Please ensure all data files are in:"
    echo "  $DATA_DIR"
fi
echo "=========================================="
