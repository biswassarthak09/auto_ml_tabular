#!/bin/bash

# Batch script to process all datasets with simple feature engineering for all folds

echo "🚀 Starting batch feature engineering for all datasets and folds..."

datasets=("bike_sharing_demand" "brazilian_houses" "superconductivity" "wine_quality" "yprop_4_1")
total_datasets=${#datasets[@]}
total_folds=10
total_tasks=$((total_datasets * total_folds))
current_task=0

for dataset in "${datasets[@]}"; do
    echo "📊 Processing dataset: $dataset"
    dataset_success=0
    dataset_failed=0
    
    for fold in {1..10}; do
        current_task=$((current_task + 1))
        echo "  🔄 Processing fold $fold/$total_folds (Task $current_task/$total_tasks)"
        
        python autofeat_processor.py --task $dataset --fold $fold --method simple
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Successfully processed $dataset fold $fold"
            dataset_success=$((dataset_success + 1))
        else
            echo "  ❌ Failed to process $dataset fold $fold"
            dataset_failed=$((dataset_failed + 1))
        fi
    done
    
    echo "📈 Dataset $dataset summary: $dataset_success/$total_folds folds successful, $dataset_failed failed"
    echo "---"
done

echo "🎉 Batch processing complete!"
echo "📁 Check the data_engineered/ folder for results"
echo "📊 Processed $total_tasks tasks total"
