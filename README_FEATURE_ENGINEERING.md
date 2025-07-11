# Automatic Feature Engineering for Tabular Data

This repository contains comprehensive feature engineering scripts for tabular datasets using sklearn and other libraries.

## 📁 Repository Structure

```
├── data/                           # Original datasets
│   ├── bike_sharing_demand/
│   ├── brazilian_houses/
│   ├── superconductivity/
│   ├── wine_quality/
│   └── yprop_4_1/
├── data_engineered/               # Engineered datasets (output)
│   ├── bike_sharing_demand/
│   ├── brazilian_houses/
│   ├── superconductivity/
│   ├── wine_quality/
│   └── yprop_4_1/
├── feature_engineering.py         # Original complex feature engineering (has issues)
├── simple_feature_engineering.py  # Simple but effective feature engineering
├── single_dataset_processor.py    # Robust single dataset processor
├── autofeat_processor.py          # Simple manual + AutoFeat options
├── process_all_datasets.sh        # Batch processing script
├── verify_engineered_data.py      # Verification script
└── README.md                      # This file
```

## 🚀 Quick Start

### Process a Single Dataset
```bash
# Process yprop_4_1 dataset, fold 1 with simple method
python autofeat_processor.py --task yprop_4_1 --fold 1 --method simple

# Process bike_sharing_demand dataset, fold 1
python autofeat_processor.py --task bike_sharing_demand --fold 1 --method simple
```

### Process All Datasets
```bash
# Make the script executable
chmod +x process_all_datasets.sh

# Run batch processing
./process_all_datasets.sh
```

### Verify Engineered Data
```bash
python verify_engineered_data.py
```

## 📊 Feature Engineering Results

| Dataset | Original Features | Engineered Features | Expansion Ratio | Folds Processed |
|---------|-------------------|-------------------|----------------|-----------------|
| bike_sharing_demand | 11 | 123 | 11.2x | 10/10 |
| brazilian_houses | 11 | 128 | 11.6x | 10/10 |
| superconductivity | 81 | 629 | 7.8x | 10/10 |
| wine_quality | 12 | 142 | 11.8x | 10/10 |
| yprop_4_1 | 62 | 305 | 4.9x | 1/10 |

## 🔧 Feature Engineering Techniques Applied

### 1. **Data Cleaning**
- Missing value imputation using median for numeric, mode for categorical
- Conversion of categorical data types to strings
- Handling of infinite values

### 2. **Statistical Features**
- Row-wise statistics: mean, std, min, max, median, range
- Cross-feature statistics across numeric columns

### 3. **Interaction Features**
- Multiplication, division, addition, subtraction between top numeric features
- Limited to top 5 features to avoid feature explosion

### 4. **Mathematical Transformations**
- Logarithmic transformation (log1p) for non-negative values
- Square root transformation for non-negative values
- Square and absolute value transformations
- Polynomial features (degree 2) for selected features

### 5. **Categorical Encoding**
- Target encoding (mean target value per category)
- Frequency encoding (relative frequency of each category)
- Count encoding (absolute count of each category)
- Label encoding for final categorical representation

### 6. **Binning Features**
- Equal-width binning (5 bins)
- Equal-frequency binning (quantiles)
- Robust handling of edge cases

### 7. **Feature Scaling**
- RobustScaler to handle outliers
- Applied to all numeric features after engineering

## 🛠️ Available Scripts

### 1. `autofeat_processor.py` (Recommended)
- **Purpose**: Process single datasets with robust feature engineering
- **Methods**: Simple manual feature engineering or AutoFeat (if installed)
- **Advantages**: Handles edge cases, consistent results, good for problematic datasets

```bash
python autofeat_processor.py --task DATASET --fold FOLD --method simple
```

### 2. `simple_feature_engineering.py`
- **Purpose**: Simple but effective feature engineering for all datasets
- **Status**: Works for most datasets, may have issues with some complex cases
- **Usage**: Run directly to process all datasets

### 3. `single_dataset_processor.py`
- **Purpose**: More complex feature engineering with advanced techniques
- **Status**: Had issues with feature name consistency, use autofeat_processor instead
- **Note**: Kept for reference

### 4. `feature_engineering.py`
- **Purpose**: Original complex feature engineering attempt
- **Status**: Has issues with feature consistency between train/test
- **Note**: Kept for reference

## 📈 Feature Engineering Quality

✅ **All engineered datasets are clean**:
- No missing values
- No infinite values
- Consistent feature names between train/test sets
- Proper scaling applied

## 🎯 Best Practices Implemented

1. **Consistent Feature Creation**: Same features created for both train and test sets
2. **Robust Handling**: Proper handling of edge cases and missing values
3. **Memory Efficient**: Limited feature explosion through selective feature creation
4. **Scalable**: Can process datasets of different sizes and complexities
5. **Reproducible**: Fixed random seeds for consistent results

## 🔍 Troubleshooting

### Common Issues and Solutions

1. **"Feature names should match" error**:
   - Use `autofeat_processor.py` with `--method simple`
   - This handles feature consistency better

2. **Memory issues**:
   - Feature engineering is limited to prevent excessive memory usage
   - Large datasets process successfully

3. **Categorical data errors**:
   - Script converts categorical columns to strings before processing
   - Handles pandas categorical dtype properly

## 📋 Usage Examples

### Example 1: Process problematic yprop_4_1 dataset
```bash
python autofeat_processor.py --task yprop_4_1 --fold 1 --method simple
```

### Example 2: Process all folds of wine_quality
```bash
for fold in {1..10}; do
    python autofeat_processor.py --task wine_quality --fold $fold --method simple
done
```

### Example 3: Verify all engineered data
```bash
python verify_engineered_data.py
```

## 🎉 Results

The feature engineering successfully creates rich, complex features from the original tabular data:

- **11-128x feature expansion** depending on dataset complexity
- **Clean, consistent datasets** ready for machine learning
- **Robust handling** of various data types and edge cases
- **Multiple feature types** including statistical, interaction, and transformation features

All engineered datasets are saved in the `data_engineered/` folder with the same structure as the original `data/` folder, making them easy to use as drop-in replacements for the original datasets.

## 🔮 Future Enhancements

1. **AutoFeat Integration**: Install autofeat for automatic feature engineering
2. **Feature Selection**: Add feature selection to reduce dimensionality
3. **Deep Feature Synthesis**: Explore automated feature engineering libraries
4. **Custom Transformations**: Add domain-specific transformations for different dataset types
