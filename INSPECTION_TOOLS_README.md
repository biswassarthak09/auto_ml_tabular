# Dataset Inspection Tools

This directory contains several tools for inspecting and analyzing the engineered datasets.

## 🔍 Quick Start

### View a specific dataset
```bash
python quick_inspect.py wine_quality 1 5
```

### Full analysis of all datasets
```bash
python inspect_dataset_heads.py
```

### Verify data quality
```bash
python verify_engineered_data.py
```

## 📋 Available Tools

### 1. `quick_inspect.py` - Quick Dataset Viewer
Simple tool to quickly view any dataset head.

**Usage:**
```bash
python quick_inspect.py <dataset_name> [fold] [rows]
```

**Examples:**
```bash
# View wine quality dataset, fold 1, first 5 rows
python quick_inspect.py wine_quality 1 5

# View bike sharing dataset, fold 3, first 3 rows
python quick_inspect.py bike_sharing_demand 3 3
```

**Available datasets:**
- `bike_sharing_demand`
- `brazilian_houses`  
- `superconductivity`
- `wine_quality`
- `yprop_4_1`

### 2. `inspect_dataset_heads.py` - Detailed Dataset Inspector
Comprehensive analysis tool with feature categorization and data quality checks.

**Usage:**
```bash
# Analyze all datasets
python inspect_dataset_heads.py

# Analyze specific dataset
python inspect_dataset_heads.py --task wine_quality --fold 1 --rows 5
```

**Features:**
- Shows original vs engineered data samples
- Categorizes features by type (Statistical, Interactions, Transformations, Categorical)
- Data quality reporting
- Summary table across all datasets

### 3. `verify_engineered_data.py` - Data Quality Verification
Validates all engineered datasets for consistency and quality.

**Usage:**
```bash
python verify_engineered_data.py
```

**Checks performed:**
- Missing values detection
- Infinite values detection
- Feature count consistency across folds
- Data type validation
- Summary statistics

## 📊 Understanding the Output

### Feature Categories

**Original Features**: Features from the original dataset, normalized to [-1, 1]

**Statistical Features (5 per dataset):**
- `row_mean`: Mean of all numerical features for each row
- `row_std`: Standard deviation across features for each row
- `row_max`: Maximum value across features for each row
- `row_min`: Minimum value across features for each row
- `row_range`: Range (max - min) across features for each row

**Interaction Features (30 per dataset):**
- Pairwise operations between top numerical features
- Format: `feature1_plus_feature2`, `feature1_mult_feature2`, `feature1_div_feature2`

**Transformation Features (variable count):**
- Mathematical transformations of numerical features
- Format: `feature_sqrt`, `feature_log`, `feature_squared`, `feature_abs`

**Categorical Features (variable count):**
- `feature_target_mean`: Mean target value for each category
- `feature_freq`: Normalized frequency of each category

### Data Quality Indicators

- ✅ **Clean**: No missing or infinite values
- ⚠️ **Issues**: Contains missing or infinite values (shouldn't occur)

### Feature Expansion Examples

| Dataset | Original | Engineered | Expansion |
|---------|----------|------------|-----------|
| bike_sharing_demand | 11 | 80 | 7.3x |
| brazilian_houses | 11 | 84 | 7.6x |
| superconductivity | 81 | 440 | 5.4x |
| wine_quality | 12 | 93 | 7.8x |
| yprop_4_1 | 62 | 305 | 4.9x |

## 🎯 Example Usage Workflow

1. **Quick overview of a dataset:**
   ```bash
   python quick_inspect.py wine_quality 1 3
   ```

2. **Detailed analysis with feature categorization:**
   ```bash
   python inspect_dataset_heads.py --task wine_quality --fold 1
   ```

3. **Verify all data is clean:**
   ```bash
   python verify_engineered_data.py
   ```

4. **Compare all datasets:**
   ```bash
   python inspect_dataset_heads.py --rows 3
   ```

## 💡 Tips

- Use `--rows` parameter to control how many sample rows to display
- The `quick_inspect.py` tool is fastest for simple checks
- The `inspect_dataset_heads.py` tool provides the most comprehensive analysis
- All tools work with the same dataset names and folder structure
- Engineered data is stored in `data_engineered/` maintaining the same structure as `data/`

## 📁 File Structure

```
data_engineered/
├── bike_sharing_demand/
│   ├── 1/
│   │   ├── X_train.parquet
│   │   ├── X_test.parquet
│   │   ├── y_train.parquet
│   │   └── y_test.parquet
│   ├── 2/
│   └── ...
├── brazilian_houses/
├── superconductivity/
├── wine_quality/
└── yprop_4_1/
```

Each dataset maintains 10 folds with train/test splits, ready for cross-validation experiments.
