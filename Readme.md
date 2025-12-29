# NutriPlan-Bench: A Benchmark Dataset and Experimental Framework for Nutritional Planning

# 1. Project Overview

NutriPlan-Bench is a comprehensive benchmark dataset and accompanying experimental framework for **intelligent nutritional planning** tasks. It aims to provide standardized support for performance evaluation and algorithm innovation of Large Language Models (LLMs) in nutritional planning scenarios. The framework integrates authoritative nutritional data from USDA/FNDDS and real recipe data from Food.com, covering three core tasks: Discriminative Ranking of Recipes (Task A), Constrained Generation (Task B), and Reflective Editing (Task C). It also offers a full-process toolchain from raw data processing, rule base construction, benchmark dataset generation to experiment execution and evaluation.

## 1.1 Core Features

- 📊 Multi-task Standardized Benchmark: Comprehensive coverage of three core tasks with unified data formats and evaluation metrics

- 🥗 Authoritative Nutritional Data Support: Integration of USDA FoodData Central (FDC) and FNDDS datasets, covering over 80 nutrient types

- 🔬 Full-process Experimental Toolchain: Complete pipeline including raw data alignment, rule base construction, dataset generation, model training, and evaluation validation

- 🧪 Enhanced Reflective Editing Dataset: Supports identification and correction of 8 nutritional violation types, with real ingredient parsing rate > 95% and correction success rate of 90-95%

- 📈 Multi-dimensional Statistical Analysis: Provides full-link statistical reports on dataset distribution, user portraits, nutritional features, and experimental results

- 🔧 High Extensibility: Supports adding new nutritional rules, extending LLM model adaptation, and integrating new task types

# 2. Complete Directory Structure

The project adopts a modular directory design with clear responsibilities and explicit dependencies for each directory. The complete structure is as follows (including all files and descriptions):

```Plain Text


NutriPlan-Bench/                  # Project root directory
├── README.md                     # Project main document (this document)
├── requirements.txt              # Complete dependency list (with version constraints)
├── .gitignore                    # Git ignore file configuration
├── LICENSE                       # License file
├── CONTRIBUTING.md               # Contribution guidelines
├── docs/                         # Project detailed documentation set
│   ├── theory_background.md      # Theoretical background (nutritional basics, task design principles)
│   ├── data_specification.md     # Dataset format specification (unified standard for all tasks)
│   ├── evaluation_metrics.md     # Detailed explanation of evaluation metrics (calculation logic, reference standards)
│   ├── faq_detailed.md           # Detailed FAQ
│   └── troubleshooting.md        # Troubleshooting guide
├── scripts/                      # Core scripts directory (full-process toolchain)
│   ├── __init__.py               # Python package initialization
│   ├── rawdataprocess/           # Raw data processing module (core of data alignment)
│   │   ├── __init__.py
│   │   ├── step1_data_preparation.py  # Data preprocessing (cleaning, standardization, format conversion)
│   │   ├── step2_ingredient_normalization.py  # Ingredient text standardization (special character, fraction processing)
│   │   ├── 3run_usda_align.py    # USDA/FNDDS data alignment main script
│   │   ├── utils_usda_align.py   # Alignment utility functions (embedding, retrieval, matching logic)
│   │   ├── config_usda_align.json  # Alignment parameter configuration file
│   │   └── test_usda_align.py    # Alignment module unit tests
│   ├── KG/                       # Knowledge graph and rule base module
│   │   ├── __init__.py
│   │   ├── build_rule_nutrition_complementarity.py  # Construction of nutritional complementarity rule base
│   │   ├── build_ingredient_knowledge_graph.py  # Construction of ingredient knowledge graph
│   │   ├── utils_kg.py           # Graph/rule base utility functions
│   │   ├── config_kg.json        # Graph construction parameter configuration
│   │   └── test_kg.py            # Graph module unit tests
│   ├── benchmark_generate/       # Benchmark dataset generation module (three tasks)
│   │   ├── __init__.py
│   │   ├── config_benchmark.json  # Dataset generation parameter configuration
│   │   ├── utils_benchmark.py    # Dataset generation utility functions
│   │   ├── test_benchmark.py     # Dataset module unit tests
│   │   ├── build_task_a_dataset_rni_FINAL.py  # Task A (Discriminative Ranking) dataset construction
│   │   ├── build_task_b_dataset_constrained.py  # Task B (Constrained Generation) dataset construction
│   │   ├── generate_c_class_full_v3_ENHANCED_FIXED.py  # Task C (Reflective Editing) enhanced version generation
│   │   ├── split_train_val_test.py  # Dataset splitting (training/validation/test sets)
│   │   └── C_CLASS_FILES_SUMMARY.md  # Detailed explanation of Task C dataset
│   ├── nutriplan_experiments/    # Core experimental framework module
│   │   ├── README_FIRST.md       # Experiment entry guide (MUST READ)
│   │   ├── EXECUTION_GUIDE.md    # Detailed execution guide (including theoretical background, step-by-step breakdown)
│   │   ├── QUICK_COMMANDS.md     # Quick command reference (common commands cheat sheet)
│   │   ├── UPDATE_RQ2_CONFIG.md  # RQ2 experiment configuration update instructions
│   │   ├── COMMANDS_CHECKLIST.md # Experiment execution command checklist (to avoid omissions)
│   │   ├── config_experiments.json  # Global experiment configuration
│   │   ├── __init__.py
│   │   ├── data/                 # Experimental data processing scripts
│   │   │   ├── __init__.py
│   │   │   ├── data_statistics.py  # Dataset statistical analysis script
│   │   │   ├── data_loader.py    # Experimental data loader (adapted to each task)
│   │   │   └── stats_report_template.md  # Statistical report template
│   │   ├── scripts/              # Experiment execution and evaluation pipeline scripts
│   │   │   ├── __init__.py
│   │   │   ├── run_rq1_experiments.sh  # RQ1 experiment execution script (batch)
│   │   │   ├── run_rq2_experiments.sh  # RQ2 baseline experiment execution script (batch)
│   │   │   ├── full_evaluation_pipeline.sh  # Complete evaluation pipeline (automatic metric calculation, report generation)
│   │   │   ├── model_adapter.py  # Model adaptation tool (supports unified call of multiple LLMs)
│   │   │   ├── evaluation_metrics.py  # Core functions for evaluation metric calculation
│   │   │   ├── result_aggregator.py  # Experimental result aggregation and formatting
│   │   │   └── visualization.py  # Result visualization script (generating charts)
│   │   └── tests/                # Experimental module test scripts
│   │       ├── __init__.py
│   │       ├── test_evaluation.py  # Evaluation metric unit tests
│   │       └── test_data_loader.py  # Data loader tests
│   └── common/                   # Common utility module (reused across the project)
│       ├── __init__.py
│       ├── logger.py             # Logging tool (unified log format)
│       ├── file_utils.py         # File operation tools (reading/writing, decompression, format conversion)
│       ├── text_processing.py    # Text processing tools (tokenization, embedding, matching)
│       └── metrics_utils.py      # General metric calculation tools
├── data/                         # Raw data storage directory (needs to be downloaded and placed by users)
│   ├── README_DATA.md            # Data download and placement guide
│   ├── usda/                     # USDA FDC data
│   │   ├── fdc_data.csv          # FDC core data (to be downloaded)
│   │   ├── nutrient_definitions.csv  # Nutrient definition table (to be downloaded)
│   │   └── food_category.csv     # Food category table (to be downloaded)
│   ├── fndds/                    # FNDDS data
│   │   ├── fndds_foods.csv       # FNDDS food table (to be downloaded)
│   │   ├── fndds_nutrients.csv   # FNDDS nutrient data (to be downloaded)
│   │   └── fndds_portions.csv    # Portion data (to be downloaded)
│   └── foodcom/                  # Food.com recipe data
│       ├── recipes.csv           # Basic recipe information (to be downloaded)
│       ├── reviews.csv           # Recipe ratings and reviews (to be downloaded)
│       └── ingredients.csv       # Recipe ingredient list (to be downloaded)
├── outputs/                      # Output file directory (automatically generated, including intermediate results and final products)
│   ├── README_OUTPUT.md          # Explanation of output files
│   ├── usda_align/               # Raw data alignment results
│   ├── nutrition_rules/          # Nutritional rule base and graph results
│   ├── benchmark/                # Benchmark dataset (stored by task)
│   │   ├── task_a/
│   │   │   ├── train.csv
│   │   │   ├── val.csv
│   │   │   ├── test.csv
│   │   │   └── stats_task_a.json
│   │   ├── task_b/
│   │   │   ├── train.csv
│   │   │   ├── val.csv
│   │   │   ├── test.csv
│   │   │   └── stats_task_b.json
│   │   └── task_c_enhanced/
│   │       ├── train.csv
│   │       ├── val.csv
│   │       ├── test.csv
│   │       ├── violation_type_stats.json
│   │       └── correction_stats.json
│   ├── experiments/              # Experimental results directory
│   │   ├── rq1/
│   │   ├── rq2/
│   │   └── visualization/        # Experimental result visualization charts
│   └── stats_reports/            # Statistical reports directory
└── models/                       # Model-related directory (pre-trained model cache, custom models)
    ├── README_MODELS.md          # Model usage guide
    ├── cache/                    # Pre-trained model cache directory (automatically generated)
    └── custom_models/            # Custom models (e.g., fine-tuned LLMs)
        ├── __init__.py
        └── model_configs/        # Custom model configuration files

```

## 2.1 Core Directory Explanation

|Top-level Directory|Core Responsibilities|Key Dependencies|
|---|---|---|
|docs/|Stores full project documentation, including theory, specifications, troubleshooting, etc.|None (independent document set)|
|scripts/|Implements core functions, including data processing, rule construction, dataset generation, and full experiment execution process|Depends on raw data in data/, outputs to outputs/|
|data/|Stores raw data, which needs to be downloaded and placed by users according to guidelines|None (data source)|
|outputs/|Automatically generated intermediate results and final products, including aligned data, rule bases, benchmark datasets, and experimental results|Depends on script execution in scripts/, input is raw data from data/|
|models/|Stores model cache and custom models, supporting calls to pre-trained or fine-tuned models in experiments|Depends on experimental script calls in scripts/nutriplan_experiments/|
# 3. Environment Preparation

## 3.1 System Requirements

- Operating System: Linux (Ubuntu 20.04+ recommended) / macOS 12+ (some scripts need adaptation) / Windows (WSL2 recommended)

- Hardware Requirements: CPU ≥ 16 cores, RAM ≥ 64GB (data alignment and experiment phases); GPU (optional, for experiment training phase, NVIDIA A10G+ recommended)

- Storage Requirements: Free space ≥ 100GB (including raw data, intermediate results, experimental results)

## 3.2 Dependency Installation

Python 3.9+ is recommended. Install complete dependencies via pip (with version constraints to avoid compatibility issues):

```Plain Text


# Method 1: One-click installation via requirements.txt (recommended)
pip install -r requirements.txt

# Method 2: Manual step-by-step installation (for individual debugging)
# Basic data processing dependencies
pip install pandas==2.1.4 numpy==1.26.3 scipy==1.11.4 scikit-learn==1.3.2
pip install pyarrow==14.0.2 fastparquet==2023.10.1 json5==0.9.14
# Text processing and embedding dependencies
pip install sentence-transformers==2.2.2 transformers==4.35.2 tokenizers==0.15.0
pip install regex==2023.10.3 nltk==3.8.1
# Retrieval and matching dependencies
pip install faiss-cpu==1.7.4 fuzzywuzzy==0.18.0 python-Levenshtein==0.23.0
# Logging and utility dependencies
pip install tqdm==4.66.1 logging==0.5.1.2 filelock==3.13.1
# Visualization dependencies
pip install matplotlib==3.8.2 seaborn==0.13.1 plotly==5.18.0
# LLM training and inference dependencies (experiment phase)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
pip install accelerate==0.24.1 peft==0.6.2
# Unit test dependencies
pip install pytest==7.4.3 pytest-cov==4.1.0

```

## 3.3 Raw Data Preparation

Download the following data according to the guidelines in `data/README_DATA.md` and place them in the corresponding directories:

|Data Type|Download Link|Target Directory|Required Files|
|---|---|---|---|
|USDA FDC Data|https://fdc.nal.usda.gov/download-datasets.html|data/usda/|fdc_data.csv, nutrient_definitions.csv, food_category.csv|
|FNDDS Data|https://www.ars.usda.gov/northeast-area/beltsville-md/beltsville-human-nutrition-research-center/food-surveys-research-group/docs/fndds/|data/fndds/|fndds_foods.csv, fndds_nutrients.csv, fndds_portions.csv|
|Food.com Data|https://www.kaggle.com/datasets/shuyangli94/foodcom-recipes-with-search-terms-and-user-reviews|data/foodcom/|recipes.csv, reviews.csv, ingredients.csv|
# 4. Complete Guide to Core Workflow

The core workflow of the project follows the sequence of "Raw Data Processing → Rule Base Construction → Benchmark Dataset Generation → Experiment Execution and Evaluation". Each step depends on the output of the previous step and must be executed in order.

## 4.1 Raw Data Processing (USDA/Food.com Alignment)

### 4.1.1 Function Explanation

Accurately align unstructured ingredient text from Food.com recipes with authoritative nutritional data from USDA/FNDDS to generate recipe data with complete nutritional labels. Core steps include: Data Cleaning and Standardization → Ingredient Text Embedding → Semantic Nearest Neighbor Retrieval → Fuzzy Matching Validation → Nutritional Data Aggregation Calculation → Statistical Report Generation.

### 4.1.2 Execution Steps

```Plain Text


# Step 1: Data preprocessing (cleaning, standardization)
python scripts/rawdataprocess/step1_data_preparation.py \
  --usda_dir data/usda \
  --fndds_dir data/fndds \
  --foodcom_dir data/foodcom \
  --out_dir outputs/usda_align/preprocessed \
  --log_file outputs/usda_align/preprocessed/step1_log.txt

# Step 2: Ingredient text standardization (processing special characters, Unicode fractions, etc.)
python scripts/rawdataprocess/step2_ingredient_normalization.py \
  --input_file outputs/usda_align/preprocessed/foodcom_ingredients_raw.csv \
  --out_file outputs/usda_align/preprocessed/foodcom_ingredients_normalized.csv \
  --synonym_file scripts/rawdataprocess/ingredient_synonyms.json \
  --log_file outputs/usda_align/preprocessed/step2_log.txt

# Step 3: Core alignment (matching Food.com ingredients with USDA/FNDDS)
python scripts/rawdataprocess/3run_usda_align.py \
  --config_file scripts/rawdataprocess/config_usda_align.json \
  --preprocessed_dir outputs/usda_align/preprocessed \
  --out_dir outputs/usda_align \
  --log_file outputs/usda_align/usda_align_log.txt

```

### 4.1.3 Complete Explanation of Output Files

|Directory|Filename|Detailed Explanation|
|---|---|---|
|outputs/usda_align/preprocessed/|foodcom_recipes_preprocessed.csv|Cleaned Food.com recipe data (invalid fields removed, missing values supplemented)|
|outputs/usda_align/preprocessed/|foodcom_ingredients_raw.csv|Raw ingredient list (unstandardized)|
|outputs/usda_align/preprocessed/|foodcom_ingredients_normalized.csv|Standardized ingredient list (special characters processed, unified format)|
|outputs/usda_align/|nutrient_catalog_usda_fndds.csv|Merged USDA+FNDDS nutrient catalog (including nutrient ID, name, unit, recommended intake)|
|outputs/usda_align/|ingredient_mapping.parquet|Ingredient-USDA FDC ID mapping table (including matching score, confidence)|
|outputs/usda_align/|recipe_with_nutrition.parquet|Core output: Main recipe file with complete nutritional data (each recipe contains over 80 nutrient values)|
|outputs/usda_align/|logs_stats.json|Alignment statistics (matching rate, number of recipes using default grams, time consumption of each step, etc.)|
|outputs/usda_align/|present_nutrients_in_recipes.csv|List of nutrients actually present in recipes (including occurrence frequency, number of covered recipes)|
|outputs/usda_align/|unmatched_ingredients.csv|List of unmatched ingredients (for subsequent manual optimization or parameter adjustment)|
## 4.2 Construction of Nutritional Complementarity Rule Base and Knowledge Graph

### 4.2.1 Function Explanation

Based on the aligned ingredient nutritional data, two core products are constructed: 1) Nutritional Complementarity Rule Base (including ingredient synergy rules, such as "High iron + High vitamin C promotes absorption"); 2) Ingredient Knowledge Graph (including ingredient-nutrient-category relationships), providing theoretical nutritional support for subsequent dataset generation and recipe evaluation.

### 4.2.2 Execution Steps

```Plain Text


# Step 1: Build nutritional complementarity rule base
python scripts/KG/build_rule_nutrition_complementarity.py \
  --config_file scripts/KG/config_kg.json \
  --ingredient_nutrition_file outputs/usda_align/recipe_with_nutrition.parquet \
  --output_dir outputs/nutrition_rules \
  --log_file outputs/nutrition_rules/complementarity_rule_log.txt

# Step 2: Build ingredient knowledge graph
python scripts/KG/build_ingredient_knowledge_graph.py \
  --config_file scripts/KG/config_kg.json \
  --ingredient_mapping_file outputs/usda_align/ingredient_mapping.parquet \
  --nutrient_catalog_file outputs/usda_align/nutrient_catalog_usda_fndds.csv \
  --output_dir outputs/nutrition_rules/kg \
  --log_file outputs/nutrition_rules/kg/kg_build_log.txt

```

### 4.2.3 Complete Explanation of Output Files

|Directory|Filename|Detailed Explanation|
|---|---|---|
|outputs/nutrition_rules/|nutrition_complementarity_pairs.csv|Ingredient nutritional complementarity pairs (including synergy score, complementarity principle, application scenarios)|
|outputs/nutrition_rules/|ingredient_nutrient_tags.csv|Ingredient nutritional tags (e.g., "high protein", "high fiber", "low sodium", "rich in vitamin C", etc.)|
|outputs/nutrition_rules/|nutrition_complementarity.json|Quick query dictionary for complementarity rules (key: ingredient pair, value: synergy score and principle)|
|outputs/nutrition_rules/|nutrition_violation_rules.csv|Nutritional violation judgment rules (corresponding to 8 core violation types in Task C, including threshold standards)|
|outputs/nutrition_rules/kg/|ingredient_kg.graphml|Ingredient knowledge graph file (GraphML format, visualizable with Gephi)|
|outputs/nutrition_rules/kg/|kg_node_attributes.csv|Graph node attribute table (including node ID, type: ingredient/nutrient/category, attribute value)|
|outputs/nutrition_rules/kg/|kg_edge_relations.csv|Graph edge relationship table (including edge ID, start node, end node, relationship type: contains/belongs to/complementary)|
## 4.3 Benchmark Dataset Generation (Three Tasks)

Based on the previously processed recipe data with nutritional labels and rule base, standardized benchmark datasets for three core tasks are generated. All datasets are split into training, validation, and test sets in an 8:1:1 ratio, and their formats uniformly follow `docs/data_specification.md`.

### 4.3.1 Task A: Discriminative Ranking

#### Task Description

Input: User profile (age, gender, physiological status, activity level), nutritional requirements (based on RNI recommended intake); Output: Ranking results of recipes that meet the requirements (sorted by matching degree from high to low). Dataset samples include "user-multi-recipe" pairs and corresponding ranking labels.

#### Execution Commands

```Plain Text


python scripts/benchmark_generate/build_task_a_dataset_rni_FINAL.py \
  --config_file scripts/benchmark_generate/config_benchmark.json \
  --recipe_nutrition_file outputs/usda_align/recipe_with_nutrition.parquet \
  --nutrition_rules_file outputs/nutrition_rules/nutrition_complementarity.json \
  --output_dir outputs/benchmark/task_a \
  --log_file outputs/benchmark/task_a/task_a_generate_log.txt

# Dataset splitting (training/validation/test sets)
python scripts/benchmark_generate/split_train_val_test.py \
  --input_file outputs/benchmark/task_a/task_a_full.csv \
  --output_dir outputs/benchmark/task_a \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42

```

### 4.3.2 Task B: Constrained Generation

#### Task Description

Input: User nutritional constraints (e.g., "Daily protein ≥ 80g, fiber ≥ 25g"), ingredient preferences (liked/disliked ingredients), scenario constraints (e.g., "weight loss", "cardiovascular-friendly", "children's meal"); Output: Complete recipes that meet all constraints (including ingredient list, steps, nutritional composition table).

#### Execution Commands

```Plain Text


python scripts/benchmark_generate/build_task_b_dataset_constrained.py \
  --config_file scripts/benchmark_generate/config_benchmark.json \
  --recipe_nutrition_file outputs/usda_align/recipe_with_nutrition.parquet \
  --nutrition_rules_file outputs/nutrition_rules/nutrition_complementarity.json \
  --output_dir outputs/benchmark/task_b \
  --log_file outputs/benchmark/task_b/task_b_generate_log.txt

# Dataset splitting
python scripts/benchmark_generate/split_train_val_test.py \
  --input_file outputs/benchmark/task_b/task_b_full.csv \
  --output_dir outputs/benchmark/task_b \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42

```

### 4.3.3 Task C: Reflective Editing

#### Task Description

Input: Original recipes with nutritional violations, explanation of violation types; Output: Optimized recipes that have been corrected for violations and achieved nutritional balance through ingredient replacement/adjustment. Enhanced dataset features: Real ingredient parsing and modification (not just annotation), accurate nutritional recalculation, covering 8 core nutritional violation types.

#### Execution Commands

```Plain Text


python scripts/benchmark_generate/generate_c_class_full_v3_ENHANCED_FIXED.py \
  --config_file scripts/benchmark_generate/config_benchmark.json \
  --recipe_nutrition_file outputs/usda_align/recipe_with_nutrition.parquet \
  --nutrition_violation_rules_file outputs/nutrition_rules/nutrition_violation_rules.csv \
  --output_dir outputs/benchmark/task_c_enhanced \
  --log_file outputs/benchmark/task_c_enhanced/task_c_generate_log.txt

# Dataset splitting
python scripts/benchmark_generate/split_train_val_test.py \
  --input_file outputs/benchmark/task_c_enhanced/task_c_full.csv \
  --output_dir outputs/benchmark/task_c_enhanced \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42

```

### 4.3.4 Common Outputs of Three Task Datasets

|File Type|Filename|Explanation|
|---|---|---|
|Complete Dataset|task_x_full.csv|Complete unsplit dataset (x = a/b/c)|
|Split Datasets|train.csv/val.csv/test.csv|Training/validation/test sets split in 8:1:1 ratio|
|Statistical Information|stats_task_x.json|Dataset statistics (number of samples, feature distribution, constraint coverage rate, etc.)|
|Data Dictionary|data_dictionary_task_x.md|Dataset field explanation (including field type, meaning, value range)|
## 4.4 Experiment Execution and Evaluation (NutriPlan Experimental Framework)

### 4.4.1 Pre-configuration

1. Edit `scripts/nutriplan_experiments/config_experiments.json` to configure experiment parameters (model type, number of training epochs, batch size, evaluation metrics, etc.); 2. Refer to `scripts/nutriplan_experiments/UPDATE_RQ2_CONFIG.md` to update the base model configuration for RQ2 experiments (e.g., specify `BEST_BASE_LLM="Qwen/Qwen2-7B"`).

### 4.4.2 Experiment Execution Workflow

```Plain Text


# 1. View experiment entry guide (MUST READ)
cat scripts/nutriplan_experiments/README_FIRST.md

# 2. Run RQ1 experiments (task performance evaluation)
bash scripts/nutriplan_experiments/run_rq1_experiments.sh

# 3. Run RQ2 baseline experiments (comparison of different models/methods)
bash scripts/nutriplan_experiments/run_rq2_experiments.sh

# 4. Execute complete evaluation pipeline (calculate metrics, generate reports, visualization)
bash scripts/nutriplan_experiments/scripts/full_evaluation_pipeline.sh

# 5. Generate dataset statistical report
python scripts/nutriplan_experiments/data/data_statistics.py \
  --benchmark_dir outputs/benchmark \
  --output_report outputs/stats_reports/benchmark_stats_report.md

```

### 4.4.3 Experiment Completeness Verification

Execute the following commands to verify whether the experiments are completely completed and avoid missing steps:

```Plain Text


# Check the number of completed RQ1 experiments (expected 15, covering 3 tasks × 5 metrics)
echo "Number of completed RQ1 experiments:"
find outputs/experiments/rq1 -path "*/training_complete.txt" | wc -l

# Check the number of RQ1 evaluation results (expected to match the number of experiments)
echo "Number of RQ1 evaluation results:"
find outputs/experiments/rq1 -path "*/eval/aggregate_metrics.json" | wc -l

# Check the number of RQ2 baseline experiments (expected 4, covering different base models)
echo "Number of RQ2 baseline evaluation results:"
ls outputs/experiments/rq2/*/eval/aggregate_metrics.json 2>/dev/null | wc -l

# Check key result files (statistical reports, visualization charts)
echo "Whether key result files exist:"
ls -lh outputs/stats_reports/benchmark_stats_report.md outputs/experiments/visualization/

```

### 4.4.4 Explanation of Experiment Output Files

|Directory|File Type|Explanation|
|---|---|---|
|outputs/experiments/rq1/[task]/[model]/|training_complete.txt|Experiment training completion marker file|
|outputs/experiments/rq1/[task]/[model]/eval/|aggregate_metrics.json|Aggregated evaluation metric results (including metric values, rankings)|
|outputs/experiments/rq2/[model]/|baseline_comparison.csv|Baseline model comparison result table|
|outputs/experiments/visualization/|*.png/*.html|Experimental result visualization charts (metric comparison charts, distribution histograms, etc.)|
|outputs/stats_reports/|benchmark_stats_report.md|Comprehensive statistical report of benchmark datasets|
# 5. Explanation of Core Metrics

## 5.1 Data Processing Metrics

|Metric Name|Calculation Logic|Reference Standard|
|---|---|---|
|map_rate_unique_norm_% (Ingredient Matching Rate)|(Number of successfully matched standardized ingredients / Total number of standardized ingredients) × 100%|≥ 85% (Adjust alignment parameters if lower)|
|n_recipes_using_default_grams (Number of Recipes Using Default Grams)|Number of recipes using default grams (100g) for nutritional calculation|≤ 10% (Lower value indicates more complete ingredient weight information)|
|nutrition_coverage_% (Nutrient Coverage Rate)|(Number of nutrients covered in recipes / Total number of nutrients) × 100%|≥ 90% (Ensure completeness of nutritional data)|
## 5.2 Experiment Evaluation Metrics

|Task|Core Metrics|Calculation Logic|
|---|---|---|
|Task A (Discriminative Ranking)|NDCG@k, Precision@k, Recall@k|Calculate the relevance and accuracy of model ranking results based on real ranking labels (k=5,10,20)|
|Task B (Constrained Generation)|Nutritional Constraint Satisfaction Rate, Ingredient Complementarity Score, Recipe Rationality Score|Constraint Satisfaction Rate: Proportion of recipes that meet all nutritional/preference constraints; Complementarity Score: Calculate ingredient synergy score based on rule base; Rationality: Manual/automatic evaluation of recipe step coherence and ingredient adaptability|
|Task C (Reflective Editing)|Violation Correction Success Rate, Nutritional Value Error Rate, Ingredient Modification Rationality|Correction Success Rate: Proportion of recipes with complete correction of violation types; Error Rate: Deviation between corrected nutritional value and ideal value; Rationality: Evaluate the scientificity and feasibility of ingredient replacement|
## 5.3 Nutritional Complementarity Score

Based on authoritative nutritional research, the synergy score (0-1 range) of ingredient complementarity pairs is defined. The core rules are as follows:

|Complementarity Pair (Ingredient/Nutrient)|Synergy Score|Scientific Principle|
|---|---|---|
|High-iron ingredients + High-vitamin C ingredients|1.0|Vitamin C can convert ferric iron|
