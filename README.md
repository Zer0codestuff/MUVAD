<div align="center">

# MUVAD: End-to-End Explainable Video Anomaly Detection with Multi-modal Large Language Models

</div>


> **Abstract:** *In intelligent video surveillance, the ability to detect anomalous events is only part of the challenge: systems must also provide clear and reliable explanations for their decisions in order to be trusted and effectively deployed. Existing approaches to explainable video anomaly detection typically rely on fragmented pipelines that separate visual understanding from semantic reasoning and prediction, often using intermediate captioning or multiple specialized models. Such multi-stage designs can introduce information loss, increased latency, and limited coherence between predictions and explanations. This thesis proposes a novel end-to-end framework for explainable video anomaly detection based on a single multimodal large language model that jointly performs visual understanding, reasoning, and anomaly prediction. Instead of decoupling perception and decision-making, the proposed approach processes raw video inputs directly and produces both anomaly scores and natural language explanations within a unified inference stage. Experiments on benchmarks such as UCF-Crime and XD-Violence demonstrate competitive detection performance compared to multi-stage baselines, achieving up to 83.17% AUC on UCF-Crime using a compact 2B model that requires 82% less memory than prior two-stage approaches. An interactive web-based interface is developed to allow users to process videos through the pipeline, visualize detection results, and inspect model-generated explanations. The results show that the single-model paradigm reduces architectural complexity while maintaining real-time processing capability, making it a reliable solution.*

---

## Table of Contents

- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Clone](#clone)
  - [Conda](#conda)
  - [Project Structure](#project-structure)
- [Data](#data)
  - [Dataset Structure](#dataset-structure)
  - [Video Labeling Convention](#video-labeling-convention)
- [Models](#models)
  - [Supported Models](#supported-models)
  - [Model Download](#model-download)
- [Inference](#inference)
  - [Single Video Prediction](#single-video-prediction)
  - [Pipeline Configurations](#pipeline-configurations)
    - [1. Single-Stage Pipeline (VLM only)](#1-single-stage-pipeline-vlm-only)
    - [2. Two-Stage Pipeline (VLM + LLM)](#2-two-stage-pipeline-vlm--llm)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
  - [Generate Charts](#generate-charts)
- [Reproducibility](#reproducibility)
  - [Effectiveness Results](#effectiveness-results)
  - [Timing Results](#timing-results)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

# Setup

We recommend the use of a Linux machine with CUDA compatible GPUs. We used 2x NVIDIA A6000 GPUs with 64GB. We provide Conda environments to configure the required libraries.

## Prerequisites

**Required:**
- Python 3.10+
- Conda
- Git

## Clone
Clone the repo with:

```bash
git clone https://github.com/tail-unica/muvad
cd muvad
```

## Conda

The framework uses **multiple isolated Conda environments** to prevent package conflicts between different model backends. Each environment is tailored for specific VLM/LLM combinations.

Install all environments with:

```bash
./setup/setup.sh
```

### Available Environments

The setup script creates the following environments:

| Environment | Purpose | Models | Backend |
|-------------|---------|--------|---------|
| `llamacpp` | Single-stage inference with llamacpp models | InternVL3.5 (2B, 4B, 8B, 30B) | llama-server |
| `florence` | Two-stage pipeline with Florence2 or BLIP2 | Florence2, BLIP2 | transformers + ollama |
| `deepseek` | Two-stage pipeline with DeepSeekVL2 | DeepSeekVL2 | transformers + ollama |

**Why multiple environments?**
- Different VLM backends have conflicting dependencies (transformers vs llama-cpp-python)
- Isolated environments ensure reproducibility and prevent library version conflicts
- You activate the appropriate environment based on which pipeline configuration you want to run
- Each environment is independent and can coexist without issues

---

## Project Structure

The framework is organized into modular components for flexibility and maintainability:

```
muvad/
├── config/              # Configuration files for different models and experiments
│   ├── config_IVL_*.yml        # InternVL3.5 single-stage pipeline configs
│   ├── config_bl_ll_lat.yml    # BLIP2 + llama3.2 two-stage pipeline
│   ├── config_ds_qw_def.yml    # DeepSeekVL2 + qwen2.5 two-stage pipeline
│   └── config_fl_ll_def.yml    # Florence2 + llama3.2 two-stage pipeline
│
├── modules/             # Core pipeline components
│   ├── extraction.py           # Video frame extraction
│   ├── selection.py            # Frame sampling and selection
│   ├── captioning.py           # Frame description generation (VLM)
│   ├── detection.py            # Anomaly detection (LLM, optional)
│   └── notification.py         # Decision logic and alerting
│
├── models/              # VLM model implementations
│   ├── florence2.py            # Microsoft Florence2 integration
│   ├── deepseekvl2.py          # DeepSeek VL2 integration
│   └── blip2.py                # Salesforce BLIP2 integration
│
├── helpers/             # Utility functions
│   ├── llamacpp_wrap.py        # llama-server wrapper and management
│   ├── ollama_wrap.py          # Ollama server wrapper and management
│   ├── logger.py               # Logging utilities
│   ├── module.py               # Base module class
│   └── structs.py              # Data structures
│
├── scripts/             # Execution scripts
│   ├── prediction/             # Single video inference
│   │   └── workflow.py         # Main prediction pipeline
│   └── evaluation/             # Dataset evaluation
│       └── evaluate.py         # Batch evaluation with metrics
│
├── setup/               # Environment setup
│   ├── setup.sh                # Main setup script
│   ├── environment-llamacpp.yml
│   ├── environment-florence.yml
│   └── environment-deepseek.yml
│
├── data/                # Datasets (user-provided)
├── results/             # Experiment outputs and metrics
├── demos/               # Demo scripts and examples
└── assets/              # Additional resources (prompts, schemas)
```

### Pipeline Architecture

The framework implements a modular pipeline with the following stages:

1. **Extractor** (`modules/extraction.py`)
   - Reads video files and extracts frames
   - Supports configurable frame resize and optional frame saving
   - Implements timeout mechanisms for real-time simulation

2. **Selector** (`modules/selection.py`)
   - Reduces frame rate by sampling frames in batches
   - Configurable batch size for optimal processing
   - Passes selected frames to the captioning stage

3. **Captioner** (`modules/captioning.py`)
   - Generates textual descriptions of frames using VLMs
   - Supports multiple backends: `llamacpp`, `transformers`, `ollama`
   - Can aggregate multiple frames in a single prompt
   - Returns JSON responses with anomaly scores and descriptions

4. **Detector** (`modules/detection.py`) - *Optional*
   - Aggregates frame captions into a unified context
   - Uses an LLM to analyze the sequence and detect anomalies
   - Only used in two-stage pipelines (VLM + LLM)

5. **Notifier** (`modules/notification.py`)
   - Analyzes JSON responses to make final decisions
   - Implements decision logic: `moving_average` or `consecutive`
   - Produces boolean output (anomaly detected: True/False)
   - Extracts descriptions for explainability

### Backend Management

**llama.cpp wrapper** (`helpers/llamacpp_wrap.py`):
- Automatically starts and manages `llama-server` instances
- Downloads models from Hugging Face using the `-hf` flag
- Handles warmup, retries, and error recovery
- Configures context length, batch size, and GPU layers

**Ollama wrapper** (`helpers/ollama_wrap.py`):
- Manages local Ollama server instances
- Creates custom model configurations with specific parameters
- Implements retry logic and error handling

---

# Data

Please download the data, including captions, temporal summaries, indexes with their textual embeddings, and scores for the UCF-Crime and XD-Violence datasets, from the links below:

| Dataset     | Link                                                                                               |
| ----------- | -------------------------------------------------------------------------------------------------- |
| UCF-Crime   | [Google Drive](https://drive.google.com/file/d/1_7juCgOoWjQruyH3S8_FBqajuRaORmnV/view?usp=sharing) |
| XD-Violence | [Google Drive](https://drive.google.com/file/d/1yzDP1lVwPlA_BS2N5Byr1PcaazBklfkI/view?usp=sharing) |

and place them in the `/data` folder. You can download the videos from the official websites ([UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) and [XD-Violence](https://roc-ng.github.io/XD-Violence/)). Please note that you need to change the paths in all the config files.



### Dataset Structure

**Expected directory structure:**

```
/muvad/data/
├── ucf_crime/
│   ├── annotations/
│   ├── captions/
│   ├── filenames/
│   ├── frames/
│   ├── index/
│   ├── scores/
│   ├── similarity/
│   ├── unzipped/
│   └── videos/           ← video files for evaluation
│
├── xd_violence/
│   ├── annotations/
│   ├── captions/
│   ├── extra_caption/
│   ├── filenames/
│   ├── index/
│   ├── similarity/
│   └── videos/           ← video files for evaluation
│
└── monserrato/
    ├── realworld_samples/
    └── simulated_samples/
```

### Video Labeling Convention

The datasets use filename-based conventions to distinguish normal from anomalous videos:

**UCF-Crime Dataset:**
- **Normal videos**: Filenames containing `Normal_Videos` (e.g., `Normal_Videos_001_x264.mp4`)
- **Anomalous videos**: All other videos (e.g., `Explosion017_x264.mp4`, `Robbery045_x264.mp4`)

**XD-Violence Dataset:**
- **Normal videos**: Filenames containing `label_A` (e.g., `label_A_002.mp4`)
- **Anomalous videos**: All other videos with different label prefixes

**Example filenames:**

```
UCF-Crime:
  Normal_Videos_001_x264.mp4       → Normal (label 0)
  Normal_Videos_950_x264.mp4       → Normal (label 0)
  Explosion017_x264.mp4            → Anomaly (label 1)
  Robbery045_x264.mp4              → Anomaly (label 1)

XD-Violence:
  label_A_002.mp4                  → Normal (label 0)
  label_B_034.mp4                  → Anomaly (label 1)
  label_C_078.mp4                  → Anomaly (label 1)
```

This naming convention is used by the evaluation script to automatically assign ground truth labels (0 = normal, 1 = anomaly) based on the `normal_video_indicator` parameter in the configuration file.

## Models

### Supported Models

The framework supports three types of models:

**Llamacpp models** (via llama-server):
- InternVL3.5-2B (Q8_0)
- InternVL3.5-4B (Q8_0)
- InternVL3.5-8B (Q8_0)
- InternVL3.5-30B-A3B (Q4_K_M)

**Transformers models** (direct Python integration):
- Florence2
- DeepSeekVL2
- Blip2

**Ollama models**:
- Qwen2.5:3b
- llama3.2:3b

### Model Download

It is recommended to download models before running the pipeline to avoid delays during execution.

**Llamacpp models:**

Llamacpp models are automatically downloaded when starting llama-server with the `-hf` flag:

```bash
# InternVL3.5 variants (recommended for vision tasks)
llama-server -hf lmstudio-community/InternVL3_5-2B-GGUF:Q8_0
llama-server -hf lmstudio-community/InternVL3_5-4B-GGUF:Q8_0
llama-server -hf lmstudio-community/InternVL3_5-8B-GGUF:Q8_0
llama-server -hf lmstudio-community/InternVL3_5-30B-A3B-GGUF:Q4_K_M
```

The models are cached in `~/.cache/llama.cpp/` for future use.

**Ollama models:**

Install Ollama from [ollama.com](https://ollama.com/), then pull the desired models:

```bash
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

**Transformers models:**

These models are downloaded automatically on first use via the Hugging Face transformers library. They are cached in `~/.cache/huggingface/hub/`.

---

## Inference

### Single Video Prediction

To perform inference on a single video, you need to run the workflow script with a configuration file. The configuration file contains all parameters including the video path, model settings, and output directories.

**Command:**

```bash
python scripts/prediction/workflow.py <config_filename>
```

where `<config_filename>` is the name of a YAML configuration file in the `config/` directory (e.g., `config.yml`, `config_IVL_2B.yml`, etc.).

**Example:**

```bash
python scripts/prediction/workflow.py config.yml
```

**Configuration File Structure:**

All parameters are specified in the YAML configuration file. Key sections include:

| Section | Parameter | Description | Example |
|---------|-----------|-------------|---------|
| `extractor` | `video_url` | Path to the input video file | `/path/to/video.mp4` |
| | `resize` | Frame resize dimensions [width, height] | `[224, 224]` |
| | `save_dir` | Directory to save extracted frames (optional) | `""` (empty = no save) |
| `selector` | `batch_size` | Number of frames to select per batch | `15` |
| `captioner` | `model_name` | VLM model name | `lmstudio-community/InternVL3_5-2B-GGUF:Q4_K_M` |
| | `backend` | Backend type | `llamacpp` or `ollama` |
| | `host` | Backend server address | `localhost:8080` |
| | `batch_size` | Batch size for captioning | `6` |
| | `aggregate` | Aggregate multiple frames in one prompt | `true` or `false` |
| `detector` | `model_name` | LLM model for detection (optional) | `null` (skip detector) or model name |
| `notifier` | `threshold` | Anomaly score threshold | `0.5` |
| | `decision_mode` | Decision logic | `moving_average` or `consecutive` |

**Output:**

The workflow logs the following information to the console:
- **Processing time**: Time taken to process the video (excluding warmup)
- **Frame statistics**: Number of frames processed, selected, and captioned
- **Warmup time**: Time taken to initialize the model
- **Final result**: Boolean value (`True` if anomaly detected, `False` otherwise)

If `save_file` is specified in the captioner/detector configuration, the intermediate responses (containing JSON with anomaly scores and descriptions) are saved to that file.

The workflow function returns a boolean indicating whether an anomaly was detected.

### Pipeline Configurations

The framework supports two types of pipeline architectures:

#### 1. Single-Stage Pipeline (VLM only)

Uses a Vision-Language Model directly to analyze frames and generate anomaly scores in a single step.

**Backend:** `llamacpp` (via llama-server)

**Environment:** `llamacpp`

**Models:** InternVL3.5 variants (2B, 4B, 8B, 30B)

**Configuration files:**
- `config_IVL_2B.yml` - InternVL3.5-2B (Q8_0)
- `config_IVL_4B.yml` - InternVL3.5-4B (Q8_0)
- `config_IVL_8B.yml` - InternVL3.5-8B (Q8_0)
- `config_IVL_30B.yml` - InternVL3.5-30B (Q4_K_M)

**Prerequisites:**

1. Activate the llamacpp environment:
```bash
conda activate llamacpp
```

2. Run the workflow:
```bash
python scripts/prediction/workflow.py config_IVL_2B.yml
```

#### 2. Two-Stage Pipeline (VLM + LLM)

Uses a Vision-Language Model to caption frames (stage 1), then an LLM to analyze the captions and detect anomalies (stage 2).

**Backend:** `transformers` (VLM) + `ollama` (LLM)

**Available configurations:**

| Config File | VLM (Captioner) | LLM (Detector) | Environment |
|-------------|-----------------|----------------|-------------|
| `config_bl_ll_lat.yml` | BLIP2 (Salesforce/blip2-flan-t5-xl) | llama3.2:3b | `florence` |
| `config_ds_qw_def.yml` | DeepSeekVL2 (deepseek-ai/deepseek-vl2-tiny) | qwen2.5:3b | `deepseek` |
| `config_fl_ll_def.yml` | Florence2 (microsoft/Florence-2-large) | llama3.2:3b | `florence` |

**Prerequisites:**

1. Activate the appropriate conda environment based on the VLM model (`florence` for Florence2 and BLIP2, `deepseek` for DeepSeekVL2):

**For BLIP2:**
```bash
conda activate florence
python scripts/prediction/workflow.py config_bl_ll_lat.yml
```

**For DeepSeekVL2:**
```bash
conda activate deepseek
python scripts/prediction/workflow.py config_ds_qw_def.yml
```

**For Florence2:**
```bash
conda activate florence
python scripts/prediction/workflow.py config_fl_ll_def.yml
```

**Note:** The VLM models (BLIP2, DeepSeekVL2, Florence2) are automatically downloaded from Hugging Face on first use and cached in `~/.cache/huggingface/hub/`.

## Evaluation

To evaluate the model on multiple videos from a dataset, use the evaluation script. This script runs the workflow on all videos in a directory and computes comprehensive metrics.

**Command:**

```bash
python scripts/evaluation/evaluate.py <config_filename>
```

where `<config_filename>` is the name of a YAML configuration file in the `config/` directory (optional, defaults to `config.yml`).

**Example:**

```bash
python scripts/evaluation/evaluate.py config.yml
```

**Configuration Requirements:**

The configuration file must include an `evaluate` section specifying:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `videos_dir` | Directory containing video files to evaluate | `/path/to/videos/` |
| `normal_video_indicator` | String to identify normal videos in filenames | `"Normal_Videos"` for ucf-crimes or `"label_A"` for xd-violence |

Videos containing the `normal_video_indicator` in their filename are labeled as normal (ground truth = 0), all others as anomalous (ground truth = 1).

**Output:**
The script creates a timestamped experiment directory (`results/experiment_<timestamp>/`) containing:

1. **`scores.csv`**: Pipe-delimited file with predictions for each video
   - Columns: `id`, `video_path`, `ground_true`, `label_prediction`, `description`
   
2. **`metrics.csv`**: Pipe-delimited file with computed metrics
   - Overall metrics (first row)
   - Per-category metrics (subsequent rows, grouped by video filename prefix)
   - Columns: `subset`, `auc`, `average_precision`, `tn`, `fp`, `fn`, `tp`, `accuracy`, `precision`, `recall`, `f1_score`

3. **`params.yml`**: Snapshot of the configuration used for the experiment

The script also prints a confusion matrix and detailed metrics to the console during execution.

---

## Visualization

After running an evaluation experiment, you can generate charts to visualize the results using the `visualize_metrics.py` script.

### Generate Charts

**Command:**

```bash
python helpers/visualize_metrics.py
```

**How it works:**

1. The script automatically finds all `metrics.csv` files in the `results/` directory
2. It displays a list of available experiments (timestamped directories)
3. You can select which experiment results to visualize by entering the corresponding number
4. The script generates various charts and saves them to `results/experiment_<timestamp>/plots/`

**Output Charts:**

The script creates the following visualizations:

1. **accuracy_per_category.png** - Bar chart showing accuracy for each category
2. **recall_f1.png** - Grouped bar chart comparing recall and F1-score per anomaly category (excludes Normal_Videos)
3. **confusion_matrix_anomalous.png** - True Positives vs False Negatives for anomalous categories (excludes Normal_Videos)
4. **confusion_matrix_normal.png** - True Negatives vs False Positives for normal videos category
5. **metrics_heatmap.png** - Heatmap showing all metrics (accuracy, precision, recall, F1-score, TNR, FPR) across categories
6. **sample_distribution.png** - Pie chart showing sample distribution (total samples per category)
7. **overall_metrics.png** - Bar chart with overall model metrics (accuracy, precision, recall, F1-score, AUC)
8. **f1_score_sorted.png** - Horizontal bar chart showing F1-scores sorted in ascending order (excludes Normal_Videos)
9. **normal_videos_metrics.png** - Specialized chart for normal videos showing TNR (Specificity), FPR (False Alarm Rate), and accuracy

**Example:**

```bash
# Run evaluation
python scripts/evaluation/evaluate.py config_IVL_2B.yml

# Wait for evaluation to complete, then generate charts
python helpers/visualize_metrics.py

# Select the experiment number when prompted
# Charts will be saved in results/experiment_<timestamp>/plots/
```

---

## Reproducibility


### Effectiveness Results

It is recommended to use `nohup` or `screen` to avoid session interruption.


#### UCF-CRIME

**Experiment 1: [InternVL3.5-2B]**

```bash
nohup python scripts/evaluation/evaluate.py config_IVL_2B.yml &
```

**Experiment 2: [InternVL3.5-4B]**

```bash
nohup python scripts/evaluation/evaluate.py config_IVL_4B.yml &
```

**Experiment 3: [InternVL3.5-8B]**

```bash
nohup python scripts/evaluation/evaluate.py config_IVL_8B.yml &
```

**Experiment 4: [InternVL3.5-30B]**

```bash
nohup python scripts/evaluation/evaluate.py config_IVL_30B.yml &
```

#### XD-VIOLENCE

To run experiments on XD-Violence dataset, you need to modify the configuration file:

1. Update `videos_dir` in the `evaluate` section to point to your XD-Violence videos directory
2. Change `normal_video_indicator` from `Normal_Videos` to `label_A`

**Example configuration changes:**

```yaml
evaluate:
  videos_dir: /path/to/xd-violence/videos/
  normal_video_indicator: label_A
```

**Experiment 1: [InternVL3.5-2B]**

```bash
nohup python scripts/evaluation/evaluate.py config_IVL_2B.yml &
```

**Experiment 2: [InternVL3.5-4B]**

```bash
nohup python scripts/evaluation/evaluate.py config_IVL_4B.yml &
```

**Experiment 3: [InternVL3.5-8B]**

```bash
nohup python scripts/evaluation/evaluate.py config_IVL_8B.yml &
```

**Experiment 4: [InternVL3.5-30B]**

```bash
nohup python scripts/evaluation/evaluate.py config_IVL_30B.yml &
```


### Timing Results

To measure timing performance on a real-time video stream simulation, you need to modify the configuration file:

1. Update `timeout` in the `extractor` section from `0.01` to `0.033` (simulates 30 FPS video stream)

**Example configuration change:**

```yaml
extractor:
  timeout: 0.033
```

**Experiment 1: [InternVL3.5-2B]**

```bash
python scripts/prediction/workflow.py config_IVL_2B.yml
```

**Experiment 2: [InternVL3.5-4B]**

```bash
python scripts/prediction/workflow.py config_IVL_4B.yml
```

**Experiment 3: [InternVL3.5-8B]**

```bash
python scripts/prediction/workflow.py config_IVL_8B.yml
```

**Experiment 4: [InternVL3.5-30B]**

```bash
python scripts/prediction/workflow.py config_IVL_30B.yml
```

---

## Demo

### CLI Demo: VAD Showcase (script)

You can run the VAD demo from the terminal and automatically generate an annotated video.

Prerequisites:
- Activate the `llamacpp` environment (`./setup/setup.sh` installs it)
- Either start `llama-server` yourself, or let the script start it for you (see below)
- ffmpeg installed is recommended; otherwise the script falls back to OpenCV video writer

Quick start:

```bash
python demos/vad_showcase.py
```

- If no input video is specified with `-i`, the script automatically uses the first video found in `assets/videos_and_frames/videos/`
- You can also pass a video path explicitly: `python demos/vad_showcase.py -i /path/to/video.mp4`

Manual server (recommended to pre-warm the model):

```bash
conda activate llamacpp
llama-server -hf lmstudio-community/InternVL3_5-2B-GGUF:Q8_0 --ctx-size 8192 -ngl 999 --port 1234 &
python demos/vad_showcase.py
```

Autostart server from the script:

```bash
python demos/vad_showcase.py \
  --autostart-server \
  --llama-url http://localhost:1234 \
  --llama-model lmstudio-community/InternVL3_5-2B-GGUF:Q8_0
```

Common options (examples):

```bash
# pass input path explicitly
python demos/vad_showcase.py -i /path/to/video.mp4

# change sampling and window params
python demos/vad_showcase.py -i /path/to/video.mp4 \
  --select-fps 2 --window-size 6 --window-step 6 --threshold 0.5

# change rendering (output fps, size)
python demos/vad_showcase.py -i /path/to/video.mp4 \
  --render-fps 2.5 --resize 1280x720

# override the anomaly prompt inline
python demos/vad_showcase.py -i /path/to/video.mp4 \
  --prompt "You are a safety analyst. Return JSON with anomaly_score and description."

# load the anomaly prompt from a text file
python demos/vad_showcase.py -i /path/to/video.mp4 \
  --prompt-file /path/to/prompt.txt
```

Outputs and workspace:
- Selected frames (≈2 fps by default): `assets/videos_and_frames/frames_selected/`
- Final annotated video (MP4): `assets/videos_and_frames/output/<video>_vad_showcase.mp4`

To keep the pipeline fast, the showcase now selects frames while decoding the video and only saves the selected frames to disk.

Show all flags and defaults:

```bash
python demos/vad_showcase.py -h
```

### Web UI Demo: VAD Showcase (Gradio)

A web-based interface is also available using Gradio, providing an interactive UI for running the VAD showcase pipeline.

Prerequisites:
- Activate the `llamacpp` environment

Quick start:

```bash
python demos/vad_showcase_gradio.py
```

The interface will be available at `http://127.0.0.1:7860` by default.

Common options:

```bash
# change host and port
python demos/vad_showcase_gradio.py --host 0.0.0.0 --port 8080

```

Features:
- Upload videos directly through the web interface
- Configure all analysis parameters via UI controls
- Edit the anomaly prompt directly from the UI
- View pipeline logs in real-time
- Download the generated annotated video

The Gradio interface uses the same underlying pipeline as the CLI script, with outputs saved to `tmp/gradio_runs/run_<timestamp>/`.

### Web UI Demo: VAD Evaluation (Gradio)

The evaluation Gradio app lets you:
- browse previously processed runs
- inspect the selected-frame windows, anomaly scores, and generated explanations
- rate the quality of each run
- process new videos from a configurable video directory
- override the anomaly prompt used for new runs
- upload a video in the evaluation tab and process it directly from there

Quick start:

```bash
python demos/vad_evaluation_gradio.py
```

The interface is available at `http://127.0.0.1:7861` by default.

By default, the "Process Videos" tab scans `data/` recursively for videos. You can override that in two portable ways:

```bash
# override from the command line
python demos/vad_evaluation_gradio.py --data-dir /path/to/videos

# or with environment variables
MUVAD_DATA_DIR=/path/to/videos python demos/vad_evaluation_gradio.py
```

Additional useful overrides:

```bash
python demos/vad_evaluation_gradio.py \
  --data-dir /path/to/videos \
  --runs-dir /path/to/gradio_runs \
  --evaluations-file /path/to/vad_evaluations.json
```

Outputs used by the evaluation UI:
- Processed runs: `tmp/gradio_runs/`
- Ratings JSON: `tmp/vad_evaluations.json`
- Ratings CSV export: `tmp/vad_evaluations.csv`

Typical workflow:
1. Process one or more videos from the "Process Videos" tab.
2. Or upload a video directly in "Evaluate Runs" and start processing from the upload section.
3. Review the anomaly timeline, frame windows, and summary.
4. Save your rating and optionally export all ratings to CSV.


## Contributing

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know.

Please feel free to file issues and pull requests on the repo and we will address them as we can.

---

## License

This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.

---

## Citation

If you use this work in your research, please cite:

```bibtex
# TODO: Add citation information
```

---

## Acknowledgments

<!-- TODO: Add acknowledgments -->
