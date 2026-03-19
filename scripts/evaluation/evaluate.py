import sys
import os
from datetime import datetime
import csv
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import shutil
import copy
try:
    import yaml
except Exception:
    yaml = None

# Add parent directory to sys.path for imports
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.prediction.workflow import MAIN_DIR, read_config, initialize_modules, workflow

# Defines constants
RESULTS_DIR = "/home/gmonni54/agenticvad/results"


def _is_video_file(name: str) -> bool:
    name_lower = (name or "").lower()
    if name_lower.startswith("."):
        return False
    valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v")
    return name_lower.endswith(valid_exts)


def _sanitize_field(text: str, max_len: int = 4096) -> str:
    if text is None:
        return ""
    s = str(text)
    # Avoid breaking the pipe-delimited CSV
    s = s.replace("\n", "   ").replace("\r", " ").replace("|", "¦")
    # Trim overly long outputs
    if len(s) > max_len:
        s = s[:max_len] + "…"
    return s


def _get_category_from_filename(name: str) -> str:
    """Extract category from a video file name.
    
    Supports two dataset formats:
    
    UCF-Crime:
      - "Arson009_x264.mp4" -> "Arson"
      - "Normal_Videos_003_x264.mp4" -> "Normal_Videos"
      - "RoadAccidents121_x264.mp4" -> "RoadAccidents"
    
    XD-Violence (categorized by label type):
      - "Rush.Hour.3.2007.BluRay__#01-00-40_01-04-00_label_A.mp4" -> "label_A"
      - "'v=QiLNvC7CIuY__#1_label_B1-0-0.mp4'" -> "label_B1"
      - "Salt.2010__#00-02-10_00-03-57_label_G-0-0.mp4" -> "label_G"
    """
    base_name = os.path.basename(name or "")
    stem = os.path.splitext(base_name)[0]
    
    # XD-Violence format: check for label_ pattern
    if "label_" in stem:
        # Extract label type (A, B1, B2, B4, B5, B6, G, etc.)
        label_parts = stem.split("label_")
        if len(label_parts) > 1:
            # Get label suffix (e.g., "A", "B1-0-0" -> "B1", "G-0-0" -> "G")
            label_suffix = label_parts[1].split("-")[0]
            return f"label_{label_suffix}"
        
        return "XD_Violence_Unknown"
    
    # UCF-Crime format: extract prefix before first digit
    prefix_chars = []
    for ch in stem:
        if ch.isdigit():
            break
        prefix_chars.append(ch)
    prefix = "".join(prefix_chars).rstrip("_")
    return prefix or "UCF_Unknown"


def run_experiment(config):
    """Function for run workflow with more videos and computed metrics"""

    # Check if have to recover an experiment
    recover_experiment = None
    for arg in sys.argv:
        if arg.startswith("recover="):
            recover_experiment = arg.removeprefix("recover=")

    # Set experiment variable
    experiment_timestamp: str = recover_experiment if recover_experiment else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(RESULTS_DIR, f"experiment_{experiment_timestamp}")
    videos_dir: str = config["evaluate"]["videos_dir"]
    os.makedirs(experiment_dir, exist_ok=True)

    # Save a snapshot of the configuration used for the experiment
    config_snapshot = copy.deepcopy(config)
    params_path = os.path.join(experiment_dir, "params.yml")
    try:
        if yaml is not None:
            with open(params_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_snapshot, f, sort_keys=False, allow_unicode=True)
        else:
            import json
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(config_snapshot, f, ensure_ascii=False, indent=2)
    except Exception:
        # If writing params fails, continue the experiment without blocking
        pass
    y_true = []
    y_score = []
    categories = []
    modules = initialize_modules(config)

    # Access notifier to read description after each run
    notifier = modules[-1]

    # Reading predictions csv (store inside experiment dir to avoid files in results root)
    predictions_path = os.path.join(experiment_dir, "scores.csv")
    if recover_experiment:
        with open(predictions_path, "r", newline="", errors="ignore", encoding="utf-8") as predictions_file:
            for i, line in enumerate(predictions_file.readlines()):
                if i == 0 or not line.strip(): continue
                fields = line.split("|")
                y_true.append(int(fields[2]))
                y_score.append(int(fields[3]))
                try:
                    prev_video_path = fields[1]
                    categories.append(_get_category_from_filename(os.path.basename(prev_video_path)))
                except Exception:
                    categories.append("UNKNOWN")
    
    # Write header
    else:
        with open(predictions_path, "w", newline="", errors="ignore", encoding="utf-8") as predictions_file:
            prediction_csv = csv.writer(predictions_file, delimiter="|")
            prediction_csv.writerow(("id", "video_path", "ground_true", "label_prediction", "description"))
        

    # Get videos not read from prediction csv
    all_entries = sorted(os.listdir(videos_dir))
    videos = [v for v in all_entries if _is_video_file(v)][len(y_true):]

    # Writing prediction csv
    with open(predictions_path, "a", newline="", errors="ignore", encoding="utf-8") as predictions_file:
        prediction_csv = csv.writer(predictions_file, delimiter="|")

        # Loop through videos
        for video_index, video_name in enumerate(videos, len(y_true)):

            # Create dir for each video
            video_dir = os.path.join(experiment_dir, video_name)
            os.makedirs(video_dir, exist_ok=True)

            # Change configuration for the video
            full_video_path = os.path.join(videos_dir, video_name)
            config["extractor"]["video_url"] = full_video_path
            config["captioner"]["save_file"] = os.path.join(video_dir, "captioner.txt")
            config["detector"]["save_file"] = os.path.join(video_dir, "detector.txt")
            
            # Obtain labels and run models
            ground_true = 0 if config["evaluate"]["normal_video_indicator"] in video_name else 1
            prediction = 1 if workflow(*modules, config) else 0

            # Save prediction and ground truth
            y_true.append(ground_true)
            y_score.append(prediction)
            categories.append(_get_category_from_filename(video_name))

            # Read concise description from notifier only when anomalous
            description = notifier.description if prediction == 1 else ""
            description = _sanitize_field(description)

            # Write video result on csv
            prediction_csv.writerow((
                video_index,
                _sanitize_field(full_video_path, max_len=2048),
                ground_true,
                prediction,
                description,
            ))

    # Calculate metrics (overall)
    # Some metrics require both classes to be present and a non-constant score
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"AUC: {auc:.4f}")
    except Exception:
        auc = 0.0
        print("AUC: N/A (insufficient class variation)")

    try:
        ap = average_precision_score(y_true, y_score)
        print(f"Average Precision: {ap:.4f}")
    except Exception:
        ap = 0.0
        print("Average Precision: N/A (insufficient class variation)")

    # Calculate and display confusion matrix (overall)
    result = confusion_matrix(y_true, y_score, labels=[0, 1])
    (tn, fp), (fn, tp) = result
    print("\nConfusion Matrix:")
    print("┌─────────────┬─────────────┬─────────────┐")
    print("│             │   Predicted │             │")
    print("│             │  Normal     │  Anomalous  │")
    print("├─────────────┼─────────────┼─────────────┤")
    print(f"│   Actual    │             │             │")
    print(f"│   Normal    │     {tn}      │     {fp}      │")
    print(f"│   Anomalous │     {fn}      │     {tp}      │")
    print("└─────────────┴─────────────┴─────────────┘")

    print(f"\nDetailed metrics:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    print(f"Accuracy: {(tn + tp) / len(y_true):.4f}")
    print(f"Precision: {tp / (tp + fp) if (tp + fp) > 0 else 0:.4f}")
    print(f"Recall: {tp / (tp + fn) if (tp + fn) > 0 else 0:.4f}")
    print(f"F1-Score: {2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.4f}")
    # Build per-category metrics and write metrics.csv like previous experiment format
    metrics_path = os.path.join(experiment_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", errors="ignore", encoding="utf-8") as metrics_file:
        csv_writer = csv.writer(metrics_file, delimiter="|")
        # Header matching the example
        csv_writer.writerow((
            "subset",
            "auc",
            "average_precision",
            "tn",
            "fp",
            "fn",
            "tp",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ))

        # Overall row
        overall_accuracy = (tn + tp) / len(y_true) if len(y_true) > 0 else 0.0
        overall_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        overall_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        overall_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        csv_writer.writerow((
            "overall",
            f"{auc:.3f}" if isinstance(auc, float) else "",
            f"{ap}" if isinstance(ap, float) else "",
            tn, fp, fn, tp,
            overall_accuracy,
            overall_precision,
            overall_recall,
            overall_f1,
        ))

        # Per-category rows
        unique_categories = sorted(set(categories))
        for cat in unique_categories:
            idxs = [i for i, c in enumerate(categories) if c == cat]
            if not idxs:
                continue
            y_true_c = [y_true[i] for i in idxs]
            y_score_c = [y_score[i] for i in idxs]

            try:
                cm_c = confusion_matrix(y_true_c, y_score_c, labels=[0, 1])
                (tn_c, fp_c), (fn_c, tp_c) = cm_c
            except Exception:
                tn_c = fp_c = fn_c = tp_c = 0

            try:
                auc_c = roc_auc_score(y_true_c, y_score_c)
            except Exception:
                auc_c = None
            try:
                ap_c = average_precision_score(y_true_c, y_score_c)
            except Exception:
                ap_c = None

            n_c = len(y_true_c)
            acc_c = (tn_c + tp_c) / n_c if n_c > 0 else 0.0
            prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * tp_c / (2 * tp_c + fp_c + fn_c) if (2 * tp_c + fp_c + fn_c) > 0 else 0.0

            csv_writer.writerow((
                cat,
                f"{auc_c:.3f}" if isinstance(auc_c, float) else "",
                f"{ap_c}" if isinstance(ap_c, float) else "",
                tn_c, fp_c, fn_c, tp_c,
                acc_c, prec_c, rec_c, f1_c,
            ))

    # Auto-delete all per-video folders inside the experiment directory
    try:
        for entry in os.listdir(experiment_dir):
            entry_path = os.path.join(experiment_dir, entry)
            if os.path.isdir(entry_path) and _is_video_file(entry):
                shutil.rmtree(entry_path, ignore_errors=True)
    except Exception:
        pass

    # Return evaluation results
    return {
        'confusion_matrix': result,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'auc': auc,
        'average_precision': ap,
        'accuracy': (tn + tp) / len(y_true),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'metrics_path': metrics_path,
        'scores_path': predictions_path,
    }



if __name__ == "__main__":

    # Read config file
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    config = read_config(config_file)

    # Execute workflow with config
    results = run_experiment(config)

    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"Results are also saved to: {results.get('metrics_path')}")
