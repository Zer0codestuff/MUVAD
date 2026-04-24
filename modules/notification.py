from queue import Queue
import json
import re as regex

from helpers.module import Module

# Create logger
from helpers.logger import getLogger
logger = getLogger("notifier")


def _extract_json_object(text: str, preferred_key: str | None = None) -> dict | None:
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for match in regex.finditer(r"\{", text or ""):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            if preferred_key and preferred_key in obj:
                return obj
            candidates.append(obj)
    return candidates[0] if candidates else None


class Notifier(Module):
    def __init__(self,
            responses_database: Queue[str],
            threshold: float,
            result_key: str,
            description_key: str,
            log: str = "INFO",
            **kwargs,
        ):
        """Initialize the notifier, loading the model"""

        # Initialize object variables
        self.threshold = threshold
        self.result_key = result_key
        self.description_key = description_key
        # Decision mode: 'moving_average' (default) or 'consecutive'
        self.decision_mode: str = str(kwargs.get("decision_mode", "moving_average")).strip().lower()
        self.avg_window_size: int = int(kwargs.get("avg_window_size", 4))
        self.consecutive_required: int = int(kwargs.get("consecutive_required", 3))
        logger.setLevel(log.upper())

        # Initialize module
        super().__init__(responses_database, self.notify, None)


    def start(self, **kwargs) -> None:
        """Start the notifier"""

        # Initialize instance variables
        self.result = False
        self._recent_scores: list[float] = []  # sliding window of per-frame scores
        self._consecutive_counter: int = 0
        self.description: str = ""
        # Track most recent description where score > threshold
        self._last_above_threshold_description: str = ""

        # Start module
        super().start()


    def _count_frames_in_response(self, response_text: str) -> int:
        """Count how many frames were processed by the model by inspecting the echoed prompt.
        We expect the detector output to be: <prompt>\n\n\n<model_response>..."""
        try:
            if "FramesCount=" in response_text:
                marker = "FramesCount="
                for line in response_text.splitlines():
                    if marker in line:
                        try:
                            return int(line.split(marker, 1)[1].split()[0])
                        except Exception:
                            continue
            prompt_part, sep, _ = response_text.partition("\n\n\n")
            if not sep:
                # Fallback: count bullet lines in the whole string
                lines = response_text.splitlines()
            else:
                lines = prompt_part.splitlines()
            return sum(1 for ln in lines if ln.strip().startswith("- "))
        except Exception:
            return 1


    def notify(self, batch: list[str]) -> list:
        """Thread function for parse the response and notify if need"""

        if not batch:
            return []

        # Extract each response
        for response in batch:

            # Determine how many frames were included in this detector call
            frames_in_response = max(1, int(self._count_frames_in_response(response)))

            # Search for a JSON object. LLMs often wrap it in markdown or add prose.
            dict_json = _extract_json_object(response, self.result_key)
            if dict_json is None:
                logger.warning("JSON not found")
                continue

            # Get anomaly score from json
            anomaly_score = dict_json.get(self.result_key, None)
            if anomaly_score is None:
                logger.error(f"{self.result_key} key not found in JSON")
                logger.debug(f"{dict_json}")
                continue

            # Get description if available
            description_value = dict_json.get(self.description_key, "")

            # Convert score to float in [0,1]
            numeric_score: float | None
            if isinstance(anomaly_score, bool):
                numeric_score = 1.0 if anomaly_score else 0.0
            elif isinstance(anomaly_score, (float, int)):
                try:
                    numeric_score = float(anomaly_score)
                except Exception:
                    numeric_score = None
            elif isinstance(anomaly_score, str):
                try:
                    numeric_score = float(anomaly_score.strip())
                except Exception:
                    numeric_score = None
            else:
                numeric_score = None

            if numeric_score is None:
                logger.error(f"Type error for '{self.result_key}' key in JSON")
                continue

            # If this response alone exceeds threshold, remember its description
            if numeric_score > float(self.threshold):
                try:
                    self._last_above_threshold_description = str(description_value) if description_value is not None else ""
                except Exception:
                    self._last_above_threshold_description = ""

            # Update state according to decision mode
            if self.decision_mode == "consecutive":
                # Increment by frames_in_response if above threshold, else reset
                if numeric_score > float(self.threshold):
                    self._consecutive_counter += frames_in_response
                else:
                    self._consecutive_counter = 0
                if (not self.result) and (self._consecutive_counter >= self.consecutive_required):
                    self.result = True
                    try:
                        # Prefer the latest description from an above-threshold response
                        self.description = self._last_above_threshold_description or (str(description_value) if description_value is not None else "")
                    except Exception:
                        self.description = ""

            else:  # moving_average (default)
                for _ in range(frames_in_response):
                    self._recent_scores.append(numeric_score)
                    if len(self._recent_scores) > max(1, self.avg_window_size):
                        self._recent_scores.pop(0)
                if (not self.result) and (len(self._recent_scores) >= max(1, self.avg_window_size)):
                    window = self._recent_scores[-max(1, self.avg_window_size):]
                    avg = sum(window) / float(len(window))
                    if avg > float(self.threshold):
                        self.result = True
                        try:
                            # Prefer the latest description from an above-threshold response
                            self.description = self._last_above_threshold_description or (str(description_value) if description_value is not None else "")
                        except Exception:
                            self.description = ""
