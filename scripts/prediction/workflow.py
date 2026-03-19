from queue import Queue
import time
import os
import sys
import torch

# Constant for main directory (must be before imports)
MAIN_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))

# Add main directory to sys.path for imports
sys.path.insert(0, MAIN_DIR)

# Import source code of each architecture modules
from helpers.module import Module
from modules.extraction import Extractor
from modules.selection import Selector
from modules.captioning import Captioner
from modules.detection import Detector
from modules.notification import Notifier

# Create logger
from helpers.logger import getLogger
logger = getLogger("workflow")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def initialize_modules(config: dict) -> list[Module]:
    """Initialize architecture modules"""

    # Set logging level
    logger.setLevel(config.get("log", "INFO").upper())

    # Initialize queues
    frames_database = Queue()
    selected_frames_database = Queue()
    captions_database = Queue()
    responses_database = Queue()

    # Initialize modules
    extractor = Extractor(frames_database, **config['extractor'])
    selector = Selector(frames_database, selected_frames_database, **config["selector"])
    captioner = Captioner(selected_frames_database, captions_database, **config["captioner"])
    detector = Detector(captions_database, responses_database, **config["detector"])
    notifier = Notifier(responses_database, **config["notifier"])

    logger.info("Architecture loaded")
    return (extractor, selector, captioner, detector, notifier)


def workflow(
        extractor: Extractor,
        selector: Selector,
        captioner: Captioner,
        detector: Detector,
        notifier: Notifier,
        config: dict
    ) -> bool:

    # Start modules in dependency-friendly order:
    # - Warm up Captioner first (blocking warmup avoids losing initial frames)
    # - Bring up downstream modules
    # - Start Extractor last so frames flow only when everyone is ready
    captioner.start(**config["captioner"])
    detector.start(**config["detector"])
    notifier.start(**config["notifier"])
    selector.start(**config["selector"])
    extractor.start(**config["extractor"]) 
    logger.info("Modules restart, workflow starts")

    # Keep track of the processing time (excludes captioner warmup)
    start = time.time()

    # Attend the end of each module
    try:
        extractor.wait()
        logger.info(f"{type(extractor).__name__} finished")
        for module in (selector, captioner, detector, notifier):
            module.queue_in_end()
            module.wait()
            logger.info(f"{type(module).__name__} finished")

    except KeyboardInterrupt as err: # If user hits Ctrl+C stop all
        logger.warning("Keyboard interrupt. Exiting...")
        torch.cuda.empty_cache()
        raise err

    # Clean cuda cache
    torch.cuda.empty_cache()

    # Print results
    result = notifier.result
    total_time = time.time() - start
    # Try to read total frames from extractor and selected/captioned counts
    try:
        total_frames = getattr(extractor, "video_total_frames", None)
        # Prefer explicit selector counter, fallback to length of times_out
        selected_frames = getattr(selector, "frames_selected_count", None)
        if selected_frames is None:
            try:
                selected_frames = len(getattr(selector, "times_out", {}))
            except Exception:
                selected_frames = None
        captioned_frames = getattr(captioner, "frames_captioned_count", None)
        if total_frames:
            if captioned_frames is not None:
                logger.info(f"Time: {total_time} | Frames: {selected_frames}/{total_frames} | Captioned: {captioned_frames}")
            else:
                logger.info(f"Time: {total_time} | Frames: {selected_frames}/{total_frames}")
        elif selected_frames is not None:
            if captioned_frames is not None:
                logger.info(f"Time: {total_time} | Frames: {selected_frames} | Captioned: {captioned_frames}")
            else:
                logger.info(f"Time: {total_time} | Frames: {selected_frames}")
        else:
            logger.info(f"Time: {total_time}")
    except Exception:
        logger.info(f"Time: {total_time}")
    try:
        warmup = float(getattr(captioner, "warmup_seconds", 0.0) or 0.0)
    except Exception:
        warmup = 0.0
    logger.info(f"Warmup: {warmup}")
    logger.info(f"TotalTime: {total_time + warmup}")
    logger.info(f"Result: {result}")

    # times = {}
    # time0 = selector.times_out[0]
    # for d in (selector.times_out, captioner.times_in, captioner.times_out, detector.times_in, detector.times_out):
    #     for k, v in d.items():
    #         if k in times:
    #             times[k].append(v - time0)
    #         else:
    #             times[k] = [v - time0]

    # for k, v in times.items():
    #     print(k, v)

    # sum1 = 0
    # sum2 = 0
    # for k, v in times.items():
    #     sum1 += v[2]-v[1]
    #     sum2 += v[4]-v[3]
    # print(sum1/len(times))
    # print(sum2/len(times))

    return result


def read_config(config_file: str = None) -> dict:
    import yaml

    if config_file is None:
        config_file = "config.yml"
    
    config_path = os.path.join(MAIN_DIR, "config", config_file)

    with open(config_path) as f:
        config = yaml.full_load(f)

    return config


if __name__ == "__main__":

    # Read config file
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    config = read_config(config_file)

    # Execute workflow with config
    modules = initialize_modules(config)
    workflow(*modules, config)
