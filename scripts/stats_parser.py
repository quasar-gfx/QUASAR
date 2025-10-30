import os
import re
import math
import json
import argparse
from collections import defaultdict

from logger import logger

def normalize_label(label):
    label = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", label)
    label = re.sub(r"\[\w\]", "", label)
    return label.strip()

def get_stats_from_file(file_path):
    data = defaultdict(list)
    pattern = r"(.+?):\s*([\d.]+)\s*(ms|MB|Proxies)"
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    label, value, unit = match.groups()
                    normalized_label = f"{normalize_label(label)} ({unit})"
                    try:
                        data[normalized_label].append(float(value))
                    except ValueError:
                        logger.debug(f"Skipping invalid value in {file_path}: {value}")
        stats = {}
        for label, values in data.items():
            if not values:
                continue
            avg = sum(values) / len(values)
            std = math.sqrt(sum((x - avg) ** 2 for x in values) / len(values)) if len(values) > 1 else 0.0
            stats[label] = {"average": avg, "std": std}
        return stats
    except Exception as e:
        logger.error(f"An error occurred while reading '{file_path}': {e}")
        return {}

def process_subdirectory(subdirectory_path):
    results = {}
    try:
        for file_name in os.listdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".log"):
                logger.info(f"Processing file: {file_path}")
                results[file_name] = get_stats_from_file(file_path)

        if results:
            output_file = os.path.join(subdirectory_path, "stats.json")
            with open(output_file, "w", encoding="utf-8") as json_file:
                json.dump(results, json_file, indent=4)
            logger.info(f"Statistics saved to: {output_file}")
    except Exception as e:
        logger.error(f"An error occurred in subdirectory '{subdirectory_path}': {e}")

def print_frame_sizes(subdirectory_path):
    result = {}
    try:
        stats_file = os.path.join(subdirectory_path, "stats.json")
        if not os.path.isfile(stats_file):
            return

        with open(stats_file, "r", encoding="utf-8") as sf:
            stats_data = json.load(sf)

        for logfile, stats in stats_data.items():
            sim_name = logfile[:-4] if logfile.lower().endswith(".log") else logfile
            fs_key = "Frame Size (MB)"
            if not stats or fs_key not in stats:
                val = 0.0
            else:
                val = stats[fs_key].get("average", 0.0) or 0.0

            val = round(float(val), 3)
            result[sim_name] = f"{val:.3f}MB/frame ({(val * 30 * 8):.1f}Mbps)"

        print(json.dumps(result, indent=4))
    except Exception as e:
        logger.error(f"An error occurred while printing frame sizes: {e}")
    return result

def process_directory(directory_path):
    try:
        for root, subdirs, _ in os.walk(directory_path):
            for subdir in subdirs:
                subdirectory_path = os.path.join(root, subdir)
                logger.info(f"Processing subdirectory: {subdirectory_path}")
                process_subdirectory(subdirectory_path)
                print_frame_sizes(subdirectory_path)
    except FileNotFoundError:
        logger.error(f"Error: Directory not found at path '{directory_path}'.")
    except Exception as e:
        logger.error(f"An error occurred while processing directory: {e}")

def run_from_config(output_path="results"):
    stats_folder = os.path.join(output_path, "stats")
    process_directory(stats_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log files in subdirectories to calculate averages and save as JSON.")
    parser.add_argument("--output-path", type=str, default="results",
                        help="Folder that stores the rendering results")
    args = parser.parse_args()

    run_from_config(output_path=args.output_path)
