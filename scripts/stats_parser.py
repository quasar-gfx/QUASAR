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
        with open(file_path, "r") as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    label, value, unit = match.groups()
                    normalized_label = f"{normalize_label(label)} ({unit})"
                    data[normalized_label].append(float(value))

        stats = {}
        for label, values in data.items():
            avg = sum(values) / len(values)
            std = math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))
            stats[label] = {"average": avg, "std": std}
        return stats
    except Exception as e:
        logger.error(f"An error occurred while processing '{file_path}': {e}")
    return {}

def process_subdirectory(subdirectory_path):
    results = {}
    try:
        for file_name in os.listdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".log"):
                logger.info(f"Processing file: {file_path}")
                results[file_name] = get_stats_from_file(file_path)
        output_file = os.path.join(subdirectory_path, "stats.json")
        with open(output_file, "w") as json_file:
            json.dump(results, json_file, indent=4)
        logger.info(f"Statistics saved to: {output_file}")
    except Exception as e:
        logger.error(f"An error occurred in subdirectory '{subdirectory_path}': {e}")

def process_directory(directory_path):
    try:
        for root, subdirs, _ in os.walk(directory_path):
            for subdir in subdirs:
                subdirectory_path = os.path.join(root, subdir)
                logger.info(f"Processing subdirectory: {subdirectory_path}")
                process_subdirectory(subdirectory_path)
    except FileNotFoundError:
        logger.error(f"Error: Directory not found at path '{directory_path}'.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

def run_from_config(output_path="results"):
    stats_folder = os.path.join(output_path, "stats")
    process_directory(stats_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process log files in subdirectories to calculate averages and save as JSON.")
    parser.add_argument("--output-path", type=str, default="results", help="Folder that stores the rendering results")
    args = parser.parse_args()

    run_from_config(output_path=args.output_path)
