import os
import random
import subprocess
import argparse
import shutil
import tempfile

from concurrent.futures import ThreadPoolExecutor, as_completed
from logger import logger

def find_files(base_dir, extension=".png"):
    files = []
    for root, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.endswith(extension):
                files.append(os.path.join(root, filename))
    return files

def run_ffmpeg(image_names, output_file, fps=60):
    abs_image_paths = [os.path.abspath(img) for img in image_names]
    output_file = os.path.abspath(output_file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frames_file = os.path.join(tmp_dir, f"frames_{random.randint(0, 100000)}.txt")

        try:
            with open(frames_file, "w") as f:
                for img_path in abs_image_paths:
                    f.write(f"file '{img_path}'\n")
                    f.write(f"duration {1.0 / fps}\n")

            command = [
                "ffmpeg",
                "-loglevel", "quiet",
                "-y",
                "-r", str(fps),
                "-f", "concat",
                "-safe", "0",
                "-i", frames_file,
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "18",
                output_file
            ]
            logger.info(f"Running ffmpeg command: {' '.join(command)}")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg command failed: {e}")

def create_video(output_dir, scene_name, video_type, simulator_name, simulator_dir):
    simulator_video_output_dir = os.path.join(output_dir, scene_name, video_type)
    os.makedirs(simulator_video_output_dir, exist_ok=True)

    output_file = os.path.join(simulator_video_output_dir, f"{simulator_name}.mp4")

    images = find_files(simulator_dir)
    if not images:
        logger.warning(f"No images found in {simulator_dir}. Skipping...")
        return

    try:
        images = sorted(images, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
        run_ffmpeg(images, output_file)
    except Exception as e:
        logger.error(f"Failed to create video for {simulator_name}: {e}")

def create_videos(scene_dir, flips_dir, output_dir):
    scene_name = os.path.basename(scene_dir)

    simulators_dirs = [
        os.path.join(scene_dir, d) for d in os.listdir(scene_dir)
        if os.path.isdir(os.path.join(scene_dir, d))
    ]

    simulators_flips_dirs = []
    if flips_dir and os.path.exists(flips_dir):
        simulators_flips_dirs = [
            os.path.join(flips_dir, d, "flip") for d in os.listdir(flips_dir)
            if os.path.isdir(os.path.join(flips_dir, d)) and os.path.isdir(os.path.join(flips_dir, d, "flip"))
        ]

    def create_color_videos(simulator_dir):
        simulator_name = os.path.basename(simulator_dir)
        create_video(output_dir, scene_name, "color", simulator_name, simulator_dir)

    def create_flip_videos(simulator_dir_flip):
        simulator_name = os.path.basename(os.path.dirname(simulator_dir_flip))
        create_video(output_dir, scene_name, "flip", simulator_name, simulator_dir_flip)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for simulator_dir in simulators_dirs:
            futures.append(executor.submit(create_color_videos, simulator_dir))

        for simulator_dir_flip in simulators_flips_dirs:
            if "scene_viewer" not in simulator_dir_flip:
                futures.append(executor.submit(create_flip_videos, simulator_dir_flip))

        for future in as_completed(futures):
            future.result()

def run_from_config(output_path="results"):
    images_dir = os.path.join(output_path, "images")
    errors_dir = os.path.join(output_path, "errors")
    video_output_dir = os.path.join(output_path, "videos")

    scenes_dirs = [
        os.path.join(images_dir, d) for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ]

    flips_dirs_dict = {}
    if os.path.exists(errors_dir):
        flips_dirs_dict = {
            d: os.path.join(errors_dir, d) for d in os.listdir(errors_dir)
            if os.path.isdir(os.path.join(errors_dir, d))
        }

    with ThreadPoolExecutor(max_workers=len(scenes_dirs)) as executor:
        futures = []
        for scene_dir in scenes_dirs:
            scene_name = os.path.basename(scene_dir)
            flips_dir = flips_dirs_dict.get(scene_name)
            logger.info(f"Creating videos for {scene_dir}" + (f" and {flips_dir}" if flips_dir else " (no flips)"))
            futures.append(executor.submit(create_videos, scene_dir, flips_dir, video_output_dir))

        for future in as_completed(futures):
            future.result()

    logger.info("Deleting images and errors directories...")
    for d in [images_dir, errors_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create videos for results.")
    parser.add_argument("--output-path", type=str, default="results", help="Folder that stores the rendering results")
    args = parser.parse_args()

    run_from_config(output_path=args.output_path)
