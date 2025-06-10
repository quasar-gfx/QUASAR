import os
import csv
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

import cv2
import numpy as np
from tqdm import tqdm
import flip_evaluator as flip
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

from logger import logger

def psnr(img1, img2, max_pixel=1.0):
    mse = mean_squared_error(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def mean_std(values):
    arr = np.array(values)
    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1) if len(values) > 1 else 0
    return mean_val, std_val

def find_files(base_dir, extension=".png"):
    return [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(base_dir)
        for filename in filenames if filename.endswith(extension)
    ]

def format_camera_pose(filename):
    parts = filename.split("_")
    return "_".join(parts[1:])[:-4] if len(parts) > 1 else filename

def clean_simulator_name(name):
    name = name.replace("_simulator", "")
    if name.startswith("atw"):
        return "ATW"
    elif name.startswith("mw"):
        return f"MeshWarp FOV {name.split('_')[1]}"
    elif name.startswith("qs"):
        return f"QuadStream VC {name.split('_')[1]}"
    elif name.startswith("qr"):
        return f"QUASAR VC {name.split('_')[1]}"
    return name

def load_image(path):
    pose = format_camera_pose(os.path.basename(path))
    image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    return pose, image

def evaluate_error_metrics(reference_image, test_image_path, pose, flip_output_dir):
    _, test_image = load_image(test_image_path)

    psnr_error = psnr(reference_image, test_image)
    if psnr_error == float('inf'):
        psnr_error = 100.0

    ssim_error = ssim(reference_image, test_image, data_range=(test_image.max() - test_image.min()), channel_axis=2)
    if ssim_error == float('inf'):
        ssim_error = 1.0

    flip_error_image, mean_flip_error, _ = flip.evaluate(reference_image, test_image, "LDR")
    flip_error_image = (cv2.cvtColor(flip_error_image, cv2.COLOR_RGB2BGR) * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(flip_output_dir, f"flip_{pose}.png"), flip_error_image)

    return pose, mean_flip_error, ssim_error, psnr_error

def compare_images(scene_dir, output_dir):
    scene = os.path.basename(scene_dir)
    scene_viewer_dir = os.path.join(scene_dir, "scene_viewer")
    if not os.path.isdir(scene_viewer_dir):
        logger.error(f"Scene viewer directory not found: {scene_viewer_dir}")
        return

    reference_image_paths = find_files(scene_viewer_dir)
    reference_images = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_image, path): path for path in reference_image_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{scene} | Loading reference images", unit="image"):
            pose, image = future.result()
            reference_images[pose] = image

    simulators = [
        d for d in os.listdir(scene_dir)
        if d != "scene_viewer" and os.path.isdir(os.path.join(scene_dir, d))
    ]

    for simulator in simulators:
        process_simulator(scene, simulator, scene_dir, output_dir, reference_images)

def process_simulator(scene, simulator_name, scene_dir, output_dir, reference_images):
    simulator_dir = os.path.join(scene_dir, simulator_name)
    simulator_output_dir = os.path.join(output_dir, simulator_name)
    os.makedirs(simulator_output_dir, exist_ok=True)

    metrics = ["flip", "ssim", "psnr"]
    paths = {metric: os.path.join(simulator_output_dir, metric) for metric in metrics}
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    metric_results = {"flip": [], "ssim": [], "psnr": []}

    tasks = []
    for pose, ref_img in reference_images.items():
        matching_test_images = [
            img for img in find_files(simulator_dir) if pose in os.path.basename(img)
        ]
        for test_img_path in matching_test_images:
            tasks.append((ref_img, test_img_path, pose, paths["flip"]))

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(evaluate_error_metrics, *task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{scene} | {simulator_name}", unit="image"):
            pose, flip_val, ssim_val, psnr_val = future.result()
            metric_results["flip"].append([pose, flip_val])
            metric_results["ssim"].append([pose, ssim_val])
            metric_results["psnr"].append([pose, psnr_val])

    for metric in metrics:
        csv_path = os.path.join(paths[metric], f"{metric}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Pose", metric.upper()])
            writer.writerows(metric_results[metric])

def calculate_average_errors(output_path, scene):
    simulators = [
        d for d in os.listdir(os.path.join(output_path, scene))
        if os.path.isdir(os.path.join(output_path, scene, d))
    ]

    errors = {"flip": defaultdict(list), "ssim": defaultdict(list), "psnr": defaultdict(list)}

    for simulator in simulators:
        for metric in errors:
            csv_path = os.path.join(output_path, scene, simulator, metric, f"{metric}.csv")
            if os.path.exists(csv_path):
                with open(csv_path, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        errors[metric][simulator].append(float(row[metric.upper()]))

    results = {"scene": scene, "total_images": sum(len(v) for v in errors["flip"].values())}

    for metric, data in errors.items():
        reverse_sort = not (metric in ["flip"])
        results[f"{metric}_errors"] = [
            {
                "simulator": clean_simulator_name(sim),
                "mean_error": mean_std(values)[0],
                "std_error": mean_std(values)[1]
            }
            for sim, values in sorted(data.items(), key=lambda x: mean_std(x[1])[0], reverse=reverse_sort)
        ]

    return results

def process_scene(scene, images_dir, output_path):
    scene_dir = os.path.join(images_dir, scene)
    output_dir = os.path.join(output_path, scene)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Processing scene: {scene}")
    compare_images(scene_dir, output_dir)
    return calculate_average_errors(output_path, scene)

def main(output_path):
    images_dir = os.path.join(output_path, "images")
    errors_dir = os.path.join(output_path, "errors")
    scenes = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    all_results = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_scene, scene, images_dir, errors_dir) for scene in scenes]
        for future in as_completed(futures):
            all_results.append(future.result())

    errors_json_file = "errors.json"
    if os.path.exists(errors_json_file):
        with open(errors_json_file, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = []

    scene_mapping = {result["scene"]: result for result in existing_results}
    for new_result in all_results:
        scene = new_result["scene"]
        if scene in scene_mapping:
            scene_mapping[scene].update(new_result)
        else:
            existing_results.append(new_result)

    with open(errors_json_file, "w") as f:
        json.dump(existing_results, f, indent=4)

    logger.info(json.dumps(existing_results, indent=4))

def run_from_config(output_path="results"):
    main(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get quality errors.")
    parser.add_argument("--output-path", type=str, default="results", help="Folder that stores the rendering results")
    args = parser.parse_args()

    run_from_config(output_path=args.output_path)
