import os
import time
import shutil
import argparse
import subprocess
from logger import logger

import simulator_runner, error_evaluator, stats_parser, video_generator

def main():
    parser = argparse.ArgumentParser(description='Run and compare all methods.')
    parser.add_argument('network_latency', type=float,
                        help='Simulated network latency in ms')
    parser.add_argument('network_jitter', type=float, nargs='?',
                        help='Simulated network jitter in ms')
    parser.add_argument('--size', type=str, default='1920x1080',
                        help='Resolution of rendering results')
    parser.add_argument('--scenes', type=str, default='robot_lab,sun_temple,viking_village,san_miguel',
                        help='Comma-separated list of scene names to evaluate')
    parser.add_argument('--output-path', type=str, default='results',
                        help='Path to store all results')
    parser.add_argument('--view-sizes', type=str, default='0.25,0.5,1.0',
                        help='Comma-separated list of viewcell sizes to evaluate')
    parser.add_argument('--num-poses', type=int, default=None,
                        help='If set, use the first N poses from the full camera path file')
    parser.add_argument('--pose-prediction', action='store_true',
                        help='Enable pose prediction')
    parser.add_argument('--pose-smoothing', action='store_true',
                        help='Enable pose smoothing')
    parser.add_argument('--no-errors', action='store_true',
                        help='Disable error evaluation')
    parser.add_argument('--no-videos', action='store_true',
                        help='Disable video generation')
    parser.add_argument('--short-paths', action='store_true',
                        help='Use short camera paths')
    args = parser.parse_args()

    scenes = args.scenes.split(',')
    view_sizes = [float(size) for size in args.view_sizes.split(',')]

    for scene in scenes:
        camera_path = f"../assets/paths/{scene}_path.txt"
        scene_file = f"../assets/scenes/{scene}.json"

        logger.info("======================================================")
        logger.info("Running simulators...")
        start_time = time.time()
        simulator_runner.run_from_config(
            scene=scene_file,
            camera_path=camera_path,
            size=args.size,
            output_path=args.output_path,
            exec_dir="../../QUASAR/build/apps",
            network_latency=args.network_latency,
            network_jitter=args.network_jitter,
            pose_prediction=args.pose_prediction,
            pose_smoothing=args.pose_smoothing,
            view_sizes=view_sizes,
            num_poses=args.num_poses
        )
        logger.info("*****************************************")
        logger.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
        logger.info("*****************************************")

        logger.info("======================================================")
        logger.info("Parsing statistics...")
        start_time = time.time()
        stats_parser.run_from_config(output_path=args.output_path)
        logger.info("*****************************************")
        logger.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
        logger.info("*****************************************")

        if not args.no_errors:
            logger.info("======================================================")
            logger.info("Calculating errors...")
            start_time = time.time()
            error_evaluator.run_from_config(output_path=args.output_path)
            logger.info("*****************************************")
            logger.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
            logger.info("*****************************************")

        if not args.no_videos:
            logger.info("======================================================")
            logger.info("Generating videos...")
            start_time = time.time()
            video_generator.run_from_config(output_path=args.output_path)
            logger.info("*****************************************")
            logger.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
            logger.info("*****************************************")

    logger.info("======================================================")
    logger.info("Creating tar file...")
    start_time = time.time()

    tar_components = [f"{args.output_path}/stats"]
    if not args.no_errors:
        tar_components.append("./errors.json")
    if not args.no_videos:
        tar_components.append(f"{args.output_path}/videos")

    tar_file = f"{args.output_path}/results_{args.network_latency}_{args.network_jitter}ms.tar.gz"
    tar_cmd = ["tar", "-zcvf", tar_file] + tar_components
    subprocess.run(tar_cmd, check=True)
    logger.info("*****************************************")
    logger.info(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes")
    logger.info("*****************************************")

    logger.info("======================================================")

    # delete tar_components
    for component in tar_components:
        if os.path.exists(component):
            if os.path.isdir(component):
                shutil.rmtree(component)
            else:
                os.remove(component)

    logger.info("All tasks completed!")

if __name__ == "__main__":
    main()
