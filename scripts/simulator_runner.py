import os
import glob
import subprocess
import argparse

from logger import logger

def read_poses_from_file(camera_path):
    poses = []
    with open(camera_path, 'r') as f:
        for line in f:
            values = [float(x.strip()) for x in line.strip().split(' ')]
            if len(values) == 7:
                poses.append(values[:6])
    return poses

def create_simulation_command(exec_dir, executable, size, scene, output_path, extra_args, camera_path):
    command = [
        os.path.join(exec_dir, executable),
        "--size", size,
        "--scene", scene,
        "--save-images",
        "--camera-path", camera_path,
        "--output-path", output_path,
        *extra_args
    ]
    logger.debug(f"Created command: {' '.join(command)}")
    return command

def run_simulator_process(command, scene_name, simulator_name, output_path, camera_path):
    command_str = ' '.join(command)
    logger.info(f"Running \"{command_str}\"...")

    try:
        poses = read_poses_from_file(camera_path)

        output_path_images = os.path.join(output_path, "images", scene_name, simulator_name)
        os.makedirs(output_path_images, exist_ok=True)

        output_path_stats = os.path.join(output_path, "stats", scene_name)
        os.makedirs(output_path_stats, exist_ok=True)

        output_file_stats = os.path.join(output_path_stats, simulator_name + ".log")
        with open(output_file_stats, 'w') as f:
            subprocess.run(command, stdout=f, stderr=f, check=True)

        # optional: sanity check log for mismatches
        frames = sorted(glob.glob(os.path.join(output_path_images, "frame_*.png")))
        if len(frames) != len(poses):
            logger.warning(f"Frame count ({len(frames)}) and pose count ({len(poses)}) differ.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {simulator_name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error for {simulator_name}: {str(e)}")

def run_simulator(
    scene,
    camera_path,
    size,
    output_path,
    exec_dir,
    simulator_name,
    executable,
    extra_args=[]
):
    scene_name = os.path.splitext(os.path.basename(scene))[0]
    output_path_images = os.path.join(output_path, "images", scene_name, simulator_name)
    os.makedirs(output_path_images, exist_ok=True)

    command = create_simulation_command(
        exec_dir, executable, size, scene, output_path_images, extra_args, camera_path
    )
    run_simulator_process(command, scene_name, simulator_name, output_path, camera_path)

def run_from_config(
    scene,
    camera_path,
    size="1920x1080",
    output_path="results",
    exec_dir="../../QUASAR/build/apps",
    network_latency=20.0,
    network_jitter=10.0,
    pose_prediction=False,
    pose_smoothing=False,
    view_sizes=[0.25, 0.5, 1.0]
):
    # Depth Peeling Simulator (Ground Truth)
    run_simulator(
        simulator_name="scene_viewer",
        executable="depth_peeling/depth_peeling",
        output_path=output_path,
        exec_dir=exec_dir,
        size=size,
        scene=scene,
        camera_path=camera_path
    )

    # ATW Simulator
    run_simulator(
        simulator_name="atw_simulator",
        executable="atw/simulator/atw_simulator",
        output_path=output_path,
        exec_dir=exec_dir,
        size=size,
        scene=scene,
        camera_path=camera_path,
        extra_args=[
            "--network-latency", str(network_latency),
            "--network-jitter", str(network_jitter),
            *(["--pose-prediction"] if pose_prediction else []),
            *(["--pose-smoothing"] if pose_smoothing else [])
        ]
    )

    # MeshWarp Simulator
    for fov in [60, 120]:
        run_simulator(
            simulator_name=f"mw_simulator_{fov}",
            executable="meshwarp/simulator/mw_simulator",
            output_path=output_path,
            exec_dir=exec_dir,
            size=size,
            scene=scene,
            camera_path=camera_path,
            extra_args=[
                "--network-latency", str(network_latency),
                "--network-jitter", str(network_jitter),
                "--rsize", "1920x1080" if fov == 60 else "3840x2160",
                "--fov", str(fov),
                *(["--pose-prediction"] if pose_prediction else []),
                *(["--pose-smoothing"] if pose_smoothing else [])
            ]
        )

    for view_size in view_sizes:
        # QuadStream Simulator
        run_simulator(
            simulator_name=f"qs_simulator_{view_size:.2f}",
            executable="quadstream/simulator/qs_simulator",
            output_path=output_path,
            exec_dir=exec_dir,
            size=size,
            scene=scene,
            camera_path=camera_path,
            extra_args=[
                "--network-latency", str(network_latency),
                "--network-jitter", str(network_jitter),
                "--view-size", str(view_size),
                *(["--pose-prediction"] if pose_prediction else []),
                *(["--pose-smoothing"] if pose_smoothing else [])
            ]
        )

        # QUASAR Simulator
        run_simulator(
            simulator_name=f"qr_simulator_{view_size:.2f}",
            executable="quasar/simulator/qr_simulator",
            output_path=output_path,
            exec_dir=exec_dir,
            size=size,
            scene=scene,
            camera_path=camera_path,
            extra_args=[
                "--network-latency", str(network_latency),
                "--network-jitter", str(network_jitter),
                "--view-size", str(view_size),
                *(["--pose-prediction"] if pose_prediction else []),
                *(["--pose-smoothing"] if pose_smoothing else [])
            ]
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulators with path file.")
    parser.add_argument("--scene", type=str, required=True, help="Path to the scene file")
    parser.add_argument("--size", type=str, default="1920x1080", help="Resolution of rendering results")
    parser.add_argument("--output-path", type=str, default="results", help="Folder that stores the rendering results")
    parser.add_argument("--exec-dir", type=str, default="../build/apps",
                        help="Directory where simulator executables are located")
    parser.add_argument("--network-latency", type=float, default=20.0, help="Network latency in ms")
    parser.add_argument("--network-jitter", type=float, default=10.0, help="Network jitter in ms")
    parser.add_argument('--pose-prediction', action='store_true')
    parser.add_argument('--pose-smoothing', action='store_true')
    parser.add_argument('--view-sizes', type=str, default='0.25,0.5,1.0')
    parser.add_argument("--camera-path", type=str, required=True, help="Camera animation file")

    args = parser.parse_args()

    view_sizes = [float(size) for size in args.view_sizes.split(',')]

    run_from_config(
        scene=args.scene,
        camera_path=args.camera_path,
        size=args.size,
        output_path=args.output_path,
        exec_dir=args.exec_dir,
        network_latency=args.network_latency,
        network_jitter=args.network_jitter,
        pose_prediction=args.pose_prediction,
        pose_smoothing=args.pose_smoothing,
        view_sizes=view_sizes,
        qr_qs_only=args.qr_qs_only
    )
