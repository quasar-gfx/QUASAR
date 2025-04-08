# QUASAR

## Install Dependencies

### Ubuntu (reccomended)

NVIDIA GPUs with CUDA are highly reccomended. For the server, 16 GB or more of VRAM is reccomended, though 8 GB could work for some scenes. Tested on RTX 3070, RTX 3090, and RTX 4090 with CUDA 12 and up.

```
sudo apt install cmake libglew-dev libao-dev libmpg123-dev ffmpeg libavdevice-dev libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
```

Optional: Follow instructions [here](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html) for installing FFMPEG from source with CUDA hardware acceleration.

### Mac

Mac works as a client for receiving and viewing streams (specifically ATW), but not reccomended for use as a streaming server.

```
brew install cmake glew ffmpeg
```

## Building
```
mkdir build ; cd build
cmake ..; make -j
```

In the `build` directory, there will be a folder called `apps`.

## Sample Apps

### Scene Viewer

The Scene Viewer app loads a scene and lets you to fly through it.

Run Scene Viewer app:
```
# in build directory
cd apps/scene_viewer
./scene_viewer --size 1920x1080 --scene ../assets/scenes/sponza.json
```

### Asynchronous Time Warp (ATW)

The ATW app warps a previously rendered frame on a plane using a homography.

Build ATW sample:

To run the simulator:
```
# in build directory
cd apps/atw/simulator
./atw_simulator --size 1920x1080 --scene ../assets/scenes/sponza.json
```

To run streamer:
```
# in build directory
cd apps/atw/streamer
./atw_streamer --size 1920x1080 --scene ../assets/scenes/sponza.json
```

In a new terminal, to run receiver:
```
# in build directory
cd apps/atw/receiver
./atw_receiver --size 1920x1080
```

### MeshWarp

The MeshWarp app warps a previously rendered frame by using a depth map to create a texture-mapped mesh.

To run the simulator:
```
# in build directory
cd apps/meshwarp/simulator
./meshwarp_simulator --size 1920x1080 --scene ../assets/scenes/sponza.json
```

To run streamer:
```
# in build directory
cd apps/meshwarp/streamer
./mw_streamer --size 1920x1080 --scene ../assets/scenes/sponza.json
```

In a new terminal, to run receiver:
```
# in build directory
cd apps/meshwarp/receiver
./mw_receiver --size 1920x1080
```

### QuadWarp

The QuadWarp app warps a previously rendered frame by fitting a series of quads from a G-Buffer.

To run the simulator:
```
# in build directory
cd apps/quadwarp/simulator
./quads_simulator --size 1920x1080 --scene ../assets/scenes/sponza.json
```

### Multi-Camera QuadWarp (QuadStream)

The Multi app fits a series of quads from multiple G-Buffers from various camera views inside a headbox.

To run the simulator:
```
# in build directory
cd apps/multi/simulator
./multi_simulator --size 1920x1080 --scene ../assets/scenes/sponza.json
```

### Depth Peeling QuadWarp with EDP

The DP app fits a series of quads from multiple G-Buffers from various layers with fragment discarding determined by Effective Depth Peeling (EDP).

To run the simulator:
```
# in build directory
cd apps/dp/simulator
./dp_simulator --size 1920x1080 --scene ../assets/scenes/sponza.json
```
