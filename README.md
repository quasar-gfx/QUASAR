# OpenGL Remote Rendering

Install dependencies:
```
# Linux (reccomended)
sudo apt-get install \
    cmake \
    libglew-dev \
    libao-dev \
    libmpg123-dev \
    ffmpeg \
    libavdevice-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev
```
NVIDIA GPUs are highly reccomended. Ensure you have CUDA.

Optional: Follow instructions [here](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html) for installing FFMPEG from source with CUDA.

```
# Mac (not reccomended for streaming, but works)
brew install cmake glew ffmpeg
```

# Building
```
mkdir build
cd build
cmake ..
make -j
```

In the `build` directory, there will be a folder called `apps`,

# Sample Apps

## Scene Viewer

The Scene Viewer app loads a scene and lets you to fly through it.

Run Scene Viewer app:
```
# in build directory
cd apps/scene_viewer
./scene_viewer --size 1920x1080 --scene ../assets/scenes/sponza.json
```

## Asynchronous Time Warp (ATW)

Build ATW sample:

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

## MeshWarp

To run MeshWarp sample:

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

## QuadWarp

To run QuadWarp sample:

```
# in build directory
cd apps/quadwarp/simulator
./quads_simulator --size 1920x1080 --scene ../assets/scenes/sponza.json
```
