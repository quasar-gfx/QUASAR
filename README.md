# OpenGL Remote Rendering

Install dependencies:
```
# Linux (reccomended)
sudo apt-get install \
    cmake \
    libglew-dev \
    libglfw3-dev \
    libao-dev \
    libmpg123-dev \
    ffmpeg \
    libavdevice-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    cuda-12
```
NVIDIA GPUs are highly reccomended.

```
# Mac (not reccomended for streaming, but works)
brew install cmake glew glfw3 ffmpeg
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
./scene_viewer --size 2048x2048 --scene ../assets/scenes/sponza.json
```

## Asynchronous Time Warp (ATW)

Build ATW sample:

To run streamer:
```
# in build directory
cd apps/atw/streamer
./atw_streamer --size 2048x2048 --scene ../assets/scenes/sponza.json --display 1
```

In a new terminal, to run receiver:
```
# in build directory
cd apps/atw/receiver
./atw_receiver --size 2048x2048
```

## MeshWarp

To run MeshWarp sample:

To run streamer:
```
# in build directory
cd apps/meshwarp/streamer
./mw_streamer --size 2048x2048 --scene ../assets/scenes/sponza.json --display 1
```

In a new terminal, to run receiver:
```
# in build directory
cd apps/meshwarp/receiver
./mw_receiver --size 2048x2048
```

## QuadWarp

To run QuadWarp sample:

```
# in build directory
cd apps/quadwarp/streamer
./quads_streamer --size 2048x2048 --size2 512x512 --scene ../assets/scenes/sponza.json
```
