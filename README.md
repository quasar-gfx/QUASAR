# OpenGL FFMPEG Video Streaming

Install dependencies:
```
# Linux (reccomended)
sudo apt-get install cmake libglew-dev libglfw3-dev libglm-dev libao-dev libmpg123-dev ffmpeg
```

```
# Mac
brew install cmake glew glfw3 glm ffmpeg
```

# Building
```
mkdir out
cd out
cmake ..
make -j
```

In the `out` directory, there will be a folder called `apps`,

# Sample Apps

## Test

The test app loads a scene and allows you to fly through it.

Run test app:
```
# in out directory
cd apps/test
./test -size 2048x2048 --scene ../assets/scenes/sponza.json
```

## Asynchronous Time Warp (ATW)

Build ATW sample:

To run streamer:
```
# in out directory
cd apps/atw/streamer
./atw_streamer --size 2048x2048 --scene ../assets/scenes/sponza.json --display 1
```

In a new terminal, to run receiver:
```
# in out directory
cd apps/atw/receiver
./atw_receiver --size 2048x2048
```

## MeshWarp

Build MeshWarp sample:

To run streamer:
```
# in out directory
cd apps/meshwarp/streamer
./mw_streamer --size 2048x2048 --scene ../assets/scenes/sponza.json --display 1
```

In a new terminal, to run receiver:
```
# in out directory
cd apps/meshwarp/receiver
./mw_receiver --size 2048x2048
```
