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
./test -w <width> -h <height> -s <path to scene json file>
```

## ATW

Build ATW sample:

To run streamer:
```
# in out directory
cd apps/atw/streamer
./atw_streamer -w <width> -h <height> -o <client's ip address>:<client's port> -s <optional path to scene json file>
```

In a new terminal, to run receiver:
```
# in out directory
cd apps/atw/receiver
./atw_receiver -w <width> -h <height> -p <server's ip address>:<server's port>
```

The streamer should render, encode, and stream its output frames to the receiver, which displays in its opengl window with a locally rendered background scene.

## MeshWarp

Build MeshWarp sample:

To run meshing:
```
# in out directory
cd apps/meshwarp/streamer
./mw_streamer -w <width> -h <height> -s <optional path to scene json file>
```

In a new terminal, to run meshingviz:
```
# in out directory
cd apps/meshwarp/receiver
./mw_receiver -w <width> -h <height>
```
