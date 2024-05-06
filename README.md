# OpenGL FFMPEG Video Streaming

Install dependencies:
```
# Linux
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

## Streamer and Receiver

Build streamer and receiver:

To run streamer:
```
# in out directory
cd apps/streamer
./streamer -w <width> -h <height>
```

In a new terminal, to run receiver:
```
# in out directory
cd apps/receiver
./receiver -w <width> -h <height>
```

The streamer should render, encode, and stream its output frames to the receiver, which displays in its opengl window with a locally rendered background scene.

## Meshing

Build meshing and meshingviz:

To run meshing:
```
# in out directory
cd apps/meshing
./meshing -w <width> -h <height> -s <path to scene json file>
```

In a new terminal, to run meshingviz:
```
# in out directory
cd apps/meshingviz
./meshingviz -w <width> -h <height>
```
