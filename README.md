# OpenGL FFMPEG Video Streaming

To run:

Install dependencies:
```
# Linux
sudo apt-get install cmake libglew-dev libglfw3-dev libglm-dev libao-dev libmpg123-dev
```

```
# Mac
brew install cmake glew glfw3 glm
```

Build streamer and receiver:
```
mkdir out
cd out
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

In the `out` directory, there will be a folder called `apps`,

To run streamer:
```
# in out directory
cd apps/streamer
./streamer
```

In a new terminal, to run receiver:
```
# in out directory
cd apps/receiver
./receiver
```

The streamer should render, encode, and stream its output frames to the receiver, which displays in its opengl window with a locally rendered background scene.

TODOs:

@edward:
- run reciever in android

@ruiyang:
- use cuda to directly stream an opengl texture from GPU
