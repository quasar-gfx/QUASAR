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

In a new terminal, to run receiever:
```
# in out directory
cd apps/receiever
./receiever
```

After the streamer says `Error reading frame: End of file`, you can manually kill the streamer and receiver programs (graceful exit is WIP).

The streamer should render some frames and encode and stream the `streamer/input.mp4` file to the receiver, which displays it on right half the screen in the opengl window.

TODOs:

@edward:
- run reciever in android

@ruiyang:
- use cuda to directly stream an opengl texture from GPU
