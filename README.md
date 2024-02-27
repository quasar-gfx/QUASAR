# FFMPEG Video Streaming

To run:

Build streamer and receiver:
```
mkdir out
cd out
cmake ..
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
- make the streamer stream an opengl texture/framebuffer instead of the input video
