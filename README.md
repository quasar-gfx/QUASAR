# FFMPEG Video Streaming

To run:

Build and run OpenGL receiver:
```
cd receiver
make run
```

Then, open a new terminal and build and run the streamer:
```
cd streamer
make run
```

After the streamer says `Error reading frame: End of file`, you have to manually kill the streamer and receiver programs (graceful exit is WIP).

The streamer should render some frames and encode and stream the `streamer/input.mp4` file to the receiver, which displays it on right half the screen in the opengl window.

TODOs:

@edward:
- run reciever in android

@ruiyang:
- make the streamer stream an opengl texture/framebuffer instead of the input video
- use nvenc_h264 for encoding (https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html)
