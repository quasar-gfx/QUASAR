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

After the streamer says `Error reading frame: End of file` and exits, you have to manually kill the receiver code (graceful exit is WIP).

The streamer should encode and stream the `streamer/input.mp4` file to the receiver, which displays it on half the screen in the opengl window.

```
# in receiver/
vlc output.mp4 # play the output file
```

TODOs:

@edward:
- run reciever in android

@ruiyang:
- make the streamer stream an opengl texture/framebuffer instead of the input video
- use nvenc_h264 for encoding (https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/ffmpeg-with-nvidia-gpu/index.html)
