# FFMPEG Video Streaming

To run:

Build and run OpenGL receiver:
```
cd receiver
make
make run
```

Then, open a new terminal and build and run the streamer:
```
cd streamer
make
make run
```

After the streamer says `Error reading frame: End of file` and exits, you have to manually kill the receiver code (graceful exit is WIP).

The streamer should encode and stream the `streamer/input.mp4` file to the receiver, which saves the decoded video frames to `receiver/output.mp4`.

```
# in receiver/
vlc output.mp4 # play the output file
```

TODOs:

@edward:
- make receiver render the video in opengl instead of saving it to a file
- run reciever in android

@ruiyang:
- make the streamer stream an opengl texture/framebuffer instead of the input video
- use nvenc_h264 for encoding
