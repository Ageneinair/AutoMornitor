# VideoCut
Version 0.0.1

[Created by Xipeng Xie](https://github.com/Ageneinair)

VideoCut is a python package builted based on OpenCV.

  - Cut down a video more easily
  - Convert the frame rate (frames per second) arbitrarily

### Instructions
The syntax of `videocut()` is:

```py
    videocut(start, end, output_fps, input_address, output_address='cuted_video.mp4')
```
`start` and `end` are the starting point and ending pnint seperately of the piece which you want from the original video in the term of second; `output_fps` is a parameter to determine frames per second of the output video (if the 'output_fps' is smaller than original fps, the redundant frames would be deleted and the time span of the output video would be `end`-`start`); `input_address` is the address of input video; `output_address` is the address of input video and it is default as creating a video file named `cuted_video.mp4` in the root directory


### Installation

Install From package installer for Python (pip).

```sh
$ pip install videocut
```

Install from github

```sh
$ git clone
$ cd videocut
$ python setup.py install
```

License
----

MIT
