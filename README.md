# project_tennis

## 程序结构

```bash
├─ball_speed
│  ├─input_videos
│  ├─sample_output
│  ├─check_transformation.py
│  └─ball_speed_analyzer.py
├─rnn_strike_isolation
│  ├─batch_processing
│  │  ├─extract_human_pose.py
│  │  ├─movenet.tflite
│  │  ├─rnn_1.py
│  │  ├─rnn_2_json_only.py
│  │  ├─rnn_2_video.py
│  │  └─tennis_rnn_rafa.keras
│  ├─sample_output
│  └─test_code
├─video_stream
│  ├─rtsp_output_1
│  ├─rtsp_output_2
│  ├─rtsp_output_3
│  ├─rtsp_output_4
│  └─video_stream_json_output.py
└─viewing_videos
```
程序一共分为四个部分，视频流读取(video_stream)->RNN分析动作(rnn_strike_isolation)->分析球速->(ball_speed)->同时浏览多路视频(viewing_videos)。其中前三个部分的大部分代码已经完成，同时浏览多路视频的界面可以根据客户需求来设计。

### 视频流读取(video_stream)

video_stream_json_output.py

输入：4个（或多个）RTSP视频流链接
```bash
rtsp_urls = [
    "rtsp://admin:123456@192.168.1.102:8554/profile0",
    "rtsp://admin:123456@192.168.1.102:8554/profile0",
    "rtsp://admin:123456@192.168.1.102:8554/profile0",
    "rtsp://admin:123456@192.168.1.102:8554/profile0"
]
```
输出：对于每一个rtsp视频流，创建一个输出文件夹。里面包括所有的输出视频，以及一个记录每段视频开始/结束时间的.json文件。（参考video_stream/rtsp_output内的文件内容)

程序逻辑；
当用户运行video_stream_json_output.py，程序会先等待一定的缓冲时间（e.g. 5s)，之后会以一定间隔（e.g. 3s), 来开始每段视频流的录制。程序的缓冲时间可以根据摄像头，以及本地电脑的性能来设置。之所以设置缓冲以及间隔时间，是为了避免4路视频同时录制，因电脑overload而带来的时间差。总体来说，这种录制方式可以让各路视频的时间差更可控。这行代码可以来设置缓冲时间，以及间隔时间。
```python
    start_delay = 5 + i * 3  # 5, 8, 11, 14 seconds
```
