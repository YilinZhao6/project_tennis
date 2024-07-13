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
程序一共分为四个部分，视频流读取(video_stream)->RNN分析动作(rnn_strike_isolation)->分析球速(ball_speed)->同时浏览多路视频(viewing_videos)。其中前三个部分的大部分代码已经完成，同时浏览多路视频的界面可以根据客户需求来设计。

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

程序逻辑:

当用户运行video_stream_json_output.py，程序会先等待一定的缓冲时间（e.g. 5s)，之后会以一定间隔（e.g. 3s), 来开始每段视频流的录制。（建议：输入RNN的视频流，可以放到最后一个rtsp link）程序的缓冲时间可以根据摄像头，以及本地电脑的性能来设置。之所以设置缓冲以及间隔时间，是为了避免4路视频同时录制，因电脑overload而带来的时间差。总体来说，这种录制方式可以让各路视频的时间差更可控。这行代码可以来设置缓冲时间，以及间隔时间。
```python
    start_delay = 5 + i * 3  # 5, 8, 11, 14 seconds
```

为了在录制过程中同时处理视频，程序会以一段时间来导出录制好的视频（例如，20s的间隔）例如，运行程序2分钟，将会得到6个长度为20s的视频，以及一个含有各个视频开始/结束时间戳的[.json](https://github.com/YilinZhao6/project_tennis/blob/main/video_stream/rtsp_output_1/video_times.json)文件。可以通过修改record_rtsp的argument来修改视频的长度。
```python
    thread = Thread(target=record_rtsp, args=(url, start_delay, 20, output_folder))
```
### RNN分析动作(rnn_strike_isolation)

根据项目要求，RNN分析动作的程序被分成了两部分：rnn_1.py负责从视频中提取人体的keypoint特征（本地执行），对于每一个视频，输出一个包含每帧信息的csv文件。将输出的csv文件上传到云端后，在云端执行rnn_2_json_only.py，来用RNN提取视频中的动作瞬间，并输出包含每个动作开始/结束时间戳的.json文件。具体的输出文件格式，可以参考rnn_strike_isolation/sample_output文件夹。

程序逻辑：
rnn_1.py（在本地运行）
当程序开始运行，会搜索source folder（即视频流读取的输出文件夹）是否存在已经保存好的mp4视频（不包括正在录制的mp4缓存文件）。如有，用movenet.tflite来逐帧提取视频中的人体特征，并输出以视频名命名的[.csv](https://github.com/YilinZhao6/project_tennis/blob/main/rnn_strike_isolation/sample_output/1_csv/1_keypoints.csv)文件。rnn_1.py和video_stream_json_output.py应同时运行，rnn_1可以动态读取视频流所保存的新视频。对于一个视频rnn_1.py的处理时间比视频时长要短（20s视频处理16s左右）

rnn_2_json_only.py（在云端运行）
读取上传的csv文件，用tennis_rnn_rafa.keras（RNN)来提取球员的正手/反手动作。输出动作开始/结束的时间戳[.json](https://github.com/YilinZhao6/project_tennis/blob/main/rnn_strike_isolation/sample_output/2_json/1_timestamp.json)文件。rnn_2_json_only.py可以使用tensorflow进行GPU加速。


### 分析球速->(ball_speed)

