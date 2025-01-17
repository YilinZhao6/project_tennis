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

RNN输出的json文件中的时间戳，是动作相对每一段视频的“相对时间”.结合视频流读取中的[video_times.json](https://github.com/YilinZhao6/project_tennis/blob/main/video_stream/rtsp_output_4/video_times.json)，我们可以得到动作的“绝对时间”。我们可以利用“绝对时间”的数据，来导出视频，进行球速分析和多视角的同步预览。

### 分析球速(ball_speed)
输入：一个视频片段
输出：一个标注了球员位置，以及球速的视频

在运行ball_speed_analyzer.py时，程序会首先让用户点击4个点（确定一个矩形），确认player和tennis ball的识别区域。此后，会让用户选择一条线（player 1和player 2的分界线）。最后，让用户点击网球场的四个角，以及中心。在完成上述步骤后，程序利用利用yolov8n.pt开始处理视频。yolov8n.pt模型会逐帧识别球的位置，并计算球的速度。另外，程序还会判断是哪一个球员的turn。假如球靠近了离player1 150px的区域，那么直到网球碰到下一个球员前，都算为player 1的turn.程序除了显示球的实时速度，还会计算出每个turn的最大速度。

在部署中需要确认的地方：
测试用视频中，大部分的球都存在虚影的情况，yolov8n.pt在该种情况下识别率较低。如果摄像头拍出来没有虚影，识别率会高很多。

当用户标注了场地的四个角，以及中心，程序会以网球场的中心做transformation，将label出的大概类似梯形的形状变换成网球场的标准形状--同时变换网球的位置。

[样例输出视频](https://drive.google.com/drive/folders/1ohPbMEhLCQOSE6XOrlFF00bfRNeW9PCK?usp=sharing)

### 同时浏览多路视频(viewing_videos)

这部分还没有完成。大概思路如下：

RNN分析动作后，将“相对时间”转化成“绝对时间”（电脑内的内置时间），这样我们就得到了每个动作的开始和结束时间。根据每个动作的开始和结束时间，从视频流读取(video_stream)的输出文件夹获取视频。每一个输出文件夹的video_times.json都包含了子视频的绝对开始和结束时间，据此可以导出多角度的视频预览。（也可以添加球速的预览）
