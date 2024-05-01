# tool

## 可以直接pip or conda
ffmpeg（用于视频切分为帧、将音频从视频中剥离）

tensorflow(因为要装spleeter)

spleeter（用于人声与背景音的分离，很垃圾，可以跳过）

pytorch

librosa（用于音频特征的抽取）

## need download
openface（用于视频图像特征的抽取）
`链接：https://pan.baidu.com/s/17kQ0ygDBNfnbIq_yFJaq9w?pwd=1234 
提取码：1234`

# notice

## pad
- 0
- norm

## Text
- 先pad再bert
- 先bert再pad

## Video

### openface 使用

#### example
- 单人
`FeatureExtraction.exe`
- 多人
`FaceLandmarkVidMulti.exe`

- 头部姿势
`-pose`
- - 单视频
`-f "C:\my videos\video1.avi"`
- 多视频
`-f "C:\my videos\video1.avi" -f "C:\my videos\video2.avi" -f "C:\my videos\video3.avi"`
- 一系列图像
`-fdir "C:\my videos\sequence1"`
- 
##### 图片
- 单图像
FaceLandmarkImg.exe -f "C:\my images\img.jpg"
- 多图像
FaceLandmarkImg.exe -f "C:\my images\img1.jpg" -f "C:\my images\img2.jpg" -f "C:\my images\img3.jpg"
- 目录
FaceLandmarkImg.exe -fdir "C:\my images"

#### 参数
`https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments`
##### 输入参数

`-f <filename>` 正在输入的视频文件，可以指定多个 -f

`-fdir <directory>` 对目录中的每个图像（.jpg、.jpeg、.png 和 .bmp）运行特征提取（输出将存储在整个目录的单个文件中）

`-out_dir <dir>` 与创建输出文件相关的根目录

##### 默认情况下，可执行文件将输出所有特征。您可以使用以下标志指定所需的输出特征：

`-2Dfp` 以像素为单位输出 2D landmark

`-3Dfp` 以毫米为单位输出 3D landmark

`-pdmparams` 输出刚性和非刚性形状参数

`-pose` 输出头部姿势（位置和旋转）

`-aus` 输出面部动作单元

`-gaze` 输出凝视和相关特征（眼睛标志的 2D 和 3D 位置）

`-hogalign` 输出提取的 HOG 特征文件

`-simalign` 简单对其

`-nobadaligned` 如果输出相似度对齐的图像，不要从检测失败或不可靠的帧输出（从而节省一些磁盘空间）

`-tracked` 带有检测到的landmark的跟踪输出视频

#### 导出数据说明
`https://blog.csdn.net/llvtingting/article/details/115839387`

## Audio

### feature
`https://www.cnblogs.com/LXP-Never/p/11561355.html`
#### 过零率librosa.feature.zero_crossing_rate

#### MFCC
`https://www.cnblogs.com/LXP-Never/p/10918590.html`

### librosa
```py
import librosa

librosa.feature.mfcc(
y=None,
sr=22050,
S=None,
n_mfcc=20,
dct_type=2,
norm='ortho')
```
y：音频时间序列
sr：音频的采样率

### wave2vector
`https://huggingface.co/docs/transformers/model_doc/wav2vec2`
`https://huggingface.co/facebook/wav2vec2-base-960h`

