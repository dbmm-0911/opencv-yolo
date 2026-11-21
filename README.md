# YOLO11 + 卡尔曼滤波跟踪（C++ / OpenCV DNN）

用 OpenCV DNN 加载 YOLO11 的 ONNX 模型，检测红/蓝圆靶（单类 `Buff`），
再用卡尔曼滤波做单目标 / 多目标跟踪；传统 HSV 圆靶识别作为 YOLO 的兜底候选。

- 纯 C++17，只依赖 OpenCV（DNN + videoio + imgproc）。
- 不依赖 ultralytics / PyTorch 运行时，`best.onnx` 直接喂给 `cv::dnn::Net`。
- 三种框同时输出：观测框（红）、预测框（黄）、滤波框（绿）。
- 参数集中在 `yolo11.hpp` 的五个结构体里（`Config` / `KalmanParams` / `KalmanAccelParams` /
  `MultiTrackParams` / `TraditionalParams`），逐字段带中文注释。

---

## 目录

1. [工程在做什么](#1-工程在做什么)
2. [文件与代码导航](#2-文件与代码导航)
3. [快速开始](#3-快速开始)
4. [整体流水线](#4-整体流水线)
5. [检测层原理](#5-检测层原理)
6. [传统识别层原理](#6-传统识别层原理)
7. [单目标卡尔曼](#7-单目标卡尔曼)
8. [多目标卡尔曼](#8-多目标卡尔曼)
9. [YOLO 主 + 传统辅](#9-yolo-主--传统辅)
10. [输出说明](#10-输出说明)
11. [参数速查表](#11-参数速查表)
12. [调参指南](#12-调参指南)
13. [模型导出（OpenCV DNN 兼容）](#13-模型导出opencv-dnn-兼容)
14. [常见问题](#14-常见问题)
15. [已知限制](#15-已知限制)
16. [文档与注释维护](#16-文档与注释维护)
17. [v1 / v2 实验变体](#17-v1--v2-实验变体)

---

## 1. 工程在做什么

输入一段比赛视频，输出两样东西：

1. **叠加视频**：每帧画上每个目标的三种框和 ID；
2. **CSV 日志**：每帧每个目标一行，记录预测/滤波/观测框、速度、年龄、命中数、丢失帧数。

每帧的处理顺序固定：

```text
YOLO 检测 ─┐
           ├─ 融合成唯一观测列表 ─ step()：预测 → 关联 → 校正 → 生命周期 ─ 画框 ─ 写视频/CSV
传统识别 ─┘
```

设计上的三个取舍（后文都有展开）：

| 取舍 | 选择 | 原因 |
| --- | --- | --- |
| 观测用什么 | 只取检测框**中心点** `(cx, cy)` | 中心点抖动远小于框边，宽高变化是“外观”不是运动，没必要进状态向量 |
| 多目标怎么关联 | 全局代价排序 + 一对一**贪心** | 本项目画面里最多 4 个目标（实测 `max tracks/frame=4`），贪心足够；匈牙利在稀疏场景收益有限，代码却复杂得多 |
| 传统识别怎么用 | **只补漏、不覆盖** YOLO | 传统会把大符上所有红色靶面都检出来，让它当主检测会产生大量假轨迹 |

---

## 2. 文件与代码导航

| 文件 | 职责 | 建议阅读顺序 |
| --- | --- | --- |
| `main.cpp` | 入口：构造检测器与跟踪器、设置参数、跑视频 | ① 先看它了解调用方式 |
| `yolo11.hpp` | 接口、状态结构体、**全部可调参数**（逐字段注释） | ② 查参数含义时翻它 |
| `yolo11.cpp` | YOLO 推理 + 单/多目标跟踪 + 传统识别的实现 | ③ 按下面的“想看什么”跳读 |
| `CMakeLists.txt` | 构建配置（C++17 + OpenCV；`main` 之外还有可选的 `main_v1` / `main_v2`） | ④ 编译不过时看 |
| `yolo11_spv1.hpp/.cpp` | v1 实验变体：只替换多目标关联（Hungarian + 两阶段 + 自适应 R） | ⑤ 想做关联对比时看 |
| `yolo11_spv2.hpp/.cpp` | v2 实验变体：v1 + 马氏门控 + 轨迹质量分 | ⑥ 想评估轨迹可信度时看 |

代码里按“想看什么”定位：

| 想看什么 | 去哪找（函数 / 分段注释） |
| --- | --- |
| 预处理、解码、NMS | `Detector::infer()`，内部标注 `1. Letterbox` ~ `8. 生成完整 Detection` |
| 单目标滤波 | `KalmanTracker::step / predict / initialize / correct` |
| 常加速单目标 | `KalmanAccelTracker::*`（结构与匀速版一一对应） |
| 多目标关联 | `MultiObjectTracker::step()`，注释分“第一步~第七步” |
| 轨迹防漂移 | `MultiObjectTracker::predict()` 与 `limitStateMotion()` |
| YOLO + 传统融合 | `MultiObjectTracker::fuseDetections()`，注释分 1/2/3 段 |
| 传统圆靶识别 | 文件末尾 `findbuff()` |
| 参数含义与量纲 | `yolo11.hpp`：`Config` / `KalmanParams` / `MultiTrackParams` / `TraditionalParams` |

### 注释约定

代码注释分三层，读的时候可以按需取舍：

- **文件头 / 函数头**：这一段在做什么、为什么这样做、背后的原理（例如为什么
  letterbox 而不是直接 resize、为什么卡尔曼用 Joseph 形式）。
- **行内注释**：只解释不显然的地方——坐标变换的先后顺序、门限为什么取这个值、
  某个分支在防什么异常。
- **参数注释**：在 `yolo11.hpp` 里逐字段写“含义 + 单位 + 调大调小的效果”。

所有注释只解释行为，不改变行为；逻辑与参数默认值都以代码为准。

---

## 3. 快速开始

```bash
# 依赖：OpenCV（建议 ≥ 4.6，需含 dnn 模块）、CMake ≥ 3.10、支持 C++17 的编译器
cmake -S . -B build && cmake --build build -j

cd build
./main        # 读 ../best.onnx 与 031.mp4，输出 ../output/a.mp4、../output/tracks.csv
```

`main.cpp` 里的路径是可运行的示例：

```cpp
// 模型只加载一次：cv::dnn::Net 的构造和 ONNX 解析是整段流程里最慢的一步。
// 第二个参数 Config 用默认值：输入 640x640、conf=0.25、NMS=0.45、CPU 推理。
yolo11::Detector detector("/mnt/e/yolo11C++/best.onnx");

yolo11::MultiTrackParams params;
params.motion_model = yolo11::MotionModel::ConstantVelocity; // 单个镜头内近似直线运动
params.dt = 1.0f;               // 以“帧”为单位：速度 = 像素/帧，与帧率无关
params.max_age = 3;             // 连续未匹配超过 3 帧就删轨迹（镜头会硬切，别留太久）
params.min_hits = 3;            // 累计命中 3 次才画框 / 写 CSV，挡掉单帧误检
params.measurement_noise = 12.0f;      // 观测噪声 R：检测中心约 3~4 像素抖动
params.process_noise_velocity = 25.0f; // 速度过程噪声 Q：允许速度快速变化

yolo11::MultiObjectTracker tracker(params);
tracker.process("/mnt/e/yolo11C++/031.mp4",   // 输入
                "../output/a.mp4",             // 输出叠加视频
                "../output/tracks.csv",        // 逐帧逐目标 CSV
                detector, 0);                  // 类别编号：单类模型固定 0
```

> 路径说明：示例用 WSL 风格 `/mnt/e/...`；在 Windows 原生编译时改成 `E:/yolo11C++/best.onnx`。
> 输出目录不需要手动创建，`process()` 会自动建父目录。

> 另有 v1 / v2 两个“只改多目标关联”的实验版本，构建与运行见
> [17. v1 / v2 实验变体](#17-v1--v2-实验变体)。

---

## 4. 整体流水线

```text
                ┌───────────────────────────── 每帧 ─────────────────────────────┐
                │                                                                │
  frame ──┬─► Detector::infer ──► yoloDetections ─┐                              │
          │   letterbox→blob→forward→decode→NMS   │                              │
          │                                       ├─► fuseDetections ─► step() ──┼─► draw ─► writer
          └─► findbuff ───────► traditionalDets ──┘        │                │    │
              HSV→形态学→轮廓→打分→去重                     │                │    └─► csv
                                                           │                └─ 轨迹生命周期
                                                    同一目标最多一个观测      预测/关联/校正/新生/过期/合并
```

关键点：

- **检测与跟踪解耦**：`Detector` 无状态，`MultiObjectTracker` 不碰模型。
  想换 TensorRT、换检测器，只需要保持 `std::vector<Detection>` 这个接口。
- **每帧都跑两路检测**，但传统候选是否进入跟踪由融合规则决定（默认只救援）。
- **关联用的是“本帧预测框”**，不是上一帧的旧框；否则快速目标一定匹配错。

---

## 5. 检测层原理

`Detector::infer()` 分 8 段，都是标准 YOLO 部署流程，但每一段都有容易踩的坑：

### 5.1 Letterbox（等比缩放 + 灰边）

```text
原图 1440×1080 ──scale=min(640/1440, 640/1080)=0.444──► 640×480 ──补边──► 640×640
                                                                   上下各 80 像素灰边
```

- **为什么不直接 resize 到 640×640**：拉伸会改变宽高比，目标被压扁，和训练分布不一致。
- **为什么补 114 灰**：这是 YOLO 训练时默认的填充值；纯黑补边会形成强边缘，反而干扰检测。
- **代价**：网络输出的坐标在“640 的 letterbox 图”上，必须做两级反变换才能回原图。

### 5.2 Blob

```cpp
// 参数含义：缩放系数、输出尺寸、减均值、swapRB、是否中心裁剪。
// 三个关键参数任一与训练不一致，网络都不报错，只会“检测变差”。
cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(640, 640),
                                      cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
```

三件事缺一不可：`1/255` 归一化到 `[0,1]`、BGR→RGB（模型训练用 RGB）、重排成 NCHW。

### 5.3 输出解码

模型输出 `[1, 5, 8400]`：

| 维度 | 含义 |
| --- | --- |
| 5 | `cx, cy, w, h, score`（单类模型没有单独的目标性通道） |
| 8400 | `80×80 + 40×40 + 20×20`，stride 8/16/32 三个检测头的网格点总数 |

坐标反变换（顺序不能反）：

```text
x_letterbox ──减 padding──► x_resized ──÷ resizeScale──► x_frame ──× original/frame──► x_original
```

- 减 padding 必须在除 scale 之前，否则多除一次比例。
- 用实际 `newW/newH` 算缩放比，而不是理论 `scale`：`newW = round(cols*scale)` 有舍入误差，
  4K 图上两者能差 1~2 像素，用错会让整幅框系统性偏移。

### 5.4 NMS

同一目标会在相邻网格、不同尺度上被重复预测。`NMSBoxes` 按分数从高到低保留框，
删掉与已保留框 IoU 超过 `nms_threshold` 的候选。`conf_threshold` 在解码和 NMS 各用一次，
结果等价，只是接口要求。

---

## 6. 传统识别层原理

`findbuff()` 不参与主检测，只在 YOLO 漏检时提供候选。六步：

```text
HSV 阈值 ──► 形态学闭/开 ──► 外轮廓 ──► 面积/圆度/填充率 ──► 环结构打分 ──► 中心去重
```

1. **为什么用 HSV 不用 RGB**：RGB 三通道随光照强耦合，同一块红靶亮处/暗处差很远；
   HSV 把颜色(H)、鲜艳度(S)、亮度(V) 拆开，H 在区间内就是“红/蓝”，对曝光更稳。
   红色在 H 轴上跨 0 点，所以用 `[0,10] ∪ [170,179]` 两段再合并；蓝色只有 `[95,125]` 一段。
   阈值里的 `S>=110` 挡白色灯条和灰色金属，`V>=80` 挡暗背景。
2. **形态学顺序**：先 7×7 闭运算（把被白十字切开的圆环连起来，核必须比裂缝宽），
   再 3×3 开运算（去掉孤立噪点，核要小，否则远处小靶会被腐蚀断）。
3. **轮廓过滤**：
   - 面积：下限取 `max(min_area, 画面面积*0.00005)`，上限 `画面面积*max_area_ratio`；
   - 圆度 `4πA/P²`：完美圆 = 1，越不规则越接近 0，用来排除灯条、方形结构件；
   - 填充率 `A/(πr²)`：断成弧的轮廓圆度可能“看着还行”，但填充率很低。
4. **环结构检查**：靶面是“红环 + 暗中心”。比较内圈（0.10r~0.35r）与外圈（0.65r~0.90r）
   的红色占比：外圈红多、内圈红少 → 环形结构 → 高分；实心红点/红灯 → 低分。
5. **打分**：`0.35*圆度 + 0.55*环结构 + 0.10*填充率`，落在 `[0,1]`，用于排序和门限比较
   （不是百分比，也不是概率）。
6. **去重**：同一块靶面常被白十字切成多段，按分数从高到低保留，中心距小于
   `0.6 * 平均半径` 的视为同一目标。

---

## 7. 单目标卡尔曼

两种模型，API 相同：

```cpp
yolo11::KalmanParams cvParams;              // 匀速：状态 [cx, cy, vx, vy]^T
yolo11::KalmanTracker cvTracker(cvParams);

yolo11::KalmanAccelParams caParams;         // 常加速：状态 [cx, cy, vx, vy, ax, ay]^T
yolo11::KalmanAccelTracker caTracker(caParams);

cvTracker.process("in.mp4", "out.mp4", "track.csv", detector, 0);
```

### 7.1 卡尔曼在做什么

滤波器同时维护“估计值 `x`”和“不确定度 `P`”，每帧两步：

```text
预测：x' = F·x            P' = F·P·Fᵀ + Q      // 按模型外推，P 变大
更新：y  = z - H·x'                            // 新息（观测 - 预测）
      S  = H·P'·Hᵀ + R                         // 新息协方差
      K  = P'·Hᵀ·S⁻¹                           // 卡尔曼增益
      x  = x' + K·y
      P  = (I-KH)·P'·(I-KH)ᵀ + K·R·Kᵀ          // Joseph 形式
```

直觉：`K ≈ P/(P+R)`。观测噪声 `R` 大 → 更信模型（平滑但滞后）；预测不确定度 `P` 大 →
更信观测（跟手但抖）。**这就是卡尔曼不需要手工设平滑系数的原因**，`K` 每帧自动算。

- `F` 是运动模型（匀速 / 常加速）；
- `H = [I₂ 0]` 只观测中心点，速度/加速度通过协方差耦合被间接修正；
- `Q` 每步加到对角线上，表示“模型不可能完全准确”；
- Joseph 形式保证 `P` 在长时间运行后仍对称正定，不会因为数值误差“跑负”导致发散。

### 7.2 单目标的三条规则

1. **选观测**：已初始化 → 离预测中心最近的检测；未初始化 → 置信度最高的检测。
2. **误差门限**：观测中心与预测中心距离超过 `max_center_distance`（默认 100 px）时，
   视为误检，跳过本次校正，并按“本帧无观测”处理（不更新边长、不清零丢失计数）。
3. **丢失与重捕**：连续丢失超过 `lost_frame_limit`（默认 30 帧）判定丢失，
   释放 `x/P`，等待下次检测重新初始化；未超限时滤波框 = 预测框（惯性滑行）。

宽高不进入状态向量：有观测时直接更新，无观测时沿用最近一次观测值。

---

## 8. 多目标卡尔曼

> 本节描述基线实现（`main`）。v1 / v2 只改这一层的关联与门控，见
> [17. v1 / v2 实验变体](#17-v1--v2-实验变体)。

```cpp
yolo11::MultiTrackParams params;
params.motion_model = yolo11::MotionModel::ConstantVelocity;
// 或：params.motion_model = yolo11::MotionModel::ConstantAcceleration;
params.dt = 1.0f;
params.max_age = 3;                 // 连续未匹配 > max_age 帧后删除轨迹
params.min_hits = 3;                // 累计命中达到该值才绘制/写 CSV
params.measurement_noise = 12.0f;   // 观测噪声 R
params.process_noise_velocity = 25.0f;

// 确认与防漂移（默认值已适配本视频，按需覆盖）
params.init_confidence = 0.50f;      // 建轨所需最低检测置信度
params.merge_iou = 0.55f;            // 重复轨迹合并阈值
params.draw_lost_frames = 2;         // 丢失后最多再画几帧
params.max_speed = 150.0f;           // 速度上限，同时作为关联速度门限
params.coast_velocity_decay = 0.80f; // 丢失期间每帧速度衰减
params.max_coast_ratio = 1.5f;       // 丢失期间预测中心离最近观测的最大距离（框对角线倍数）

yolo11::MultiObjectTracker tracker(params);
tracker.process("input.mp4", "tracked.mp4", "tracks.csv", detector, 0);
```

### 8.1 每帧七步

| 步骤 | 做什么 | 细节 |
| --- | --- | --- |
| 1 预测 | 所有轨迹 `kf.predict()` | 丢失时先衰减速度、夹紧 `max_speed/max_accel`、把预测中心拉回 `max_coast_ratio*对角线` 内 |
| 2 代价 | 每个“轨迹-检测”组合算分 | `cost = (w_iou·(1-IoU) + w_dist·min(1, d/gate)) / (w_iou+w_dist)` |
| 3 匹配 | 按 `cost` 升序贪心，一对一 | 每条轨迹、每个检测最多用一次 |
| 4 新生 | 未匹配且 `conf >= init_confidence` 的检测建新轨 | 传统救援框 `allow_new_track=false`，不会在这里建轨 |
| 5 清理 | `time_since_update > max_age` 删除 | erase-remove 惯用法 |
| 6 合并 | 框 IoU ≥ `merge_iou` 的重复轨迹合并 | 保留小 ID（创建更早），命中数取 max |
| 7 视图 | `refreshViews()` 生成只读快照 | 外部通过 `tracks()` 查询，不直接碰 `KalmanFilter` |

关联门限有三层，任何一层不过就直接丢弃该候选：

```text
d > max_speed * dt                 → 拒绝（一帧内物理上不可能移动这么远，通常是镜头切换）
d > gate                           → 拒绝（gate = max(max_center_distance, 对角线*max_distance_ratio)）
IoU < min_iou 且 d > 0.45*gate     → 拒绝（离得不近、形状也不像）
```

### 8.2 轨迹生命周期

```text
新建（hits=1, time_since_update=0）
   │
   ├─ 本帧命中 ──► hits++ / hit_streak++ / time_since_update=0 ──► 继续跟踪
   │
   └─ 本帧未命中 ──► time_since_update++
                        ├─ > draw_lost_frames ──► 不再绘制（轨迹仍存活）
                        └─ > max_age ──────────► 删除轨迹
```

- **确认**：`hits >= min_hits` 才绘制、才写 CSV；未确认的轨迹只在内部流转，避免误检“闪框”。
- **不画幽灵框**：丢失超过 `draw_lost_frames` 帧不再绘制，但轨迹对象还活着，可能在 `max_age` 内被救援。
- **重新捕获**（之前至少丢过一帧）：速度减半、加速度清零，防止遮挡后“冲过头”。

### 8.3 为什么能防漂移

卡尔曼在没有观测时会纯靠模型外推，常加速模型尤其容易发散。这里叠了四道保险：

1. 丢失期间每帧速度乘 `coast_velocity_decay`（默认 0.8），外推逐渐“停住”；
2. 速度/加速度按模长夹到 `max_speed`、`max_accel`（保留方向，只压缩幅值）；
3. 预测中心离最近一次观测中心超过 `max_coast_ratio * 框对角线` 就拉回边界；
4. 重新捕获时衰减速度、清加速度，让滤波器快速收敛到新观测。

---

## 9. YOLO 主 + 传统辅

`MultiObjectTracker::process()` 每帧都跑 YOLO，同时用 `findbuff()` 产生传统红/蓝圆靶候选。
融合规则是“YOLO 主、传统只补漏”：

- 与 YOLO 框重叠的传统候选只标记 `source=2`，最终框仍用 YOLO，传统不能覆盖主检测。
- 传统候选只在**已确认且当前丢失**的轨迹预测框附近、且尺寸/分数通过门限时，才作为该轨迹的观测
  （救援，保持 ID）。
- 默认 `TraditionalParams::allow_new_tracks=false`：传统不会新建轨迹。大符上五块红色靶面会被
  传统同时检出，如果允许新建，031 视频会多出一批假轨迹（实测 `max tracks/frame` 从 4 涨到 7）。
- 传统救援框的 `allow_new_track=false`，即使关联失败也不会在 `step()` 里反过来新建轨迹。
- `findbuff()` 的 `confidence` 是 `[0,1]` 分数（圆度 + 环结构 + 填充率），不是写死的 90；
  传统路径抛异常时只清空传统候选，不影响 YOLO 主流程。
- CSV 的 `source` 列：`0=YOLO`，`1=传统救援`，`2=两路匹配后保留的 YOLO 框`。

融合的判定顺序（`fuseDetections`）：

```text
1) 每条 YOLO 框先配一个最像的传统框（IoU 或中心距），配上只改 source=2，框仍用 YOLO
2) 剩下的传统框：先找“已确认 + 当前丢失”的轨迹做救援；否则看是否允许新建（默认否）
3) 所有要进入 fused 的框再查一遍重叠，保证同一目标最多一个观测
```

`031.mp4`（1440×1080、30 fps、5225 帧）实测：YOLO 8294 个检测，传统候选 9770 个，
融合后 `both=8140`、传统单独进入观测 114 次；轨迹 196 条，`max tracks/frame=4`，
没有出现大符五靶面同时建轨。

---

## 10. 输出说明

### 10.1 叠加视频

| 颜色 (BGR) | 框 | 标签 | 含义 |
| --- | --- | --- | --- |
| 红 `(0,0,255)` | 观测框 | `ID n obs` | 本帧真实检测（单目标版为 `obs`） |
| 黄 `(0,255,255)` | 预测框 | `ID n pred` | 纯模型外推 |
| 绿 `(0,255,0)` | 滤波框 | `ID n filt` | 融合后的最优估计；丢失时后缀 `(lost k)` |

> 画面文字是英文：OpenCV 的 `putText` 不含中文字形，画中文会变成 `???`。

### 10.2 多目标 CSV

列（共 22 列）：

```text
frame, timestamp, id, class_id, source, observed,
pred_x, pred_y, pred_w, pred_h,
filtered_x, filtered_y, filtered_w, filtered_h,
observed_x, observed_y, observed_w, observed_h,
confidence, age, hits, lost_frames
```

- 每个目标每帧一行；`hits < min_hits` 的轨迹不写，保证 CSV 与画面一致。
- 缺失的框写空字段（`,,,`），读进 pandas 是 `NaN` 而不是 0，不会和真实坐标混淆。
- `age` 是总存活帧数，`hits` 是累计命中数，`lost_frames` 是连续未匹配帧数。
- v2 变体在这 22 列之后追加 `track_quality`、`mahalanobis_sq` 两列，见
  [17. v1 / v2 实验变体](#17-v1--v2-实验变体)。

### 10.3 单目标 CSV

匀速版（15 列）：

```text
frame, timestamp, observed,
predicted_cx, predicted_cy, filtered_cx, filtered_cy,
observed_cx, observed_cy, velocity_x, velocity_y,
width, height, lost_frames, lost
```

常加速版在此基础上多两列 `acceleration_x, acceleration_y`。被误差门限拒绝的观测
`observed=0`，观测中心留空。

---

## 11. 参数速查表

全部参数及其默认值以 `yolo11.hpp` 为准，这里列常用项；v1 / v2 新增的关联参数见第 17 节。

### 11.1 检测 `Config`

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `input_size` | 640 | 网络固定输入边长，改它不会让模型自适应 |
| `conf_threshold` | 0.25 | 候选框最低分数 |
| `nms_threshold` | 0.45 | IoU 超过该值视为重复框 |
| `use_gpu` | false | 切 CUDA 后端（需要带 CUDA 的 OpenCV，否则静默回退 CPU） |

### 11.2 单目标 `KalmanParams`

| 参数 | 默认 | 含义 / 调大调小 |
| --- | --- | --- |
| `dt` | 1.0 | 时间步长（帧）；速度单位 = 像素/帧 |
| `process_noise_position` | 1e-2 | 位置过程噪声；大 → 更信观测、更跟手也更抖 |
| `process_noise_velocity` | 1e-2 | 速度过程噪声；大 → 允许速度快速变化（机动目标） |
| `measurement_noise` | 10 | 观测噪声 R（像素²）；大 → 更平滑、更滞后 |
| `max_center_distance` | 100 | 误差门限（像素），≤0 关闭 |
| `initial_position_variance` | 10 | 初始位置不确定度 |
| `initial_velocity_variance` | 10 | 初始速度不确定度（速度未知时给大一点） |
| `lost_frame_limit` | 30 | 连续丢失多少帧后判定丢失 |
| `output_fps` / `output_fourcc` | 0 / `mp4v` | ≤0 跟随源帧率 |

### 11.3 多目标 `MultiTrackParams`

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `motion_model` | `ConstantVelocity` | 匀速 / 常加速（构造时固定，运行中不能换维度） |
| `max_age` | 10 | 连续未匹配超过该值删除轨迹 |
| `min_hits` | 3 | 累计命中达到该值才绘制/写 CSV |
| `init_confidence` | 0.50 | 建轨所需最低置信度（挡误检建轨） |
| `merge_iou` | 0.55 | 重复轨迹合并阈值 |
| `draw_lost_frames` | 2 | 丢失后最多再画几帧 |
| `max_speed` | 150 | 速度上限（像素/帧），同时是关联速度门限 |
| `max_accel` | 2000 | 加速度上限（像素/帧²） |
| `coast_velocity_decay` | 0.80 | 丢失期间每帧速度衰减 |
| `max_coast_ratio` | 1.5 | 丢失期间预测中心离最近观测的最大距离（对角线倍数） |
| `min_iou` / `max_center_distance` / `max_distance_ratio` | 0.01 / 140 / 2.5 | 关联门限 |
| `iou_weight` / `distance_weight` | 0.65 / 0.35 | 代价权重（内部会归一化，比例才有意义） |
| `measurement_noise` | 8.0 | 观测噪声 R |
| `size_smoothing` | 0.65 | 尺寸指数平滑：`new = α·obs + (1-α)·old` |

### 11.4 传统识别 `TraditionalParams`

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `enabled` | true | 关掉就只跑 YOLO |
| `red` | true | true 检测红色靶，false 检测蓝色 |
| `min_score` | 0.55 | 进入融合的最低分数 |
| `new_track_score` | 0.70 | 允许新建轨迹所需的更高分数 |
| `allow_new_tracks` | false | 是否允许传统在没有轨迹先验时建轨（默认禁止） |
| `confirm_frames` | 2 | 无先验时需连续出现的帧数 |
| `min_area` / `max_area_ratio` | 120 / 0.08 | 面积下限（像素）/ 上限（占画面比例） |
| `min_circularity` / `min_fill_ratio` | 0.55 / 0.45 | 圆度、填充率门限 |
| `iou_match` / `center_match_ratio` | 0.30 / 0.50 | 与 YOLO 框配对的门限 |
| `max_size_change_ratio` | 0.35 | 救援时与轨迹尺寸的最大相对变化 |
| `memo_max_missed` | 5 | 传统候选连续多少帧不出现就遗忘 |

---

## 12. 调参指南

按“看到什么现象 → 先动哪个参数”的顺序查：

| 现象 | 先调 | 方向 | 说明 |
| --- | --- | --- | --- |
| 框抖、画框乱跳 | `measurement_noise` | ↑ | 更信模型，平滑但滞后 |
| 目标快速机动时跟不上 | `process_noise_velocity` | ↑ | 允许速度变化更快；仍不够就换常加速模型 |
| 遮挡后框“飞”出去 | `max_coast_ratio` / `coast_velocity_decay` | ↓ / ↓ | 限制外推距离、加速衰减 |
| 画面满屏幽灵框 | `init_confidence` / `min_hits` | ↑ | 提高建轨门槛；再降 `draw_lost_frames`、`max_age` |
| 同一目标出现两条 ID | `merge_iou` | ↓ | 更容易合并；但要防止把相邻目标也合并 |
| 相邻目标交叉后 ID 互换 | `iou_weight` / `max_distance_ratio` | ↑ / ↓ | 更依赖形状重合、收紧距离门限；彻底解决需加外观特征或匈牙利 |
| 镜头切换后轨迹错乱 | `max_age` / `max_speed` | ↓ / ↓ | 旧轨迹尽快过期；速度门限按 `max_speed*dt` 生效 |
| 目标出现后很久才画框 | `min_hits` | ↓ | 代价是误检更容易被画出来 |
| 检测框大小跳变 | `size_smoothing` | ↓ | 更平滑（更不跟随观测） |
| 传统误检混进来 | `min_score` / `new_track_score` | ↑ | 并确认 `allow_new_tracks=false` |

调参的基本顺序：先让**检测**稳定（置信度阈值、NMS），再调**关联门限**（别让错目标配上），
最后才动**噪声参数**（抖动 vs 滞后的折中）。一次只改一个参数，否则无法归因。

---

## 13. 模型导出（OpenCV DNN 兼容）

`best.onnx` 由 `export_onnx.py` 从训练权重导出，默认权重为
`/mnt/e/rm_runs/v4/weights/best.pt`（buff_v4 数据集，单类 `Buff`）：

```bash
python export_onnx.py --weights /mnt/e/rm_runs/v4/weights/best.pt --out best.onnx
```

导出的模型输入 `images[1,3,640,640]`，输出 `output0[1,5,8400]`，每列为 `(cx, cy, w, h, score)`，
坐标为 640 letterbox 图像上的像素值。

### 13.1 为什么要修补 strides

ultralytics 直接导出的 ONNX 在检测头末尾有一步 strides 广播乘法：

```text
dbox(1,4,8400) * strides(1,8400)
```

OpenCV 4.6 的 Eltwise(Mul) 不支持这种“少一维”的广播，forward 会抛：

```text
shape_utils.hpp:170: (-215:Assertion failed) start <= (int)shape.size() && end <= (int)shape.size()
```

而且 OpenCV 4.6 只支持 opset ≤ 12，无法靠提高 opset 绕过。

`export_onnx.py` 在导出后把 strides 常量从 `(1,8400)` 复制成 `(4,8400)`，乘法变成同形状逐元素乘，
语义完全等价（脚本用 onnxruntime 校验修补前后输出 `max|diff| = 0`），这样 OpenCV 4.6 与
onnxruntime 都能直接推理。

---

## 14. 常见问题

**Q：打开视频时出现 `GStreamer warning: cannot link elements`？**
那是 OpenCV 先用 GStreamer 后端探测、失败后回退 FFMPEG 的提示，不是错误。只要进程正常退出
（exit code 0）就可以忽略。

**Q：`forward()` 抛 `shape_utils.hpp:170` 断言？**
模型是 ultralytics 原样导出的，strides 广播没修补。按 [13.1](#131-为什么要修补-strides) 重新导出。

**Q：`readNetFromONNX` 之后 `net_.empty()` 为真？**
路径错、文件不是 ONNX、或当前 OpenCV DNN 不支持模型里的算子。先用 onnxruntime 验证模型可推理，
再确认 OpenCV 版本 ≥ 4.6。

**Q：为什么只跟踪 `class_id=0`？**
模型是单类别（`Buff`），输出只有 5 个通道；请求其它类别时 `infer()` 直接返回空。

**Q：为什么宽高不放进卡尔曼状态？**
宽高是“外观量”，与运动模型无关；而且框边抖动比中心大得多，放进状态会让速度估计变差。
现在的做法是“有观测就更新、没观测就沿用”，多目标版额外做指数平滑。

**Q：为什么镜头切换后 ID 全变了？**
这是设计行为：切换瞬间整屏位移超过 `max_speed*dt`，关联会拒绝这些候选，旧轨迹过期后在新位置
新建 ID。强行保持 ID 需要跨镜头的重识别，本工程不做。

**Q：Windows 上编译找不到 OpenCV？**
需要 Windows 版 OpenCV（官方 installer 或 vcpkg），并让 CMake 找到 `OpenCVConfig.cmake`。
本工程原本在 WSL Ubuntu 24.04 + apt 的 OpenCV 4.6.0 上跑通。

---

## 15. 已知限制

- 检测观测量只使用框中心 `(cx, cy)`，宽高独立维护，不做形状滤波。
- 多目标关联是“全局候选代价排序 + 一对一贪心”，不是 Hungarian 最优指派；目标高度密集、
  交叉遮挡严重时可能失配，可进一步替换为匈牙利算法或加入外观特征（v1/v2 变体已提供
  Hungarian 版本，见 [第 17 节](#17-v1--v2-实验变体)）。
- 传统识别只针对红/蓝圆靶这一种目标，参数（HSV 区间、圆度）换场景需要重新标定。
- 单类别模型：`infer()` 只支持 `class_id=0`。
- `MultiObjectTracker` 的运动模型维度在构造时固定，运行中不能从匀速切到常加速。
- 原有 `MultiTrackParams::process_noise` 保留为兼容字段；旧代码显式修改它时，会同时覆盖
  位置和速度过程噪声。

---

## 16. 文档与注释维护

- **代码注释**：`yolo11.cpp` / `yolo11.hpp` / `main.cpp` / `CMakeLists.txt` 已按
  “文件头 → 函数头 → 行内”三层补充中文注释，重点是“这段在做什么 / 为什么这样做 / 原理是什么”，
  例如 letterbox 为什么补 114 灰、卡尔曼为什么用 Joseph 形式、关联门限为什么分三层；
  v1 / v2（`yolo11_spv1.*` / `yolo11_spv2.*`）沿用同一套注释规范，并在改动处标了“spv1/spv2 新增”。
- **本 README**：从“接口速记”扩成“可独立阅读的工程说明”，新增流水线、检测/传统识别/卡尔曼原理、
  输出格式、参数速查、调参对照、FAQ，以及 [第 17 节](#17-v1--v2-实验变体) 的变体用法。
- **保持一致**：参数默认值以 `yolo11.hpp` 为准（变体以各自的 `yolo11_spv*.hpp` 为准）；
  改参数后请顺手更新 [11. 参数速查表](#11-参数速查表)。
  注释只解释行为，不改变行为——如果注释和代码冲突，以代码为准并修正注释。

---

## 17. v1 / v2 实验变体

仓库里有三套可运行的实现，差别**只在多目标关联**；检测层、传统识别、单目标卡尔曼、
融合规则和画框配色完全一致，参数校验沿用基线规则、并对新增字段追加了检查。

| 实现 | 源文件 | 可执行文件 | 多目标关联 | 多目标 CSV |
| --- | --- | --- | --- | --- |
| 基线 | `yolo11.hpp/.cpp` | `main` | 排序贪心 + 单阶段 | 22 列 |
| v1 | `yolo11_spv1.hpp/.cpp` | `main_v1` | Hungarian + ByteTrack 两阶段 + 置信度自适应 R | 22 列 |
| v2 | `yolo11_spv2.hpp/.cpp` | `main_v2` | v1 的全部逻辑 + Mahalanobis 门控 + 不确定度门限 + 轨迹质量分 | 24 列 |

三套头文件的公共类名和函数签名相同（`yolo11::Detector`、`yolo11::MultiObjectTracker`、
`process()` 等），所以同一份 `main.cpp` 能编译出三个可执行文件，只靠编译宏切换头文件。

### 17.1 构建

基线仍是默认目标，构建方式和以前完全一样：

```bash
cmake -S . -B build
cmake --build build -j          # 只编 main
```

v1 / v2 是 `EXCLUDE_FROM_ALL` 目标，按需单独构建：

```bash
cmake --build build --target main_v1
cmake --build build --target main_v2
```

产物为 `build/main_v1`、`build/main_v2`。默认不编变体的原因：它们是实验分支，
默认构建保持和原来一样快；某个变体改坏时也不会连累主流程，最多是显式构建那个目标失败。

`CMakeLists.txt` 里的接线：

```cmake
add_executable(main_v1 EXCLUDE_FROM_ALL main.cpp yolo11_spv1.cpp)
target_compile_definitions(main_v1 PRIVATE YOLO11_VARIANT_SPV1)
target_link_libraries(main_v1 PRIVATE ${OpenCV_LIBS})
```

`main.cpp` 顶部按宏选择头文件，并给出变体专属的默认输出名：

```cpp
#if defined(YOLO11_VARIANT_SPV1)
#include "yolo11_spv1.hpp"
#define YOLO11_OUTPUT_VIDEO "../output/a_v1.mp4"
#define YOLO11_OUTPUT_CSV   "../output/tracks_v1.csv"
#elif defined(YOLO11_VARIANT_SPV2)
#include "yolo11_spv2.hpp"
#define YOLO11_OUTPUT_VIDEO "../output/a_v2.mp4"
#define YOLO11_OUTPUT_CSV   "../output/tracks_v2.csv"
#else
#include "yolo11.hpp"
#define YOLO11_OUTPUT_VIDEO "../output/a.mp4"
#define YOLO11_OUTPUT_CSV   "../output/tracks.csv"
#endif
```

不用 CMake 时也可以直接编译（单行版）：

```bash
g++ -std=c++17 -O2 -DYOLO11_VARIANT_SPV1 main.cpp yolo11_spv1.cpp -o main_v1 \
    $(pkg-config --cflags --libs opencv4)
```

> 注意：一个可执行文件只能 include 一个实现。不要把 `yolo11.cpp` 和
> `yolo11_spv1.cpp` 同时编进去，否则 `yolo11` 命名空间里的同名类会重复定义。

### 17.2 运行

```bash
cd build
./main        # 基线：../output/a.mp4、../output/tracks.csv
./main_v1     # v1  ：../output/a_v1.mp4、../output/tracks_v1.csv
./main_v2     # v2  ：../output/a_v2.mp4、../output/tracks_v2.csv
```

- 三个程序读同一个输入（`main.cpp` 里的 `/mnt/e/yolo11C++/031.mp4`）和同一个模型；
  换视频/模型仍然改 `main.cpp` 里那两处路径。
- 输出默认分开命名，跑完三遍可以直接并排比较视频和 CSV，不会互相覆盖。
- 控制台摘要（`processed frames / yolo detections / fused both / traditional-only`）
  三个版本格式相同，可以直接对比。
- 跟踪参数在 `main.cpp` 里设置，三个目标共用。想给某个变体单独调参，用
  `#if defined(YOLO11_VARIANT_SPV1)` 包住一段覆盖代码，或复制一份 `main.cpp`。

### 17.3 v1：Hungarian + 两阶段 + 自适应 R

v1 只改多目标层：`step()` 里的关联方式，以及 `update()` 写进卡尔曼滤波器的观测噪声。
代价公式和三层门限仍与基线相同：

```text
cost = (w_iou*(1-IoU) + w_dist*min(1, d/gate)) / (w_iou + w_dist)
门限：d > max_speed*dt  或  d > gate  或  (IoU < min_iou 且 d > 0.45*gate)  → 拒绝
```

新增参数（默认值即推荐起点）：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `use_hungarian` | true | 对代价矩阵做全局最优一对一指派；false 退回基线式“按代价排序 + 贪心” |
| `enable_two_stage` | true | 高分/低分两轮匹配 |
| `high_confidence_threshold` | 0.50 | 第一阶段下限，收 `[high, 1]` |
| `low_confidence_threshold` | 0.10 | 第二阶段下限，收 `[low, high)`，须 ≤ `high` |
| `adaptive_measurement_noise` | true | 按检测置信度放大观测噪声 R |
| `confidence_noise_power` | 2.0 | 惩罚指数：`(1/conf)^p` |
| `min_measurement_noise_scale` / `max_measurement_noise_scale` | 0.75 / 8.0 | R 缩放倍数的上下限 |

行为要点：

1. **两阶段**：第一阶段用 `conf >= high` 的框和全部轨迹做全局指派；第二阶段只拿
   `low <= conf < high` 的框，去救第一阶段没配上的轨迹。低分框永远不能建新轨
   （建轨门槛是 `max(init_confidence, high)`，默认 0.50）。
2. **默认阈值不会丢检测**：`Detector` 已经在 `conf_threshold=0.25` 处过滤过一次，
   默认 `low=0.10` 比它还低，所以第二阶段实际收的是 `0.25 ~ 0.50` 的中低分框。
   只有把 `low` 抬到 0.25 以上，才会真正把一部分低分框排除在关联之外。
3. **自适应 R**：`R_eff = R * clamp((1/max(0.05, conf))^power, min_scale, max_scale)`。
   低分框只能轻微修正状态（救回轨迹、不把中心拽偏），高分框照常更新。
4. **Hungarian** 的“未匹配”用虚拟列实现，禁用边填 `1e6` 哨兵，不会被选走；
   关闭 Hungarian 时，贪心按 `(cost, row, col)` 排序，同样输入的结果可复现。
5. 输出（视频配色、标签、CSV 22 列）与基线完全一致。

想单独验证“关联策略本身”的收益，可以先把 v1 退化成基线行为做 A/B：

```cpp
params.use_hungarian = false;
params.enable_two_stage = true;          // 保持 true
params.high_confidence_threshold = 0.0f; // 第一阶段收全部检测
params.low_confidence_threshold = 0.0f;
params.adaptive_measurement_noise = false;
```

此时与基线的贪心行为等价，唯一差别是代价并列时的排序更确定
（基线只按 `cost` 排序，`std::sort` 不稳定；v1 补了行列号作为次级键）。

### 17.4 v2：在 v1 上再加马氏门控与质量分

v2 = v1 的全部开关 + 马氏/质量相关的逻辑，并把代价函数换成了固定权重的三项式：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `enable_mahalanobis_gate` | true | 用平方马氏距离做第四层门控 |
| `mahalanobis_gate_sq` | 9.21 | 2 自由度卡方 99% 分位（填的是平方值，不是 3.03） |
| `uncertainty_gate_scale` | 2.0 | 预测越不确定、空间门限放大越多：`gate *= 1 + scale*sigma/对角线` |
| `max_uncertainty_gate_scale` | 3.0 | 空间门限放大倍率上限，防止幽灵轨迹吸附远处误检 |
| `quality_hit_gain` | 0.18 | 命中时质量分的提升速度 |
| `quality_miss_decay` | 0.88 | 每帧未命中时质量分的衰减系数 |
| `min_track_quality` | 0.05 | 低于该质量分的轨迹不绘制 |

行为要点：

1. **代价公式**（固定权重；`iou_weight` / `distance_weight` 不再参与计算，只保留作参数校验）：

   ```text
   cost = 0.55*(1-IoU) + 0.25*min(1, d/gate) + 0.20*min(1, d²/9.21)
   ```

   关闭 `enable_mahalanobis_gate` 时马氏项退化为归一化距离，等价于
   `0.55*(1-IoU) + 0.45*d/gate`。
2. **马氏门控**：`d² = dx²/(P00+R) + dy²/(P11+R)`，`P` 是本帧预测协方差，`R` 按置信度自适应。
   `d² > mahalanobis_gate_sq` 的观测直接拒绝——欧氏距离够近、但在统计上太“意外”的框
   也不会被用来更新轨迹（默认只接受落在预测椭圆 99% 区域内的观测）。
3. **动态空间门限**：丢失过的轨迹 `P` 变大，门限按 `sigma/对角线` 放宽，便于重新捕获；
   `max_uncertainty_gate_scale` 封顶，防止已经飘远的轨迹把附近误检吸过来。
4. **质量分**：建轨时 `q = conf`；每帧预测时先衰减 `q *= 0.88`，命中后再按
   `q += 0.18*(1-q)*conf` 提升。`draw()` 跳过 `q < min_track_quality` 的轨迹，
   但 CSV 仍按 `hits >= min_hits` 记录——质量分只影响画面，不影响关联、删除和日志。
5. **新增两列诊断量**：`track_quality`（质量分 0~1）和 `mahalanobis_sq`
   （最近一次匹配的平方马氏距离；`observed=0` 的帧沿用上一次的值）。

### 17.5 v2 的 CSV（24 列）

在基线 22 列之后**追加**两列，老脚本按列名或前缀取列不会被顶位：

```text
frame, timestamp, id, class_id, source, observed,
pred_x, pred_y, pred_w, pred_h,
filtered_x, filtered_y, filtered_w, filtered_h,
observed_x, observed_y, observed_w, observed_h,
confidence, age, hits, lost_frames, track_quality, mahalanobis_sq
```

`MultiTrackView` 也相应多出 `quality` 与 `association_distance` 两个字段；
v1 的视图和 CSV 与基线一致（22 列）。

### 17.6 怎么选

| 场景 | 建议 |
| --- | --- |
| 目标稀疏（≤4 个）、只求稳定 | 基线：代码最短、参数最熟，行为已有实测基线 |
| 目标交叉/遮挡后 ID 频繁互换 | v1：Hungarian 避免贪心“先到先得”，两阶段让低分框救轨迹 |
| 误检多、需要判断“这条轨迹可不可信” | v2：马氏门控 + `track_quality` / `mahalanobis_sq` 可直接做离线评估 |
| 想做 A/B 对照 | 三个目标默认输出分开，跑完对比 CSV 的 ID 切换次数与 `lost_frames` 分布 |

变体专属的调参对照：

| 现象 | 变体 | 先调 | 方向 |
| --- | --- | --- | --- |
| 低分框救不回轨迹 | v1/v2 | `low_confidence_threshold` / `high_confidence_threshold` | ↓ / ↓ |
| 误检被救回、框被拉偏 | v1/v2 | `low_confidence_threshold`、`confidence_noise_power`、`max_measurement_noise_scale` | ↑ / ↑ / ↑ |
| 遮挡后重新捕获慢 | v2 | `mahalanobis_gate_sq`、`uncertainty_gate_scale` | ↑ / ↑ |
| 幽灵轨迹吸附远处误检 | v2 | `max_uncertainty_gate_scale`、`quality_miss_decay`、`min_track_quality` | ↓ / ↓ / ↑ |
| CSV 有轨迹但画面不画 | v2 | `min_track_quality` | ↓ |

### 17.7 边界与坑

- v2 的 `iou_weight` / `distance_weight` 目前**只参与构造校验**，改它们不会改变代价；
  想调关联偏好只能改 `step()` 里的固定权重（0.55 / 0.25 / 0.20）。
- 质量分不参与关联、删除、合并，只影响绘制；`hits` 够的轨迹即使质量分很低也会写 CSV。
- v1/v2 的头文件给 `MultiTrackParams` / `MultiTrackView` 加了字段，因此不要把不同变体的
  头文件和 `.cpp` 混进同一个可执行文件。
- 三个版本都不做外观特征/ReID，镜头切换后仍然会换 ID；`max_age`、`max_speed` 仍按
  基线的方式控制“镜头切换后旧轨迹多快过期”。
- 参数校验比基线更严：`0 <= low <= high <= 1`、质量参数在 `[0,1]`、马氏门限为正，
  越界会直接抛 `std::invalid_argument`（见各变体构造函数）。
