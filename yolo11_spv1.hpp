// ============================================================================
// yolo11_spv1.hpp —— 公共接口与全部可调参数（多目标关联实验变体 spv1）
//
// 本头文件只放“契约”，实现全在 yolo11_spv1.cpp：
//   - Detection / Config / Detector        : 检测层的输入输出
//   - TraditionalParams / findbuff         : 传统红蓝圆靶识别（YOLO 的兜底）
//   - KalmanParams / KalmanState / KalmanTracker            : 单目标匀速
//   - KalmanAccelParams / KalmanAccelTracker                : 单目标常加速
//   - MultiTrackParams / MultiTrackView / MultiObjectTracker: 多目标
//
// 三套结构体的分工是固定的，读代码时按这个规则找变量即可：
//   xxxParams : 可调参数（构造时校验一次，运行期只读）
//   xxxState  : 随时间变化的运行时状态（只有 tracker 会改）
//   xxxView   : 对外导出的只读快照（避免外部直接动 KalmanFilter）
//
// 单位约定（所有几何量都在“原图坐标系”下）：
//   像素 / 像素每帧（速度）/ 像素每帧²（加速度，dt 默认 1 帧）。
//   噪声参数是方差或协方差，不是标准差；量纲为像素² 或 (像素/帧)²。
//   颜色参数是 OpenCV 的 BGR 顺序，不是 RGB。
//
// 与 yolo11.hpp 的差异：本版本只改了多目标关联，单目标卡尔曼部分保持一致。
// 多目标关联在原来的单阶段匹配之外，额外支持：
//   - Hungarian 全局指派（use_hungarian），避免贪心匹配“先到先得”；
//   - ByteTrack 式两阶段匹配（enable_two_stage + 高/低分阈值）：
//     高分检测先配，低分检测只补救已有轨迹、不建新轨；
//   - 按检测置信度自适应观测噪声 R（adaptive_measurement_noise + power/scale）：
//     低分检测放大 R，削弱它对状态的影响。
// ============================================================================
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolo11
{

// 检测结果结构体
struct Detection
{
    // 原图坐标下的检测框
    cv::Rect2f rect;

    // 中心点
    cv::Point2f center;

    // 四个顶点（顺时针：左上 -> 左下 -> 右下 -> 右上）
    cv::Point2f LT{ 0.0f, 0.0f };  // 左上 (Left-Top)
    cv::Point2f LB{ 0.0f, 0.0f };  // 左下 (Left-Bottom)
    cv::Point2f RB{ 0.0f, 0.0f };  // 右下 (Right-Bottom)
    cv::Point2f RT{ 0.0f, 0.0f };  // 右上 (Right-Top)

    // 类别编号
    int class_id = 0;

    // 宽高
    float width = 0.0f;
    float height = 0.0f;

    // 置信度，范围 [0, 1]
    float confidence = 0.0f;

    // 检测来源：0=YOLO，1=传统识别，2=两路匹配后保留的 YOLO 框。
    // 仅用于日志/调试，不参与跟踪代价计算。
    int source = 0;

    // 是否允许该检测在 step() 中新建轨迹。
    // YOLO 默认为 true；传统救援候选为 false，防止救援框在关联失败时
    // 反过来新建一条假轨迹。
    bool allow_new_track = true;
};

// 传统红/蓝圆靶检测参数。
// 传统识别只作为 YOLO 的补充：YOLO 有框时优先使用 YOLO，
// 传统候选只在 YOLO 漏检、且通过分数/形状/时序门限时进入跟踪。
struct TraditionalParams
{
    // 是否启用传统识别。关闭后 process() 只跑 YOLO。
    bool enabled = true;

    // true 检测红色靶面，false 检测蓝色靶面。
    bool red = true;

    // 传统候选进入融合的最低分数，以及新建轨迹所需的更高分数。
    float min_score = 0.55f;        // 进入救援/融合的最低分
    float new_track_score = 0.70f;  // 允许传统新建轨迹所需的更高分

    // 是否允许传统候选在没有轨迹先验时新建轨迹。
    // 默认关闭：传统只救援 YOLO 已建立的轨迹，避免把大符上所有红色靶面
    // 都建成新目标（031 视频里会一次多出四个假轨迹）。
    bool allow_new_tracks = false;

    // 无轨迹先验时，传统候选必须连续出现的帧数。
    // 已有丢失轨迹附近的救援只需 1 帧（轨迹本身提供了先验）。
    // 仅在 allow_new_tracks=true 时生效。
    int confirm_frames = 2;

    // 面积范围：下限为绝对像素，上限按整帧面积比例，避免分辨率/距离变化。
    float min_area = 120.0f;       // 最小面积（像素²）
    float max_area_ratio = 0.08f;  // 最大面积占整帧比例

    // 形状门限：外轮廓圆度、轮廓面积占最小外接圆面积的比例。
    float min_circularity = 0.55f; // 圆度下限：4πA/P²，完美圆为 1
    float min_fill_ratio = 0.45f;  // 填充率下限：A/(πr²)，排除碎弧

    // 与 YOLO 框匹配的门限：IoU 达到该值，或中心距小于
    // center_match_ratio * 两框对角线的较大值，视为同一目标。
    float iou_match = 0.30f;          // IoU 配对门限
    float center_match_ratio = 0.50f; // 中心距配对门限（相对对角线）

    // 救援时传统框与轨迹上一帧尺寸的最大相对变化。
    float max_size_change_ratio = 0.35f;

    // 传统候选连续多少帧没再出现就从时序记忆中删除。
    int memo_max_missed = 5;
};

// 传统红/蓝圆靶检测。
// 返回的 Detection::confidence 是 [0,1] 分数，source=1。
// 该函数只做单帧候选生成；是否进入跟踪由 MultiObjectTracker 的融合逻辑决定。
std::vector<Detection> findbuff(const cv::Mat& frame, const TraditionalParams& params = TraditionalParams());

// 模型配置
// 只影响“怎么推理”，改完必须与模型导出参数一致。
// 这里的字段在 Detector 构造时校验，非法值直接抛异常。
struct Config
{
    // 网络固定输入边长（模型导出时写死的 640）。
    // 改它不会让模型自适应，只会让 letterbox 的输出和网络期望不一致。
    int input_size = 640;

    // 单类别模型的类别置信度阈值，范围 [0, 1]。
    // 低于该值的候选在解码阶段直接丢弃，不进入 NMS。
    float conf_threshold = 0.25f;

    // NMS IoU 阈值，范围 [0, 1]。
    // 两个框重叠超过该值视为同一目标的重复检测，只保留分数高的一个。
    float nms_threshold = 0.45f;

    // true 时切 CUDA 后端，false 走 CPU。
    // 需要带 CUDA 的 OpenCV 构建；没有 CUDA 时会被静默忽略并回退 CPU。
    bool use_gpu = false;
};

class Detector
{
public:

    // 构造函数
    // 校验 Config -> 读入 ONNX -> 选推理后端。
    // 模型加载失败、配置非法都会在这里抛异常（而不是等到 forward 才炸）。
    explicit Detector(const std::string& model_path, const Config& config = Config());

    // 析构用默认实现：cv::dnn::Net 自己管理底层资源。
    ~Detector() = default;

    // 禁止复制
    // cv::dnn::Net 体积大且内部持有模型资源，复制代价高、语义也不清晰。
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // 允许移动
    // 转移 Net 的所有权，不复制模型。
    Detector(Detector&&) = default;
    Detector& operator=(Detector&&) = default;

    // 简单检测接口：只返回框，适合不需要类别/顶点/来源的调用方。
    // 单类别模型只支持 class_id=0；其他类别编号返回空结果。
    // 支持 ONNX 输出 [1, 5, N] 和 [1, N, 5] 两种布局。
    // 检测接口返回原图坐标下的 Rect2f。
    std::vector<cv::Rect2f> detect(const cv::Mat& frame, int class_id, int originalWidth, int originalHeight);

    // 完整检测接口
    // 返回中心点、四顶点、类别、宽高等信息
    // originalWidth/originalHeight 是“希望结果落在哪个坐标系”，
    // 与 frame 尺寸不同时会自动做一次比例换算。
    std::vector<Detection> detectDetailed(const cv::Mat& frame, int class_id, int originalWidth, int originalHeight);

private:

    // 已加载的 ONNX 网络。构造后只读，逐帧复用（不要每帧 readNetFromONNX）。
    cv::dnn::Net net_;

    // 构造时固化的配置（输入尺寸、阈值、后端选择）。
    Config config_;

    // 真正的推理实现：letterbox -> blob -> forward -> 解码 -> 反变换 -> NMS。
    // 两个公有接口都只是它的薄封装。
    std::vector<Detection> infer(const cv::Mat& frame, int class_id, int originalWidth, int originalHeight);
};

// ============================================================
// 2D 匀速卡尔曼跟踪
//
// 状态模型：x = [cx, cy, vx, vy]^T
//   cx, cy : 目标中心点（像素）
//   vx, vy : 中心点速度（像素/帧）
//
// 卡尔曼的两个方程（代码里分别对应 predict() 和 correct()）：
//   预测：x' = F*x
//         P' = F*P*F^T + Q
//   更新：y  = z - H*x'          // 新息：观测 - 预测
//         S  = H*P'*H^T + R      // 新息协方差
//         K  = P'*H^T*S^-1       // 卡尔曼增益
//         x  = x' + K*y
//         P  = (I-K*H)*P'*(I-K*H)^T + K*R*K^T   // Joseph 形式
// P 是“估计的不确定度”，Q 是模型噪声，R 是观测噪声；K 由 P 与 R 的相对大小
// 自动决定“更信模型还是更信观测”，这就是卡尔曼不需要手工调平滑系数的原因。
//
// 只对中心点做匀速卡尔曼预测，目标宽高不进入状态向量：
//   - 当前帧有观测时，用 Detection 的宽高更新边长；
//   - 当前帧无观测时，沿用最近一次观测到的边长。
//
// 代码分工：
//   KalmanParams  : 所有可调参数（噪声、绘制、输出）
//   KalmanState   : 所有运行时状态（x、P、丢失计数、绘制框）
//   KalmanTracker : 只负责推进/改变 KalmanState
// ============================================================

// 卡尔曼参数：所有可调参数集中在这里。
// 噪声参数的单位与状态单位一致：位置为像素，速度为像素/帧。
struct KalmanParams
{
    // 匀速模型的时间步长（单位：帧）。
    // 一般保持 1.0，表示每来一帧预测一次；
    // 此时速度 vx、vy 的单位就是“像素/帧”。
    float dt = 1.0f;

    // 位置过程噪声方差。
    // 越大越不相信匀速假设，滤波器对观测的响应越快，但结果越抖。
    float process_noise = 1e-2f; // 旧版兼容参数；非默认时同时覆盖位置/速度过程噪声
    float process_noise_position = 1e-2f;

    // 速度过程噪声方差。
    // 越大表示允许目标速度变化得越快，适合机动目标。
    float process_noise_velocity = 1e-2f;

    // 观测噪声方差（像素^2）。
    // 越大越不相信检测框中心，滤波结果越平滑，但滞后也越明显。
    float measurement_noise = 10.0f;

    // 观测误差门限（像素）。
    // 预测中心与观测中心的距离超过该值时，认为检测是误检/异常观测，
    // correct() 会跳过本次修正。<= 0 表示不启用门限。
    // 只在滤波器已初始化时生效；首次检测没有预测中心，不参与门限判断。
    // 被拒绝的观测按“无观测”处理：不更新边长、不重置丢失计数。
    float max_center_distance = 100.0f;

    // 初始位置协方差。
    // 表示第一次观测时对中心点的不确定程度。
    float initial_position_variance = 10.0f;

    // 初始速度协方差。
    // 目标初始速度未知时保留一定不确定度；
    // 若确定目标从静止开始，可以适当调小。
    float initial_velocity_variance = 10.0f;

    // 连续丢失多少帧后判定目标丢失。
    // 初始化/重置时会复制到 KalmanState::lost_frame_limit，
    // 运行期以状态结构体中的值为准。
    int lost_frame_limit = 30;

    // ---- 绘制参数 ----
    // 注意：cv::Scalar 的颜色顺序是 BGR，不是 RGB。
    cv::Scalar predicted_color{ 0, 255, 255 };  // 预测框：黄色
    cv::Scalar filtered_color{ 0, 255, 0 };     // 滤波框：绿色
    cv::Scalar observed_color{ 0, 0, 255 };     // 观测框：红色

    // 框线宽和线型（cv::LINE_AA 表示抗锯齿线）。
    int thickness = 2;
    int line_type = cv::LINE_AA;

    // 标签文字参数，标签内容为 pred / filt / obs。
    double label_font_scale = 0.5;
    int label_thickness = 1;
    bool draw_labels = true;   // 是否在框旁边绘制标签
    bool draw_status = true;   // 是否在左上角绘制丢失状态

    // ---- 输出视频参数 ----
    // <= 0 时沿用源视频帧率；> 0 时使用指定帧率。
    double output_fps = 0.0;

    // 输出视频编码 FourCC，必须是 4 个字符，例如 "mp4v"、"XVID"。
    std::string output_fourcc = "mp4v";
};

// 卡尔曼状态：所有随时间变化的数据集中在这里。
// 匀速/变速跟踪器共用该结构体，外部可通过 state() 只读访问。
struct KalmanState
{
    // 状态向量，类型 CV_32F，列向量。
    // 匀速模型：[cx, cy, vx, vy]^T，4x1；
    // 变速模型：[cx, cy, vx, vy, ax, ay]^T，6x1。
    // 未初始化时为空矩阵；判定丢失后会被释放。
    cv::Mat x;

    // 状态协方差矩阵，类型 CV_32F，维度与 x 对应（4x4 或 6x6）。
    // 对角线元素分别对应状态向量各分量的方差。
    cv::Mat P;

    // 是否已经由观测完成初始化。
    // false 时不会预测，也不会绘制预测框/滤波框。
    bool initialized = false;

    // 当前帧是否有通过误差门限的有效观测。
    // 该标记决定是否绘制观测框、是否更新边长和丢失计数。
    // correct() 发现距离超过门限时会把该标记置为 false。
    bool observed = false;

    // 当前连续丢失帧数。
    // 每收到一次观测清零；无观测且已初始化时累加。
    int lost_frames = 0;

    // 丢失帧上限：连续丢失超过该值后判定目标丢失。
    // 运行期以本字段为准，由 KalmanParams::lost_frame_limit 复制而来。
    int lost_frame_limit = 30;

    // 是否已经判定目标丢失。
    // 丢失后 initialized 变为 false，等待目标重新出现后重新初始化。
    bool lost = false;

    // 最近一次观测到的目标边长。
    // 卡尔曼只预测中心点，宽高不参与滤波：
    // 有观测时用 Detection 的宽高更新，无观测时沿用这里的值。
    float width = 0.0f;    // 最近一次观测的宽（像素）
    float height = 0.0f;   // 最近一次观测的高（像素）
    bool has_size = false; // 是否已经拿到过有效边长（决定 CSV 里宽高列是否留空）

    // 当前帧用于绘制的三个框，均由“中心点 + 边长”生成。
    // predicted_rect：预测阶段的中心 + 上一次观测的边长
    //                 （首次观测时用观测框占位）
    // filtered_rect ：校正/预测后的中心 + 当前（或上一次）观测的边长
    // observed_rect ：当前检测框，仅在 observed == true 时有效
    cv::Rect2f predicted_rect;  // 预测框（黄）
    cv::Rect2f filtered_rect;   // 滤波框（绿）
    cv::Rect2f observed_rect;   // 观测框（红）
};

// 2D 匀速卡尔曼跟踪器。
//
// 设计约定：
//   - 所有参数放在 KalmanParams；
//   - 所有运行时状态放在 KalmanState；
//   - 类只持有这两个结构体，并提供推进状态的方法；
//   - 不在类中散落 last_center、last_box 等影子状态。
//
// 用法示例：
//   yolo11::Detector detector("yolo11n.onnx");
//   yolo11::KalmanTracker tracker;
//   tracker.process("in.mp4", "out.mp4", "track.csv", detector, 0);
class KalmanTracker
{
public:

    // 构造函数：校验参数并初始化状态。
    explicit KalmanTracker(const KalmanParams& params = KalmanParams());

    // 完整流程：逐帧检测 -> 卡尔曼预测/更新 -> 叠加绘制 -> 输出视频与 CSV。
    // 参数：
    //   input_video  : 输入视频路径
    //   output_video : 输出叠加视频路径
    //   csv_log      : CSV 日志路径
    //   detector     : 已构造好的 YOLO11 检测器（本类不持有模型资源）
    //   class_id     : 要跟踪的类别编号
    //
    // CSV 列：
    //   frame,timestamp,observed,
    //   predicted_cx,predicted_cy,
    //   filtered_cx,filtered_cy,
    //   observed_cx,observed_cy,
    //   velocity_x,velocity_y,
    //   width,height,
    //   lost_frames,lost
    //
    // 连续丢失超过 lost_frame_limit 后停止绘制预测/滤波框，
    // 直到重新检测到目标并重新初始化滤波器。
    void process(const std::string& input_video, const std::string& output_video, const std::string& csv_log, Detector& detector, int class_id = 0);

    // 单帧推进：执行预测、观测选择、误差门限、校正和绘制框计算。
    // detections 为当前帧检测结果；没有检测时传空 vector 即可。
    void step(const std::vector<Detection>& detections);

    // 根据 KalmanState 中保存的框绘制到 frame 上。
    // 应在 step() 之后调用。
    void draw(cv::Mat& frame) const;

    // 清空状态，重新开始跟踪；丢失上限会从 KalmanParams 重新复制。
    void reset();

    // 只读访问当前状态，便于外部记录或调试。
    const KalmanState& state() const { return state_; }

    // 只读访问构造时使用的参数。
    const KalmanParams& params() const { return params_; }

private:

    // 执行匀速预测：x = F*x，P = F*P*F^T + Q。
    void predict();

    // 用首次观测（或丢失后的首次观测）初始化 x 和 P。
    void initialize(const Detection& detection);

    // 用当前观测做卡尔曼校正，更新 x 和 P。
    // 若观测中心与预测中心的距离超过 KalmanParams::max_center_distance，
    // 则跳过本次修正，并把 state.observed 置为 false。
    void correct(const Detection& detection);

    // 从当前帧的多个检测中选出一个观测：
    // 已初始化时选离预测中心最近的；未初始化时选置信度最高的。
    const Detection* selectObservation(const std::vector<Detection>& detections) const;

    // 生成 CSV 表头。
    std::string csvHeader() const;

    // 将当前状态格式化为一行 CSV。
    std::string csvRow(long long frame_index, double timestamp) const;

    KalmanParams params_;   // 只读参数（构造后不再修改）

    KalmanState state_;     // 唯一的运行时状态
};

// ============================================================
// 2D 变速卡尔曼跟踪（常加速度模型）
//
// 状态模型：x = [cx, cy, vx, vy, ax, ay]^T
//   cx, cy : 目标中心点（像素）
//   vx, vy : 中心点速度（像素/帧）
//   ax, ay : 中心点加速度（像素/帧^2）
//
// 状态转移采用常加速度模型：
//   cx' = cx + dt*vx + 0.5*dt^2*ax
//   cy' = cy + dt*vy + 0.5*dt^2*ay
//   vx' = vx + dt*ax
//   vy' = vy + dt*ay
//   ax' = ax
//   ay' = ay
//
// 与匀速模型相同：
//   - 只预测中心点，宽高不进入状态向量；
//   - 有观测时更新边长，无观测时沿用上次边长；
//   - 有观测时校正，无观测时按丢失帧处理；
//   - 参数/状态全部放在结构体中，类只负责推进状态。
// ============================================================

// 2D 常加速度卡尔曼参数。
// 复用 KalmanParams 中的 dt、位置/速度过程噪声、观测噪声、
// 误差门限、丢失帧上限、绘制和输出参数；
// 这里只增加加速度相关的两个参数。
struct KalmanAccelParams : public KalmanParams
{
    // 加速度过程噪声方差。
    // 越大表示允许目标加速度变化得越快，对机动目标响应越快，但结果越抖。
    float process_noise_acceleration = 1e-2f;

    // 初始加速度协方差。
    // 第一次观测时目标加速度未知，保留一定不确定度。
    float initial_acceleration_variance = 10.0f;
};

// 2D 常加速度卡尔曼跟踪器（单目标）。
// 使用方式与 KalmanTracker 相同：
//   yolo11::Detector detector("yolo11n.onnx");
//   yolo11::KalmanAccelTracker tracker;
//   tracker.process("in.mp4", "out.mp4", "track.csv", detector, 0);
class KalmanAccelTracker
{
public:

    // 构造函数：校验参数并初始化状态。
    explicit KalmanAccelTracker(const KalmanAccelParams& params = KalmanAccelParams());

    // 完整流程：逐帧检测 -> 卡尔曼预测/更新 -> 叠加绘制 -> 输出视频与 CSV。
    // CSV 列：
    //   frame,timestamp,observed,
    //   predicted_cx,predicted_cy,
    //   filtered_cx,filtered_cy,
    //   observed_cx,observed_cy,
    //   velocity_x,velocity_y,
    //   acceleration_x,acceleration_y,
    //   width,height,
    //   lost_frames,lost
    //
    // 连续丢失超过 lost_frame_limit 后停止绘制预测/滤波框，
    // 直到重新检测到目标并重新初始化滤波器。
    void process(const std::string& input_video, const std::string& output_video, const std::string& csv_log, Detector& detector, int class_id = 0);

    // 单帧推进：执行预测、观测选择、误差门限、校正和绘制框计算。
    void step(const std::vector<Detection>& detections);

    // 根据 KalmanState 中保存的框绘制到 frame 上。
    void draw(cv::Mat& frame) const;

    // 清空状态，重新开始跟踪。
    void reset();

    // 只读访问当前状态（x/P 为 6x1、6x6）。
    const KalmanState& state() const { return state_; }

    // 只读访问构造时使用的参数。
    const KalmanAccelParams& params() const { return params_; }

private:

    // 常加速度预测：x = F*x，P = F*P*F^T + Q。
    void predict();

    // 用首次观测（或丢失后的首次观测）初始化 6 维状态。
    void initialize(const Detection& detection);

    // 用当前观测做卡尔曼校正，更新 x 和 P。
    // 若观测中心与预测中心的距离超过 KalmanParams::max_center_distance，
    // 则跳过本次修正，并把 state.observed 置为 false。
    void correct(const Detection& detection);

    // 从当前帧的多个检测中选出一个观测：
    // 已初始化时选离预测中心最近的；未初始化时选置信度最高的。
    const Detection* selectObservation(const std::vector<Detection>& detections) const;

    // 生成 CSV 表头。
    std::string csvHeader() const;

    // 将当前状态格式化为一行 CSV。
    std::string csvRow(long long frame_index, double timestamp) const;

    KalmanAccelParams params_;   // 只读参数（构造后不再修改）

    KalmanState state_;          // 唯一的运行时状态（x/P 为 6 维）
};


// ============================================================
// 统一运动模型：单目标与多目标均可选择匀速或常加速。
// ============================================================
enum class MotionModel
{
    ConstantVelocity,       // 匀速：状态 [cx, cy, vx, vy]^T
    ConstantAcceleration    // 常加速：状态 [cx, cy, vx, vy, ax, ay]^T
};

// 多目标跟踪参数：算法、噪声、关联、生命周期、绘图和视频输出集中管理。
struct MultiTrackParams
{
    MotionModel motion_model = MotionModel::ConstantVelocity; // 多目标运动模型
    float dt = 1.0f;                    // 帧间隔；匀速模型速度单位为像素/帧
    int max_age = 10;                   // 未匹配超过该帧数则删除轨迹
    int min_hits = 3;                   // 达到该命中次数后才绘制轨迹 / 写入 CSV

    // ---- 轨迹确认、去重与防漂移 ----
    // 只有置信度不低于该值的检测才允许新建轨迹，抑制低置信误检造成的“幽灵轨迹”。
    float init_confidence = 0.50f;
    // 两条轨迹框的 IoU 超过该值时视为同一目标的重复轨迹，保留 ID 更小（创建更早）的一条。
    float merge_iou = 0.55f;
    // 轨迹丢失后最多再绘制几帧；(0 表示丢失后立即停止绘制)
    int draw_lost_frames = 2;
    // 状态速度/加速度上限（像素/帧、像素/帧^2），防止滤波器在丢失期间发散；
    // 速度上限同时用作关联门限：一帧内中心位移超过该值视为镜头切换或误检。
    float max_speed = 150.0f;   // 速度模长上限（像素/帧）
    float max_accel = 2000.0f;  // 加速度模长上限（像素/帧²）
    // 丢失期间每帧的速度衰减系数（<1 使外推逐渐停住）。
    float coast_velocity_decay = 0.80f;
    // 丢失期间预测中心距最近一次观测中心的最大距离，按目标框对角线倍数计。
    float max_coast_ratio = 1.5f;

    // 关联门限：先用运动距离排除不可能匹配，再综合 IoU 与归一化距离评分。
    // 评分公式（越小越可能是同一个目标，见 step()）：
    //   cost = (iou_weight*(1-IoU) + distance_weight*(distance/gate))
    //          / (iou_weight + distance_weight)
    // 除以权重和把 cost 归一到 [0,1]，这样改权重只改变偏好，不改变数值范围。
    // 速度门限额外用 max_speed*dt：一帧内物理上不可能移动那么远。
    float min_iou = 0.01f;              // IoU 低于该值且中心也不近时拒绝匹配
    float max_center_distance = 140.0f; // 中心距门限的固定下限（像素）
    float max_distance_ratio = 2.5f;    // 中心距门限 = 该值 * 目标框对角线
    float iou_weight = 0.65f;           // 代价中 IoU 项的权重
    float distance_weight = 0.35f;      // 代价中归一化中心距的权重

    // 高阶关联：Hungarian 全局指派 + ByteTrack 双阶段匹配。
    // 高分检测先匹配；低分检测只允许补充已有轨迹，不允许创建新轨迹。
    // use_hungarian：true 时对代价矩阵做全局最优指派，避免贪心“先到先得”
    //   把高分检测配给错误轨迹；false 时退回逐对贪心匹配。
    // enable_two_stage：true 时按高分/低分两轮匹配：
    //   第一轮只用 conf >= high_confidence_threshold 的检测做全局指派；
    //   第二轮用 low <= conf < high 的检测，只补救第一轮后仍未匹配的轨迹。
    // enable_two_stage=false 时 low_confidence_threshold 不生效，
    // 只跑 high 以上的单阶段匹配；低分检测仍然不能新建轨迹。
    // 两个阈值都在 [0,1]，且必须满足 low <= high（构造时校验，否则抛异常）。
    //   high 调高 -> 更多检测被推到第二轮，主关联更干净但更依赖低分补救；
    //   low  调高 -> 第二轮可用检测变少，误检更难混进来，但漏检时更容易断轨。
    // 与 init_confidence 的关系：新建轨迹的门限取
    //   max(init_confidence, high_confidence_threshold)，
    //   即低分检测永远不能建轨；init_confidence 更高时以它为准。
    bool use_hungarian = true;                  // true=Hungarian 全局指派，false=贪心
    bool enable_two_stage = true;               // true=高分/低分两阶段匹配
    float high_confidence_threshold = 0.50f;    // 第一阶段门限，范围 [0,1]
    float low_confidence_threshold = 0.10f;     // 第二阶段门限，须 <= high 且 >= 0

    // 按检测置信度自适应测量噪声：低置信检测对应更大的 R，降低其对状态的拉动。
    // 缩放公式：R_eff = R * clamp((1/conf)^power, min_scale, max_scale)
    //   conf 为参与校正的检测置信度（[0,1]，实现里用 0.05 兜底防止除零），
    //   R 为 measurement_noise；
    //   conf 越低 -> (1/conf)^power 越大 -> R_eff 越大 -> 卡尔曼增益越小，
    //   该检测对中心/速度的修正越弱，避免低分框把轨迹拉偏。
    // confidence_noise_power：惩罚强度，一般取 1~3；1 表示与置信度成反比，
    //   2 更陡、对低分惩罚更强；调大后低分框几乎不参与校正。
    // min_measurement_noise_scale：R 缩放下限（倍数），保证 R_eff 不低于该比例，
    //   避免过度相信单帧观测；调大整体更平滑。
    // max_measurement_noise_scale：R 缩放上限（倍数），阻止 conf 接近 0 时
    //   R_eff 无限放大；调大等于更彻底地忽略极低分检测。
    bool adaptive_measurement_noise = true;         // true=按置信度放大观测噪声 R
    float confidence_noise_power = 2.0f;            // 惩罚指数，越大越不信任低分检测
    float min_measurement_noise_scale = 0.75f;      // R 缩放下限（倍数）
    float max_measurement_noise_scale = 8.0f;       // R 缩放上限（倍数）

    // 卡尔曼噪声均为协方差/方差，必须与像素及帧时间单位匹配。
    // 旧版兼容字段：如果调用方仍设置 process_noise（且不等于默认值），
    // 构造函数会用它同时覆盖位置和速度过程噪声。新代码建议分别设置下方参数。
    float process_noise = 1e-2f;                 // 旧版兼容字段（同时覆盖位置/速度）
    float process_noise_position = 1e-2f;        // 位置过程噪声方差（像素²）
    float process_noise_velocity = 1e-2f;        // 速度过程噪声方差（(像素/帧)²）
    float process_noise_acceleration = 1e-2f;    // 加速度过程噪声方差（(像素/帧²)²）
    float measurement_noise = 8.0f;              // 观测噪声 R（像素²）
    float initial_position_variance = 25.0f;     // 初始位置协方差（像素²）
    float initial_velocity_variance = 25.0f;     // 初始速度协方差（(像素/帧)²）
    float initial_acceleration_variance = 25.0f; // 初始加速度协方差（(像素/帧²)²）

    // 尺寸指数平滑系数：new = alpha*观测 + (1-alpha)*旧值。
    // 1.0 完全跟随观测（抖），0.0 完全不动（僵）。
    float size_smoothing = 0.65f;       // 尺寸指数平滑：new=alpha*obs+(1-alpha)*old

    // 绘图参数。OpenCV 颜色通道顺序为 BGR。
    cv::Scalar predicted_color{0, 255, 255}; // 黄色：预测框
    cv::Scalar filtered_color{0, 255, 0};    // 绿色：滤波框
    cv::Scalar observed_color{0, 0, 255};    // 红色：观测框
    int thickness = 2;                       // 框线宽（像素）
    int line_type = cv::LINE_AA;             // 线型；LINE_AA 为抗锯齿
    double label_font_scale = 0.5;           // 标签字号缩放
    int label_thickness = 1;                 // 标签文字线宽
    bool draw_labels = true;                 // 是否画 "ID n pred/filt/obs" 标签
    bool draw_predicted_box = true;          // 是否画黄色预测框
    bool draw_filtered_box = true;           // 是否画绿色滤波框
    bool draw_observed_box = true;           // 是否画红色观测框（仅本帧匹配到检测时）

    // 视频输出参数。
    double output_fps = 0.0;            // <=0 跟随源视频帧率
    std::string output_fourcc = "mp4v"; // 编码 FourCC，必须正好 4 个字符
};

// 多目标轨迹的只读快照（对外导出用）。
// 为什么单独定义一个 View 而不是暴露内部 Track：Track 里有 cv::KalmanFilter，
// 外部拿到引用就可能绕过 tracker 改状态；View 只拷贝标量和小矩形，
// 外部可以随意读取/记录，不会破坏跟踪逻辑。
struct MultiTrackView
{
    int id = -1;                        // 轨迹 ID
    int class_id = 0;                   // 类别编号
    int source = 0;                     // 0=YOLO，1=传统，2=两者匹配
    cv::Rect2f box;                     // 滤波框
    cv::Rect2f predicted_box;           // 本帧预测框
    cv::Rect2f observed_box;            // 本帧真实检测框
    float confidence = 0.0f;            // 最近一次观测的置信度
    int age = 0;                        // 轨迹总存活帧数
    int hits = 0;                       // 累计命中次数
    int time_since_update = 0;          // 连续未匹配帧数
    bool observed = false;              // 本帧是否匹配到检测
};

class MultiObjectTracker
{
public:
    // 构造时传入参数；运动模型在构造时确定，运行中不动态改变维度。
    // traditional_params 只影响 process() 中 YOLO + 传统识别的融合。
    // 构造函数会一次性校验全部参数（含传统识别参数），非法值直接抛异常。
    explicit MultiObjectTracker(const MultiTrackParams& params = MultiTrackParams(), const TraditionalParams& traditional_params = TraditionalParams());
    // 清空所有轨迹、传统候选记忆和 ID 计数。
    // process() 开头会自动调用，保证同一个 tracker 连续处理多段视频时互不污染。
    void reset();

    // 输入当前帧所有检测；内部依次执行预测、全局候选关联、一对一匹配、
    // 卡尔曼校正、新轨迹创建、过期轨迹删除。
    void step(const std::vector<Detection>& detections);

    // 按预测/滤波/观测三种颜色绘制所有可见轨迹。
    void draw(cv::Mat& frame) const;
    // 当前所有轨迹的只读快照（step() 结束时刷新）。
    // 注意返回的是引用，下一次 step() 后内容会变化；需要长期保存请自己拷贝。
    const std::vector<MultiTrackView>& tracks() const { return visible_tracks_; }

    // 完整视频处理入口：检测 -> 跟踪 -> 绘图 -> 视频/CSV 输出。
    void process(const std::string& input_video, const std::string& output_video,
                 const std::string& csv_log, Detector& detector, int class_id = 0);
    // 构造时使用的跟踪参数（只读，运行中不可改）。
    const MultiTrackParams& params() const { return params_; }
    // 构造时使用的传统识别/融合参数（只读）。
    const TraditionalParams& traditionalParams() const { return traditional_params_; }

private:
    struct Track
    {
        int id = -1;                    // 轨迹唯一 ID，从 1 递增；合并时保留小 ID
        int class_id = 0;               // 类别；关联时要求类别一致
        int source = 0;                 // 最近一次观测来源，供日志/绘制使用
        cv::KalmanFilter kf;            // 根据 motion_model 创建 4 维或 6 维滤波器
        cv::Rect2f box;                 // 当前滤波框（后验中心 + 平滑尺寸）
        cv::Rect2f predicted_box;       // predict() 得到的预测框（先验中心 + 上次尺寸）
        cv::Rect2f observed_box;        // 当前帧匹配到的原始检测框（仅 observed 时有效）
        cv::Point2f last_center;        // 最近一次观测到的中心，用于限制丢失期间的外推
        float confidence = 0.0f;        // 最近一次观测的置信度
        int age = 0;                    // 轨迹总存活帧数（从 1 开始）
        int hits = 0;                   // 累计命中帧数；达到 min_hits 才视为确认
        int hit_streak = 0;             // 连续命中帧数（当前用于合并时取 max）
        int time_since_update = 0;      // 连续未匹配帧数；决定过期删除与丢失绘制
        bool observed = false;          // 本帧是否匹配到检测
    };

    // 预测一条轨迹：推进卡尔曼状态，生成 predicted_box；丢失期间做衰减/限幅/回拉。
    void predict(Track& track);
    // 用匹配到的检测做卡尔曼校正，更新 box / observed_box / 尺寸平滑和生命周期计数。
    void update(Track& track, const Detection& detection);
    // 为未匹配的检测新建轨迹：分配 ID、创建 4/6 维 KalmanFilter 并初始化状态。
    void createTrack(const Detection& detection);
    // 把内部 tracks_ 复制成只读的 MultiTrackView 列表，供外部查询。
    void refreshViews();

    // YOLO + 传统识别的融合：YOLO 优先；传统候选只用于补漏/救援。
    // 返回的列表直接交给 step()，因此同一目标最多只有一个观测框。
    std::vector<Detection> fuseDetections(const std::vector<Detection>& yolo,
                                          const std::vector<Detection>& traditional);

    // 传统候选的跨帧记忆，用于“连续 N 帧一致才新建轨迹”。
    // 单帧的传统候选证据太弱，必须靠多帧一致性过滤掉偶发误检。
    struct TraditionalMemo
    {
        cv::Point2f center;     // 最近一次命中的中心
        float radius = 0.0f;    // 最近一次命中的等效半径（用于匹配容差）
        int hits = 0;           // 连续命中帧数
        int missed = 0;         // 连续未命中帧数（超过 memo_max_missed 就遗忘）
        float score = 0.0f;     // 最近一次命中的分数
    };

    // params_ / traditional_params_ 构造后只读；tracks_ 等运行时容器由 step() 改写。
    MultiTrackParams params_;           // 跟踪参数
    TraditionalParams traditional_params_; // 传统识别/融合参数
    std::vector<Track> tracks_;         // 每条轨迹各自拥有独立卡尔曼状态
    std::vector<MultiTrackView> visible_tracks_; // 供外部读取的快照
    std::vector<TraditionalMemo> traditional_memo_; // 传统候选的跨帧记忆
    int next_id_ = 1;                   // 下一条轨迹的 ID（reset 时回到 1）
};

} // namespace yolo11
