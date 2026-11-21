// ============================================================================
// main.cpp —— 程序入口：构造检测器 + 多目标跟踪器，跑完一整段视频
//
// 数据流：
//   best.onnx  --Detector-->  每帧若干 Detection
//                              --MultiObjectTracker.step--> 轨迹（ID + 三种框）
//                              --draw/writer--> 叠加视频
//                              --csv--> 逐帧逐目标的状态日志
//
// 运行前注意：
//   - 下面的绝对路径是 WSL 下的写法（/mnt/e/...）；在 Windows 原生环境编译时
//     需要改成 "E:/yolo11C++/best.onnx" 之类，否则构造 Detector 会抛异常。
//   - 输入视频和模型文件都要存在；输出目录由 MultiObjectTracker::process()
//     自动创建（见各实现里的 ensureParentDirectory）。
// ============================================================================

// ============================================================================
// 实现选择：同一份 main.cpp 对应三种实现（公共接口完全一致，main() 不用改）。
//   - 默认            : yolo11.*      （基线：排序贪心 + 单阶段关联）
//   - YOLO11_VARIANT_SPV1 : yolo11_spv1.*（v1：Hungarian + 两阶段 + 自适应 R）
//   - YOLO11_VARIANT_SPV2 : yolo11_spv2.*（v2：v1 + 马氏门控 + 轨迹质量分）
// 宏由 CMake 的 main_v1 / main_v2 目标定义；也可以手动 -D 编译，见 README 第 17 节。
// 三个版本默认输出到不同文件，方便并排比较；基线路径保持原样不变。
// ============================================================================
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

int main()
{
    // 只读入一次模型：Detector 内部持有 cv::dnn::Net，构造/加载 ONNX
    // 是整段流程里最慢的一步，逐帧复用同一个对象，不要放进循环里。
    // 第二个参数 Config 使用默认值：输入 640x640、conf 0.25、NMS 0.45、CPU。
    yolo11::Detector detector("/mnt/e/yolo11C++/best.onnx");


    // 多目标跟踪参数。这里只覆盖和本视频（031.mp4，比赛集锦，含镜头切换）
    // 相关的几项，其余用 MultiTrackParams 的默认值。
    yolo11::MultiTrackParams params;

    // 运动模型：匀速。目标在单个镜头内近似直线运动；镜头会硬切，
    // 用常加速模型反而会把切换瞬间的巨大位移当成“加速度”学进去。
    params.motion_model = yolo11::MotionModel::ConstantVelocity;

    // 时间步长：按“帧”为单位，1.0 表示每来一帧预测一次。
    // 这样速度单位就是像素/帧，和帧率无关；换 60fps 视频也不用改这里。
    params.dt = 1.0f;

    // 轨迹在连续多少帧没有匹配后删除。
    // 3 是偏小的值：镜头切换后旧目标不应该“苟活”太久，否则会和
    // 新位置上的新轨迹抢同一个检测框（关联门限虽然能挡，但越早删越干净）。
    params.max_age = 3;

    // 连续命中多少帧才认为轨迹“确认”，之后才绘制、写 CSV。
    // 3 帧可以把单帧误检挡掉，又不会让新出现的目标等太久。
    params.min_hits = 3;

    // 观测噪声 R（像素²）：检测框中心的抖动程度。
    // 12 大致对应“检测中心有 3~4 像素标准差”的画面；调大 -> 轨迹更平滑但滞后，
    // 调小 -> 更跟手但会把检测抖动原样传进轨迹。
    params.measurement_noise = 12.0f;

    // 速度过程噪声 Q（(像素/帧)²）：允许目标速度变化多快。
    // 25 意味着每帧速度可以有几个像素/帧的改变，适合快速机动的小目标；
    // 调小会让轨迹在目标加速时明显滞后，调大会让静止目标也“抖”。
    params.process_noise_velocity = 25.0f;

    // 构造跟踪器（参数在构造时校验并固化，运行中不再改变运动模型维度）。
    yolo11::MultiObjectTracker tracker(params);

    // 处理整段视频：
    //   "/mnt/e/yolo11C++/031.mp4"  输入
    //   YOLO11_OUTPUT_VIDEO         输出叠加视频（相对 build 目录；变体自动改名）
    //   YOLO11_OUTPUT_CSV           逐帧逐目标 CSV（变体自动改名）
    //   detector                    已加载好的检测器
    //   0                           要跟踪的类别编号（单类别模型固定 0）
    // 这个调用是同步的：视频跑完才返回，控制台会打印检测/融合统计摘要。
    tracker.process("/mnt/e/yolo11C++/031.mp4", YOLO11_OUTPUT_VIDEO, YOLO11_OUTPUT_CSV, detector, 0);
}
