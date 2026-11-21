// ============================================================================
// yolo11.cpp —— YOLO11 检测 + 卡尔曼滤波跟踪的完整实现
//
// 本文件按“数据从原始像素到轨迹”的顺序分成四个部分；检测/传统识别是
// 两条并行的观测来源，单目标与多目标跟踪则是两种并列的使用方式：
//
//   1. 检测层 Detector（文件开头）
//      输入一帧 BGR 图像，输出“原图坐标”下的 Detection。
//      内部流程：letterbox 预处理 -> blobFromImage 归一化 -> ONNX 前向推理
//      -> 解码头解析 (cx,cy,w,h,score) -> 坐标反变换回原图 -> NMS 去重。
//
//   2. 传统识别层 findbuff（文件末尾）
//      在 HSV 空间分割红/蓝圆靶，用“圆度 + 环形结构 + 填充率”打分，
//      作为 YOLO 的兜底候选。它只产生单帧候选，是否进入跟踪完全由
//      MultiObjectTracker::fuseDetections 的融合规则决定。
//
//   3. 单目标跟踪层 KalmanTracker / KalmanAccelTracker
//      状态只含中心点（匀速：+速度；常加速：+加速度），宽高不参与滤波，
//      只做“有观测就更新、没观测就沿用”。每帧流程：
//      预测 -> 选择观测 -> 误差门限 -> 校正 -> 维护丢失计数。
//
//   4. 多目标跟踪层 MultiObjectTracker
//      每条轨迹各自持有一个 cv::KalmanFilter；每帧先预测全部轨迹，再用
//      “IoU + 归一化中心距”代价做一对一贪心关联，最后处理新生、过期、合并。
//      它是 1、2 两路观测的消费者，也是 3 的推广（多套独立卡尔曼状态）。
//
// 阅读时请一直记住的几条全局约定：
//   - 所有几何量都在“原图坐标系”下，单位是像素；速度是像素/帧，
//     加速度是像素/帧²（因为 dt 默认 1 帧，帧率信息不进入滤波器）。
//   - OpenCV 颜色通道顺序是 BGR，所以“黄色”写成 Scalar(0,255,255)。
//   - 噪声参数都是方差/协方差而不是标准差，量纲是像素² 或 (像素/帧)²；
//     标准差换算方差要平方，量级差两个数量级，改动时容易出错。
//   - 卡尔曼只滤波中心点；框的宽高属于“外观量”，与运动模型无关，
//     单独用观测值/指数平滑维护（原因见各 Track::update 的注释）。
// ============================================================================

#include "yolo11.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <cmath>
#include <utility>
#include <filesystem>
#include <iostream>



namespace yolo11
{

// 构造检测器：校验配置 -> 加载 ONNX -> 选择推理后端。
// 所有检查都放在加载模型之前：这些参数直接决定后续解码和 NMS 的数学含义，
// 非法值必须在对象构造阶段就暴露，而不是等到 forward() 时才出现难查的错误。
Detector::Detector(const std::string& model_path, const Config& config) : config_(config)
{
    // input_size 是网络固定输入边长（模型导出时写死的 640，训练时也是 640）。
    // 改这个值不会让模型自动适配，只会让 letterbox 的输出和网络期望不一致。
    if (config_.input_size <= 0)
        throw std::invalid_argument("Config::input_size must be positive");
    // 两个阈值都是 [0,1] 的分数：置信度阈值筛候选，NMS 阈值判“重叠到多少算重复”。
    // NaN 会让所有比较都为 false，必须显式拦掉。
    if (!std::isfinite(config_.conf_threshold) || config_.conf_threshold < 0.0f || config_.conf_threshold > 1.0f)
        throw std::invalid_argument("Config::conf_threshold must be in [0, 1]");
    if (!std::isfinite(config_.nms_threshold) || config_.nms_threshold < 0.0f || config_.nms_threshold > 1.0f)
        throw std::invalid_argument("Config::nms_threshold must be in [0, 1]");

    // 读入 ONNX 并建立层图（DNN 模块会按算子逐个映射成内部层）。
    // 失败时不会抛异常，而是返回空 Net，所以要用 net_.empty() 再判一次。
    net_ = cv::dnn::readNetFromONNX(model_path);

    if (net_.empty())
        throw std::runtime_error("Failed to load YOLO11 ONNX model");

    // 后端/目标只决定“在哪算”，不改变结果：
    //   - CUDA 后端需要带 CUDA 的 OpenCV 构建；没有 CUDA 时 setPreferable*
    //     会被静默忽略并退回 CPU，所以这里不需要额外探测。
    //   - DNN_TARGET_OPENCL 等其它目标同理，这里只区分 GPU/CPU 两条最常用的路径。
    if (config_.use_gpu)
    {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}


// 简单检测接口：只把框挑出来给不需要类别/顶点的调用方。
// 真正的推理逻辑都在 infer() 里，这里只做字段筛选和预留容量，避免拷贝膨胀。
std::vector<cv::Rect2f> Detector::detect(const cv::Mat& frame, int class_id, int originalWidth, int originalHeight)
{
    std::vector<cv::Rect2f> boxes;

    // detect() 与 detectDetailed() 共用 infer()：这里只把 rect 挑出来，
    // 不会做第二次推理；需要中心点/顶点时请直接用 detectDetailed()。
    auto detections = infer(frame, class_id, originalWidth, originalHeight);

    // 预留容量，避免逐个 push_back 触发多次扩容拷贝。
    boxes.reserve(detections.size());

    for (const auto& detection : detections)
    {
        // Detection::rect 已经是“原图坐标 + 左上角 + 宽高”，可直接交给绘制/IoU 使用。
        boxes.push_back(detection.rect);
    }

    return boxes;
}


// 完整检测接口：直接把 infer() 的结果（中心点、四顶点、宽高、来源）交给调用方。
std::vector<Detection> Detector::detectDetailed(const cv::Mat& frame, int class_id, int originalWidth, int originalHeight)
{
    return infer(frame, class_id, originalWidth, originalHeight);
}


// YOLO11 推理核心：一帧图像 -> 若干原图坐标下的检测框。
// 下面八个阶段（Letterbox / Blob / Forward / Decode / 去 Padding /
// 回原图 / NMS / 组装）各自都有注释块，读的时候按顺序看即可。
// 出错时尽量抛出带上下文信息的异常，因为这里是“模型输出和代码假设
// 不一致”最容易出问题的地方，静默返回空结果反而更难排查。
std::vector<Detection> Detector::infer(const cv::Mat& frame, int class_id, int originalWidth, int originalHeight)
{
    std::vector<Detection> results;

    // 单类别模型只有 class_id=0（输出 5 维 = cx,cy,w,h,score）；
    // 请求其它类别时没有对应的分数通道，直接返回空而不是瞎猜。
    // frame 为空或原始尺寸非法时也缺少坐标变换的基准，一并拒绝。
    if (frame.empty() || originalWidth <= 0 || originalHeight <= 0 || class_id != 0)
        return results;
    if (frame.cols <= 0 || frame.rows <= 0)
        return results;

    // ==========================================
    // 1. Letterbox（等比缩放 + 灰边填充）
    //
    // 为什么不直接把整帧拉伸到 640x640：拉伸会改变宽高比，目标形状被压扁，
    // 和训练时的数据分布不一致，精度会下降。letterbox 用同一个 scale
    // 同时缩放宽高，再把不足的部分用 114 灰色补到 640x640。
    // 114 是 YOLO 训练时默认的填充灰度（BGR 三通道相同），这样补边区域
    // 在统计上和训练一致；用纯黑补边会形成强边缘，反而干扰检测。
    // 代价是：网络输出的坐标必须先减去补边、再除以 scale 才能回到原图。
    // ==========================================

    const int inputSize = config_.input_size;

    // 取两个方向中较小的比例：缩放后长边正好是 inputSize、短边 <= inputSize，
    // 图像完整落在 640x640 内，不会被裁掉任何内容。
    float scale = std::min((float)inputSize / frame.cols, (float)inputSize / frame.rows);

    // 缩放后的实际像素尺寸（四舍五入到整数，因为 resize 只接受整数尺寸）。
    int newW = std::round(frame.cols * scale);

    int newH = std::round(frame.rows * scale);

    // 补边居中：左上补 left/top，右下补“剩余量”（用减法算，保证两边之和精确
    // 等于 640，不会因为整除截断而少补或多补几个像素）。
    int left = (inputSize - newW) / 2;
    int top = (inputSize - newH) / 2;

    cv::Mat resized;

    cv::resize(frame, resized, cv::Size(newW, newH));

    cv::Mat input;

    cv::copyMakeBorder(resized, input, top, inputSize - newH - top, left, inputSize - newW - left, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // ==========================================
    // 2. Blob
    //
    // blobFromImage 一次完成三件事，缺一不可：
    //   1) 像素乘 1/255：把 uint8 [0,255] 归一化到 float [0,1]，与训练一致；
    //   2) swapRB=true：OpenCV 解码出来是 BGR，而模型训练用 RGB，
    //      必须交换 R/B 通道，否则颜色通道完全反了；
    //   3) 重排成 NCHW 的 4 维 blob（1x3x640x640），这是 DNN 模块的输入约定。
    // 第四个参数 mean 传空 Scalar 表示不做减均值；crop=false 表示输入已经
    // 是 640x640，不需要再裁剪。三个参数只要有一个和训练时不一致，
    // 网络输出就会整体偏移，而这种错误不会抛异常，只会表现为“检测变差”。
    // ==========================================

    cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(inputSize, inputSize), cv::Scalar(), true, false);

    // ==========================================
    // 3. Forward
    //
    // 输出形状由导出脚本决定，本项目的 best.onnx 是 [1, 5, N]：
    //   5    = (cx, cy, w, h, score)：单类别模型没有单独的目标性（objectness）
    //          通道，第 5 维直接就是类别分数；坐标是相对 640 letterbox 图的
    //          像素值（导出时已乘过 stride，不在 0~1 归一化空间里）。
    //   8400 = 80x80 + 40x40 + 20x20：stride 8/16/32 三个检测头在 640 输入上
    //          的网格点总数，每个格点对应一个候选框。
    // 先校验维度再分支，避免把 [1,N,5] 误当成 [1,5,N] 解析出完全错误的框。
    // ==========================================

    net_.setInput(blob);

    cv::Mat output = net_.forward();

    if (output.dims != 3 || output.size[0] != 1)
        throw std::runtime_error("Unexpected YOLO11 output shape: expected a 3D batch-1 tensor");

    // 单类别 YOLO11 通常为 [1, 5, N]；部分导出器会输出 [1, N, 5]。
    const int channels_first = output.size[1];
    const int channels_last = output.size[2];

    // DNN 输出是三维 Mat：[1, C, N] 或 [1, N, C]。
    // 不要对三维 Mat 直接调用 reshape(1, rows)：部分 OpenCV 版本会在
    // reshape 内部计算 total(start, end) 时触发 shape_utils.hpp 断言。
    // 这里先用输出数据指针构造二维 Mat 视图，再按布局决定是否转置。
    // 先转 CV_32F 并保证连续：后面用 ptr<float>() 按 float 步进，
    // 非连续内存（比如张量被子矩阵切过）会导致逐行指针错位。
    cv::Mat outputFloat;
    if (output.depth() == CV_32F)
        outputFloat = output;
    else
        output.convertTo(outputFloat, CV_32F);
    if (!outputFloat.isContinuous())
        outputFloat = outputFloat.clone();

    cv::Mat predictions;

    if (channels_first == 5 && channels_last > 0)
    {
        // [1, 5, N] -> 建 5xN 的“通道优先”视图，再转置成 Nx5。
        // t() 得到的 Mat 不连续，但后续只按行读取，逐行访问是安全的；
        // 不 clone 是为了省一次 8400x5 的拷贝。
        cv::Mat channelMajor(5, channels_last, CV_32F, outputFloat.ptr<float>());
        predictions = channelMajor.t();
    }
    else if (channels_last == 5 && channels_first > 0)
    {
        // [1, N, 5] -> 内存本身就是“每个候选连续 5 个数”，直接建视图即可；
        // clone() 让 predictions 拥有独立内存，避免它的生命周期依赖 outputFloat。
        cv::Mat channelMinor(channels_first, 5, CV_32F, outputFloat.ptr<float>());
        predictions = channelMinor.clone();
    }
    else
    {
        // 形状既不是 [1,5,N] 也不是 [1,N,5]：多半是导出了多类别模型，
        // 或检测头结构不同。带上实际维度抛错，方便对照导出脚本排查。
        std::ostringstream message;
        message << "Unexpected YOLO11 output shape: expected [1,5,N] or [1,N,5], got ["
                << output.size[0] << "," << channels_first << "," << channels_last << "]";
        throw std::runtime_error(message.str());
    }

    // ==========================================
    // 4. Decode
    //
    // 每个候选行是 (cx, cy, w, h, score)，坐标仍在 640x640 的 letterbox 图上，
    // 要回到调用方要的“原图坐标”需要两级反变换：
    //   第 1 级：减 padding、除以实际 resize 比例 -> 回到 frame 坐标；
    //   第 2 级：乘 original/frame 比例        -> 回到调用方声称的原图坐标。
    // 分两级写而不是合并成一个系数，是因为 frame 和 original 的尺寸可能不同
    // （调用方可能传入已经缩放过的帧），分开写便于核对每一级的量纲。
    // ==========================================

    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;

    // frame -> original 的比例
    float originalScaleX = (float)originalWidth / frame.cols;

    float originalScaleY = (float)originalHeight / frame.rows;

    // 使用实际 resize 尺寸计算，避免取整误差：
    // newW = round(cols*scale) 有舍入，真正的缩放比是 newW/cols 而不是 scale；
    // 两者在 4K 图上可能差 1~2 像素，用错会让整幅框系统性地偏移。
    float resizeScaleX = (float)newW / frame.cols;

    float resizeScaleY = (float)newH / frame.rows;

    for (int i = 0; i < predictions.rows; ++i)
    {
        // 每行 5 个数，顺序固定为 cx,cy,w,h,score（由导出脚本决定）。
        const float* data = predictions.ptr<float>(i);

        float cx = data[0];
        float cy = data[1];
        float w  = data[2];
        float h  = data[3];

        // 单类别置信度
        float confidence = data[4];

        // 过滤三类坏候选，避免它们进入后面的几何运算：
        //   1) NaN/Inf：网络数值异常或输出未初始化时会传播脏数据；
        //   2) 非正宽高：xyxy 变换后会出现负面积框；
        //   3) 分数越界或低于阈值：分数 >1 说明导出时漏了 sigmoid，
        //      不能当正常置信度使用。
        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) ||
            !std::isfinite(h) || !std::isfinite(confidence) || w <= 0.0f || h <= 0.0f ||
            confidence < config_.conf_threshold || confidence > 1.0f)
            continue;

        // xywh（中心点 + 宽高，网络输出格式）-> xyxy（左上 + 右下，
        // 方便做裁剪、IoU 和后续 Rect2f 构造）。
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // ======================================
        // 5. 去除 Padding
        //
        // 先减去左上补边（把坐标平移回缩放图的左上角），再除以实际缩放比，
        // 就得到该点在 frame（原始输入帧）坐标系中的位置。
        // 减 padding 必须在除以 scale 之前：padding 是在缩放后的图上加的，
        // 顺序反了会多除一次 scale。
        // ======================================

        x1 = (x1 - left) / resizeScaleX;
        x2 = (x2 - left) / resizeScaleX;

        y1 = (y1 - top) / resizeScaleY;
        y2 = (y2 - top) / resizeScaleY;

        // ======================================
        // 6. 转换到原图坐标
        //
        // frame 与调用方指定的 original 尺寸可能不同（例如帧被预缩放），
        // 这一步补上这层比例，保证返回的框可以直接画在 original 尺寸的图上。
        // ======================================

        x1 *= originalScaleX;
        x2 *= originalScaleX;

        y1 *= originalScaleY;
        y2 *= originalScaleY;

        // 限制坐标：反变换后贴边的框可能略微越界（负坐标或超过宽高），
        // 不裁剪会产生负宽高，导致 IoU/绘制/CSV 全部异常。
        x1 = std::max(0.0f, std::min(x1, (float)originalWidth));
        x2 = std::max(0.0f, std::min(x2, (float)originalWidth));

        y1 = std::max(0.0f, std::min(y1, (float)originalHeight));
        y2 = std::max(0.0f, std::min(y2, (float)originalHeight));

        float boxW = x2 - x1;
        float boxH = y2 - y1;

        // 裁剪后面积归零的框（完全落在画面外）直接丢弃。
        if (boxW <= 0 || boxH <= 0)
            continue;

        boxes.emplace_back(x1, y1, boxW, boxH);

        scores.push_back(confidence);
    }

    // ==========================================
    // 7. NMS（非极大值抑制）
    //
    // 同一目标会在相邻网格点、不同尺度（stride 8/16/32）上被重复预测。
    // NMSBoxes 按分数从高到低遍历：保留当前最高分框，并把与它 IoU 超过
    // nms_threshold 的候选删掉，从而每堆重叠框只留一个。
    // 这里再传一次 conf_threshold 是 NMSBoxes 的接口要求（它会先按该阈值
    // 过滤），与上面的 Decode 阶段重复，但不会改变结果。
    // ==========================================

    std::vector<int> indices;

    // NMSBoxes 没有 Rect2f 重载，先转成 Rect2d（double 精度不会丢信息）
    std::vector<cv::Rect2d> nms_boxes;
    nms_boxes.reserve(boxes.size());
    for (const auto& b : boxes)
        nms_boxes.emplace_back(b.x, b.y, b.width, b.height);

    cv::dnn::NMSBoxes(nms_boxes, scores, config_.conf_threshold, config_.nms_threshold, indices);

    // ==========================================
    // 8. 生成完整 Detection
    //
    // 只把 NMS 保留下来的索引组装成 Detection：rect/中心点/四顶点/宽高
    // 全部由同一个框推导，保证几个字段之间不会自相矛盾。
    // source 默认 0（YOLO），allow_new_track 默认 true（YOLO 框可以建轨）。
    // ==========================================

    results.reserve(indices.size());

    for (int index : indices)
    {
        const cv::Rect2f& box = boxes[index];

        Detection detection;

        detection.rect = box;

        detection.class_id = class_id;

        detection.width = box.width;

        detection.height = box.height;

        // 中心点
        detection.center = cv::Point2f(box.x + box.width * 0.5f, box.y + box.height * 0.5f);

        // 四个顶点（顺时针：左上 -> 左下 -> 右下 -> 右上）
        detection.LT = cv::Point2f(box.x, box.y);

        detection.LB = cv::Point2f(box.x, box.y + box.height);

        detection.RB = cv::Point2f(box.x + box.width, box.y + box.height);

        detection.RT = cv::Point2f(box.x + box.width, box.y);

        // 顶点顺序固定为 左上 -> 左下 -> 右下 -> 右上（顺时针），
        // 方便需要画四边形、做透视变换或取“靶面朝向”的调用方直接使用。
        detection.confidence = scores[index];

        results.push_back(detection);
    }

    return results;
}


// ============================================================
// 2D 匀速卡尔曼跟踪实现
//
// 状态模型：x = [cx, cy, vx, vy]^T
//   - 只对中心点做匀速卡尔曼预测，宽高不进入状态向量；
//   - 有观测时用 Detection 的宽高更新边长；
//   - 无观测时沿用最近一次观测到的边长。
//
// 先说清楚卡尔曼在做什么，后面读 predict()/correct() 会轻松很多：
//   滤波器同时维护“估计值 x”和“估计的不确定度 P”。每帧做两件事：
//     预测：用运动模型把上一帧的 x 推到当前帧，P 因模型不精确而变大；
//     校正：用当前帧观测把预测值拉向真实值，P 因获得信息而变小。
//   两者加权比例由卡尔曼增益 K 自动决定：
//     K ≈ P / (P + R)
//   R（观测噪声）大 -> K 小 -> 更相信模型，结果平滑但滞后；
//   P（预测不确定度）大 -> K 大 -> 更相信观测，响应快但抖。
//   本项目里 R 就是 KalmanParams::measurement_noise，P 的初始值来自
//   initial_position/velocity_variance，Q 决定 P 每帧涨得多快。
//
// 每帧流程：
//   1. 预测中心点，得到预测框（使用上次观测边长）
//   2. 从当前帧检测结果中选择一个观测
//   3. 有观测：先做误差门限判断；通过则更新边长并校正中心
//   4. 无观测或被门限拒绝：累加丢失计数，未超限时滤波框 = 预测框
//   5. 超过丢失上限：判定丢失，停止输出框，等待重新检测
// ============================================================

namespace
{

// VideoWriter / ofstream 不会自动建目录：输出路径的父目录不存在时，
// 打开输出文件会直接失败（而且不同平台报错信息不一致，很难定位）。
// 这里在打开输出前先把父目录创建出来，把“路径不存在”变成不可能发生的错误。
void ensureParentDirectory(const std::string& path)
{
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();

    // 只有文件名、没有目录部分时无需处理（相对路径直接写在当前目录）。
    if (parent.empty())
        return;

    // 用 error_code 版本而不是抛异常的版本：目录已存在时 create_directories
    // 返回 false 且不报错，但并发/权限场景仍可能置位 ec，所以这里不直接
    // 依赖返回值，而是“ec 有错且目录确实不是目录”才认为是真失败。
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);

    if (ec && !std::filesystem::is_directory(parent))
        throw std::runtime_error("Failed to create output directory: " + parent.string() + " (" + ec.message() + ")");
}


// 由中心点和边长构造矩形框。
// 注意：cv::Rect2f 保存的是左上角坐标，所以这里要减去半宽、半高。
// 卡尔曼的状态是“中心点”，而 OpenCV 的绘图/IoU 接口吃“左上角 + 宽高”，
// 所有状态到框的转换都必须经过这个函数，避免两种表示混用导致整体偏移半个框。
cv::Rect2f makeBox(float center_x, float center_y, float width, float height)
{
    return cv::Rect2f(center_x - width * 0.5f, center_y - height * 0.5f, width, height);
}


// 判断矩形框是否有有效面积（宽高都必须大于 0）。
// 用 > 0 而不是 >= 0：零宽高的框画不出来，IoU 恒为 0，会被当成“未观测”，
// 早点过滤掉比让它们进入后续计算更安全。
bool hasArea(const cv::Rect2f& rect)
{
    return rect.width > 0.0f && rect.height > 0.0f;
}


// 浮点坐标四舍五入为 int，供 OpenCV 的绘制接口使用。
// 不用 static_cast<int>（截断）的原因：截断会让框整体向左上缩半个像素，
// 小目标上肉眼可见地“框偏了”；lround 是对称的。
int toInt(float value)
{
    return static_cast<int>(std::lround(value));
}


// 优先使用 Detection 的 width/height 字段；
// 如果调用方只填了 rect，则退回使用 rect 的宽高。
// 这样 YOLO 的检测和外部手工构造的 Detection 都能被同一套代码处理。
float detectionWidth(const Detection& detection)
{
    return std::isfinite(detection.width) && detection.width > 0.0f ? detection.width : detection.rect.width;
}


// 同上，取得检测框的高度。
float detectionHeight(const Detection& detection)
{
    return std::isfinite(detection.height) && detection.height > 0.0f ? detection.height : detection.rect.height;
}


// rect 有效时以 rect 为准，避免调用方漏填 center 后误用默认 (0,0)。
cv::Point2f detectionCenter(const Detection& detection)
{
    if (hasArea(detection.rect) && std::isfinite(detection.rect.x) && std::isfinite(detection.rect.y) && std::isfinite(detection.rect.width) && std::isfinite(detection.rect.height))
        return cv::Point2f(detection.rect.x + detection.rect.width * 0.5f, detection.rect.y + detection.rect.height * 0.5f);
    return detection.center;
}


// 统一的“这个检测能不能用”判定。所有进入跟踪/融合的检测都必须先过这里：
//   - class_id 必须为 0：本项目的模型是单类别（Buff），其它编号的检测框
//     没有对应的语义，混进来只会污染关联；
//   - 中心点/宽高/分数必须是有限数：NaN 参与任何比较都返回 false，
//     会让“距离最近”“分数最高”这类选择逻辑悄悄失效；
//   - 宽高必须为正、分数必须在 [0,1]：负面积框和越界分数都会破坏 IoU 和门限。
bool validDetection(const Detection& detection)
{
    const cv::Point2f center = detectionCenter(detection);
    const float width = detectionWidth(detection);
    const float height = detectionHeight(detection);
    return detection.class_id == 0 && std::isfinite(center.x) && std::isfinite(center.y) && std::isfinite(width) && std::isfinite(height) && width > 0.0f && height > 0.0f && std::isfinite(detection.confidence) && detection.confidence >= 0.0f && detection.confidence <= 1.0f;
}


// 构造当前帧的观测框：
// 优先使用检测结果自带的 rect；如果 rect 无效，
// 再用 Detection::center 和宽高重新拼一个框。
// 之所以不直接信任 rect：YOLO 和传统识别都会填 rect，但如果调用方手工
// 构造 Detection 只填了 center/width/height，直接用 rect 会得到全零框，
// 画出来就是左上角一个点，很难排查。
cv::Rect2f detectionBox(const Detection& detection)
{
    if (hasArea(detection.rect) && std::isfinite(detection.rect.x) && std::isfinite(detection.rect.y) && std::isfinite(detection.rect.width) && std::isfinite(detection.rect.height))
        return detection.rect;

    const cv::Point2f center = detectionCenter(detection);
    return makeBox(center.x, center.y, detectionWidth(detection), detectionHeight(detection));
}


// 在帧上绘制一个跟踪框：
//   - 框无效或帧为空时直接返回；
//   - 框坐标从 float 四舍五入到 int 后调用 cv::rectangle；
//   - 可选在框旁边绘制标签；
//   - OpenCV 的 putText 不支持中文字形，所以标签用短英文。
void drawTrackBox(cv::Mat& frame, const cv::Rect2f& rect, const cv::Scalar& color, const std::string& label, const KalmanParams& params)
{
    if (frame.empty() || !hasArea(rect))
        return;

    // OpenCV 的 rectangle 接口使用整数坐标，这里做一次四舍五入。
    const cv::Rect box(toInt(rect.x), toInt(rect.y), toInt(rect.width), toInt(rect.height));

    cv::rectangle(frame, box, color, params.thickness, params.line_type);

    if (!params.draw_labels)
        return;

    int baseline = 0;

    const cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, params.label_font_scale, params.label_thickness, &baseline);

    // 标签默认放在框的左上角上方。
    int text_x = box.x;
    int text_y = box.y - 4;

    // 如果框太靠上，放到框内，避免文字被裁掉。
    if (text_y - text_size.height < 0)
        text_y = box.y + text_size.height + 4;

    // 防止标签超出画面右边界。
    text_x = std::max(0, std::min(text_x, frame.cols - text_size.width));

    cv::putText(frame, label, cv::Point(text_x, text_y), cv::FONT_HERSHEY_SIMPLEX, params.label_font_scale, color, params.label_thickness, params.line_type);
}

} // namespace


// 构造时把参数一次性校验完，然后重置状态。
// 这些检查不是形式主义：dt 决定 F 矩阵里速度/位置的耦合系数，噪声要求
// 非负（负方差会让 P 失去正定性，S.inv() 直接产生垃圾值），
// 绘制参数为 0 会让 cv::rectangle 抛断言。参数错了应该在构造阶段就失败，
// 而不是跑到第 3000 帧时突然 throw 一个看不懂的 OpenCV 断言。
KalmanTracker::KalmanTracker(const KalmanParams& params) : params_(params)
{
    // dt 是“两帧之间经过了多少个时间单位”，必须为正；本项目取 1 帧。
    if (!std::isfinite(params_.dt) || params_.dt <= 0.0f)
        throw std::invalid_argument("KalmanParams::dt must be finite and positive");

    // 过程噪声 Q 的对角元素，允许为 0（表示完全相信匀速模型），但不能为负。
    if (!std::isfinite(params_.process_noise_position) || !std::isfinite(params_.process_noise_velocity) ||
        params_.process_noise_position < 0.0f || params_.process_noise_velocity < 0.0f)
        throw std::invalid_argument("KalmanParams process noise must be finite and non-negative");

    // 观测噪声 R 必须严格正：R=0 意味着“检测绝对准确”，S 可能奇异，
    // 而且滤波结果会完全跟随观测，失去平滑意义。
    if (!std::isfinite(params_.measurement_noise) || params_.measurement_noise <= 0.0f)
        throw std::invalid_argument("KalmanParams::measurement_noise must be finite and positive");

    // 误差门限允许为 0 或负数（<= 0 表示关闭门限），只要求不是 NaN/Inf。
    if (!std::isfinite(params_.max_center_distance))
        throw std::invalid_argument("KalmanParams::max_center_distance must be finite");

    // 初始协方差 P0 的对角元素：只要求有限且非负。
    if (!std::isfinite(params_.initial_position_variance) || !std::isfinite(params_.initial_velocity_variance) ||
        params_.initial_position_variance < 0.0f || params_.initial_velocity_variance < 0.0f)
        throw std::invalid_argument("KalmanParams initial variance must be finite and non-negative");

    if (params_.lost_frame_limit < 0)
        throw std::invalid_argument("KalmanParams::lost_frame_limit must be non-negative");

    // 绘制参数：线宽/字号为 0 或负会让 OpenCV 断言失败。
    if (!std::isfinite(params_.label_font_scale) || params_.label_font_scale <= 0.0 ||
        params_.thickness <= 0 || params_.label_thickness <= 0)
        throw std::invalid_argument("KalmanParams drawing scale and thickness values must be positive");

    reset();
}


// 重置：把状态整体换回默认构造的 KalmanState（x/P 为空、initialized=false、
// lost_frames=0），然后从参数重新复制丢失上限。
// 这样同一个 tracker 对象可以被 process() 反复使用，每次都从“等待目标出现”开始。
void KalmanTracker::reset()
{
    state_ = KalmanState();

    // 丢失帧上限保存在状态结构体中（运行期以状态里的值为准）
    state_.lost_frame_limit = params_.lost_frame_limit;
}


// 处理整段视频：检测 -> 单帧状态推进 -> 叠加绘制 -> 写视频/CSV。
//
// 顺序上有两个容易被忽略的细节：
//   1. 必须先读一帧再建 VideoWriter：输出分辨率取自首帧，而且读首帧失败
//      比“写了一半才发现输入坏了”更容易定位问题；
//   2. 必须 reset() 再写表头：同一个 tracker 被复用时，上一轮的轨迹状态
//      不能带进新视频，否则 CSV 前几行会混入旧目标的输出。
//
// 循环用 do-while 而不是 while：首帧已经在建 writer 之前读出来了，
// 循环体处理的是“当前手里的 frame”，末尾再读下一帧。
// 时间戳按“帧序号 / 源帧率”换算，和输出帧率无关：
//   params_.output_fps 只影响播放速度，不改变每帧对应的真实时间。
void KalmanTracker::process(const std::string& input_video, const std::string& output_video, const std::string& csv_log, Detector& detector, int class_id)
{
    // 三个路径都不能为空：空字符串会让 VideoCapture/VideoWriter 返回未打开，
    // 但错误信息会很含糊（"Failed to open input video: "），提前拦截更清楚。
    if (input_video.empty() || output_video.empty() || csv_log.empty())
        throw std::invalid_argument("KalmanTracker::process requires non-empty paths");

    // FourCC 必须是 4 个字符，否则 fourcc() 取到的编码是垃圾值；
    // fps 必须是有限数，否则 VideoWriter 内部会算出负数或 NaN 帧间隔。
    if (params_.output_fourcc.size() != 4)
        throw std::invalid_argument("KalmanParams::output_fourcc must contain exactly 4 characters");
    if (!std::isfinite(params_.output_fps))
        throw std::invalid_argument("KalmanParams::output_fps must be finite");

    cv::VideoCapture capture(input_video);

    if (!capture.isOpened())
        throw std::runtime_error("Failed to open input video: " + input_video);

    cv::Mat frame;

    if (!capture.read(frame) || frame.empty())
        throw std::runtime_error("Failed to read the first frame from: " + input_video);

    const int width = frame.cols;
    const int height = frame.rows;

    double source_fps = capture.get(cv::CAP_PROP_FPS);

    // 容器里没有写帧率（或写了 0）时回退到 25fps：
    // 只影响 CSV 时间戳和默认输出帧率，不影响跟踪本身的逻辑（dt 固定按帧算）。
    if (!std::isfinite(source_fps) || source_fps <= 0.0)
        source_fps = 25.0;

    const double output_fps = params_.output_fps > 0.0 ? params_.output_fps : source_fps;

    const std::string& fourcc = params_.output_fourcc;

    // 输出目录可能不存在（如 ../output），先建目录再打开文件。
    ensureParentDirectory(output_video);
    ensureParentDirectory(csv_log);

    cv::VideoWriter writer(output_video, cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]), output_fps, cv::Size(width, height));

    if (!writer.isOpened())
        throw std::runtime_error("Failed to open output video: " + output_video);

    std::ofstream csv(csv_log, std::ios::out | std::ios::trunc);

    if (!csv.is_open())
        throw std::runtime_error("Failed to open csv log: " + csv_log);

    // 清空上一轮运行留下的状态（同一个 tracker 可能被 process() 调用多次）。
    reset();

    csv << csvHeader() << '\n';

    long long frame_index = 0;

    do
    {
        // 每帧都重新检测：检测器无状态，跟踪状态全部在 state_ 里。
        // 这里传 frame.cols/rows 作为“原图尺寸”，因为 frame 就是原图。
        const std::vector<Detection> detections = detector.detectDetailed(frame, class_id, frame.cols, frame.rows);

        // 推进一帧：预测 -> 选观测 -> 门限 -> 校正 -> 更新丢失计数。
        step(detections);

        // 绘制会直接改写 frame，所以必须在 writer.write 之前调用。
        draw(frame);

        writer.write(frame);

        // CSV 的时间戳用源帧率换算；frame_index 从 0 开始，第 0 帧对应 0 秒。
        csv << csvRow(frame_index, static_cast<double>(frame_index) / source_fps) << '\n';

        frame_index++;

    } 
    while (capture.read(frame) && !frame.empty());
}


// 单帧状态推进：这是单目标跟踪的“心脏”，按 5 个步骤顺序执行。
// 注意 detections 里可能同时有多个框（例如画面里出现两个靶），
// 单目标模式只选一个（见 selectObservation），另一个不会建第二条轨迹。
void KalmanTracker::step(const std::vector<Detection>& detections)
{
    // 新的一帧：清空当前帧的绘制框和观测标记。
    // 这三个框是“本帧的临时结果”，必须每帧重置；否则上一帧的框会残留，
    // 在丢帧/未检测到目标时被误画成本帧结果（常见 bug：幽灵框）。
    state_.observed = false;
    state_.observed_rect = cv::Rect2f();
    state_.predicted_rect = cv::Rect2f();
    state_.filtered_rect = cv::Rect2f();

    // 1. 只用中心点做匀速预测
    //    宽高不参与滤波，所以预测框的尺寸直接沿用上一次观测到的边长；
    //    目标远近变化带来的尺寸变化由“有观测时更新边长”这一步跟上。
    if (state_.initialized)
    {
        predict();

        // 预测框使用上一次观测到的边长（当前帧还没有观测）
        state_.predicted_rect = makeBox(state_.x.at<float>(0, 0), state_.x.at<float>(1, 0), state_.width, state_.height);
    }

    // 2. 选择当前帧的观测（单目标：已初始化取离预测最近的，未初始化取最可信的）
    const Detection* observation = selectObservation(detections);

    // 3. 已初始化时先做误差门限判断和卡尔曼修正。
    //    correct() 内部比较预测中心与观测中心的距离：
    //    - 距离未超限：执行修正，observed 保持 true；
    //    - 距离超限：不修正，并把 observed 置为 false（视为本帧没有观测）。
    //    先“置 true 再让 correct 否决”的写法，是因为只有 correct 才知道
    //    预测值（未初始化时没有预测值，不存在门限问题）。
    if (observation != nullptr)
    {
        state_.observed = true;

        if (state_.initialized)
            correct(*observation);
    }

    // 4. 有效观测：更新边长，初始化或校正，生成滤波框。
    if (observation != nullptr && state_.observed)
    {
        // 有观测时更新边长；之后丢失也沿用这里的值（宽高不进入状态向量，
        // 所以“最近一次观测的尺寸”就是丢失期间唯一可用的尺寸估计）
        const float observed_width = detectionWidth(*observation);
        const float observed_height = detectionHeight(*observation);

        if (observed_width > 0.0f && observed_height > 0.0f)
        {
            state_.width = observed_width;
            state_.height = observed_height;
            state_.has_size = true;
        }

        state_.observed_rect = detectionBox(*observation);

        if (!state_.initialized)
        {
            // 首次观测，或丢失后重新出现：重新初始化滤波器。
            // 注意初始化会把速度（以及常加速模型下的加速度）置 0、把 P 重置为
            // 初始大方差，相当于“忘掉”丢失前的运动状态。因为丢失期间的外推
            // 本来就不可信，继续沿用只会让新轨迹一上来就带着错误速度
            // （遮挡后目标常常已经反向运动，沿用旧速度会连续几帧冲过头）。
            initialize(*observation);

            // 第一帧没有预测值，用观测框占位，保证三个框里至少有一个是真实观测
            state_.predicted_rect = state_.observed_rect;
        }
        else
        {
            // 已初始化且 correct() 已执行：清除丢失标记。
            // 只有“通过门限的观测”才能走到这里，被门限拒绝的观测不会清零，
            // 否则一次误检就能洗掉丢失计数，让幽灵轨迹一直不消失。
            state_.lost = false;
            state_.lost_frames = 0;
        }

        // 滤波框使用更新后的中心点和当前边长
        state_.filtered_rect = makeBox(state_.x.at<float>(0, 0), state_.x.at<float>(1, 0), state_.width, state_.height);
    }
    else
    {
        // 5. 无观测，或观测被误差门限拒绝：按丢失帧处理。
        state_.observed = false;
        state_.observed_rect = cv::Rect2f();

        if (state_.initialized)
        {
            state_.lost_frames++;

            // 超过上限：判定目标丢失，并主动释放 x/P。
            // 释放而不是保留的原因：如果留着旧状态，下一次观测会被当成
            // “已初始化 + 距离超限”处理，从而永远进不了 initialize()，
            // 轨迹就卡死在“丢失”状态。清空后下次检测到目标会走重新初始化。
            if (state_.lost_frames > state_.lost_frame_limit)
            {
                state_.lost = true;
                state_.initialized = false;
                state_.x.release();
                state_.P.release();
                state_.predicted_rect = cv::Rect2f();
                state_.filtered_rect = cv::Rect2f();
            }
            else
            {
                // 还没到上限：用纯预测值当滤波输出（惯性外推）。
                // 这段时间没有观测校正，框会按最后的速度继续滑行，
                // 短时遮挡（几帧）就能靠它维持住位置。
                state_.filtered_rect = state_.predicted_rect;
            }
        }
    }
}


// 匀速模型预测：x' = F*x，P' = F*P*F^T + Q。
//
// F 的物理含义就是“位置 += 速度*dt，速度不变”：
//   [1 0 dt 0 ]   cx' = cx + dt*vx
//   [0 1 0 dt ]   cy' = cy + dt*vy
//   [0 0 1  0 ]   vx' = vx
//   [0 0 0  1 ]   vy' = vy
// 协方差传播 P' = F*P*F^T 表示“把不确定性也按同样的线性关系传下去”：
// 位置的不确定度会因为速度的不确定度而随 dt 增长。
void KalmanTracker::predict()
{
    const float dt = params_.dt;

    const cv::Mat F = (cv::Mat_<float>(4, 4) <<
        1.0f, 0.0f, dt,   0.0f,
        0.0f, 1.0f, 0.0f, dt,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f);

    state_.x = F * state_.x;
    state_.P = F * state_.P * F.t();

    // 对角过程噪声 Q：模型不可能完美，每预测一步都要给协方差“加一点不确定”。
    // 位置项代表目标不完全按匀速走；速度项代表速度本身会漂移。
    // 这里只加到对角线上（简化处理，忽略位置-速度相关性），
    // 对像素级跟踪足够用，而且能保证 P 始终对角占优、数值稳定。
    state_.P.at<float>(0, 0) += params_.process_noise_position;
    state_.P.at<float>(1, 1) += params_.process_noise_position;
    state_.P.at<float>(2, 2) += params_.process_noise_velocity;
    state_.P.at<float>(3, 3) += params_.process_noise_velocity;
}


// 用一次观测初始化滤波器（首次检测到目标，或丢失后重新捕获）。
//
// 状态向量取 [观测中心 x, 观测中心 y, 0, 0]：
//   - 位置直接用观测值，因为这是唯一可用的信息；
//   - 速度没有任何观测可用，先置 0，并用较大的初始方差表示“完全不知道”。
//     如果硬猜一个速度（例如用前两帧差分），一旦第一帧检测有偏差，
//     错误速度会被 P 的小值“锁住”，反而需要很多帧才能纠正。
// P 取对角阵，含义是“位置/速度各自独立且不确定度已知”，非对角项为 0
// 表示初始时假设位置误差和速度误差不相关。
void KalmanTracker::initialize(const Detection& detection)
{
    state_.x = (cv::Mat_<float>(4, 1) <<
        detectionCenter(detection).x,
        detectionCenter(detection).y,
        0.0f,
        0.0f);

    state_.P = cv::Mat::zeros(4, 4, CV_32F);

    state_.P.at<float>(0, 0) = params_.initial_position_variance;
    state_.P.at<float>(1, 1) = params_.initial_position_variance;
    state_.P.at<float>(2, 2) = params_.initial_velocity_variance;
    state_.P.at<float>(3, 3) = params_.initial_velocity_variance;

    // 标记为已初始化，并清掉丢失状态（这次观测就是“重新出现”的证据）。
    state_.initialized = true;
    state_.lost = false;
    state_.lost_frames = 0;
}


// 卡尔曼修正。修正前先做误差门限判断：
// 预测中心与观测中心的距离超过 max_center_distance 时，
// 认为该检测是误检，直接返回，不修改 x、P。
void KalmanTracker::correct(const Detection& detection)
{
    // 误差门限：预测中心（当前 state_.x 的前两维）与观测中心的距离。
    // 使用平方距离比较，省去开方；<= 0 表示不启用门限。
    if (params_.max_center_distance > 0.0f)
    {
        const cv::Point2f center = detectionCenter(detection);
        const float dx = center.x - state_.x.at<float>(0, 0);
        const float dy = center.y - state_.x.at<float>(1, 0);
        const float distance_sq = dx * dx + dy * dy;
        const float limit_sq = params_.max_center_distance * params_.max_center_distance;

        if (distance_sq > limit_sq)
        {
            // 标记本帧观测无效，step() 会按“无观测”处理：
            // 不更新边长、不重置丢失计数、不绘制观测框。
            state_.observed = false;
            return;
        }
    }

    // 观测矩阵 H：只从状态里“取中心点”，其余分量（速度）不直接参与观测。
    // H 的形状是 2x4，把 4 维状态投影成 2 维观测 z = [cx, cy]^T。
    const cv::Mat H = (cv::Mat_<float>(2, 4) <<
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f);

    // 观测噪声 R = measurement_noise * I：x、y 两个方向独立同分布，
    // 单位是像素²。R 越大表示越不相信检测框中心（画质差、抖动大时调大）。
    const cv::Mat R = cv::Mat::eye(2, 2, CV_32F) * params_.measurement_noise;

    const cv::Point2f center = detectionCenter(detection);
    const cv::Mat z = (cv::Mat_<float>(2, 1) << center.x, center.y);

    // 新息 innovation = z - H*x：观测与预测的差，卡尔曼只用这个“偏差”修正，
    // 而不是直接用观测覆盖状态，所以整条轨迹是平滑的。
    const cv::Mat innovation = z - H * state_.x;

    // 新息协方差 S = H*P*H^T + R：预测的不确定度 + 观测的不确定度。
    const cv::Mat S = H * state_.P * H.t() + R;

    // 卡尔曼增益 K = P*H^T*S^-1，形状 4x2。
    // K 把“2 维的新息”放大/缩小后分配到“4 维状态”上：
    // 新息大且 P 大时，速度分量也会被明显修正（因为 P 的非对角项把
    // 位置误差和速度误差关联起来了）。
    const cv::Mat K = state_.P * H.t() * S.inv();

    // 状态修正：x = x + K*innovation
    state_.x = state_.x + K * innovation;

    // 协方差更新用 Joseph 形式：P = (I-KH)P(I-KH)^T + K*R*K^T。
    // 教科书公式 (I-KH)P 在数值上容易失去对称性/正定性（K 接近 1 时尤甚），
    // Joseph 形式对任意 K 都保持对称半正定，长时间跑视频不会因为
    // 协方差“跑负”而发散。
    const cv::Mat I = cv::Mat::eye(4, 4, CV_32F);

    const cv::Mat IKH = I - K * H;

    state_.P = IKH * state_.P * IKH.t() + K * R * K.t();
}


// 从当前帧的多个检测里挑一个作为观测（单目标跟踪的数据关联）。
//
// 单目标和多目标的区别就在这里：多目标要“多对多”匹配，单目标只需要
// 一个“最像”的检测，所以用两条简单规则就够：
//   - 已初始化：取离预测中心最近的。用平方距离比较即可，省一次开方，
//     因为开方是单调的，不影响谁最近；
//   - 未初始化：取置信度最高的。此时还没有预测位置，只能相信检测本身。
// 都没有可用检测（空列表 / 全部无效）时返回 nullptr，表示本帧无观测。
const Detection* KalmanTracker::selectObservation(const std::vector<Detection>& detections) const
{
    if (detections.empty())
        return nullptr;

    if (state_.initialized)
    {
        // 预测中心：step() 中已经执行过 predict()，x 的前两维就是当前帧的预测位置。
        const float predicted_x = state_.x.at<float>(0, 0);
        const float predicted_y = state_.x.at<float>(1, 0);

        const Detection* nearest = nullptr;
        float nearest_distance = std::numeric_limits<float>::max();

        for (const auto& detection : detections)
        {
            // 无效检测（类别不对、NaN、零面积）直接跳过，不能参与“最近”比较：
            // NaN 会让所有比较为 false，导致最近检测被悄悄忽略。
            if (!validDetection(detection))
                continue;
            const cv::Point2f center = detectionCenter(detection);
            const float dx = center.x - predicted_x;
            const float dy = center.y - predicted_y;
            const float distance = dx * dx + dy * dy;

            if (distance < nearest_distance)
            {
                nearest_distance = distance;
                nearest = &detection;
            }
        }

        return nearest;
    }

    // 尚未初始化时取置信度最高的有效单类别观测。
    const Detection* best = nullptr;
    for (const auto& detection : detections)
        if (validDetection(detection) && (!best || detection.confidence > best->confidence))
            best = &detection;
    return best;
}


// 绘制：左上角状态文字 + 预测框 / 滤波框 / 观测框。
// 三种颜色对应跟踪器内部的三条数据流：
//   黄色 = 纯模型外推（没有观测时唯一能看的东西）；
//   绿色 = 融合后的最优估计（正常跟踪时应该贴着目标）；
//   红色 = 本帧真实检测（用来对照“滤波是否跟得上”）。
void KalmanTracker::draw(cv::Mat& frame) const
{
    if (frame.empty())
        return;

    // 左上角状态：未初始化时区分“还没见过目标”和“跟丢后等待重新出现”；
    // 已初始化时显示“连续丢失帧数/上限”，方便调 lost_frame_limit。
    // OpenCV 的 putText 不支持中文字形，所以状态文字用英文。
    if (params_.draw_status)
    {
        std::ostringstream status;

        if (!state_.initialized)
            status << (state_.lost ? "target: lost" : "target: waiting");
        else
            status << "lost: " << state_.lost_frames << '/' << state_.lost_frame_limit;

        cv::putText(frame, status.str(), cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, params_.label_font_scale, cv::Scalar(255, 255, 255), params_.label_thickness, params_.line_type);
    }

    // 已判定丢失（且状态已释放）时不再画框，避免长期挂着一个不动的旧框。
    if (!state_.initialized || state_.lost)
        return;

    drawTrackBox(frame, state_.predicted_rect, params_.predicted_color, "pred", params_);

    drawTrackBox(frame, state_.filtered_rect, params_.filtered_color, "filt", params_);

    // 只有当前帧有通过误差门限的有效观测时才画观测框
    if (state_.observed)
        drawTrackBox(frame, state_.observed_rect, params_.observed_color, "obs", params_);
}


// CSV 表头；其中 observed 表示当前帧是否有通过误差门限的有效观测。
std::string KalmanTracker::csvHeader() const
{
    return "frame,timestamp,observed,predicted_cx,predicted_cy,filtered_cx,filtered_cy,"
           "observed_cx,observed_cy,velocity_x,velocity_y,width,height,lost_frames,lost";
}


// 将当前状态写成一行 CSV；被门限拒绝的观测 observed=0，观测中心留空。
// 列顺序必须与 csvHeader() 严格一致，否则数据会错位（而且不会报错）。
std::string KalmanTracker::csvRow(long long frame_index, double timestamp) const
{
    std::ostringstream row;

    // 时间戳保留 3 位小数（毫秒级），避免科学计数法影响 Excel/pandas 解析。
    row << frame_index << ',' << std::fixed << std::setprecision(3) << timestamp << ',' << (state_.observed ? 1 : 0);

    // 本地 lambda：输出一个框的中心点；框无效时输出两个空字段（保持列数不变）。
    // 用“宽高 > 0”判断有效性，而不是看坐标：未初始化的 Rect2f 是 (0,0,0,0)，
    // 直接输出会把“没有框”写成坐标 (0,0)。
    const auto add_center = [&row](const cv::Rect2f& rect)
    {
        if (rect.width > 0.0f && rect.height > 0.0f)
            row << ',' << (rect.x + rect.width * 0.5f) << ',' << (rect.y + rect.height * 0.5f);
        else
            row << ",,";
    };

    add_center(state_.predicted_rect);
    add_center(state_.filtered_rect);
    add_center(state_.observed_rect);

    // 速度只有滤波器活着时才有意义；丢失后 x 已释放，这里必须留空而不是写 0。
    if (state_.initialized && !state_.x.empty())
        row << ',' << state_.x.at<float>(2, 0) << ',' << state_.x.at<float>(3, 0);
    else
        row << ",,";

    // 宽高只要有观测过就有值（丢失期间沿用最后一次观测）。
    if (state_.has_size)
        row << ',' << state_.width << ',' << state_.height;
    else
        row << ",,";

    // 连续丢失帧数与丢失标记放最后，便于用表格软件筛选。
    row << ',' << state_.lost_frames << ',' << (state_.lost ? 1 : 0);

    return row.str();
}

// ============================================================
// 2D 常加速度卡尔曼跟踪实现
//
// 状态模型：x = [cx, cy, vx, vy, ax, ay]^T
//   - 只对中心点做常加速度卡尔曼预测，宽高不进入状态向量；
//   - 有观测时用 Detection 的宽高更新边长；
//   - 无观测时沿用最近一次观测到的边长。
//
// 为什么还要一个常加速模型：匀速模型假设“速度不变”，目标一旦加速/转向，
// 滤波器总会滞后半个节拍（预测值永远落在真实轨迹外侧）。把加速度也放进
// 状态后，滤波器可以自己“学到”加速度并用它外推，适合快速机动目标。
// 代价是状态从 4 维变 6 维、Q 多一个加速度噪声项：加速度噪声给大了会放大
// 检测抖动，给小了又跟不上机动，所以默认值比匀速模型更依赖调参。
//
// 每帧流程与匀速模型相同，只是状态转移矩阵包含加速度项：
//   cx' = cx + dt*vx + 0.5*dt^2*ax
//   cy' = cy + dt*vy + 0.5*dt^2*ay
//   vx' = vx + dt*ax
//   vy' = vy + dt*ay
//   ax' = ax
//   ay' = ay
// ============================================================

// 构造跟踪器：校验公共参数和加速度参数，然后清空状态。
// 校验项与 KalmanTracker 一一对应（两个类不共享实现，所以各自校验一次），
// 再加两项加速度专属检查。参数错了要在构造阶段就失败，而不是跑到中途才炸。
KalmanAccelTracker::KalmanAccelTracker(const KalmanAccelParams& params) : params_(params)
{
    // ---- 继承自 KalmanParams 的公共参数校验 ----
    // dt 是 F 矩阵里速度/加速度的耦合系数，必须为正。
    if (!std::isfinite(params_.dt) || params_.dt <= 0.0f)
        throw std::invalid_argument("KalmanAccelParams::dt must be finite and positive");

    // 过程噪声是方差，允许为 0，但不能为负（负方差会让 P 失去正定性）。
    if (!std::isfinite(params_.process_noise_position) || !std::isfinite(params_.process_noise_velocity) ||
        params_.process_noise_position < 0.0f || params_.process_noise_velocity < 0.0f)
        throw std::invalid_argument("KalmanAccelParams process noise must be finite and non-negative");

    // 观测噪声 R 必须严格为正：R=0 时 S 可能奇异，滤波会退化成“完全跟随观测”。
    if (!std::isfinite(params_.measurement_noise) || params_.measurement_noise <= 0.0f)
        throw std::invalid_argument("KalmanAccelParams::measurement_noise must be finite and positive");

    // 误差门限允许 <=0（表示关闭门限），只要求不是 NaN/Inf。
    if (!std::isfinite(params_.max_center_distance))
        throw std::invalid_argument("KalmanAccelParams::max_center_distance must be finite");

    // 初始协方差只要求有限且非负。
    if (!std::isfinite(params_.initial_position_variance) || !std::isfinite(params_.initial_velocity_variance) ||
        params_.initial_position_variance < 0.0f || params_.initial_velocity_variance < 0.0f)
        throw std::invalid_argument("KalmanAccelParams initial variance must be finite and non-negative");

    if (params_.lost_frame_limit < 0)
        throw std::invalid_argument("KalmanAccelParams::lost_frame_limit must be non-negative");

    // 绘制参数：线宽/字号为 0 或负会让 cv::rectangle / cv::putText 断言失败。
    if (!std::isfinite(params_.label_font_scale) || params_.label_font_scale <= 0.0 ||
        params_.thickness <= 0 || params_.label_thickness <= 0)
        throw std::invalid_argument("KalmanAccelParams drawing scale and thickness values must be positive");

    // ---- 加速度相关参数校验 ----
    // 加速度过程噪声：常加速模型发散与否主要看这一项，允许为 0 但不能为负。
    if (!std::isfinite(params_.process_noise_acceleration) || params_.process_noise_acceleration < 0.0f)
        throw std::invalid_argument("KalmanAccelParams::process_noise_acceleration must be finite and non-negative");

    // 初始加速度协方差：首次观测时加速度完全未知，用较大的方差表达不确定度。
    if (!std::isfinite(params_.initial_acceleration_variance) || params_.initial_acceleration_variance < 0.0f)
        throw std::invalid_argument("KalmanAccelParams::initial_acceleration_variance must be finite and non-negative");

    reset();
}


// 清空所有运行时状态，并重新从参数中复制丢失帧上限。
void KalmanAccelTracker::reset()
{
    state_ = KalmanState();

    // 丢失帧上限保存在状态结构体中
    state_.lost_frame_limit = params_.lost_frame_limit;
}


// 处理整个视频：
//   打开输入视频 -> 逐帧检测 -> 推进卡尔曼状态 -> 绘制叠加 -> 写输出视频和 CSV。
// 每帧先读取原图，检测得到原图坐标下的 Detection，再交给 step() 更新状态。
void KalmanAccelTracker::process(const std::string& input_video, const std::string& output_video, const std::string& csv_log, Detector& detector, int class_id)
{
    // 路径参数必须有效。
    if (input_video.empty() || output_video.empty() || csv_log.empty())
        throw std::invalid_argument("KalmanAccelTracker::process requires non-empty paths");

    // 输出编码必须正好是 4 个字符。
    if (params_.output_fourcc.size() != 4)
        throw std::invalid_argument("KalmanAccelParams::output_fourcc must contain exactly 4 characters");
    if (!std::isfinite(params_.output_fps))
        throw std::invalid_argument("KalmanAccelParams::output_fps must be finite");

    // 1. 打开输入视频。
    cv::VideoCapture capture(input_video);

    if (!capture.isOpened())
        throw std::runtime_error("Failed to open input video: " + input_video);

    // 2. 读取第一帧，用它的尺寸创建输出视频。
    cv::Mat frame;

    if (!capture.read(frame) || frame.empty())
        throw std::runtime_error("Failed to read the first frame from: " + input_video);

    const int width = frame.cols;
    const int height = frame.rows;

    // 3. 读取源视频帧率；无效值（0、NaN、inf）回退到 25fps。
    double source_fps = capture.get(cv::CAP_PROP_FPS);

    if (!std::isfinite(source_fps) || source_fps <= 0.0)
        source_fps = 25.0;

    // 输出帧率：参数为正时使用参数，否则跟随源视频。
    const double output_fps = params_.output_fps > 0.0 ? params_.output_fps : source_fps;

    const std::string& fourcc = params_.output_fourcc;

    // 4. 创建叠加视频的写入器。
    ensureParentDirectory(output_video);
    ensureParentDirectory(csv_log);
    cv::VideoWriter writer(output_video, cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]), output_fps, cv::Size(width, height));

    if (!writer.isOpened())
        throw std::runtime_error("Failed to open output video: " + output_video);

    // 5. 打开 CSV 日志。
    std::ofstream csv(csv_log, std::ios::out | std::ios::trunc);

    if (!csv.is_open())
        throw std::runtime_error("Failed to open csv log: " + csv_log);

    // 同一个 tracker 可能被重复使用，这里先清空上一轮的状态。
    reset();

    // CSV 第一行是表头，列顺序与 csvRow() 保持一致。
    csv << csvHeader() << '\n';

    long long frame_index = 0;

    // 6. 逐帧处理：检测 -> 状态推进 -> 叠加绘制 -> 写视频 -> 写日志。
    do
    {
        // 对当前原始帧做检测，得到原图坐标下的中心点、宽高和检测框。
        const std::vector<Detection> detections = detector.detectDetailed(frame, class_id, frame.cols, frame.rows);

        // 推进卡尔曼状态：预测、选择观测、误差门限、校正、更新绘制框。
        step(detections);

        // 把预测框/滤波框/观测框叠加到原图上。
        draw(frame);

        // 写出叠加后的视频帧。
        writer.write(frame);

        // 记录当前帧的滤波状态；时间戳按源视频帧率换算。
        csv << csvRow(frame_index, static_cast<double>(frame_index) / source_fps) << '\n';

        frame_index++;

    } 
    while (capture.read(frame) && !frame.empty());
}


// 单帧状态推进。detections 为当前帧的检测结果，可以为空。
void KalmanAccelTracker::step(const std::vector<Detection>& detections)
{
    // 新的一帧：清空当前帧的绘制框和观测标记
    state_.observed = false;
    state_.observed_rect = cv::Rect2f();
    state_.predicted_rect = cv::Rect2f();
    state_.filtered_rect = cv::Rect2f();

    // 1. 只用中心点做常加速度预测
    if (state_.initialized)
    {
        predict();

        // 预测框使用上一次观测到的边长（当前帧还没有观测）
        state_.predicted_rect = makeBox(state_.x.at<float>(0, 0), state_.x.at<float>(1, 0), state_.width, state_.height);
    }

    // 2. 选择当前帧的观测（单目标：取最近/最可信的一个）
    const Detection* observation = selectObservation(detections);

    // 3. 已初始化时先做误差门限判断和卡尔曼修正。
    //    correct() 内部比较预测中心与观测中心的距离：
    //    - 距离未超限：执行修正，observed 保持 true；
    //    - 距离超限：不修正，并把 observed 置为 false。
    if (observation != nullptr)
    {
        state_.observed = true;

        if (state_.initialized)
            correct(*observation);
    }

    // 4. 有效观测：更新边长，初始化或校正，生成滤波框。
    if (observation != nullptr && state_.observed)
    {
        // 有观测时更新边长；之后丢失也沿用这里的值
        const float observed_width = detectionWidth(*observation);
        const float observed_height = detectionHeight(*observation);

        if (observed_width > 0.0f && observed_height > 0.0f)
        {
            state_.width = observed_width;
            state_.height = observed_height;
            state_.has_size = true;
        }

        state_.observed_rect = detectionBox(*observation);

        if (!state_.initialized)
        {
            // 首次观测，或丢失后重新出现：重新初始化滤波器。
            // 注意初始化会把速度（以及常加速模型下的加速度）置 0、把 P 重置为
            // 初始大方差，相当于“忘掉”丢失前的运动状态。因为丢失期间的外推
            // 本来就不可信，继续沿用只会让新轨迹一上来就带着错误速度
            // （遮挡后目标常常已经反向运动，沿用旧速度会连续几帧冲过头）。
            initialize(*observation);

            // 第一帧没有预测值，用观测框占位，保证三个框里至少有一个是真实观测
            state_.predicted_rect = state_.observed_rect;
        }
        else
        {
            // 已初始化且 correct() 已执行：清除丢失标记。
            // 只有“通过门限的观测”才能走到这里，被门限拒绝的观测不会清零，
            // 否则一次误检就能洗掉丢失计数，让幽灵轨迹一直不消失。
            state_.lost = false;
            state_.lost_frames = 0;
        }

        // 滤波框使用更新后的中心点和当前边长
        state_.filtered_rect = makeBox(state_.x.at<float>(0, 0), state_.x.at<float>(1, 0), state_.width, state_.height);
    }
    else
    {
        // 5. 无观测，或观测被误差门限拒绝：按丢失帧处理。
        state_.observed = false;
        state_.observed_rect = cv::Rect2f();

        if (state_.initialized)
        {
            state_.lost_frames++;

            // 超过上限：判定目标丢失，并主动释放 x/P。
            // 释放而不是保留的原因：如果留着旧状态，下一次观测会被当成
            // “已初始化 + 距离超限”处理，从而永远进不了 initialize()，
            // 轨迹就卡死在“丢失”状态。清空后下次检测到目标会走重新初始化。
            if (state_.lost_frames > state_.lost_frame_limit)
            {
                state_.lost = true;
                state_.initialized = false;
                state_.x.release();
                state_.P.release();
                state_.predicted_rect = cv::Rect2f();
                state_.filtered_rect = cv::Rect2f();
            }
            else
            {
                // 还没到上限：用纯预测值当滤波输出（惯性外推）。
                // 这段时间没有观测校正，框会按最后的速度继续滑行，
                // 短时遮挡（几帧）就能靠它维持住位置。
                state_.filtered_rect = state_.predicted_rect;
            }
        }
    }
}


// 常加速度模型预测（只预测中心点）：
//   F = [1 0 dt 0 0.5*dt^2 0;
//        0 1 0 dt 0 0.5*dt^2;
//        0 0 1 0 dt 0;
//        0 0 0 1 0 dt;
//        0 0 0 0 1 0;
//        0 0 0 0 0 1]
//   x' = F * x
//   P' = F * P * F^T + Q
void KalmanAccelTracker::predict()
{
    const float dt = params_.dt;
    const float half_dt2 = 0.5f * dt * dt;

    const cv::Mat F = (cv::Mat_<float>(6, 6) <<
        1.0f, 0.0f, dt,   0.0f, half_dt2, 0.0f,
        0.0f, 1.0f, 0.0f, dt,   0.0f,     half_dt2,
        0.0f, 0.0f, 1.0f, 0.0f, dt,       0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f,     dt,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f,     0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,     1.0f);

    // 状态向量预测：中心点、速度、加速度分别按常加速度模型推进。
    state_.x = F * state_.x;

    // 协方差预测：不确定性随预测步长增大。
    state_.P = F * state_.P * F.t();

    // 叠加对角过程噪声 Q（位置、速度、加速度各自的方差）。
    state_.P.at<float>(0, 0) += params_.process_noise_position;
    state_.P.at<float>(1, 1) += params_.process_noise_position;
    state_.P.at<float>(2, 2) += params_.process_noise_velocity;
    state_.P.at<float>(3, 3) += params_.process_noise_velocity;
    state_.P.at<float>(4, 4) += params_.process_noise_acceleration;
    state_.P.at<float>(5, 5) += params_.process_noise_acceleration;
}


// 用一次观测初始化常加速度滤波器：
//   x = [cx, cy, 0, 0, 0, 0]^T（初始速度和加速度未知，先置 0）
//   P = diag(位置方差, 位置方差, 速度方差, 速度方差, 加速度方差, 加速度方差)
// 同时清除丢失标记，表示轨迹重新建立。
void KalmanAccelTracker::initialize(const Detection& detection)
{
    // 只取 Detection 的中心点；宽高不进入状态向量。
    state_.x = (cv::Mat_<float>(6, 1) <<
        detectionCenter(detection).x,
        detectionCenter(detection).y,
        0.0f,
        0.0f,
        0.0f,
        0.0f);

    // 初始协方差：位置、速度、加速度分别使用配置的初始方差。
    state_.P = cv::Mat::zeros(6, 6, CV_32F);

    state_.P.at<float>(0, 0) = params_.initial_position_variance;
    state_.P.at<float>(1, 1) = params_.initial_position_variance;
    state_.P.at<float>(2, 2) = params_.initial_velocity_variance;
    state_.P.at<float>(3, 3) = params_.initial_velocity_variance;
    state_.P.at<float>(4, 4) = params_.initial_acceleration_variance;
    state_.P.at<float>(5, 5) = params_.initial_acceleration_variance;

    // 滤波器已建立，清除丢失标记和丢失计数。
    state_.initialized = true;
    state_.lost = false;
    state_.lost_frames = 0;
}


// 常加速度卡尔曼修正。观测向量只包含中心点：
//   z = [检测中心x, 检测中心y]^T
//   H = [1 0 0 0 0 0; 0 1 0 0 0 0]
//
// 修正前先做误差门限判断：
// 预测中心与观测中心的距离超过 max_center_distance 时，
// 认为该检测是误检，直接返回，不修改 x、P。
void KalmanAccelTracker::correct(const Detection& detection)
{
    // 误差门限：预测中心（当前 state_.x 的前两维）与观测中心的距离。
    // 使用平方距离比较，省去开方；<= 0 表示不启用门限。
    if (params_.max_center_distance > 0.0f)
    {
        const cv::Point2f center = detectionCenter(detection);
        const float dx = center.x - state_.x.at<float>(0, 0);
        const float dy = center.y - state_.x.at<float>(1, 0);
        const float distance_sq = dx * dx + dy * dy;
        const float limit_sq = params_.max_center_distance * params_.max_center_distance;

        if (distance_sq > limit_sq)
        {
            // 标记本帧观测无效，step() 会按“无观测”处理：
            // 不更新边长、不重置丢失计数、不绘制观测框。
            state_.observed = false;
            return;
        }
    }

    // 观测矩阵：只观测中心点 cx、cy。
    const cv::Mat H = (cv::Mat_<float>(2, 6) <<
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // 观测噪声协方差 R = measurement_noise * I。
    const cv::Mat R = cv::Mat::eye(2, 2, CV_32F) * params_.measurement_noise;

    // 观测向量 z：中心点从 Detection 中取得。
    const cv::Point2f center = detectionCenter(detection);
    const cv::Mat z = (cv::Mat_<float>(2, 1) << center.x, center.y);

    // 新息：观测值与预测值之差。
    const cv::Mat innovation = z - H * state_.x;

    // 新息协方差 S。
    const cv::Mat S = H * state_.P * H.t() + R;

    // 卡尔曼增益 K（6x2）。
    const cv::Mat K = state_.P * H.t() * S.inv();

    // 用新息修正状态向量（中心点、速度、加速度）。
    state_.x = state_.x + K * innovation;

    // 协方差更新采用 Joseph 形式，保证矩阵对称正定。
    const cv::Mat I = cv::Mat::eye(6, 6, CV_32F);

    const cv::Mat IKH = I - K * H;

    state_.P = IKH * state_.P * IKH.t() + K * R * K.t();
}


// 从当前帧的检测结果中选出一个观测。
// 单目标跟踪策略：
//   - 已初始化：选离预测中心最近的检测，保证轨迹连续；
//   - 未初始化：选置信度最高的检测，作为新轨迹的起点。
const Detection* KalmanAccelTracker::selectObservation(const std::vector<Detection>& detections) const
{
    if (detections.empty())
        return nullptr;

    if (state_.initialized)
    {
        // 预测中心：step() 中已经执行过 predict()。
        const float predicted_x = state_.x.at<float>(0, 0);
        const float predicted_y = state_.x.at<float>(1, 0);

        const Detection* nearest = nullptr;
        float nearest_distance = std::numeric_limits<float>::max();

        // 遍历所有检测，找离预测中心最近的一个（比较平方距离）。
        for (const auto& detection : detections)
        {
            if (!validDetection(detection))
                continue;
            const cv::Point2f center = detectionCenter(detection);
            const float dx = center.x - predicted_x;
            const float dy = center.y - predicted_y;
            const float distance = dx * dx + dy * dy;

            if (distance < nearest_distance)
            {
                nearest_distance = distance;
                nearest = &detection;
            }
        }

        return nearest;
    }

    // 尚未初始化时取置信度最高的有效单类别观测。
    const Detection* best = nullptr;
    for (const auto& detection : detections)
        if (validDetection(detection) && (!best || detection.confidence > best->confidence))
            best = &detection;
    return best;
}


// 根据 KalmanState 中已经计算好的框绘制叠加层。
// 绘制规则与匀速跟踪器一致：
//   预测框 -> 滤波框 -> 观测框（仅有有效观测时）。
void KalmanAccelTracker::draw(cv::Mat& frame) const
{
    if (frame.empty())
        return;

    // 左上角状态文字：显示当前连续丢失帧数/上限。
    if (params_.draw_status)
    {
        std::ostringstream status;

        if (!state_.initialized)
            status << (state_.lost ? "target: lost" : "target: waiting");
        else
            status << "lost: " << state_.lost_frames << '/' << state_.lost_frame_limit;

        cv::putText(frame, status.str(), cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX,
                    params_.label_font_scale, cv::Scalar(255, 255, 255), params_.label_thickness, params_.line_type);
    }

    // 没有有效轨迹时不画任何框。
    if (!state_.initialized || state_.lost)
        return;

    // 预测框：预测中心 + 上次观测边长（黄色）。
    drawTrackBox(frame, state_.predicted_rect, params_.predicted_color, "pred", params_);

    // 滤波框：校正后中心 + 当前/上次观测边长（绿色）。
    drawTrackBox(frame, state_.filtered_rect, params_.filtered_color, "filt", params_);

    // 只有当前帧有通过误差门限的有效观测时才画观测框（红色）。
    if (state_.observed)
        drawTrackBox(frame, state_.observed_rect, params_.observed_color, "obs", params_);
}


// CSV 表头；其中 observed 表示当前帧是否有通过误差门限的有效观测。
std::string KalmanAccelTracker::csvHeader() const
{
    return "frame,timestamp,observed,predicted_cx,predicted_cy,filtered_cx,filtered_cy,"
           "observed_cx,observed_cy,velocity_x,velocity_y,acceleration_x,acceleration_y,"
           "width,height,lost_frames,lost";
}


// 将当前状态写成一行 CSV；被门限拒绝的观测 observed=0，观测中心留空。
std::string KalmanAccelTracker::csvRow(long long frame_index, double timestamp) const
{
    std::ostringstream row;

    row << frame_index << ',' << std::fixed << std::setprecision(3) << timestamp << ',' << (state_.observed ? 1 : 0);

    // 本地 lambda：输出一个框的中心点；框无效时输出两个空字段。
    const auto add_center = [&row](const cv::Rect2f& rect)
    {
        if (rect.width > 0.0f && rect.height > 0.0f)
            row << ',' << (rect.x + rect.width * 0.5f) << ',' << (rect.y + rect.height * 0.5f);
        else
            row << ",,";
    };

    add_center(state_.predicted_rect);
    add_center(state_.filtered_rect);
    add_center(state_.observed_rect);

    // 速度 (vx, vy) 和加速度 (ax, ay)：只有滤波器已初始化时才存在。
    if (state_.initialized && !state_.x.empty())
        row << ',' << state_.x.at<float>(2, 0) << ',' << state_.x.at<float>(3, 0)
            << ',' << state_.x.at<float>(4, 0) << ',' << state_.x.at<float>(5, 0);
    else
        row << ",,,,";

    // 边长：最近一次观测到的值（丢失时仍保留）。
    if (state_.has_size)
        row << ',' << state_.width << ',' << state_.height;
    else
        row << ",,";

    // 连续丢失帧数和丢失标记。
    row << ',' << state_.lost_frames << ',' << (state_.lost ? 1 : 0);

    return row.str();
}

} // namespace yolo11

namespace yolo11
{

// ============================================================
// 多目标卡尔曼跟踪实现
// 设计原则：每条轨迹拥有独立 KalmanFilter；所有轨迹共享一份参数。
// 运动模型由 MultiTrackParams::motion_model 决定：
//   CV: [cx, cy, vx, vy]^T
//   CA: [cx, cy, vx, vy, ax, ay]^T
// ============================================================

namespace
{
// 计算两个矩形的交并比 IoU = 交集面积 / 并集面积，取值 [0,1]。
//
// 直观含义：两个框“重合得有多好”。IoU=1 表示完全重合，0 表示不相交。
// 它是多目标关联最常用的外观无关相似度：目标大小/位置相近时 IoU 高，
// 与颜色、纹理无关，所以对光照变化不敏感。
// 计算时先求交集矩形的“左下/右上”两个角（用 max/min 夹取），
// 再对负宽度取 0：两个框不相交时 x2-x1 会为负，不夹取会算出负面积。
float rectIoU(const cv::Rect2f& a, const cv::Rect2f& b)
{
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    const float iw = std::max(0.0f, x2 - x1);
    const float ih = std::max(0.0f, y2 - y1);
    const float intersection = iw * ih;
    const float uni = a.area() + b.area() - intersection;
    return uni > 0.0f ? intersection / uni : 0.0f;
}

// 根据轨迹 ID 生成稳定颜色，确保同一目标在视频中始终使用同一种颜色。
// 做法：把 ID 乘以一个奇数（Knuth 乘法散列），再取不同位的字节映射到
// B/G/R 三个通道。乘奇数让相邻 ID 的哈希值差异很大，避免 ID 1/2/3
// 得到几乎一样的颜色；每个通道限制在 60~219，保证在亮背景上也看得清。
cv::Scalar trackColor(int id)
{
    const unsigned int h = static_cast<unsigned int>(id) * 2654435761u;
    return cv::Scalar(60 + (h & 0x9F), 60 + ((h >> 8) & 0x9F), 60 + ((h >> 16) & 0x9F));
}

// 读取滤波器状态中的中心点，并结合宽高构造矩形框。
// cv::KalmanFilter 的 statePost 是列向量，前两维固定是 cx、cy（两种模型都一样），
// 所以这个函数不用关心是 4 维还是 6 维状态。
cv::Rect2f stateBox(const cv::Mat& state, float width, float height)
{
    return makeBox(state.at<float>(0, 0), state.at<float>(1, 0), width, height);
}

// 把状态向量里的速度/加速度限制在物理上合理的范围内。
// 卡尔曼滤波在目标被遮挡、观测中断时会靠模型外推，常加速模型尤其容易发散
// （加速度被噪声推大后，位置会指数级跑飞）；直接修改状态比事后裁剪框更稳定，
// 因为后续所有预测都基于这个状态，裁剪框只是遮住了症状。
// 做法是按模长等比缩放，保留运动方向、只压缩幅值：
//   v *= max_speed / |v|
// 注意缩放前后方向不变，所以“目标往哪个方向走”这条信息不会丢。
void limitStateMotion(cv::Mat& state, const MultiTrackParams& params)
{
    // 状态至少要含速度（前 4 维：cx,cy,vx,vy）。
    if (state.empty() || state.rows < 4)
        return;

    const float vx = state.at<float>(2, 0);
    const float vy = state.at<float>(3, 0);
    const float speed = std::sqrt(vx * vx + vy * vy);
    if (speed > params.max_speed && speed > 0.0f)
    {
        const float scale = params.max_speed / speed;
        state.at<float>(2, 0) = vx * scale;
        state.at<float>(3, 0) = vy * scale;
    }

    // 6 维状态（常加速模型）才有加速度分量。
    if (state.rows >= 6)
    {
        const float ax = state.at<float>(4, 0);
        const float ay = state.at<float>(5, 0);
        const float accel = std::sqrt(ax * ax + ay * ay);
        if (accel > params.max_accel && accel > 0.0f)
        {
            const float scale = params.max_accel / accel;
            state.at<float>(4, 0) = ax * scale;
            state.at<float>(5, 0) = ay * scale;
        }
    }
}

// 创建匀速/常加速模型共用的转移矩阵。
// F 描述状态从 k-1 帧到 k 帧的运动关系：x(k)=F*x(k-1)。
// 两种模型只差“是否包含加速度行/列”，所以用同一个函数按枚举返回，
// 避免在两处各写一份容易写错的常量矩阵。
cv::Mat makeTransitionMatrix(MotionModel model, float dt)
{
    if (model == MotionModel::ConstantVelocity)
    {
        // 匀速：位置由“原位置 + 速度*时间”得到，速度保持不变。
        return (cv::Mat_<float>(4, 4) <<
            1, 0, dt, 0,
            0, 1, 0, dt,
            0, 0, 1, 0,
            0, 0, 0, 1);
    }

    // 常加速：位置增加 0.5*a*dt^2，速度增加 a*dt，加速度保持不变。
    const float half_dt2 = 0.5f * dt * dt;
    return (cv::Mat_<float>(6, 6) <<
        1, 0, dt, 0, half_dt2, 0,
        0, 1, 0, dt, 0, half_dt2,
        0, 0, 1, 0, dt, 0,
        0, 0, 0, 1, 0, dt,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 1);
}

// 绘制单个框及其标签；颜色由调用者传入，以区分观测、预测和滤波结果。
void drawMultiBox(cv::Mat& frame, const cv::Rect2f& rect, const cv::Scalar& color,
                  const std::string& label, const MultiTrackParams& params)
{
    if (!hasArea(rect)) return;

    const cv::Rect box(toInt(rect.x), toInt(rect.y), toInt(rect.width), toInt(rect.height));
    cv::rectangle(frame, box, color, params.thickness, params.line_type);

    if (params.draw_labels)
    {
        int textY = std::max(15, box.y - 5);
        cv::putText(frame, label, cv::Point(std::max(0, box.x), textY),
                    cv::FONT_HERSHEY_SIMPLEX, params.label_font_scale,
                    color, params.label_thickness, params.line_type);
    }
}
} // namespace

// 构造时校验所有参数，防止非法矩阵、除零或无意义的关联阈值进入跟踪流程。
MultiObjectTracker::MultiObjectTracker(const MultiTrackParams& params,
                                       const TraditionalParams& traditional_params)
    : params_(params), traditional_params_(traditional_params)
{
    // 兼容旧代码：旧版只有 process_noise，一个值同时控制位置和速度过程噪声。
    // 若调用方显式改过旧参数，则优先采用旧值；新代码应分别设置 position/velocity。
    if (params_.process_noise != 1e-2f)
    {
        params_.process_noise_position = params_.process_noise;
        params_.process_noise_velocity = params_.process_noise;
    }
    // ---- 轨迹生命周期 ----
    // dt 必须为正：它决定速度/位置在 F 矩阵里的耦合系数，0 会让状态永远不动。
    if (!std::isfinite(params_.dt) || params_.dt <= 0.0f)
        throw std::invalid_argument("MultiTrackParams::dt must be finite and positive");
    // max_age 允许为 0（丢一帧就删），min_hits 至少为 1（否则轨迹永远不会被确认）。
    if (params_.max_age < 0 || params_.min_hits < 1)
        throw std::invalid_argument("MultiTrackParams max_age/min_hits is invalid");

    // ---- 关联门限 ----
    // 两个距离门限都必须为正，否则 gate 会退化成 0，任何候选都匹配不上。
    if (!std::isfinite(params_.max_center_distance) || params_.max_center_distance <= 0.0f ||
        !std::isfinite(params_.max_distance_ratio) || params_.max_distance_ratio <= 0.0f)
        throw std::invalid_argument("MultiTrackParams distance gates must be positive");
    // min_iou 是“最低形状重合度”，必须落在 [0,1]。
    if (!std::isfinite(params_.min_iou) || params_.min_iou < 0.0f || params_.min_iou > 1.0f)
        throw std::invalid_argument("MultiTrackParams min_iou must be in [0,1]");
    // 两个权重非负且不能同时为 0（否则 cost 会除以 0）。
    if (!std::isfinite(params_.iou_weight) || !std::isfinite(params_.distance_weight) ||
        params_.iou_weight < 0.0f || params_.distance_weight < 0.0f ||
        params_.iou_weight + params_.distance_weight <= 0.0f)
        throw std::invalid_argument("MultiTrackParams association weights are invalid");

    // ---- 卡尔曼噪声 ----
    // 过程噪声允许为 0（完全相信模型），观测噪声必须严格为正（否则 S 可能奇异）。
    if (!std::isfinite(params_.process_noise_position) || params_.process_noise_position < 0.0f ||
        !std::isfinite(params_.process_noise_velocity) || params_.process_noise_velocity < 0.0f ||
        !std::isfinite(params_.process_noise_acceleration) || params_.process_noise_acceleration < 0.0f ||
        !std::isfinite(params_.measurement_noise) || params_.measurement_noise <= 0.0f)
        throw std::invalid_argument("MultiTrackParams noise values are invalid");
    // 初始协方差只要求有限且非负：0 表示“初始状态完全确定”（一般不这么用）。
    if (!std::isfinite(params_.initial_position_variance) || params_.initial_position_variance < 0.0f ||
        !std::isfinite(params_.initial_velocity_variance) || params_.initial_velocity_variance < 0.0f ||
        !std::isfinite(params_.initial_acceleration_variance) || params_.initial_acceleration_variance < 0.0f)
        throw std::invalid_argument("MultiTrackParams initial variances are invalid");

    // ---- 尺寸平滑、建轨确认、轨迹合并 ----
    // size_smoothing 是插值系数，必须在 [0,1]；越界会导致尺寸外推/震荡。
    if (!std::isfinite(params_.size_smoothing) || params_.size_smoothing < 0.0f || params_.size_smoothing > 1.0f)
        throw std::invalid_argument("MultiTrackParams::size_smoothing must be in [0,1]");
    // 建轨置信度与合并阈值都是分数，必须落在 [0,1]。
    if (!std::isfinite(params_.init_confidence) || params_.init_confidence < 0.0f || params_.init_confidence > 1.0f)
        throw std::invalid_argument("MultiTrackParams::init_confidence must be in [0,1]");
    if (!std::isfinite(params_.merge_iou) || params_.merge_iou < 0.0f || params_.merge_iou > 1.0f)
        throw std::invalid_argument("MultiTrackParams::merge_iou must be in [0,1]");
    if (params_.draw_lost_frames < 0)
        throw std::invalid_argument("MultiTrackParams::draw_lost_frames must be non-negative");

    // ---- 防漂移 ----
    // 速度/加速度上限必须为正（0 会让目标完全不能动）。
    if (!std::isfinite(params_.max_speed) || params_.max_speed <= 0.0f ||
        !std::isfinite(params_.max_accel) || params_.max_accel <= 0.0f)
        throw std::invalid_argument("MultiTrackParams::max_speed/max_accel must be positive");
    // 衰减系数取 (0,1]：1 表示不衰减，>1 会让丢失期间速度越滚越大。
    if (!std::isfinite(params_.coast_velocity_decay) ||
        params_.coast_velocity_decay <= 0.0f || params_.coast_velocity_decay > 1.0f)
        throw std::invalid_argument("MultiTrackParams::coast_velocity_decay must be in (0,1]");
    if (!std::isfinite(params_.max_coast_ratio) || params_.max_coast_ratio <= 0.0f)
        throw std::invalid_argument("MultiTrackParams::max_coast_ratio must be positive");

    // ---- 绘图与视频输出 ----
    // 线宽/字号为 0 或负会让 OpenCV 的绘制接口断言失败。
    if (params_.thickness <= 0 || params_.label_thickness <= 0 ||
        !std::isfinite(params_.label_font_scale) || params_.label_font_scale <= 0.0)
        throw std::invalid_argument("MultiTrackParams drawing settings are invalid");
    // FourCC 必须 4 个字符；fps 必须有限（0 表示跟随源视频）。
    if (params_.output_fourcc.size() != 4 || !std::isfinite(params_.output_fps))
        throw std::invalid_argument("MultiTrackParams output settings are invalid");

    // 传统识别/融合参数校验。
    // 分数门限：new_track_score 必须不低于 min_score，否则“建轨门槛”形同虚设。
    if (!std::isfinite(traditional_params_.min_score) || traditional_params_.min_score < 0.0f ||
        traditional_params_.min_score > 1.0f ||
        !std::isfinite(traditional_params_.new_track_score) ||
        traditional_params_.new_track_score < traditional_params_.min_score ||
        traditional_params_.new_track_score > 1.0f)
        throw std::invalid_argument("TraditionalParams score thresholds are invalid");
    // 帧计数：至少要确认 1 帧，遗忘阈值不能为负。
    if (traditional_params_.confirm_frames < 1 || traditional_params_.memo_max_missed < 0)
        throw std::invalid_argument("TraditionalParams frame counters are invalid");
    // 面积：下限是绝对像素，上限是占画面比例（换分辨率时只需调下限）。
    if (!std::isfinite(traditional_params_.min_area) || traditional_params_.min_area <= 0.0f ||
        !std::isfinite(traditional_params_.max_area_ratio) ||
        traditional_params_.max_area_ratio <= 0.0f || traditional_params_.max_area_ratio > 1.0f)
        throw std::invalid_argument("TraditionalParams area thresholds are invalid");
    // 形状：圆度与填充率都是 [0,1] 的比例。
    if (!std::isfinite(traditional_params_.min_circularity) ||
        traditional_params_.min_circularity < 0.0f || traditional_params_.min_circularity > 1.0f ||
        !std::isfinite(traditional_params_.min_fill_ratio) ||
        traditional_params_.min_fill_ratio < 0.0f || traditional_params_.min_fill_ratio > 1.0f)
        throw std::invalid_argument("TraditionalParams shape thresholds are invalid");
    // 匹配门限：IoU 是 [0,1]；中心距比例为正；尺寸变化比例非负。
    if (!std::isfinite(traditional_params_.iou_match) ||
        traditional_params_.iou_match < 0.0f || traditional_params_.iou_match > 1.0f ||
        !std::isfinite(traditional_params_.center_match_ratio) ||
        traditional_params_.center_match_ratio <= 0.0f ||
        !std::isfinite(traditional_params_.max_size_change_ratio) ||
        traditional_params_.max_size_change_ratio < 0.0f)
        throw std::invalid_argument("TraditionalParams match thresholds are invalid");
}

// 重置所有轨迹和 ID 计数，开始一段全新的跟踪序列。
// 三个容器要一起清：只清 tracks_ 会留下上一段视频的传统候选记忆，
// 导致新视频开头几十帧被旧位置“救援”；ID 回到 1 是为了日志可读。
void MultiObjectTracker::reset()
{
    tracks_.clear();          // 所有轨迹（含各自的 KalmanFilter）全部丢弃
    visible_tracks_.clear();  // 清掉旧快照，避免外部读到已删除的轨迹
    traditional_memo_.clear();// 清掉传统候选的跨帧记忆
    next_id_ = 1;             // ID 计数归位
}

// 预测阶段：对每一条已有轨迹执行 x'=F*x、P'=F*P*F^T+Q。
// KalmanFilter::predict() 内部完成状态与协方差传播；这里将预测中心转成预测框。
// 目标丢失期间额外做三件事，避免滤波框“飞”到画面外：
//   1. 每帧按 coast_velocity_decay 衰减速度/加速度；
//   2. 把速度、加速度限制在参数上限内；
//   3. 预测中心偏离最近一次观测中心超过 max_coast_ratio*对角线 时拉回边界。
//
// 为什么要在调用 predict() 之前改 statePost：cv::KalmanFilter::predict()
// 用的是 statePre = F * statePost，也就是“先看后验、再外推”。所以想影响
// 这一帧的预测，必须在外推之前修改后验速度，事后改预测值只影响本帧显示。
void MultiObjectTracker::predict(Track& t)
{
    ++t.age;                              // 轨迹总存活帧数加一
    ++t.time_since_update;                // 连续未匹配帧数加一
    t.observed = false;                   // 新帧开始时，尚未有检测与该轨迹匹配
    t.observed_box = cv::Rect2f();        // 清空上一帧的观测框，避免误绘

    // 只在“已经丢了一帧以上”时衰减速度。
    // time_since_update 在进入本函数时已经 +1，所以 >1 表示上一帧就没匹配上；
    // 刚匹配过的轨迹保持原速度，避免正常跟踪时被无谓地“踩刹车”。
    if (t.time_since_update > 1)
    {
        const float decay = params_.coast_velocity_decay;
        t.kf.statePost.at<float>(2, 0) *= decay;
        t.kf.statePost.at<float>(3, 0) *= decay;
        if (params_.motion_model == MotionModel::ConstantAcceleration)
        {
            t.kf.statePost.at<float>(4, 0) *= decay;
            t.kf.statePost.at<float>(5, 0) *= decay;
        }
    }

    // 外推之前先把速度/加速度压回物理上限，防止一帧就冲出画面。
    limitStateMotion(t.kf.statePost, params_);

    cv::Mat prediction = t.kf.predict();
    // 外推结果再夹一次：对常加速模型来说，预测这一步会把加速度累加进速度
    // （v' = v + a*dt），所以预测后的速度可能超过上限；匀速模型下这行是空操作。
    limitStateMotion(prediction, params_);

    float cx = prediction.at<float>(0, 0);
    float cy = prediction.at<float>(1, 0);

    // 限制“丢失期间能飘多远”：预测中心与最近一次观测中心的距离
    // 不允许超过 max_coast_ratio * 框对角线（至少 1 像素，避免退化框把限制压成 0）。
    // 这是最后一道保险：即使速度上限放过了某个异常状态，框也不会离开目标
    // 最后出现的位置太远。拉回时同时改 prediction 和 statePre，保证
    // 后续 correct() 用的预测中心与这里看到的一致。
    const float diagonal = std::sqrt(t.box.width * t.box.width + t.box.height * t.box.height);
    const float maxCoast = std::max(1.0f, params_.max_coast_ratio * diagonal);
    const float dx = cx - t.last_center.x;
    const float dy = cy - t.last_center.y;
    const float distance = std::sqrt(dx * dx + dy * dy);
    if (distance > maxCoast)
    {
        const float scale = maxCoast / distance;
        cx = t.last_center.x + dx * scale;
        cy = t.last_center.y + dy * scale;
        prediction.at<float>(0, 0) = cx;
        prediction.at<float>(1, 0) = cy;
        t.kf.statePre.at<float>(0, 0) = cx;
        t.kf.statePre.at<float>(1, 0) = cy;
    }

    // 预测框沿用当前框尺寸；未匹配前滤波框先等于预测框，
    // 这样“丢失一两帧”时画面上的绿框会继续按惯性滑行，而不是突然消失。
    t.predicted_box = makeBox(cx, cy, t.box.width, t.box.height);
    t.box = t.predicted_box;              // 未匹配前，滤波框暂时采用纯预测结果
}

// 更新阶段：将匹配检测的中心点 z=[cx,cy]^T 输入 correct()。
// 卡尔曼增益 K=P*H^T*(H*P*H^T+R)^-1 决定预测与观测各占多少权重。
// 注意 update() 只处理“已经确认匹配”的轨迹，关联逻辑在 step() 里完成，
// 这样轨迹生命周期管理（hits/age）和状态修正就分开了，便于单独调试。
void MultiObjectTracker::update(Track& t, const Detection& d)
{
    const cv::Point2f c = detectionCenter(d);
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << c.x, c.y);
    // correct() 用测量值修正状态，并把 P 按 Joseph 形式更新（OpenCV 内部实现）。
    const cv::Mat corrected = t.kf.correct(measurement);

    // 检测框尺寸不属于当前运动状态向量，因此用指数平滑单独更新宽高：
    //   new = alpha*观测 + (1-alpha)*旧值
    // alpha=1 完全跟随观测（抖），alpha=0 完全不动（僵）；
    // 0.65 是本视频调出来的折中：目标远近变化跟得上，单帧误检又不会撑大框。
    const float alpha = params_.size_smoothing;
    const float w = detectionWidth(d);
    const float h = detectionHeight(d);
    const float newW = alpha * w + (1.0f - alpha) * t.box.width;
    const float newH = alpha * h + (1.0f - alpha) * t.box.height;

    // correct() 输出后验状态；后验中心 + 平滑尺寸构成最终滤波框。
    t.box = stateBox(corrected, newW, newH);
    t.observed_box = detectionBox(d);     // 保留原始检测框，供红色观测框绘制
    t.confidence = d.confidence;
    t.class_id = d.class_id;
    t.source = d.source;
    t.last_center = c;

    // 丢失若干帧后重新捕获，滤波器里的速度往往已经不代表真实运动
    // （例如目标在遮挡期间掉头）：衰减速度、清掉加速度，避免下一帧又冲出去。
    // 这里用 0.5 的固定衰减而不是参数，因为“重新捕获”本身已经是异常事件，
    // 更需要快速收敛而不是可调性。
    if (t.time_since_update >= 2)
    {
        t.kf.statePost.at<float>(2, 0) *= 0.5f;
        t.kf.statePost.at<float>(3, 0) *= 0.5f;
        if (params_.motion_model == MotionModel::ConstantAcceleration)
        {
            t.kf.statePost.at<float>(4, 0) = 0.0f;
            t.kf.statePost.at<float>(5, 0) = 0.0f;
        }
    }
    limitStateMotion(t.kf.statePost, params_);

    t.time_since_update = 0;              // 成功匹配，丢失计数清零
    ++t.hits;                              // 累计命中次数加一
    ++t.hit_streak;                        // 连续命中帧数加一
    t.observed = true;
}

// 新建轨迹：为该检测分配唯一 ID，并按选定运动模型创建 4/6 维 KalmanFilter。
//
// cv::KalmanFilter(stateSize, 2, 0) 的三个参数分别是：
//   stateSize    状态维数（4 匀速 / 6 常加速）
//   measureSize  观测维数（固定 2：中心 x/y）
//   controlSize  控制量维数（本项目没有控制输入，填 0）
// 构造后必须把 F/H/Q/R/P0/x0 全部显式赋值：构造函数只保证矩阵尺寸正确，
// 各矩阵的默认值（全零或单位阵）不是本工程想要的，漏赋值会出现
// “预测永远等于初始状态”或“协方差保持为零、滤波器完全不更新”这类静默错误。
void MultiObjectTracker::createTrack(const Detection& d)
{
    Track t;
    t.id = next_id_++;
    t.class_id = d.class_id;
    t.source = d.source;

    const bool useAccel = params_.motion_model == MotionModel::ConstantAcceleration;
    const int stateSize = useAccel ? 6 : 4;

    // 状态维数为 4（匀速）或 6（常加速），观测维数固定为 2（中心 x/y）。
    t.kf = cv::KalmanFilter(stateSize, 2, 0, CV_32F);
    t.kf.transitionMatrix = makeTransitionMatrix(params_.motion_model, params_.dt);

    // H 只选择状态中的中心位置分量；速度、加速度通过协方差耦合间接被修正。
    t.kf.measurementMatrix = cv::Mat::zeros(2, stateSize, CV_32F);
    t.kf.measurementMatrix.at<float>(0, 0) = 1.0f;
    t.kf.measurementMatrix.at<float>(1, 1) = 1.0f;

    // Q 表示模型未能解释的随机运动；R 表示检测中心本身的测量误差。
    // Q 表示模型未能解释的随机运动；R 表示检测中心本身的测量误差。
    // 两者都只取对角阵：x/y 两个方向独立，且位置/速度/加速度之间不设相关。
    // 对像素级目标跟踪来说，非对角项带来的收益远小于调参成本。
    t.kf.processNoiseCov = cv::Mat::zeros(stateSize, stateSize, CV_32F);
    t.kf.processNoiseCov.at<float>(0, 0) = params_.process_noise_position;
    t.kf.processNoiseCov.at<float>(1, 1) = params_.process_noise_position;
    t.kf.processNoiseCov.at<float>(2, 2) = params_.process_noise_velocity;
    t.kf.processNoiseCov.at<float>(3, 3) = params_.process_noise_velocity;
    if (useAccel)
    {
        t.kf.processNoiseCov.at<float>(4, 4) = params_.process_noise_acceleration;
        t.kf.processNoiseCov.at<float>(5, 5) = params_.process_noise_acceleration;
    }
    t.kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * params_.measurement_noise;

    // 初始位置来自首个检测；速度/加速度未知，先设为 0，并用较大的方差表达不确定性。
    t.kf.errorCovPost = cv::Mat::zeros(stateSize, stateSize, CV_32F);
    t.kf.errorCovPost.at<float>(0, 0) = params_.initial_position_variance;
    t.kf.errorCovPost.at<float>(1, 1) = params_.initial_position_variance;
    t.kf.errorCovPost.at<float>(2, 2) = params_.initial_velocity_variance;
    t.kf.errorCovPost.at<float>(3, 3) = params_.initial_velocity_variance;
    if (useAccel)
    {
        t.kf.errorCovPost.at<float>(4, 4) = params_.initial_acceleration_variance;
        t.kf.errorCovPost.at<float>(5, 5) = params_.initial_acceleration_variance;
    }

    const cv::Point2f c = detectionCenter(d);
    t.kf.statePost = cv::Mat::zeros(stateSize, 1, CV_32F);
    t.kf.statePost.at<float>(0, 0) = c.x;
    t.kf.statePost.at<float>(1, 0) = c.y;
    t.kf.statePre = t.kf.statePost.clone();

    // 第一帧中预测、滤波、观测框重合；从下一帧开始三者才可能分离。
    t.box = detectionBox(d);
    t.predicted_box = t.box;
    t.observed_box = t.box;
    t.last_center = c;
    t.confidence = d.confidence;
    t.age = 1;
    t.hits = 1;
    t.hit_streak = 1;
    t.time_since_update = 0;
    t.observed = true;
    tracks_.push_back(std::move(t));
}

    // 单帧主流程：先预测所有轨迹，再建立候选匹配，执行一对一关联，最后处理新生/过期轨迹。
    // 六个步骤对应 SORT 系列跟踪器的标准结构：
    //   预测 -> 代价 -> 匹配 -> 新生 -> 清理 -> 合并。
    // 顺序不能乱：关联必须基于“本帧预测框”，否则运动快的目标会匹配到上一帧位置。
void MultiObjectTracker::step(const std::vector<Detection>& detections)
{
    // 第一步：预测所有已有目标的位置。关联必须使用预测框，而不是上一帧旧框。
    for (auto& t : tracks_)
        predict(t);

    // 第二步：为每个“轨迹-检测”组合计算匹配代价。
    // 代价越小越好：cost = w_iou*(1-IoU) + w_dist*(distance/gate)。
    //   - (1-IoU) 越小说明两个框形状/位置越重合；
    //   - distance/gate 把中心距归一化到 [0,1]，避免像素值和 IoU 量纲混用；
    //   - 两项加权后再除以权重和，保证 cost 仍在 [0,1]，权重怎么调都不改变
    //     “阈值含义”，只改变偏好。
    // 这里不追求全局最优（Hungarian），因为目标数很少（<10）且检测偶尔缺失，
    // 贪心在这个规模下的失配率可以接受，代码也好调试（见 README 的说明）。
    struct Candidate { float cost; size_t trackIndex; size_t detectionIndex; };
    std::vector<Candidate> candidates;

    for (size_t ti = 0; ti < tracks_.size(); ++ti)
    {
        const Track& t = tracks_[ti];
        const cv::Point2f predictedCenter(t.predicted_box.x + t.predicted_box.width * 0.5f, t.predicted_box.y + t.predicted_box.height * 0.5f);

        for (size_t di = 0; di < detections.size(); ++di)
        {
            const Detection& d = detections[di];
            if (!validDetection(d) || d.class_id != t.class_id)
                continue;

            const cv::Rect2f detectionRect = detectionBox(d);
            const cv::Point2f detectionCtr = detectionCenter(d);
            const float dx = detectionCtr.x - predictedCenter.x;
            const float dy = detectionCtr.y - predictedCenter.y;
            const float distance = std::sqrt(dx * dx + dy * dy);

            // 距离门限同时参考固定像素值与目标框对角线，适应不同尺寸目标。
            const float diagonal = std::sqrt(t.box.width * t.box.width + t.box.height * t.box.height);
            const float gate = std::max(params_.max_center_distance, diagonal * params_.max_distance_ratio);
            const float iou = rectIoU(t.predicted_box, detectionRect);

            // 物理速度上限：目标一帧内不可能移动超过 max_speed*dt 像素。
            // 超过该值的候选（典型情况是镜头切换后目标整体换位置）直接拒绝，
            // 让旧轨迹自然过期、新位置另建轨迹，而不是把框从画面一头拖到另一头。
            if (distance > params_.max_speed * params_.dt)
                continue;

            // 运动距离明显超限时直接拒绝；低 IoU 但中心仍接近时允许继续参与评分。
            // 第二个条件（iou < min_iou 且 distance > 0.45*gate）是为了过滤
            // “离得不近、形状也不像”的组合：这种情况下 IoU 已经失去区分度，
            // 硬算出来的代价只会让贪心匹配乱点鸳鸯。
            if (distance > gate || (iou < params_.min_iou && distance > gate * 0.45f))
                continue;

            // 归一化距离并加权：归一化保证 cost 不会被“像素距离”量级主导。
            const float normalizedDistance = std::min(1.0f, distance / gate);
            const float weightSum = params_.iou_weight + params_.distance_weight;
            const float cost = (params_.iou_weight * (1.0f - iou) +
                                params_.distance_weight * normalizedDistance) / weightSum;
            candidates.push_back({cost, ti, di});
        }
    }

    // 第三步：按代价从小到大排序，再贪心执行一对一匹配。
    // 每条轨迹和每个检测最多使用一次，避免多个目标抢占同一个检测框。
    // 贪心的顺序很关键：先配“最像”的一对，再配次像的；代价相同的候选
    // 按 std::sort 的不稳定顺序处理，对结果影响可以忽略（本项目目标稀疏）。
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });

    std::vector<bool> usedTrack(tracks_.size(), false);
    std::vector<bool> usedDetection(detections.size(), false);
    for (const Candidate& candidate : candidates)
    {
        // 任一侧已被占用就跳过——这就是“一对一”约束的实现。
        if (usedTrack[candidate.trackIndex] || usedDetection[candidate.detectionIndex])
            continue;

        usedTrack[candidate.trackIndex] = true;
        usedDetection[candidate.detectionIndex] = true;
        update(tracks_[candidate.trackIndex], detections[candidate.detectionIndex]);
    }

    // 第四步：未匹配的有效检测视为新目标，创建独立 ID 和独立卡尔曼状态。
    // 只有置信度足够的检测才允许建轨，避免误检催生大量幽灵轨迹。
    for (size_t di = 0; di < detections.size(); ++di)
    {
        if (!usedDetection[di] && validDetection(detections[di]) &&
            detections[di].allow_new_track &&
            detections[di].confidence >= params_.init_confidence)
            createTrack(detections[di]);
    }

    // 第五步：连续丢失超过 max_age 的轨迹删除，避免旧轨迹无限积累。
    // 用 erase-remove 惯用法：remove_if 只把“要保留的元素”前移到容器前部，
    // 真正删除必须再调 erase；否则容器尾部会留下未定义状态的元素。
    // 判据是 time_since_update（连续未匹配帧数），不是 age（总存活帧数）：
    // 长期存在但一直匹配得上的轨迹不应该被删。
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [this](const Track& t) { return t.time_since_update > params_.max_age; }),
        tracks_.end());

    // 第六步：合并重复轨迹。同一个目标偶尔会被拆成两条轨迹（关联失败后重新建轨），
    // 两条框高度重叠时保留 ID 更小（创建更早）的一条，把命中数继承过来。
    // 先标记 removed 再统一重建，而不是边遍历边 erase：erase 会让下标失效，
    // 而这里的双层循环还要继续用 j 往后找，直接删会踩内存。
    if (params_.merge_iou > 0.0f && tracks_.size() > 1)
    {
        std::vector<bool> removed(tracks_.size(), false);
        for (size_t i = 0; i < tracks_.size(); ++i)
        {
            if (removed[i] || !hasArea(tracks_[i].box))
                continue;

            for (size_t j = i + 1; j < tracks_.size(); ++j)
            {
                if (removed[j] || !hasArea(tracks_[j].box))
                    continue;
                if (tracks_[j].class_id != tracks_[i].class_id)
                    continue;
                if (rectIoU(tracks_[i].box, tracks_[j].box) < params_.merge_iou)
                    continue;

                // 保留 i（ID 更小），把 j 的有价值信息并进来后丢弃 j。
                // 命中数取 max 是为了不让“确认状态”因为合并而退化：
                // 一条刚建的新轨迹并入老轨迹后，不能因为 hits 变小而从画面消失。
                removed[j] = true;
                tracks_[i].hits = std::max(tracks_[i].hits, tracks_[j].hits);
                tracks_[i].hit_streak = std::max(tracks_[i].hit_streak, tracks_[j].hit_streak);
                if (!tracks_[i].observed && tracks_[j].observed)
                {
                    tracks_[i].observed = true;
                    tracks_[i].observed_box = tracks_[j].observed_box;
                    tracks_[i].confidence = tracks_[j].confidence;
                }
            }
        }

        // 有任何一条被标记才重建容器，避免每帧都做一次无谓的 vector 搬运。
        if (std::find(removed.begin(), removed.end(), true) != removed.end())
        {
            std::vector<Track> kept;
            kept.reserve(tracks_.size());
            for (size_t i = 0; i < tracks_.size(); ++i)
                if (!removed[i])
                    kept.push_back(std::move(tracks_[i]));
            tracks_.swap(kept);
        }
    }

    // 第七步：刷新只读视图，供外部查询当前每条轨迹的三种框和生命周期信息。
    refreshViews();
}

// 将内部轨迹复制为轻量只读视图，避免外部直接修改 KalmanFilter。
// 每次 step() 结束调用一次：这样外部拿到的 tracks() 永远和刚处理完的那一帧一致。
void MultiObjectTracker::refreshViews()
{
    // 先清空再按当前轨迹重建，保证已删除的轨迹不会残留在视图里。
    visible_tracks_.clear();
    visible_tracks_.reserve(tracks_.size());
    for (const auto& t : tracks_)
    {
        // 逐字段拷贝：只复制标量和小矩形，不复制 cv::KalmanFilter（那才是状态本体）。
        MultiTrackView view;
        view.id = t.id;
        view.class_id = t.class_id;
        view.source = t.source;
        view.box = t.box;
        view.predicted_box = t.predicted_box;
        view.observed_box = t.observed_box;
        view.confidence = t.confidence;
        view.age = t.age;
        view.hits = t.hits;
        view.time_since_update = t.time_since_update;
        view.observed = t.observed;
        visible_tracks_.push_back(view);
    }
}

// 绘制阶段：黄色预测框、绿色滤波框、红色观测框；标签显示目标 ID。
// 只画“已确认”（hits >= min_hits）且没有丢失太久的轨迹，原因：
//   - 未确认轨迹可能只是误检，画出来会闪；
//   - 丢失太久的轨迹位置纯靠外推，继续画等于在画“幽灵”。
void MultiObjectTracker::draw(cv::Mat& frame) const
{
    if (frame.empty()) return;

    for (const auto& t : tracks_)
    {
        // hits 是累计命中数：达不到 min_hits 的轨迹不参与输出，也不写 CSV。
        if (t.hits < params_.min_hits || !hasArea(t.box))
            continue;

        // 丢失的轨迹只在短暂预测窗口内继续绘制，避免满屏“幽灵框”。
        if (t.time_since_update > params_.draw_lost_frames)
            continue;

        // 标签：ID + 框类型；丢失时额外标出丢失了几帧，方便对照 CSV。
        std::ostringstream idLabel;
        idLabel << "ID " << t.id;
        const std::string suffix = t.observed ? "" : " (lost " + std::to_string(t.time_since_update) + ")";

        if (params_.draw_predicted_box)
            drawMultiBox(frame, t.predicted_box, params_.predicted_color,
                         idLabel.str() + " pred", params_);

        if (params_.draw_filtered_box)
            drawMultiBox(frame, t.box, params_.filtered_color,
                         idLabel.str() + " filt" + suffix, params_);

        // 观测框只在本帧确实匹配到检测时绘制，不能把上一帧检测框误当成本帧观测。
        if (params_.draw_observed_box && t.observed)
            drawMultiBox(frame, t.observed_box, params_.observed_color,
                         idLabel.str() + " obs", params_);
    }
}

// 将 YOLO 与传统识别结果融合成一份观测列表。
//
// 设计前提：YOLO 是学习模型，漏检少、框更贴边，但小目标/极端光照下会漏；
// 传统识别是 HSV + 形状启发式，只在“红/蓝圆靶”这一种目标上有效，
// 而且会把大符上的多块靶面、R 标灯条一起检出来，误检远多于 YOLO。
// 所以融合策略必须是“主 YOLO、辅传统”，绝不能让传统覆盖 YOLO 的框。
//
// 规则（主 YOLO、辅传统）：
//   1. 每条 YOLO 框先与最匹配的传统框配对；配对成功只标记 source=both，
//      最终框仍用 YOLO，传统不能覆盖主检测。
//   2. 未被 YOLO 覆盖的传统候选，若落在已确认且当前丢失的轨迹附近，
//      可以直接作为该轨迹的观测（救援，保持 ID）。
//   3. 没有轨迹先验时，传统候选必须连续 confirm_frames 帧出现，
//      且分数达到 new_track_score，才允许新建轨迹。
//   4. 所有进入 fused 的框之后再统一做一次重叠检查，保证同一目标只有一个观测。
//
// 为什么第 3 条默认是关闭的：传统会在同一块大符上一次检出多个靶面，
// 允许它建轨会让 031 视频的每帧轨迹数从 4 涨到 7（README 里有实测数据）。
std::vector<Detection> MultiObjectTracker::fuseDetections(
    const std::vector<Detection>& yolo,
    const std::vector<Detection>& traditional)
{
    std::vector<Detection> fused;
    fused.reserve(yolo.size() + traditional.size());

    // 传统候选每帧先老化一次；超过保留窗口的记忆必须在本帧匹配前删除，
    // 防止“已经过期”的旧候选被当前检测重新匹配而复活。
    for (auto& memo : traditional_memo_)
        ++memo.missed;
    traditional_memo_.erase(
        std::remove_if(traditional_memo_.begin(), traditional_memo_.end(),
            [this](const TraditionalMemo& memo)
            { return memo.missed > traditional_params_.memo_max_missed; }),
        traditional_memo_.end());

    std::vector<bool> traditionalUsed(traditional.size(), false);

    // ---- 1. YOLO 优先：把每条 YOLO 框和传统框配对 ----
    for (auto y : yolo)
    {
        const cv::Point2f yc = detectionCenter(y);
        const float ydiag = std::sqrt(y.rect.width * y.rect.width +
                                      y.rect.height * y.rect.height);

        int bestIndex = -1;
        float bestIoU = 0.0f;
        float bestDistance = std::numeric_limits<float>::max();

        for (size_t i = 0; i < traditional.size(); ++i)
        {
            // 已被别的 YOLO 框配走的传统候选不再参与：传统候选数量少，
            // 但同一块靶面可能被切成多段，不独占会出现“一个 YOLO 框配多个传统框”。
            if (traditionalUsed[i] || !validDetection(traditional[i]))
                continue;

            const Detection& t = traditional[i];
            const float iou = rectIoU(y.rect, t.rect);

            const cv::Point2f tc = detectionCenter(t);
            const float dx = tc.x - yc.x;
            const float dy = tc.y - yc.y;
            const float distance = std::sqrt(dx * dx + dy * dy);

            // 门限用“两框对角线较大者”而不是固定像素：远处小靶的框只有几十像素，
            // 近处大靶有几百像素，固定门限要么放过远处误差、要么卡死近处匹配。
            const float tdiag = std::sqrt(t.rect.width * t.rect.width +
                                          t.rect.height * t.rect.height);
            const float gate = traditional_params_.center_match_ratio *
                               std::max(ydiag, tdiag);

            // 双条件：IoU 够高，或者中心够近。传统框常因轮廓断裂比 YOLO 框小一圈，
            // 单看 IoU 会漏配，所以补一条中心距条件。
            if (iou < traditional_params_.iou_match && distance > gate)
                continue;

            // 优先 IoU 更高的；IoU 接近时优先中心更近的。
            if (iou > bestIoU + 1e-4f ||
                (std::abs(iou - bestIoU) <= 1e-4f && distance < bestDistance))
            {
                bestIndex = static_cast<int>(i);
                bestIoU = iou;
                bestDistance = distance;
            }
        }

        if (bestIndex >= 0)
        {
            traditionalUsed[static_cast<size_t>(bestIndex)] = true;
            y.source = 2;              // 两路都看到，但保留 YOLO 框
        }
        fused.push_back(y);
    }

    // ---- 2. 传统候选：只补 YOLO 没覆盖的目标 ----
    for (size_t i = 0; i < traditional.size(); ++i)
    {
        if (traditionalUsed[i])
            continue;

        const Detection& t = traditional[i];
        if (!validDetection(t) || t.confidence < traditional_params_.min_score)
            continue;

        const cv::Point2f tc = detectionCenter(t);
        const float tRadius = 0.5f * std::max(detectionWidth(t), detectionHeight(t));
        const float tdiag = std::sqrt(t.rect.width * t.rect.width +
                                      t.rect.height * t.rect.height);

        // 2a. 找最近的“已确认且当前丢失”的轨迹；用于救援。
        int lostTrackIndex = -1;
        float lostTrackDistance = std::numeric_limits<float>::max();

        for (size_t k = 0; k < tracks_.size(); ++k)
        {
            const Track& track = tracks_[k];

            // 只救援已经确认过的轨迹，避免传统候选把未确认轨迹“扶正”。
            if (track.time_since_update <= 0 || track.hits < params_.min_hits)
                continue;

            const cv::Point2f predictedCenter(
                track.predicted_box.x + track.predicted_box.width * 0.5f,
                track.predicted_box.y + track.predicted_box.height * 0.5f);

            const float dx = tc.x - predictedCenter.x;
            const float dy = tc.y - predictedCenter.y;
            const float distance = std::sqrt(dx * dx + dy * dy);

            const float diag = std::sqrt(track.box.width * track.box.width +
                                         track.box.height * track.box.height);
            const float gate = std::max(params_.max_center_distance,
                                        diag * params_.max_distance_ratio);
            if (distance > gate)
                continue;

            // 尺寸必须和轨迹上一帧接近，防止把 R 标/LED 当成靶面接上。
            const float referenceSize = 0.5f * (track.box.width + track.box.height);
            const float candidateSize = 0.5f * (detectionWidth(t) + detectionHeight(t));
            if (referenceSize > 0.0f &&
                std::abs(candidateSize - referenceSize) / referenceSize >
                    traditional_params_.max_size_change_ratio)
                continue;

            if (distance < lostTrackDistance)
            {
                lostTrackDistance = distance;
                lostTrackIndex = static_cast<int>(k);
            }
        }

        // 2b. 更新传统候选的跨帧记忆。
        // 为什么需要记忆：单帧传统候选的证据太弱（一块红色圆斑就可能是靶面），
        // 只有“同一个位置连续多帧都被检出”才能说明它是个稳定目标。
        // 匹配用“中心距 + 半径”的宽松阈值，因为靶面被白十字切开时，
        // 相邻帧算出的外接圆半径会有几像素的抖动。
        int memoIndex = -1;
        float memoDistance = std::numeric_limits<float>::max();
        for (size_t m = 0; m < traditional_memo_.size(); ++m)
        {
            const TraditionalMemo& memo = traditional_memo_[m];
            const float dx = tc.x - memo.center.x;
            const float dy = tc.y - memo.center.y;
            const float distance = std::sqrt(dx * dx + dy * dy);
            // +8 像素的固定容差用于吸收小目标在 1~2 帧内的位移。
            const float mergeRadius = 0.6f * std::max(tRadius, memo.radius) + 8.0f;

            if (distance <= mergeRadius && distance < memoDistance)
            {
                memoIndex = static_cast<int>(m);
                memoDistance = distance;
            }
        }

        if (memoIndex >= 0)
        {
            // 命中已有的记忆：更新位置/分数，命中数 +1，漏检清零。
            TraditionalMemo& memo = traditional_memo_[static_cast<size_t>(memoIndex)];
            // 只有上一帧也命中（本帧老化后 missed==1）才算连续命中；
            // 中间漏检过就从当前帧重新累计，不能把间断命中拼成 confirm_frames。
            memo.hits = (memo.missed == 1) ? (memo.hits + 1) : 1;
            memo.center = tc;
            memo.radius = tRadius;
            memo.missed = 0;
            memo.score = t.confidence;
        }
        else
        {
            // 第一次出现：新建一条记忆，hits=1。
            TraditionalMemo memo;
            memo.center = tc;
            memo.radius = tRadius;
            memo.hits = 1;
            memo.missed = 0;
            memo.score = t.confidence;
            traditional_memo_.push_back(memo);
            memoIndex = static_cast<int>(traditional_memo_.size()) - 1;
        }

        const int hits = traditional_memo_[static_cast<size_t>(memoIndex)].hits;

        // 2c. 是否允许进入跟踪：丢失轨迹附近 1 帧即可（轨迹本身提供了先验，
        //     位置、尺寸、存在性都已经被历史确认过）；
        //     全新目标默认不允许，只有显式打开 allow_new_tracks 时才按连续帧确认。
        const bool rescue = lostTrackIndex >= 0;
        const bool accept =
            (rescue && t.confidence >= traditional_params_.min_score) ||
            (traditional_params_.allow_new_tracks &&
             hits >= traditional_params_.confirm_frames &&
             t.confidence >= traditional_params_.new_track_score);
        if (!accept)
            continue;

        // 2d. 不重复：即使没有和 YOLO 配成对，也不能和已保留框高度重叠。
        // 这一步是最后一道去重，覆盖“传统框和 YOLO 框勉强错开、配对失败、
        // 但实际仍是同一目标”的情况。阈值故意比 1 号规则宽松（IoU 0.2），
        // 因为这里的目标是“宁可错杀重复框，也不能让同一目标出现两个观测”。
        bool duplicate = false;
        for (const auto& f : fused)
        {
            const cv::Point2f fc = detectionCenter(f);
            const float dx = tc.x - fc.x;
            const float dy = tc.y - fc.y;
            const float distance = std::sqrt(dx * dx + dy * dy);
            const float fdiag = std::sqrt(f.rect.width * f.rect.width +
                                          f.rect.height * f.rect.height);
            if (rectIoU(t.rect, f.rect) > 0.2f ||
                distance < 0.5f * std::max(tdiag, fdiag))
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            continue;

        Detection d = t;
        d.source = 1;
        // 只有显式允许“传统新建轨迹”时才让它在 step() 里建轨；
        // 救援框永远只能更新已有轨迹。
        d.allow_new_track = (!rescue && traditional_params_.allow_new_tracks);
        fused.push_back(d);
    }

    // ---- 3. 清理长时间没再出现的传统候选 ----
    traditional_memo_.erase(
        std::remove_if(traditional_memo_.begin(), traditional_memo_.end(),
            [this](const TraditionalMemo& memo)
            {
                return memo.missed > traditional_params_.memo_max_missed;
            }),
        traditional_memo_.end());

    return fused;
}

// 视频入口：统一读取帧率/尺寸，逐帧检测和跟踪，并输出叠加视频及逐目标 CSV。
//
// 每帧的固定流水线（A~F，与循环体内的注释字母对应）：
//   A. YOLO 检测（主）
//   B. 传统识别（辅，可选，异常自行吞掉）
//   C. 融合成唯一的一份观测列表
//   D. step()：预测 -> 关联 -> 校正 -> 轨迹生命周期
//   E. 绘制并写视频
//   F. 写 CSV（每目标一行）
// 检测和跟踪是解耦的：Detector 不持有状态，MultiObjectTracker 不碰模型，
// 所以可以单独替换其中一层（例如换成 TensorRT 推理）。
void MultiObjectTracker::process(const std::string& input_video, const std::string& output_video, const std::string& csv_log, Detector& detector, int class_id)
{
    // 空路径会让 VideoCapture/VideoWriter 返回“未打开”，但报错信息含糊，
    // 这里提前给出明确的参数错误。
    if (input_video.empty() || output_video.empty() || csv_log.empty())
        throw std::invalid_argument("MultiObjectTracker::process requires non-empty paths");

    // 打开视频并读取首帧，首帧尺寸用于创建固定分辨率的输出文件。
    cv::VideoCapture capture(input_video);
    if (!capture.isOpened())
        throw std::runtime_error("Failed to open input video: " + input_video);

    cv::Mat frame;
    if (!capture.read(frame) || frame.empty())
        throw std::runtime_error("Failed to read first video frame");

    const int width = frame.cols;
    const int height = frame.rows;

    // 读取源 FPS；部分视频容器可能返回 0/NaN，此时回退到常用的 25 FPS。
    double sourceFps = capture.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(sourceFps) || sourceFps <= 0.0)
        sourceFps = 25.0;
    const double outputFps = params_.output_fps > 0.0 ? params_.output_fps : sourceFps;

    // 创建输出视频写入器，并检查编码器是否成功打开。
    ensureParentDirectory(output_video);
    ensureParentDirectory(csv_log);
    cv::VideoWriter writer(output_video,
        cv::VideoWriter::fourcc(params_.output_fourcc[0], params_.output_fourcc[1], params_.output_fourcc[2], params_.output_fourcc[3]), outputFps, cv::Size(width, height));
    if (!writer.isOpened())
        throw std::runtime_error("Failed to open output video: " + output_video);

    // CSV 每行记录一个目标在一帧中的预测/滤波/观测框，未观测字段留空。
    std::ofstream csv(csv_log, std::ios::out | std::ios::trunc);
    if (!csv.is_open())
        throw std::runtime_error("Failed to open CSV: " + csv_log);

    reset();
    csv << "frame,timestamp,id,class_id,source,observed,"
           "pred_x,pred_y,pred_w,pred_h,"
           "filtered_x,filtered_y,filtered_w,filtered_h,"
           "observed_x,observed_y,observed_w,observed_h,"
           "confidence,age,hits,lost_frames\n";

    long long frameIndex = 0;
    long long yoloCount = 0;
    long long traditionalCount = 0;
    long long traditionalOnlyCount = 0;
    long long bothCount = 0;

    do
    {
        // A. 主检测：YOLO 每帧都跑，所有检测框统一为原图坐标。
        std::vector<Detection> yoloDetections = detector.detectDetailed(frame, class_id, frame.cols, frame.rows);

        // B. 辅助检测：传统识别每帧都跑，但只作为候选，是否进入跟踪由 fuse 决定。
        //    兜底路径必须自己吞异常，不能因为传统识别失败而中断 YOLO 主流程。
        std::vector<Detection> traditionalDetections;
        if (traditional_params_.enabled)
        {
            try
            {
                traditionalDetections = findbuff(frame, traditional_params_);
            }
            catch (const std::exception&)
            {
                traditionalDetections.clear();
            }
            catch (...)
            {
                traditionalDetections.clear();
            }
        }

        // C. 融合：YOLO 优先；传统只补漏/救援；同一目标最多一个观测。
        std::vector<Detection> detections = fuseDetections(yoloDetections, traditionalDetections);

        yoloCount += static_cast<long long>(yoloDetections.size());
        traditionalCount += static_cast<long long>(traditionalDetections.size());
        for (const auto& d : detections)
        {
            if (d.source == 1)
                traditionalOnlyCount++;
            else if (d.source == 2)
                bothCount++;
        }

        // D. 多目标跟踪：预测、关联、卡尔曼校正和轨迹生命周期维护。
        step(detections);

        // E. 将三类框叠加到当前帧，并写入视频。
        //    必须先 draw 再 write：draw 直接改写 frame 的像素，
        //    顺序反了写出去的是没有叠加层的原视频。
        draw(frame);
        writer.write(frame);

        // F. 写 CSV；预测框和滤波框始终记录，观测框仅在本帧匹配成功时记录。
        //    每个目标每帧一行，字段与表头一一对应；缺失的框写成空字段（",,,"），
        //    这样 pandas/Excel 读进来是 NaN 而不是 0，不会和真实坐标混淆。
        for (const auto& t : tracks_)
        {
            // 未确认的轨迹（命中次数不足）不写入日志，保证 CSV 与画面一致。
            if (t.hits < params_.min_hits)
                continue;

            // 本地 lambda：写一个矩形。valid=false 或框无效时输出 4 个空字段。
            const auto writeRect = [&csv](const cv::Rect2f& r, bool valid)
            {
                if (valid && hasArea(r))
                    csv << r.x << ',' << r.y << ',' << r.width << ',' << r.height;
                else
                    csv << ",,,";
            };

            csv << frameIndex << ',' << std::fixed << std::setprecision(3)
                << static_cast<double>(frameIndex) / sourceFps << ','
                << t.id << ',' << t.class_id << ',' << t.source << ','
                << (t.observed ? 1 : 0) << ',';
            writeRect(t.predicted_box, hasArea(t.predicted_box));
            csv << ',';
            writeRect(t.box, hasArea(t.box));
            csv << ',';
            writeRect(t.observed_box, t.observed);
            csv << ',' << t.confidence << ',' << t.age << ',' << t.hits
                << ',' << t.time_since_update << '\n';
        }

        frameIndex++;
    } while (capture.read(frame) && !frame.empty());

    // 控制台摘要：便于判断传统识别每帧跑了多少、真正进入融合的又有多少。
    std::cout << "processed frames: " << frameIndex
              << ", yolo detections: " << yoloCount
              << ", traditional candidates: " << traditionalCount
              << ", fused both: " << bothCount
              << ", traditional-only: " << traditionalOnlyCount
              << std::endl;
}


// ============================================================
// 传统识别 findbuff：红/蓝圆靶候选生成
//
// 它是 YOLO 的兜底路径，不参与主检测；是否进入跟踪由
// MultiObjectTracker::fuseDetections 的 4 条融合规则决定。
//
// 目标：在没有 YOLO 的时候也能找到“红/蓝圆靶”，作为兜底候选。
// 为什么用 HSV 而不是 RGB：RGB 三个通道随光照强耦合成一个颜色点，
// 同一块红靶在亮处/暗处 RGB 差得很远；HSV 把“颜色(H)、鲜艳度(S)、
// 亮度(V)”拆开，H 在一定范围内就代表“红/蓝”，对曝光变化更稳。
//
// 单帧流程（只产生候选，不碰跟踪状态）：
//   1. HSV 阈值分割出红/蓝像素 -> 二值 mask；
//   2. 形态学闭/开：把被白色十字切开的圆环连起来、去掉噪点；
//   3. 找外轮廓，按面积、圆度、填充率过滤；
//   4. 圆心内外两圈的红色占比，判断“红环 + 暗中心”的靶面结构；
//   5. 打分并排序，按中心距去重（一块靶面常被切成多个环段）。
//
// 返回的 Detection::confidence 是 [0,1] 分数（不是 90/100 那种百分制），
// source=1，allow_new_track=false（默认只能用于救援已有轨迹）。
// ============================================================
std::vector<Detection> findbuff(const cv::Mat& frame, const TraditionalParams& params)
{
    std::vector<Detection> result;
    // 只接受 8 位 3 通道 BGR：其它格式（灰度、浮点、RGBA）先由调用方转换，
    // 这里不做隐式转换，避免“色彩空间不对导致检测全空”的隐蔽 bug。
    if (frame.empty() || frame.type() != CV_8UC3)
        return result;

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // 红色在 OpenCV 的 H 轴上跨 0 点（H ∈ [0,10] ∪ [170,179]），需要两段再合并；
    // 蓝色只有一段（H ≈ 95~125）。
    // S 下限 110：排除白色灯条、灰色金属（它们的 S 很低）；
    // V 下限 80：排除暗背景、阴影里的红色（亮度太低时颜色不可信）；
    // V 上限取 255：靶面高光即使过曝也保留，过曝区域会被后面的环形检查处理。
    cv::Mat mask;
    if (params.red)
    {
        cv::Mat low, high;
        cv::inRange(hsv, cv::Scalar(0, 110, 80), cv::Scalar(10, 255, 255), low);
        cv::inRange(hsv, cv::Scalar(170, 110, 80), cv::Scalar(179, 255, 255), high);
        cv::bitwise_or(low, high, mask);
    }
    else
    {
        cv::inRange(hsv, cv::Scalar(95, 110, 80), cv::Scalar(125, 255, 255), mask);
    }

    // 形态学处理，顺序不能反：
    //   先闭运算（膨胀后腐蚀，7x7 圆核）：把被白色十字/高光切开的圆环连成整块，
    //     核必须比裂缝宽，7 是“能连上 3~4 像素裂缝”的经验值；
    //   再开运算（腐蚀后膨胀，3x3 圆核）：去掉闭运算后残留的孤立小点，
    //     核要小，否则会把远处小靶的圆环腐蚀断。
    const cv::Mat closeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    const cv::Mat openKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, closeKernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, openKernel);

    // 只取最外层轮廓（RETR_EXTERNAL）：靶面内部还有白色十字/内环，
    // 取全部层级会得到大量互相嵌套的轮廓，反而更难判断。
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 面积门限：下限取“绝对像素”和“画面比例”的较大值，上限只按画面比例。
    // 这样同一套参数在 720p 和 1080p 上都大致可用：绝对下限挡住远处噪点，
    // 比例上限挡住整屏红色（例如有人举着红布走过镜头）。
    const double frameArea = static_cast<double>(frame.cols) * frame.rows;
    const double minArea = std::max(static_cast<double>(params.min_area),
                                    frameArea * 0.00005);
    const double maxArea = frameArea * params.max_area_ratio;

    // 候选先收集再统一排序，避免“先到先得”导致小靶抢在大靶前面。
    struct Candidate
    {
        Detection detection;
        float score = 0.0f;
    };
    std::vector<Candidate> candidates;

    for (const auto& contour : contours)
    {
        const double area = cv::contourArea(contour);
        if (area < minArea || area > maxArea)
            continue;

        // 周长（closed=true，即把首尾连起来）。
        const double perimeter = cv::arcLength(contour, true);
        if (perimeter < 1e-6)
            continue;

        // 圆度 = 4πA / P²。这个公式的来历：圆的面积 A=πr²、周长 P=2πr，
        // 代入得 4π·πr²/(4π²r²) = 1，所以完美圆是 1，越不规则越接近 0。
        // 用它排除方形/长条（灯条、机械臂的红色结构件）。
        const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        if (circularity < params.min_circularity)
            continue;

        // 最小外接圆给出圆心和半径；Detection 的框直接用 2r x 2r 的正方形。
        float radius = 0.0f;
        cv::Point2f center;
        cv::minEnclosingCircle(contour, center, radius);
        if (radius <= 0.0f)
            continue;

        // 断成弧的轮廓也能算出外接圆，但 fill（轮廓面积 / 外接圆面积）很低；
        // 用它排除半圆、碎弧——这类形状圆度可能也“看着还行”，但明显不是靶面。
        const double fill = area / (CV_PI * radius * radius);
        if (fill < params.min_fill_ratio)
            continue;

        // 靶面是“红环 + 暗中心”（白十字把圆心分成亮/暗交替的几块）。
        // 这里比较内圈（0.10r~0.35r）与外圈（0.65r~0.90r）的红色占比：
        //   outer 高、inner 低 -> 环形结构 -> 给高分；
        //   否则（实心红点、整块红色、红色灯）-> 低分。
        const auto circleMean = [&mask](const cv::Point2f& c, float r0, float r1) -> float
        {
            // 采样外接矩形的包围盒，再按“到圆心距离在 [r0,r1] 的圆环”筛选像素。
            // 直接遍历矩形会包含圆环外的角点，所以必须用 d2 判断。
            // 边界用 floor/ceil 并夹到图像范围内：靶面贴边时圆心可能离边界
            // 不到一个半径，不夹会越界访问 mask。
            const int x0 = std::max(0, static_cast<int>(std::floor(c.x - r1)));
            const int x1 = std::min(mask.cols - 1, static_cast<int>(std::ceil(c.x + r1)));
            const int y0 = std::max(0, static_cast<int>(std::floor(c.y - r1)));
            const int y1 = std::min(mask.rows - 1, static_cast<int>(std::ceil(c.y + r1)));
            int total = 0;
            int on = 0;
            for (int y = y0; y <= y1; ++y)
            {
                for (int x = x0; x <= x1; ++x)
                {
                    const float dx = x - c.x;
                    const float dy = y - c.y;
                    const float d2 = dx * dx + dy * dy;
                    // 只要圆环内的像素，圆环外的（角点）跳过。
                    if (d2 < r0 * r0 || d2 > r1 * r1)
                        continue;
                    ++total;
                    if (mask.at<uchar>(y, x) != 0)
                        ++on;
                }
            }
            // 返回圆环内“红/蓝像素占比”；采样点太少（total=0）时返回 0，
            // 让调用方按“没有环结构”处理，而不是除零。
            return total > 0 ? static_cast<float>(on) / static_cast<float>(total) : 0.0f;
        };

        // 内圈取 0.10r~0.35r（避开正中心的白色高光），外圈取 0.65r~0.90r
        // （避开轮廓边缘的抗锯齿过渡带）。这两个比例是靶面几何决定的：
        // 红环的宽度大致落在外圈，中心的白十字/暗孔落在内圈。
        const float innerRed = circleMean(center, radius * 0.10f, radius * 0.35f);
        const float outerRed = circleMean(center, radius * 0.65f, radius * 0.90f);
        // 环形结构得分：外圈红多、内圈红少 -> 1；否则 0.35。
        // 不直接给 0 是因为“内圈被高光打爆”时内圈也可能很红，
        // 保留 0.35 让这类候选还能靠圆度进入融合，只是优先级更低。
        const float ringScore = (outerRed > 0.25f && innerRed < 0.55f) ? 1.0f : 0.35f;

        // 分数：圆度 35% + 环结构 55% + 填充率 10%。
        // 权重这样分配的原因：环结构最能区分“靶面”和“普通红色圆斑”，
        // 所以权重最大；圆度次之；填充率只用来排除碎弧，给个很小的权重即可。
        // 最终分数表示“候选的可信度”，不是概率，只用于排序和门限比较。
        // 实心红点/红色灯不会得到高分：它们的 innerRed 也高，ringScore 只有 0.35。
        float score = 0.35f * static_cast<float>(std::min(1.0, circularity)) +
                      0.55f * ringScore +
                      0.10f * static_cast<float>(std::min(1.0, fill));
        score = std::max(0.0f, std::min(1.0f, score));

        // 用外接圆构造一个正方形 Detection：传统识别没有四边形信息，
        // 宽高都取直径，保证 rect 和 center 自洽（下游 IoU/滤波直接可用）。
        Detection d;
        d.center = center;
        d.width = 2.0f * radius;
        d.height = 2.0f * radius;
        d.LT = cv::Point2f(center.x - radius, center.y - radius);
        d.LB = cv::Point2f(center.x - radius, center.y + radius);
        d.RB = cv::Point2f(center.x + radius, center.y + radius);
        d.RT = cv::Point2f(center.x + radius, center.y - radius);
        d.rect = cv::Rect2f(d.LT.x, d.LT.y, d.width, d.height);
        d.confidence = score;
        // source=1 标记“传统来源”；allow_new_track=false 表示即使关联失败，
        // 这条候选也不能自己新建轨迹——是否建轨由 fuseDetections 决定。
        d.source = 1;
        d.allow_new_track = false;
        // 单类别模型，传统识别也统一记为类别 0，才能和 YOLO 检测一起关联。
        d.class_id = 0;
        candidates.push_back({d, score});
    }

    // 同一块靶面可能被白十字切成多个环段，形态学闭运算也未必能完全连起来，
    // 所以按分数从高到低再做一次“中心距去重”：
    // 保留高分候选，低分候选只要离已保留候选太近就丢弃。
    // 先排序再遍历，保证保留的总是“最像靶面”的那个环段。
    std::sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    for (const auto& candidate : candidates)
    {
        bool duplicate = false;
        for (const auto& accepted : result)
        {
            const float dx = candidate.detection.center.x - accepted.center.x;
            const float dy = candidate.detection.center.y - accepted.center.y;
            const float distance = std::sqrt(dx * dx + dy * dy);
            // 阈值取两者平均半径的 0.6 倍：小于这个距离说明两个候选
            // 基本套在同一块靶面上，而不是画面里相邻的两个靶。
            const float radiusSum = 0.5f * (candidate.detection.width + accepted.width);
            if (distance < 0.6f * radiusSum)
            {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            result.push_back(candidate.detection);
    }

    return result;
}















} // namespace yolo11
