# ============================================================================
# CMake 构建脚本：只依赖 OpenCV 的 DNN 模块（cv::dnn）+ 核心模块
#
# 为什么要求 C++17：yolo11.cpp 用到了 std::filesystem（自动创建输出目录），
# 这是 C++17 才进标准库的组件；C++14 下没有，配置阶段就会失败。
# ============================================================================
cmake_minimum_required(VERSION 3.10)

# 工程名只影响生成的工程文件/变量名，和可执行文件名（main）无关。
project(task LANGUAGES CXX)

# 全工程统一 C++17；REQUIRED ON 表示编译器不支持时直接报错，
# 而不是悄悄降级到 C++14（那样会在 filesystem 处出现难懂的编译错误）。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找系统安装的 OpenCV（apt install libopencv-dev / vcpkg / 官方 installer 均可）。
# REQUIRED 表示找不到就终止配置，而不是生成一个“编译到链接阶段才失败”的工程。
# 这里没有写死最低版本，但工程实际在 OpenCV 4.6 上验证过；更早的版本
# 对 YOLO11 导出图的算子支持不全，还可能触发 shape_utils.hpp 断言，建议 4.6+。
find_package(OpenCV REQUIRED)

# 两个源文件：main.cpp（入口/参数） + yolo11.cpp（检测与跟踪实现）。
# yolo11.hpp 是被包含的头文件，不需要单独列出。
add_executable(main main.cpp yolo11.cpp)

# 头文件搜索路径：OpenCV 的头文件不一定在系统默认路径（尤其是自定义安装）。
include_directories(${OpenCV_INCLUDE_DIRS})

# 链接 OpenCV 的各个模块库。DNN 推理、视频读写（videoio）、图像处理（imgproc）
# 都在这里；漏掉任何一个都会在链接阶段报 undefined reference。
target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
)

# ----------------------------------------------------------------------------
# 实验变体（v1 / v2）：与基线共用同一个 main.cpp，只换跟踪实现和编译宏。
# 两个目标都标了 EXCLUDE_FROM_ALL，所以默认的 `cmake --build build` 仍然只
# 编译基线 main，变体代码出问题不会影响主流程；需要时显式构建：
#   cmake --build build --target main_v1
#   cmake --build build --target main_v2
# 宏 YOLO11_VARIANT_SPV1/SPV2 在 main.cpp 里选择头文件并切换默认输出文件名。
# ----------------------------------------------------------------------------
add_executable(main_v1 EXCLUDE_FROM_ALL main.cpp yolo11_spv1.cpp)
target_compile_definitions(main_v1 PRIVATE YOLO11_VARIANT_SPV1)
target_link_libraries(main_v1 PRIVATE ${OpenCV_LIBS})

add_executable(main_v2 EXCLUDE_FROM_ALL main.cpp yolo11_spv2.cpp)
target_compile_definitions(main_v2 PRIVATE YOLO11_VARIANT_SPV2)
target_link_libraries(main_v2 PRIVATE ${OpenCV_LIBS})
