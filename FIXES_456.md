# 修复记录（4 / 5 / 6）

1. **传统候选连续确认**：每帧老化后先清理超出 `memo_max_missed` 的记忆；重新出现但中间漏检的候选将 `hits` 重置为 1，只有真正连续命中才累计 `confirm_frames`。
2. **SPV2 自适应噪声诊断**：`association_distance` 的平方马氏距离分母改为 `P^- + R_eff`，与当前观测实际写入 KalmanFilter 的 `measurementNoiseCov` 使用同一个 `R_eff`。
3. **SPV2 CSV / 视频一致性**：CSV 输出筛选与 `draw()` 对齐，统一应用 `min_hits`、`min_track_quality`、有效框和 `draw_lost_frames` 四项条件。

原始上传文件未覆盖。基线文件按 CMake 预期规范命名为 `yolo11.cpp` / `yolo11.hpp`，其余 SPV1/SPV2 文件名保持不变。
