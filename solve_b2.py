# -*- coding: utf-8 -*-
"""B.2（正解 + 下料反解）：已知法兰位姿 → 求工件位姿，再求放置时法兰走到哪。

安装链（刚性夹持）：

    T_base_part = T_base_flange @ T_flange_tcp @ T_tcp_part

正向：已知 T_base_flange，求 T_base_tcp 与 T_base_part；
反向校验：T_base_flange' = T_base_part @ (T_flange_tcp @ T_tcp_part)^{-1} ≡ T_base_flange（< 1e-9）；
常量链合并（TCP 标定）：T_flange_part = T_flange_tcp @ T_tcp_part（一次算好、缓存复用）；
下料放置：目标 T_base_part_target 已知 → T_base_flange_target = T_base_part_target @ T_flange_part^{-1}，
并回代验证 T_base_flange_target @ T_flange_part ≡ T_base_part_target。
"""

from __future__ import annotations

import numpy as np

from se3 import T_from_rpy, T_inv, T_is_close, T_mul, fmt_vec, print_pose

TOL = 1e-9

# ---------------- 已知量（题面，deg / m） ----------------
T_base_flange = T_from_rpy([0.420, 0.150, 0.560], (150, 0, 45))   # 法兰当前位姿
T_flange_tcp = T_from_rpy([0.000, 0.000, 0.165], (0, 0, 90))      # 夹持中心相对法兰
T_tcp_part = T_from_rpy([0.025, 0.000, 0.060], (0, 0, 0))         # 工件相对夹持中心

# ---------------- 1. 正解 ----------------
T_base_tcp = T_mul(T_base_flange, T_flange_tcp)
T_base_part = T_mul(T_base_tcp, T_tcp_part)
print_pose("【B.2-1】夹持中心 TCP 位姿 T_base_tcp", T_base_tcp)
print()
print_pose("【B.2-1】工件位姿 T_base_part", T_base_part)

# ---------------- 2. 常量链合并（工具坐标系 / TCP 标定） ----------------
T_flange_part = T_mul(T_flange_tcp, T_tcp_part)
print()
print_pose("【B.2-2】合并后的常量链 T_flange_part = T_flange_tcp @ T_tcp_part", T_flange_part)

# ---------------- 反向校验 ----------------
T_base_flange_back = T_mul(T_base_part, T_inv(T_flange_part))
err = float(np.max(np.abs(T_base_flange_back - T_base_flange)))
print("\n反向校验：T_base_part @ T_flange_part^{-1} 与给定 T_base_flange 的最大元素差")
print(f"  max|ΔT| = {err:.3e}  （要求 < {TOL:g}）")
assert err < TOL
print("  => 反解与正解完全互逆 ✔")

# ---------------- 3. 下料放置 ----------------
T_base_part_target = T_from_rpy([0.350, -0.450, 0.090], (180, 0, 0))
T_base_flange_target = T_mul(T_base_part_target, T_inv(T_flange_part))
print()
print_pose("【B.2-3】放置时法兰目标位姿 T_base_flange_target", T_base_flange_target)

# 回代验证：法兰走到该位姿后，工件应精确落在目标位姿
T_check = T_mul(T_base_flange_target, T_flange_part)
err2 = float(np.max(np.abs(T_check - T_base_part_target)))
print("\n回代验证：T_base_flange_target @ T_flange_part 与目标 T_base_part_target 的最大元素差")
print(f"  max|ΔT| = {err2:.3e}  （要求 < {TOL:g}）")
assert err2 < TOL
print("  => 工件精确落在料盘目标位姿 ✔")

# 直观性核对：法兰应在工件正上方 0.225 m（工具 Z 朝下）
print(f"\n直观核对：法兰 z − 工件 z = "
      f"{T_base_flange_target[2, 3] - T_base_part_target[2, 3]:.6f} m "
      f"（= 常量链沿法兰 Z 的 0.225 m 伸出，工具朝下 ✔）")
print(f"  工件 +Z（法兰系常量链第三列）在基座系下 = {fmt_vec(T_flange_part[:3, 2])} "
      f"→ 目标姿态下指向 {fmt_vec(np.round(T_base_flange_target[:3, :3] @ T_flange_part[:3, 2]))}（竖直向下）")

if __name__ == "__main__":
    pass
