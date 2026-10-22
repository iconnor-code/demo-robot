# -*- coding: utf-8 -*-
"""B.3（eye-to-hand 视觉引导定位）：像素 → 相机系 → 基座系 → 法兰目标位姿。

链路：
    1) 像素 (u, v) + 深度 z_c --内参反投影--> 圆盘中心在相机系 P_cam；
    2) P_cam --固定手眼外参 T_base_cam--> 基座系 P_base（与机器人当前位姿无关）；
    3) 圆盘平放、法向竖直、绕 Z 旋转对称 → 工件位姿取 RPY (180, 0, 0)；
    4) 套 B.1 公式：T_base_flange_target = T_base_disc @ T_flange_tcp^{-1}，
       并回代验证夹持中心落在圆盘中心。
"""

from __future__ import annotations

import numpy as np

from se3 import T_from_rpy, T_inv, T_mul, T_point, fmt_vec, print_pose

TOL = 1e-9

# ---------------- 已知量（题面） ----------------
fx = fy = 800.0            # 内参（像素），1280×720，忽略畸变
cx, cy = 640.0, 360.0
u, v = 720.0, 280.0        # 圆盘中心像素
z_c = 0.400                # 深度（m，深度相机给出）
T_base_cam = T_from_rpy([0.700, -0.200, 1.000], (180, 0, 15))   # 手眼外参（固定）
T_flange_tcp = T_from_rpy([0.000, 0.000, 0.165], (0, 0, 45))    # 与 B.1 同一把夹爪

# ---------------- 1. 像素 → 相机系（反投影） ----------------
# 针孔模型：u = fx·X/Z + cx, v = fy·Y/Z + cy → 反解
x_c = (u - cx) / fx * z_c
y_c = (v - cy) / fy * z_c
P_cam = np.array([x_c, y_c, z_c])
print("【B.3-1】像素 → 相机系（反投影）")
print(f"  x_c = (u − cx)/fx · z_c = ({u:.0f} − {cx:.0f})/{fx:.0f} × {z_c} = {x_c:+.6f} m")
print(f"  y_c = (v − cy)/fy · z_c = ({v:.0f} − {cy:.0f})/{fy:.0f} × {z_c} = {y_c:+.6f} m")
print(f"  z_c = {z_c:.6f} m")
print(f"  => 圆盘中心（相机系）P_cam = {fmt_vec(P_cam)} m")

# ---------------- 2. 相机系 → 基座系 ----------------
P_base = T_point(T_base_cam, P_cam)
print("\n【B.3-2】相机系 → 基座系：P_base = T_base_cam @ P_cam")
print(f"  => 圆盘中心（基座系）P_base = {fmt_vec(P_base)} m")

# ---------------- 3. 工件完整位姿（6 DoF） ----------------
# 圆盘平放、法向竖直（+Z 朝上），绕基座 Z 旋转对称 → 取 rz = 0
T_base_disc = T_from_rpy(P_base, (180, 0, 0))
print("\n【B.3-3】工件完整位姿 T_base_disc（平移取圆心，姿态取 RPY(180, 0, 0)）")
print_pose("  T_base_disc", T_base_disc)

# ---------------- 4. 法兰目标位姿 + 回代验证 ----------------
T_base_flange_target = T_mul(T_base_disc, T_inv(T_flange_tcp))
print()
print_pose("【B.3-4】机器人抓取时法兰目标位姿 T_base_flange_target", T_base_flange_target)

T_check = T_mul(T_base_flange_target, T_flange_tcp)
err_t = float(np.max(np.abs(T_check[:3, 3] - P_base)))
err_R = float(np.max(np.abs(T_check[:3, :3] - T_base_disc[:3, :3])))
print("\n回代验证：T_base_flange_target @ T_flange_tcp 的夹持中心应落在圆盘中心")
print(f"  夹持中心 = {fmt_vec(T_check[:3, 3])} m，与圆盘中心最大差 {err_t:.3e} m")
print(f"  姿态与 T_base_disc 最大差 {err_R:.3e}")
assert err_t < TOL and err_R < TOL
print(f"  => 差异 < {TOL:g}，夹持中心精确落在圆盘中心 ✔")

if __name__ == "__main__":
    pass
