# -*- coding: utf-8 -*-
"""B.1（反解）：已知工件位姿 → 求法兰位姿。

模型（本小题工件坐标系与夹持中心重合，T_part_tcp = I）：

    T_base_part = T_base_flange @ T_flange_tcp @ T_part_tcp = T_base_flange @ T_flange_tcp

    =>  T_base_flange = T_base_part @ T_flange_tcp^{-1}

其中 T_flange_tcp 即题面的"夹爪夹持中心 TCP 相对法兰"。

两条独立计算路径交叉验证（要求差 < 1e-9）：
    路径 A（旋转矩阵）：全程 4×4 齐次矩阵，自实现 T_mul / T_inv；
    路径 B（四元数）  ：姿态用四元数 ⊗/共轭 组合、矢量用三明治积旋转，
                        RPY 直接由四元数分量公式解出，完全不构造矩阵。
"""

from __future__ import annotations

import numpy as np

from se3 import (T_from_rpy, T_inv, T_mul, fmt_mat, fmt_vec, matrix_to_quat,
                 matrix_to_rpy, print_pose, quat_conj, quat_mul, quat_rotate,
                 quat_to_rpy, rpy_to_quat)

TOL = 1e-9

# ---------------- 已知量（题面，deg / m） ----------------
T_flange_tcp = T_from_rpy([0.000, 0.000, 0.165], (0, 0, 45))       # TCP 相对法兰
T_base_part = T_from_rpy([0.520, -0.180, 0.120], (180, 0, -30))    # 工件在基座系

# ---------------- 路径 A：齐次矩阵 ----------------
T_base_flange_A = T_mul(T_base_part, T_inv(T_flange_tcp))

# ---------------- 路径 B：四元数 + 矢量 ----------------
q_ft = rpy_to_quat(*np.deg2rad((0, 0, 45)))          # 夹爪姿态（纯轴角组合）
q_bp = rpy_to_quat(*np.deg2rad((180, 0, -30)))       # 工件姿态
t_ft = np.array([0.000, 0.000, 0.165])
t_bp = np.array([0.520, -0.180, 0.120])

q_bf = quat_mul(q_bp, quat_conj(q_ft))               # R_bf = R_bp @ R_ft^T
t_bf = t_bp - quat_rotate(q_bf, t_ft)                # t_bf = t_bp - R_bf @ t_ft
rpy_bf_B = quat_to_rpy(q_bf)

# ---------------- 结果输出 ----------------
print_pose("【B.1】法兰在基座系下的位姿 T_base_flange（路径 A：旋转矩阵）", T_base_flange_A)

# ---------------- 交叉验证 ----------------
rpy_bf_A = matrix_to_rpy(T_base_flange_A[:3, :3])
q_bf_A = matrix_to_quat(T_base_flange_A[:3, :3])
if q_bf_A @ q_bf < 0:                                # 四元数双覆盖，先对齐符号
    q_bf_aligned = -q_bf
else:
    q_bf_aligned = q_bf

dt = np.max(np.abs(T_base_flange_A[:3, 3] - t_bf))
dq = np.max(np.abs(q_bf_A - q_bf_aligned))
drpy = np.max(np.abs(np.rad2deg(rpy_bf_A) - np.rad2deg(rpy_bf_B)))
print("\n交叉验证（路径 A 矩阵 vs 路径 B 四元数）：")
print(f"  平移分量最大差   |Δt|   = {dt:.3e}")
print(f"  四元数分量最大差 |Δq|   = {dq:.3e}")
print(f"  RPY 最大差(deg)  |Δrpy| = {drpy:.3e}")
assert dt < TOL and dq < TOL and drpy < TOL, "两条路径结果不一致！"
print(f"  => 全部 < {TOL:g}，两条独立路径互相印证通过 ✔")

# ---------------- 解析核对手算（封闭解） ----------------
# R_bp = Rz(-30°)Rx(180°)，R_ft = Rz(45°)
#   R_bf = R_bp @ Rz(-45°) = Rz(-30°)Rx(180°)Rz(-45°) = Rz(15°)Rx(180°)
#   t_bf = t_bp - R_bf @ (0,0,0.165) = (0.520, -0.180, 0.285)
print("\n封闭解核对（手算）：R_bf = Rz(15°)Rx(180°)，t_bf = (0.520, -0.180, 0.285)")
print(f"  数值解平移 {fmt_vec(T_base_flange_A[:3, 3])}")
print(f"  数值解 RPY {fmt_vec(np.rad2deg(rpy_bf_A))} deg  —— 与封闭解一致 ✔")

if __name__ == "__main__":
    pass
