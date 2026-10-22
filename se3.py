# -*- coding: utf-8 -*-
"""se3.py —— 刚体位姿核心运算库（自实现）。

约定（与考卷 B 部分一致）：
  * 平移单位：米；角度：题面用 deg，本库一律用 rad；
  * 姿态：固定角 ZYX（RPY），R = Rz(rz) @ Ry(ry) @ Rx(rx)，记作 (rx, ry, rz)；
  * 四元数：Hamilton 约定，顺序 (x, y, z, w)，q ⊗ p 表示先施加 p 再施加 q，
    即 R(q ⊗ p) = R(q) @ R(p)；旋转矢量 v' = q ⊗ (0,v) ⊗ q*；
  * 变换记法：T_AB 表示 B 系在 A 系下的位姿，p_A = T_AB @ p_B；
    链式法则 T_AC = T_AB @ T_BC；求逆 T_BA = T_AB^-1。

核心运算（矩阵相乘/求逆、RPY ⇄ 旋转矩阵 ⇄ 四元数）全部自己实现，
numpy 仅用于数组容器与逐元素算术；scipy 只在单元测试里做交叉验证。
"""

from __future__ import annotations

import numpy as np


# ============================================================
# 基本旋转矩阵
# ============================================================

def rot_x(a: float) -> np.ndarray:
    """绕 X 轴旋转 a（rad）。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c, -s],
                     [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    """绕 Y 轴旋转 a（rad）。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    """绕 Z 轴旋转 a（rad）。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


# ============================================================
# RPY ⇄ 旋转矩阵
# ============================================================

def rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """固定角 ZYX（RPY）→ 旋转矩阵：R = Rz(rz) @ Ry(ry) @ Rx(rx)。"""
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 → 固定角 ZYX（RPY），与 rpy_to_matrix 互逆。

    元素对照（R = Rz Ry Rx）：
        R[2,0] = -sin(ry),  R[2,1] = cos(ry) sin(rx),  R[2,2] = cos(ry) cos(rx)
        R[1,0] = sin(rz) cos(ry),  R[0,0] = cos(rz) cos(ry)
    取 ry ∈ [-pi/2, pi/2] 的主值分支；万向锁（cos(ry)≈0）时约定 rz = 0。
    """
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    if abs(sy) <= 1.0 - 1e-12:                       # 常规分支
        ry = np.arcsin(sy)
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:                                            # 万向锁：R = Ry(±90°)Rx(rx ∓ rz)，
        ry = np.pi / 2.0 if sy > 0 else -np.pi / 2.0 # 约定 rz = 0，由 R01/R11 定 rx
        rz = 0.0
        s = 1.0 if sy > 0 else -1.0
        rx = np.arctan2(s * R[0, 1], R[1, 1])
    return float(rx), float(ry), float(rz)


# ============================================================
# 四元数（Hamilton，顺序 x, y, z, w）
# ============================================================

def quat_mul(q1, q2) -> np.ndarray:
    """四元数乘法 q1 ⊗ q2，满足 R(q1 ⊗ q2) = R(q1) @ R(q2)。"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_conj(q) -> np.ndarray:
    """单位四元数的逆 = 共轭。"""
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    """由旋转轴（单位向量）与转角构造四元数。"""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    half = 0.5 * angle
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def quat_rotate(q, v) -> np.ndarray:
    """用三明治积旋转三维矢量：v' = q ⊗ (0, v) ⊗ q*（纯四元数路径，不经过矩阵）。"""
    v = np.asarray(v, dtype=float)
    p = np.concatenate([v, [0.0]])                   # 纯四元数 (v, 0)
    r = quat_mul(quat_mul(q, p), quat_conj(q))
    return r[:3]


def quat_to_matrix(q) -> np.ndarray:
    """四元数 → 旋转矩阵。"""
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> np.ndarray:
    """旋转矩阵 → 四元数（Shepperd 法：取分母最大的一列解算，数值最稳）。"""
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:                                     # w 最大
        s = np.sqrt(tr + 1.0) * 2.0                  # s = 4w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:    # x 最大
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0   # s = 4x
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
        w = (R[2, 1] - R[1, 2]) / s
    elif R[1, 1] > R[2, 2]:                          # y 最大
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0   # s = 4y
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
        w = (R[0, 2] - R[2, 0]) / s
    else:                                            # z 最大
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0   # s = 4z
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        w = (R[1, 0] - R[0, 1]) / s
    q = np.array([x, y, z, w])
    if q[3] < 0.0:                                   # 规范化：w >= 0
        q = -q
    return q


def rpy_to_quat(rx: float, ry: float, rz: float) -> np.ndarray:
    """RPY → 四元数（纯四元数路径：q = qz(rz) ⊗ qy(ry) ⊗ qx(rx)，不经过矩阵）。"""
    qx = quat_from_axis_angle([1.0, 0.0, 0.0], rx)
    qy = quat_from_axis_angle([0.0, 1.0, 0.0], ry)
    qz = quat_from_axis_angle([0.0, 0.0, 1.0], rz)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_to_rpy(q) -> tuple[float, float, float]:
    """四元数 → RPY（直接由四元数分量公式解算，不构造旋转矩阵）。

    与矩阵路径的元素公式对照（q = (x, y, z, w)）：
        sin(ry) = 2(w*y - x*z)
        tan(rx) = 2(y*z + w*x) / (1 - 2(x^2 + y^2))
        tan(rz) = 2(x*y + w*z) / (1 - 2(y^2 + z^2))
    """
    x, y, z, w = quat_normalize(q)
    sy = 2.0 * (w * y - x * z)
    if abs(sy) <= 1.0 - 1e-12:
        ry = np.arcsin(sy)
        rx = np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
        rz = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
    else:                                            # 万向锁，约定 rz = 0（同矩阵路径）
        ry = np.pi / 2.0 if sy > 0 else -np.pi / 2.0
        rz = 0.0
        s = 1.0 if sy > 0 else -1.0
        rx = np.arctan2(s * 2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z))
    return float(rx), float(ry), float(rz)


# ============================================================
# 齐次变换（SE(3)）
# ============================================================

def make_T(t, R) -> np.ndarray:
    """由平移 t（3 维）与旋转 R（3×3）构造 4×4 齐次变换矩阵。"""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def T_from_rpy(t, rpy_deg) -> np.ndarray:
    """便捷构造：平移（m）+ RPY（deg）。"""
    rx, ry, rz = np.deg2rad(rpy_deg)
    return make_T(t, rpy_to_matrix(rx, ry, rz))


def T_mul(Ta: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """链式法则：T_AC = T_AB @ T_BC。自实现分块相乘。"""
    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Rb, tb = Tb[:3, :3], Tb[:3, 3]
    return make_T(ta + Ra @ tb, Ra @ Rb)


def T_inv(T: np.ndarray) -> np.ndarray:
    """求逆：T^-1 = [R^T, -R^T t; 0 1]（利用旋转矩阵正交性，自实现）。"""
    R, t = T[:3, :3], T[:3, 3]
    return make_T(-R.T @ t, R.T)


def T_point(T: np.ndarray, p) -> np.ndarray:
    """把 B 系中的点变换到 A 系：p_A = T_AB @ [p_B; 1]。"""
    p = np.asarray(p, dtype=float)
    return T[:3, :3] @ p + T[:3, 3]


def T_is_close(Ta, Tb, tol: float = 1e-9) -> bool:
    """两个位姿是否在容差内一致（用于校验）。"""
    return bool(np.max(np.abs(Ta - Tb)) < tol)


# ============================================================
# 输出格式化
# ============================================================

def _clean(x: float, nd: int) -> float:
    """显示层把 −1e−17 这类舍入噪声归零，避免打印出 −0.000000。"""
    return 0.0 if abs(x) < 0.5 * 10.0 ** (-nd) else x


def fmt_vec(v, nd: int = 6) -> str:
    return "(" + ", ".join(f"{_clean(float(x), nd):.{nd}f}" for x in np.asarray(v)) + ")"


def fmt_mat(M, nd: int = 6) -> str:
    return "\n".join("  [" + "  ".join(f"{_clean(float(x), nd):{nd + 4}.{nd}f}" for x in row) + "]"
                     for row in np.asarray(M))


def print_pose(title: str, T: np.ndarray, nd: int = 6) -> None:
    """按考题要求打印：平移、旋转矩阵、RPY（deg）、四元数 (x, y, z, w)。"""
    R = T[:3, :3]
    rx, ry, rz = matrix_to_rpy(R)
    q = matrix_to_quat(R)
    print(f"{title}")
    print(f"  平移 (x, y, z) / m : {fmt_vec(T[:3, 3], nd)}")
    print(f"  旋转矩阵 R         :\n{fmt_mat(R, nd)}")
    print(f"  RPY (rx, ry, rz)/deg: {fmt_vec(np.rad2deg([rx, ry, rz]), nd)}")
    print(f"  四元数 (x, y, z, w): {fmt_vec(q, nd)}")
