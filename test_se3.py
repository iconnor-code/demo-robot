# -*- coding: utf-8 -*-
"""test_se3.py —— se3.py 核心运算的单元测试。

覆盖：
  1. RPY ⇄ 旋转矩阵 ⇄ 四元数的往返一致性（随机采样 + 特殊角 + 万向锁）；
  2. 四元数代数（乘法、共轭、三明治积）与旋转矩阵的一致性；
  3. 齐次变换的链式法则、求逆、点变换；
  4. 与 scipy.spatial.transform.Rotation 的交叉验证（考题允许的用法）。

运行：python test_se3.py   （或 pytest test_se3.py）
"""

from __future__ import annotations

import unittest

import numpy as np

from se3 import (T_inv, T_mul, T_point, make_T, matrix_to_quat, matrix_to_rpy,
                 quat_conj, quat_mul, quat_rotate, quat_to_matrix,
                 quat_to_rpy, rpy_to_matrix, rpy_to_quat)

TOL = 1e-12
RNG = np.random.default_rng(42)


def _rand_rpy(n, avoid_gimbal=True):
    """随机 RPY 样本。

    ry 限制在主值分支 (−π/2, π/2) 内 —— 因为 (rx, ry, rz) 与
    (rx±π, π−ry, rz±π) 表示同一旋转，matrix_to_rpy / quat_to_rpy
    按约定返回 ry ∈ [−π/2, π/2] 的那一支；avoid_gimbal 再避开 |ry| ≈ 90°。
    """
    rpy = np.column_stack([
        RNG.uniform(-np.pi, np.pi, size=n),
        RNG.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, size=n),
        RNG.uniform(-np.pi, np.pi, size=n),
    ])
    if avoid_gimbal:
        while True:
            bad = np.abs(np.cos(rpy[:, 1])) < 1e-6
            if not bad.any():
                break
            rpy[bad, 1] = RNG.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, size=bad.sum())
    return rpy


class TestRpyMatrixQuat(unittest.TestCase):

    def test_rpy_to_matrix_matches_explicit_formula(self):
        """R = Rz Ry Rx 的元素展开式逐项核对。"""
        for rx, ry, rz in _rand_rpy(200):
            R = rpy_to_matrix(rx, ry, rz)
            cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
            expect = np.array([
                [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                [-sy, cy * sx, cy * cx],
            ])
            np.testing.assert_allclose(R, expect, atol=TOL)

    def test_rpy_matrix_roundtrip(self):
        """rpy → R → rpy 往返一致（主值分支内）。"""
        for rx, ry, rz in _rand_rpy(500):
            back = matrix_to_rpy(rpy_to_matrix(rx, ry, rz))
            np.testing.assert_allclose(back, [rx, ry, rz], atol=1e-10)

    def test_rpy_quat_roundtrip(self):
        """rpy → q → rpy 往返一致。"""
        for rx, ry, rz in _rand_rpy(500):
            np.testing.assert_allclose(quat_to_rpy(rpy_to_quat(rx, ry, rz)),
                                       [rx, ry, rz], atol=1e-10)

    def test_matrix_quat_roundtrip(self):
        """R → q → R 往返一致；w 规范化为非负。"""
        for rx, ry, rz in _rand_rpy(500):
            R = rpy_to_matrix(rx, ry, rz)
            q = matrix_to_quat(R)
            self.assertGreaterEqual(q[3], 0.0)
            np.testing.assert_allclose(quat_to_matrix(q), R, atol=1e-12)

    def test_rpy_to_quat_equals_matrix_path(self):
        """纯四元数路径（轴角 ⊗）与矩阵路径给出同一姿态。"""
        for rx, ry, rz in _rand_rpy(500):
            R = rpy_to_matrix(rx, ry, rz)
            np.testing.assert_allclose(quat_to_matrix(rpy_to_quat(rx, ry, rz)), R, atol=1e-12)

    def test_known_rotations(self):
        """几个手算特例：单位元、单轴 90°、180° 组合。"""
        identity = np.eye(3)
        np.testing.assert_allclose(rpy_to_matrix(0, 0, 0), identity, atol=TOL)
        np.testing.assert_allclose(matrix_to_rpy(identity), [0, 0, 0], atol=TOL)
        # Rz(90°)
        np.testing.assert_allclose(rpy_to_matrix(0, 0, np.pi / 2),
                                   [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=TOL)
        # Rx(180°) → RPY (180, 0, 0)，四元数 (1, 0, 0, 0)
        np.testing.assert_allclose(matrix_to_rpy(rpy_to_matrix(np.pi, 0, 0)),
                                   [np.pi, 0, 0], atol=TOL)
        np.testing.assert_allclose(matrix_to_quat(rpy_to_matrix(np.pi, 0, 0)),
                                   [1, 0, 0, 0], atol=TOL)
        # Rz(45°)Rx(180°) = 绕 (cos22.5°, sin22.5°, 0) 转 180°
        R = rpy_to_matrix(np.pi, 0, np.pi / 4)
        q = matrix_to_quat(R)
        np.testing.assert_allclose(np.abs(q), [np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0], atol=1e-12)

    def test_gimbal_lock(self):
        """万向锁（ry = ±90°）下按约定 rz = 0，仍能还原同一旋转矩阵。"""
        for ry_deg in (90.0, -90.0):
            R = rpy_to_matrix(0.0, np.deg2rad(ry_deg), 0.3)   # 传入任意 rz，锁死处只有和有定义
            rx, ry2, rz = matrix_to_rpy(R)
            self.assertEqual(rz, 0.0)
            np.testing.assert_allclose(rpy_to_matrix(rx, ry2, rz), R, atol=1e-12)


class TestQuaternionAlgebra(unittest.TestCase):

    def test_mul_matches_matrix_product(self):
        """R(q1 ⊗ q2) == R(q1) @ R(q2)。"""
        for rx, ry, rz in _rand_rpy(300):
            qa = rpy_to_quat(rx, ry, rz)
            qb = rpy_to_quat(*RNG.uniform(-np.pi, np.pi, 3))
            R = quat_to_matrix(qa) @ quat_to_matrix(qb)
            np.testing.assert_allclose(quat_to_matrix(quat_mul(qa, qb)), R, atol=1e-12)

    def test_conj_is_inverse(self):
        """q ⊗ q* = 单位四元数。"""
        for rx, ry, rz in _rand_rpy(300):
            q = rpy_to_quat(rx, ry, rz)
            np.testing.assert_allclose(quat_mul(q, quat_conj(q)), [0, 0, 0, 1], atol=1e-12)

    def test_sandwich_matches_matrix(self):
        """三明治积旋转矢量 == 矩阵乘矢量。"""
        for rx, ry, rz in _rand_rpy(300):
            q = rpy_to_quat(rx, ry, rz)
            R = quat_to_matrix(q)
            for _ in range(5):
                v = RNG.uniform(-1, 1, 3)
                np.testing.assert_allclose(quat_rotate(q, v), R @ v, atol=1e-12)


class TestTransforms(unittest.TestCase):

    def test_chain_rule(self):
        """T_AC = T_AB @ T_BC，与逐点变换一致。"""
        for _ in range(300):
            Ta = make_T(RNG.uniform(-1, 1, 3), rpy_to_matrix(*RNG.uniform(-np.pi, np.pi, 3)))
            Tb = make_T(RNG.uniform(-1, 1, 3), rpy_to_matrix(*RNG.uniform(-np.pi, np.pi, 3)))
            Tac = T_mul(Ta, Tb)
            p = RNG.uniform(-1, 1, 3)
            np.testing.assert_allclose(T_point(Tac, p), T_point(Ta, T_point(Tb, p)), atol=1e-12)

    def test_inverse(self):
        """T @ T^-1 = I；点的往返变换还原。"""
        for _ in range(300):
            T = make_T(RNG.uniform(-1, 1, 3), rpy_to_matrix(*RNG.uniform(-np.pi, np.pi, 3)))
            np.testing.assert_allclose(T_mul(T, T_inv(T)), np.eye(4), atol=1e-12)
            p = RNG.uniform(-1, 1, 3)
            np.testing.assert_allclose(T_point(T_inv(T), T_point(T, p)), p, atol=1e-12)


class TestScipyCrossValidation(unittest.TestCase):
    """用 scipy 成熟库交叉验证（考题允许：仅用于验证，不参与求解）。"""

    def test_vs_scipy_rotation(self):
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self.skipTest("scipy 不可用，跳过交叉验证")
        for rx, ry, rz in _rand_rpy(500):
            R = rpy_to_matrix(rx, ry, rz)
            # scipy 小写 'xyz' = 绕固定轴 x→y→z = Rz @ Ry @ Rx，与本卷约定一致
            R_sci = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
            np.testing.assert_allclose(R, R_sci, atol=1e-12)
            # 四元数（scipy 默认标量在后 = (x, y, z, w)）
            q_sci = Rotation.from_matrix(R).as_quat()
            q = matrix_to_quat(R)
            if q_sci[3] < 0:
                q_sci = -q_sci
            np.testing.assert_allclose(q, q_sci, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
