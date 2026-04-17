"""
Python port of example.m
论文: Liangzu Peng, Christian Kümmerle, René Vidal,
     "On the Convergence of IRLS and Its Variants in Outlier-Robust Estimation", CVPR 2023

本文件实现了两种鲁棒点云配准算法：
  1. MS-GNC-TLS —— 多尺度渐进非凸截断最小二乘
  2. GNC-IRLS0  —— 渐进非凸迭代重加权最小二乘（IRLS-0 权函数）

依赖: numpy, scipy
"""

import numpy as np
import scipy.io
from scipy.spatial.transform import Rotation
from scipy.stats import chi2


# ─────────────────────────────────────────────
#  辅助函数
# ─────────────────────────────────────────────

def rand_rotation(rotation_bound: float = 2 * np.pi) -> np.ndarray:
    """生成随机旋转矩阵 (3×3)。

    参数
    ----
    rotation_bound : 旋转角范围上界，默认 2π（无限制）。

    返回
    ----
    R : ndarray, shape (3, 3)，正交旋转矩阵，det(R) = +1。
    """
    # 在 [-rotation_bound/2, rotation_bound/2] 内随机采样旋转角
    angle = rotation_bound * np.random.rand() - rotation_bound / 2.0
    # 随机单位旋转轴
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    # 用轴角表示构造旋转矩阵（等价于 MATLAB 的 axang2rotm）
    R = Rotation.from_rotvec(angle * axis).as_matrix()
    return R


def gen_point_cloud_registration(N: int,
                                  outlier_ratio: float = 0.0,
                                  translation_bound: float = 10.0,
                                  noise_sigma: float = 0.01) -> dict:
    """随机生成点云配准问题。

    对应 MATLAB 中的 gen_point_cloud_registration()，
    原始代码来自 MIT-SPARK/CertifiablyRobustPerception（Heng Yang, 2021）。

    参数
    ----
    N               : 对应点对数量。
    outlier_ratio   : 外点比例 (0~1)。
    translation_bound : 平移向量的范数上界。
    noise_sigma     : 内点高斯噪声标准差。

    返回
    ----
    problem : dict，包含点云、真值变换及噪声界等信息。
    """
    # 随机生成点云 A（3×N）
    cloud_A = np.random.randn(3, N)

    # 随机真值旋转与平移
    R_gt = rand_rotation()
    t_gt = np.random.randn(3, 1)
    t_gt = t_gt / np.linalg.norm(t_gt)
    t_gt = translation_bound * np.random.rand() * t_gt

    # 点云 B = R * A + t + 噪声
    cloud_B = R_gt @ cloud_A + t_gt + noise_sigma * np.random.randn(3, N)

    # 添加外点（随机替换末尾 nrOutliers 列）
    nr_outliers = round(N * outlier_ratio)
    if (N - nr_outliers) < 3:
        raise ValueError("点云配准至少需要 3 对内点对应关系。")

    if nr_outliers > 0:
        print(f"点云配准：随机生成 {nr_outliers} 个外点。")
        outlier_B = np.random.randn(3, nr_outliers)
        center_B = cloud_B.mean(axis=1, keepdims=True)          # (3,1)
        outlier_ids = list(range(N - nr_outliers, N))
        cloud_B[:, outlier_ids] = outlier_B + center_B
    else:
        outlier_ids = []

    # chi2inv(0.99, 3) 为自由度为 3 的卡方分布 0.99 分位点
    noise_bound_sq = noise_sigma ** 2 * chi2.ppf(0.99, df=3)
    noise_bound_sq = max(4e-2, noise_bound_sq)                  # 保证数值稳定性

    print(f"N: {N}, outlierRatio: {outlier_ratio}, translationBound: {translation_bound}, "
          f"noiseBoundSq: {noise_bound_sq:.6g}, noiseBound: {np.sqrt(noise_bound_sq):.6g}.")

    problem = {
        "type":            "point cloud registration",
        "N":               N,
        "outlier_ratio":   outlier_ratio,
        "noise_sigma":     noise_sigma,
        "translation_bound": translation_bound,
        "cloud_A":         cloud_A,
        "cloud_B":         cloud_B,
        "nr_outliers":     nr_outliers,
        "outlier_ids":     outlier_ids,
        "R_gt":            R_gt,
        "t_gt":            t_gt,
        "noise_bound_sq":  noise_bound_sq,
        "noise_bound":     np.sqrt(noise_bound_sq),
    }
    return problem


def get_angular_error(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    """计算两个旋转矩阵之间的角度误差（单位：度）。

    公式：error = arccos( (trace(R_gt^T * R_est) - 1) / 2 )
    """
    val = (np.trace(R_gt.T @ R_est) - 1.0) / 2.0
    # 数值裁剪，防止浮点误差导致 arccos 域外
    val = np.clip(val, -1.0, 1.0)
    return float(np.degrees(np.abs(np.arccos(val))))


# ─────────────────────────────────────────────
#  最小二乘旋转求解
# ─────────────────────────────────────────────

def ls_rotation_search(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """用 SVD 求解加权最小二乘旋转 R̂ = argmin ||Y - R X||_F。

    参数
    ----
    Y, X : ndarray, shape (3, m)。

    返回
    ----
    R_hat : ndarray, shape (3, 3)，最优旋转矩阵。
    """
    # 交叉协方差矩阵
    M = Y @ X.T                           # 3×3

    U, _, Vt = np.linalg.svd(M)
    V = Vt.T

    # 修正行列式，确保 det(R) = +1（非反射）
    D = np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V)])

    R_hat = U @ D @ Vt
    return R_hat


def ls_point_cloud_registration(Y: np.ndarray, X: np.ndarray,
                                 weights: np.ndarray) -> tuple:
    """加权最小二乘点云配准，求解 R, t。

    算法：
      1. 用权重计算加权质心；
      2. 去质心后用 SVD 求最优旋转；
      3. 由质心关系计算平移。

    参数
    ----
    Y, X    : ndarray, shape (3, m)，目标点云与源点云。
    weights : ndarray, shape (m,)，每对点的非负权重。

    返回
    ----
    R : ndarray (3, 3)，t : ndarray (3, 1)。
    """
    s = weights.sum()

    # 加权质心，shape (3, 1)
    x = (X @ weights[:, None]) / s        # 等价于 MATLAB: X * weights' / s
    y = (Y @ weights[:, None]) / s

    # 去质心
    X_ = X - x
    Y_ = Y - y

    # 将权重的平方根融入点云（等价于加权 Frobenius 范数）
    sqrtw = np.sqrt(weights)              # shape (m,)
    R = ls_rotation_search(sqrtw * Y_, sqrtw * X_)

    t = y - R @ x
    return R, t


# ─────────────────────────────────────────────
#  GNC 权重更新
# ─────────────────────────────────────────────

def gnc_weights_update(weights: np.ndarray, mu: float,
                        residuals: np.ndarray, barc2: float) -> np.ndarray:
    """标准 GNC-TLS 权重更新（对应 MATLAB gncWeightsUpdate）。

    参数
    ----
    weights   : 当前权重向量，shape (m,)。
    mu        : 控制鲁棒函数形状的参数。
    residuals : 每对点的残差平方，shape (m,)。
    barc2     : 噪声界的平方（截断阈值）。

    返回
    ----
    更新后的权重向量。
    """
    ub = (mu + 1) / mu * barc2
    lb = mu / (mu + 1) * barc2

    new_weights = np.empty_like(weights)
    for k in range(len(residuals)):
        if residuals[k] - ub >= 0:
            new_weights[k] = 0.0                               # 外点
        elif residuals[k] - lb <= 0:
            new_weights[k] = 1.0                               # 内点
        else:
            # 过渡区域，连续权重
            new_weights[k] = np.sqrt(barc2 * mu * (mu + 1) / residuals[k]) - mu
    return new_weights


def m_gnc_weights_update(weights: np.ndarray, mu: float,
                          residuals: np.ndarray, barc2: float) -> np.ndarray:
    """多尺度（Majorize）GNC-TLS 权重更新（对应 MATLAB M_gncWeightsUpdate）。

    与标准版本的区别在于上下界的定义（使用二阶近似）。
    """
    ub = (mu + 1) ** 2 / mu ** 2 * barc2
    lb = barc2

    new_weights = np.empty_like(weights)
    for k in range(len(residuals)):
        if residuals[k] - ub >= 0:
            new_weights[k] = 0.0
        elif residuals[k] - lb <= 0:
            new_weights[k] = 1.0
        else:
            new_weights[k] = np.sqrt(barc2 / residuals[k]) * (mu + 1) - mu
    return new_weights


# ─────────────────────────────────────────────
#  主算法
# ─────────────────────────────────────────────

def gnc_tls_point_cloud_registration(Y: np.ndarray, X: np.ndarray,
                                      noise_bound_squared: float,
                                      stop_th: float = 1e-10,
                                      majorize: bool = True,
                                      superlinear: bool = True):
    """(MS-)GNC-TLS 鲁棒点云配准。

    通过渐进非凸（GNC）策略逐步收紧截断最小二乘（TLS）目标函数，
    迭代估计最优旋转 R 和平移 t。

    参数
    ----
    Y                    : ndarray (3, m)，目标点云。
    X                    : ndarray (3, m)，源点云。
    noise_bound_squared  : 噪声界的平方，用于区分内点与外点。
    stop_th              : 代价函数变化量的收敛阈值。
    majorize             : True 使用多尺度（MS）权重更新，否则使用标准更新。
    superlinear          : True 时 mu 采用超线性增长策略。

    返回
    ----
    R_hat : ndarray (3, 3)，t_hat : ndarray (3, 1)。
    """
    m = X.shape[1]
    weights = np.ones(m)                  # 初始化均匀权重

    pre_cost = np.inf
    R_hat, t_hat = None, None

    for i in range(100):
        # 1. 加权最小二乘求解 R, t
        R_hat, t_hat = ls_point_cloud_registration(Y, X, weights)

        # 2. 计算每对点的残差范数和残差平方
        abs_residual = np.linalg.norm(Y - R_hat @ X - t_hat, axis=0)  # (m,)
        squared_residual = abs_residual ** 2

        # 3. 第一次迭代时根据最大残差初始化 mu
        if i == 0:
            max_res = squared_residual.max()
            mu = max(1.0 / (5.0 * max_res / noise_bound_squared - 1.0), 1e-6)

        # 4. 更新权重（多尺度或标准 GNC-TLS）
        if majorize:
            weights = m_gnc_weights_update(weights, mu, squared_residual,
                                           noise_bound_squared)
        else:
            weights = gnc_weights_update(weights, mu, squared_residual,
                                         noise_bound_squared)

        # 5. 计算加权代价并检查收敛
        cost = float(weights @ squared_residual)
        cost_diff = abs(cost - pre_cost)

        if cost_diff <= stop_th:
            break

        # 6. 更新 mu（控制鲁棒函数逐渐趋近 TLS）
        if superlinear:
            if mu < 1.0:
                mu = min(np.sqrt(mu) * 1.4, 1e16)
            else:
                mu = min(mu * 1.4, 1e16)
        else:
            mu = min(mu * 1.4, 1e16)

        pre_cost = cost

    return R_hat, t_hat


def gnc_irls0_point_cloud_registration(Y: np.ndarray, X: np.ndarray,
                                        epsilon: float, beta: float,
                                        epsilon_min: float,
                                        stop_th: float = 1e-10):
    """GNC-IRLS0 鲁棒点云配准。

    使用 IRLS-0（零范数松弛的迭代重加权最小二乘）并结合 GNC 策略，
    通过逐步减小 epsilon 使权函数趋近于截断平方损失的次梯度。

    IRLS-0 权重：w_i = 1 / max(r_i, epsilon)，再平方。
    epsilon 更新规则：epsilon = max(beta * epsilon^2, epsilon_min)。

    参数
    ----
    Y            : ndarray (3, m)，目标点云。
    X            : ndarray (3, m)，源点云。
    epsilon      : IRLS-0 的初始光滑化参数（越小越逼近 L0 范数）。
    beta         : epsilon 衰减因子（0 < beta < 1 时指数衰减）。
    epsilon_min  : epsilon 的最小值（通常为噪声界）。
    stop_th      : 收敛阈值。

    返回
    ----
    R_hat : ndarray (3, 3)，t_hat : ndarray (3, 1)。
    """
    m = X.shape[1]
    weights = np.ones(m)                  # 初始化均匀权重

    pre_cost = np.inf
    R_hat, t_hat = None, None

    for i in range(100):
        # 1. 加权最小二乘求解 R, t
        R_hat, t_hat = ls_point_cloud_registration(Y, X, weights)

        # 2. 计算残差范数
        D = Y - R_hat @ X - t_hat        # 残差矩阵 (3, m)
        residual = np.linalg.norm(D, axis=0)   # 每列的 L2 范数 (m,)

        # 3. 更新 IRLS-0 权重：w_i = 1 / max(r_i, ε)，再平方
        weights = 1.0 / np.maximum(residual, epsilon)
        weights = weights ** 2

        # 4. 计算加权代价并检查收敛
        cost = float(weights @ (residual ** 2))
        cost_diff = abs(cost - pre_cost)

        if cost_diff <= stop_th:
            break

        # 5. 更新光滑化参数 epsilon（逐步趋近 epsilon_min）
        epsilon = max(beta * epsilon ** 2, epsilon_min)

        pre_cost = cost

    return R_hat, t_hat


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(0)                     # 固定随机种子，便于复现

    # ── 参数设置 ──────────────────────────────
    m = 1000                              # 总点对数
    nr_outliers = 900                     # 外点数量（外点比例 = 90%）
    sigma = 0.01                          # 高斯噪声标准差

    # 噪声界：5.54σ ≈ sqrt(chi2inv(0.99,3)) * sigma（经验倍数）
    noise_bound = 5.54 * sigma

    stop_th = 1e-10                       # 收敛阈值

    # ── 生成随机点云配准问题 ──────────────────
    problem = gen_point_cloud_registration(
        N=m,
        outlier_ratio=nr_outliers / m,
        noise_sigma=sigma,
        translation_bound=10.0,
    )
    X    = problem["cloud_A"]             # 源点云，3×m
    Y    = problem["cloud_B"]             # 目标点云，3×m
    R_gt = problem["R_gt"]               # 真值旋转
    t_gt = problem["t_gt"]               # 真值平移

    # ── 将问题数据保存到 problem_data.mat ────────
    # MATLAB 端直接加载此文件，保证两种实现使用完全相同的输入数据
    scipy.io.savemat(
        "problem_data.mat",
        {
            "cloudA": X,          # 3×m 源点云
            "cloudB": Y,          # 3×m 目标点云
            "R_gt":   R_gt,       # 3×3 真值旋转
            "t_gt":   t_gt,       # 3×1 真值平移
        },
    )
    print("问题数据已保存到 problem_data.mat（MATLAB 端请先运行 Python 再运行 MATLAB）\n")

    # ── 运行 MS-GNC-TLS ───────────────────────
    R_mgnc, t_mgnc = gnc_tls_point_cloud_registration(
        Y, X,
        noise_bound_squared=noise_bound ** 2,
        stop_th=stop_th,
        majorize=True,
        superlinear=True,
    )

    # ── 运行 GNC-IRLS0 ────────────────────────
    epsilon = 1.0
    beta    = 0.8
    R_irls0, t_irls0 = gnc_irls0_point_cloud_registration(
        Y, X,
        epsilon=epsilon,
        beta=beta,
        epsilon_min=noise_bound,
        stop_th=stop_th,
    )

    # ── 计算误差 ──────────────────────────────
    ang_err_mgnc  = get_angular_error(R_gt, R_mgnc)
    ang_err_irls0 = get_angular_error(R_gt, R_irls0)

    t_err_mgnc  = float(np.linalg.norm(t_mgnc  - t_gt) / np.linalg.norm(t_gt))
    t_err_irls0 = float(np.linalg.norm(t_irls0 - t_gt) / np.linalg.norm(t_gt))

    # ── 打印结构化结果（与 MATLAB 格式对齐，便于对比）──
    def _fmt_matrix(M: np.ndarray) -> str:
        """将矩阵每行格式化为固定精度字符串，行间用换号分隔。"""
        rows = []
        for row in M:
            rows.append("  " + "  ".join(f"{v:.15g}" for v in row))
        return "\n".join(rows)

    def _fmt_vec(v: np.ndarray) -> str:
        return "  " + "  ".join(f"{x:.15g}" for x in v.ravel())

    print("=== MS-GNC-TLS ===")
    print("R:")
    print(_fmt_matrix(R_mgnc))
    print("t:")
    print(_fmt_vec(t_mgnc))
    print(f"Rotation error (deg):          {ang_err_mgnc:.15g}")
    print(f"Translation relative error:    {t_err_mgnc:.15g}")

    print()
    print("=== GNC-IRLS0 ===")
    print("R:")
    print(_fmt_matrix(R_irls0))
    print("t:")
    print(_fmt_vec(t_irls0))
    print(f"Rotation error (deg):          {ang_err_irls0:.15g}")
    print(f"Translation relative error:    {t_err_irls0:.15g}")
