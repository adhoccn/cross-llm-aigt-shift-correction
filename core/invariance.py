# -*- coding: utf-8 -*-
"""
跨 LLM 不变性工具箱：散度矩阵、CLIDA 投影、测试时自适应与不变性赤字 Ξ。

符号约定（对应论文 §3–§4）：
    S_B   类间散度（Human vs AI）
    S_W   类内散度
    S_E   给定类后的环境间散度（LLM 身份 / LLM×领域）
    CLIDA 解广义特征问题 S_B w = λ (S_W + γ S_E) w，
          逐维方法 diag_fisher / diag_heuristic 是其对角受限版本
    Ξ_eff = −uᵀδ，u = Σ_W^{-1}(μ₁−μ₀)/Δ，δ = μ_AI,tgt − μ_AI,src，
          即漂移在判别方向上的投影；Δ' = Δ − Ξ_eff 为跨 LLM 后的有效分离度
"""
from __future__ import annotations
import numpy as np


def scatter_matrices(X, y, env=None):
    """
    X   : (n, d) 特征矩阵
    y   : (n,)  类别标签 {0,1}
    env : (n,)  环境标签（LLM 身份 或 LLM×领域）；None 则 S_E = 0

    返回 (S_B, S_W, S_E)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, d = X.shape
    mu = X.mean(axis=0)

    S_B = np.zeros((d, d))
    S_W = np.zeros((d, d))
    classes = np.unique(y)
    for c in classes:
        Xc = X[y == c]
        muc = Xc.mean(axis=0)
        nc = len(Xc)
        dv = (muc - mu).reshape(-1, 1)
        S_B += nc * (dv @ dv.T)
        Xc0 = Xc - muc
        S_W += Xc0.T @ Xc0
    S_B /= max(n, 1)
    S_W /= max(n - len(classes), 1)

    S_E = np.zeros((d, d))
    if env is not None:
        env = np.asarray(env)
        for c in classes:
            m_c = (y == c)
            muc = X[m_c].mean(axis=0)
            for e in np.unique(env[m_c]):
                m = m_c & (env == e)
                if m.sum() < 2:
                    continue
                mu_e = X[m].mean(axis=0)
                dv = (mu_e - muc).reshape(-1, 1)
                S_E += m.sum() * (dv @ dv.T)
        S_E /= max(n, 1)
    return S_B, S_W, S_E


def _regularize(Q, ridge=1e-6):
    d = Q.shape[0]
    tr = np.trace(Q)
    scale = (tr / d) if d > 0 else 1.0
    return Q + ridge * scale * np.eye(d)


class CLIDA:
    """
    Cross-LLM Invariant Discriminant Analysis。

    gamma        环境惩罚系数。gamma=0 退化为标准 Fisher/LDA；
                 gamma→∞ 趋于纯不变性投影
    n_components 投影维度 k；None 则取满秩 d
    """

    def __init__(self, gamma=1.0, n_components=None, ridge=1e-6, whiten=False,
                 normalize_se=True):
        self.gamma = gamma
        self.n_components = n_components
        self.ridge = ridge
        self.whiten = whiten
        self.normalize_se = normalize_se

    def fit(self, X, y, env=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        d = X.shape[1]
        self.S_B_, self.S_W_, self.S_E_ = scatter_matrices(X, y, env)
        S_E_used, self.se_scale_ = (normalize_SE(self.S_W_, self.S_E_)
                                    if self.normalize_se else (self.S_E_, 1.0))
        self.S_E_used_ = S_E_used
        Q = _regularize(self.S_W_ + self.gamma * S_E_used, self.ridge)
        try:
            from scipy.linalg import eigh
            evals, evecs = eigh(self.S_B_, Q)
        except Exception:
            evals, evecs = np.linalg.eig(np.linalg.solve(Q, self.S_B_))
            evecs = np.real(evecs)
            evals = np.real(evals)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        k = self.n_components or d
        k = int(min(k, d))
        self.evals_ = evals[:k]
        self.W_ = evecs[:, :k]
        self.J_ = float(np.trace(self.W_.T @ self.S_B_ @ self.W_) /
                        max(np.trace(self.W_.T @ Q @ self.W_), 1e-12))
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) @ self.W_
        if self.whiten:
            s = np.sqrt(np.maximum(self.evals_, 1e-12))
            Z = Z / s
        return Z


def diag_fisher(X, y, env=None, gamma=1.0, power=0.5, ridge=1e-6, normalize_se=True):
    """逐维最优缩放：u_i = (S_B[i,i] / (S_W[i,i] + γ S_E[i,i]))^power"""
    S_B, S_W, S_E = scatter_matrices(X, y, env)
    if normalize_se:
        S_E, _ = normalize_SE(S_W, S_E)
    a = np.diag(S_B).copy()
    b = np.diag(S_W).copy() + gamma * np.diag(S_E).copy()
    b = np.maximum(b, 1e-12)
    u = (np.maximum(a, 0.0) / b) ** power
    # 归一化，避免整体尺度漂移影响后续正则化强度
    if np.mean(u) > 0:
        u = u / np.mean(u)
    return u


def diag_heuristic(X, y, env=None, gamma=1.0):
    """
    论文 §3.2.4 特征调制的实例化：
      V_c,i = Var_y ( E[z_i | y] )        类间方差
      V_d,i = Var_L ( E[z_i | y=A, L] )   给定 AI 类后的 LLM 间方差
      M_i   = V_c,i / (V_c,i + γ V_d,i)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if env is None:
        raise ValueError('diag_heuristic 需要 env（LLM 身份）以估计 V_d')
    env = np.asarray(env)
    Vc = np.zeros(X.shape[1])
    Vd = np.zeros(X.shape[1])
    means = []
    for c in np.unique(y):
        means.append(X[y == c].mean(axis=0))
    Vc = np.var(np.array(means), axis=0)          # 两类之间的均值方差
    ai = (y == 1)
    per_llm = []
    for e in np.unique(env[ai]):
        m = ai & (env == e)
        if m.sum() >= 2:
            per_llm.append(X[m].mean(axis=0))
    if len(per_llm) >= 2:
        Vd = np.var(np.array(per_llm), axis=0)
    M = Vc / (Vc + gamma * Vd + 1e-12)
    if np.mean(M) > 0:
        M = M / np.mean(M)
    return M


def adaptive_standardize(X_te, mu_tr=None, sd_tr=None):
    """
    测试时自适应标准化（TTA-AdaBN 式）：用测试批次自身的均值/标准差做
    z-score；若给出 mu_tr/sd_tr，则再把测试分布仿射回训练坐标系。
    """
    X_te = np.asarray(X_te, dtype=np.float64)
    mu = X_te.mean(axis=0)
    sd = X_te.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X_te - mu) / sd
    if mu_tr is not None and sd_tr is not None:
        Z = Z * np.asarray(sd_tr) + np.asarray(mu_tr)
    return Z


def normalize_SE(S_W, S_E):
    """
    把环境间散度缩放到与类内散度同 trace 尺度。两者的原始量级通常相差
    两个数量级，直接相加会使 γ 的相对权重失去意义；归一化后 γ 表示
    「环境散度 vs 类内散度」的相对权重，便于跨特征集调参。
    """
    tE = np.trace(S_E)
    if tE <= 1e-12:
        return S_E, 0.0
    scale = np.trace(S_W) / tE
    return S_E * scale, scale


def estimate_Xi(X_src, y_src, X_tgt_ai, X_tgt_hu=None, ridge=1e-2):
    """
    条件不变性赤字。返回 dict：
      xi_raw      全空间漂移 (1/d)·δᵀΣ_W^{-1}δ，不区分方向
      xi_eff      判别方向投影漂移 −uᵀδ，u = Σ_W^{-1}(μ₁−μ₀)/Δ
      Delta       源域马氏分离度
      Delta_prime Δ − xi_eff，跨 LLM 后的有效分离度
      xi_hu_*     人类侧的对应量（传入 X_tgt_hu 时才有值）
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    y_src = np.asarray(y_src)
    d = X_src.shape[1]
    mu = {}
    SW = np.zeros((d, d))
    for c in (0, 1):
        Xc = X_src[y_src == c]
        if len(Xc) < 2:
            continue
        mu[c] = Xc.mean(axis=0)
        Xc0 = Xc - mu[c]
        SW += Xc0.T @ Xc0
    SW /= max(len(X_src) - 2, 1)
    SW = SW + ridge * (np.trace(SW) / max(d, 1)) * np.eye(d)
    SWi = np.linalg.pinv(SW)

    dmu = mu[1] - mu[0]
    w = SWi @ dmu
    Delta2 = float(dmu @ SWi @ dmu)
    Delta = float(np.sqrt(max(Delta2, 0.0)))
    u = w / Delta if Delta > 1e-9 else np.zeros_like(w)

    def maha(a, b):
        dv = np.asarray(a) - np.asarray(b)
        return float(dv @ SWi @ dv) / d

    d_ai = np.asarray(X_tgt_ai).mean(axis=0) - mu[1]
    xi_raw = maha(np.asarray(X_tgt_ai).mean(axis=0), mu[1])
    xi_eff = float(-u @ d_ai)
    Delta_prime = Delta + float(u @ d_ai)

    xi_hu_raw = xi_hu_eff = float('nan')
    if X_tgt_hu is not None and len(X_tgt_hu) > 1:
        xi_hu_raw = maha(np.asarray(X_tgt_hu).mean(axis=0), mu[0])
        xi_hu_eff = float(-u @ (np.asarray(X_tgt_hu).mean(axis=0) - mu[0]))

    return {'xi_raw': xi_raw, 'xi_eff': xi_eff, 'Delta': float(Delta),
            'Delta_prime': Delta_prime, 'xi_hu_raw': xi_hu_raw, 'xi_hu_eff': xi_hu_eff}
