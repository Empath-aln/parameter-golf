# Gauss-Newton 优化器：2 层 Momentum 数学解析

## 整体结构

本优化器（`gauss_newton.py`）采用 **Blended GN-Frank-Wolfe with Momentum** 框架，信号从 blended gradient 到最终参数更新经过 2 层独立的 momentum 平滑。

---

## 输入：Blended Gradient（混合梯度）

每步先计算 blended gradient（一阶 + 二阶混合），**不经过 EMA，直接送入 LMO**：

$$\tilde{g}_t = J^\top \left[ \underbrace{(p - y)}_{\text{一阶项}} + \underbrace{\eta \alpha \cdot H \cdot J v_{t-1}}_{\text{二阶 GN 项}} \right] \Big/ (B \cdot S)$$

其中：
- $p = \text{softmax}(z)$，$y$ 是 one-hot 标签
- $H = \text{diag}(p) - pp^\top$：CE 对 logits 的 Hessian（半正定）
- $J$：logits 对参数的 Jacobian
- $v_{t-1}$：上一步的方向向量（**反馈回路**，见下文）
- $\eta$ = `self.lr`（由 train loop 设为 `base_lr * schedule_scale`），$\alpha$ = `so_ratio`

---

## 第 1 层：LMO 内部的 Momentum

$\tilde{g}_t$ 直接送进 LMO（Linear Minimization Oracle），LMO 内部有独立的 momentum 机制，对 matrix 参数和 scalar 参数分别处理。

### 1a. Matrix 参数：Muon LMO（`_muon_lmo`）

Muon 内部维护自己的动量缓冲 `buf`：

$$\text{buf}_t = 0.95 \cdot \text{buf}_{t-1} + (1 - 0.95) \cdot \tilde{g}_t$$

Nesterov lookahead：

$$g_{\text{eff}} = \tilde{g}_t + 0.95 \cdot \text{buf}_t$$

Newton-Schulz 正交化 + 缩放：

$$s_t = -\sqrt{\max(1,\; \text{rows}/\text{cols})} \cdot \text{NS5}(g_{\text{eff}})$$

其中 NS5 是 5 步 Newton-Schulz 迭代，将矩阵映射到最近的正交矩阵（零幂近似）。

### 1b. Scalar/Embed 参数：Adam LMO（`_adam_lmo`）

标准 Adam，以 $\tilde{g}_t$ 作为输入梯度：

$$\hat{m}_t^{(\text{adam})} = \frac{\beta_1 \cdot \hat{m}_{t-1}^{(\text{adam})} + (1-\beta_1) \cdot \tilde{g}_t}{1 - \beta_1^t}$$

$$\hat{v}_t^{(\text{adam})} = \frac{\beta_2 \cdot \hat{v}_{t-1}^{(\text{adam})} + (1-\beta_2) \cdot \tilde{g}_t^2}{1 - \beta_2^t}$$

$$s_t = -\frac{\hat{m}_t^{(\text{adam})}}{\sqrt{\hat{v}_t^{(\text{adam})}} + \epsilon}$$

- 系数：$\beta_1 = 0.9$，$\beta_2 = 0.95$，$\epsilon = 10^{-8}$

---

## 第 2 层：方向向量的 Frank-Wolfe 凸组合（`direction`）

LMO 输出 $s_t$ 后，通过 Frank-Wolfe 风格的凸组合更新方向向量：

$$v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot s_t$$

- 系数：$\beta$ = `GN_BETA`（默认 0.9）

### 最终参数更新

$$\theta_{t+1} = \theta_t + \eta \cdot v_t$$

其中 $\eta$ = `self.lr`，由 train loop 每步设为 `base_lr * schedule_scale`。

---

## 信号流总图

```
                    ┌──────────────────────────────────┐
                    │          反馈回路                  │
                    │  v_{t-1} 参与下一步 JVP 计算       │
                    │                                    │
raw grad + GN term  │                                    │
  (p-y) + ηα·H·Jv  ◄────────────────────────────────┐   │
        │                                            │   │
        │ tilde_g (直接送入 LMO，无 EMA)              │   │
        ▼                                            │   │
 ┌──────────────────────┐                            │   │
 │  第1层 LMO momentum   │                           │   │
 │                        │                           │   │
 │  matrix: Muon          │                           │   │
 │    buf += (1-0.95)·g̃_t │                           │   │
 │    g_eff = g̃_t+0.95·buf│                           │   │
 │    s_t = -NS5(g_eff)   │                           │   │
 │                        │                           │   │
 │  scalar: Adam          │                           │   │
 │    s_t = -m̂/(√v̂+ε)    │                           │   │
 └──────────┬─────────────┘                           │   │
            │ s_t                                     │   │
            ▼                                         │   │
 ┌────────────────────┐                               │   │
 │  第2层 FW 凸组合    │  v_t = β·v_{t-1} + (1-β)·s_t │   │
 │  (β=0.9)           │                               │   │
 └──────────┬─────────┘                               │   │
            │ v_t                                     │   │
            ├─────────────────────────────────────────┘   │
            │                                             │
            ▼                                             │
     θ_{t+1} = θ_t + η·v_t ──────────────────────────────┘
```

---

## Momentum 系数汇总

| 层级 | 机制 | 系数 | 作用 |
|------|------|------|------|
| 第1层 (matrix) | Muon Nesterov momentum | 0.95 (硬编码) | 正交化前的动量加速 |
| 第1层 (scalar) | Adam β1/β2 | 0.9 / 0.95 (硬编码) | 自适应学习率 |
| 第2层 | FW convex combination | β = 0.9 (`GN_BETA`) | 方向向量的平滑更新 |

---

## 关键设计：反馈回路

$v_t$ 有双重作用：
1. **参数更新**：$\theta_{t+1} = \theta_t + \eta \cdot v_t$
2. **下一步二阶项计算**：$Jv_t$ 出现在 $\tilde{g}_{t+1}$ 的二阶 GN 项中

这形成了一个闭环：方向向量 → 参数更新 & JVP tangent → 新的 blended gradient → LMO → 新的方向向量。

当 $\alpha = 0$ 时，二阶项消失，JVP 不再需要，退化为纯一阶 Frank-Wolfe + Muon/Adam。

---

## 与旧版（3 层 EMA）的区别

旧版在 blended gradient 和 LMO 之间有一层额外的 EMA 平滑：

$$m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot \tilde{g}_t$$

现在去掉了这层，$\tilde{g}_t$ 直接送入 LMO。原因：LMO 内部已有 momentum（Muon 的 Nesterov / Adam 的 β1），额外的 EMA 是冗余的平滑。

---

## 学习率调度

学习率调度由 train loop 负责，与默认优化器保持一致的模式：

```python
# train_gpt.py
scale = lr_mul(step, elapsed_ms)          # 计算 schedule 系数
gn_optimizer.lr = args.gn_lr * scale       # 设置 scheduled lr
gn_optimizer.step()                        # optimizer 直接使用 self.lr
```

GN optimizer 内部不维护 `base_lr`，不关心 schedule 逻辑。

---

## JVP-VJP 计算避免显式构造 Jacobian

直接计算 $J^\top H J$ 不可行（$J$ 大小为 $(B \times S \times V) \times N_{\text{params}}$），采用隐式分解：

```
1. JVP:  Jv = J @ v_{t-1}          → 正向模式 AD，O(forward)
2. 手动: w  = H @ (Jv)             → 利用 H 的结构，O(B·S·V)
         w  = diag(p)·Jv - p·(p^T·Jv)
3. VJP:  g  = J^T @ [(p-y) + ηα·w] → 反向模式 AD，O(backward)
```

总代价 ≈ 2× 标准 forward+backward。
