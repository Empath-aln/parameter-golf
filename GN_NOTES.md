# Gauss-Newton 优化器移植笔记

## 概述

本项目将 `nanogpt-gn-senmiao` 中的 Gauss-Newton (GN) 二阶优化方法移植到 `parameter-golf-AIStat-GN`。
后者模型更小（9层 × 512维，vocab=1024，约5M参数），基础设施更完善（Muon+Adam 分组优化、wallclock 停止机制、int8 量化管线），
更适合快速实验和验证 GN 方法的有效性。

## 算法原理

### 标准一阶方法 vs Gauss-Newton

| 方面 | 标准方法 (Adam/Muon) | Gauss-Newton |
|------|---------------------|-------------|
| 更新方向 | 一阶梯度 ∇L | 利用 Hessian 信息的二阶方向 |
| 收敛速度 | 一阶收敛 | 拟牛顿收敛（局部更快） |
| 计算代价 | 1次 forward + 1次 backward | 1次 JVP forward + 1次 VJP backward + Hessian 计算 |
| 内存开销 | 标准 | 额外存储方向向量 v_t |

### 核心数学推导

对于交叉熵损失 `L = -Σ y_c log(softmax(z)_c)`，其中 z 是 logits：

1. **一阶梯度**（对 logits）：`∂L/∂z = p - y`，其中 `p = softmax(z)`，`y` 是 one-hot 标签

2. **Hessian 矩阵**（对 logits 的二阶导）：`H = diag(p) - pp^T`
   - 这是一个半正定矩阵，这保证了 Gauss-Newton 方向是下降方向

3. **Gauss-Newton 更新**：利用 `J^T H J` 近似完整 Hessian，其中 J 是 logits 对参数的 Jacobian

### Senmiao GN 算法（Blended GN-Frank-Wolfe with Momentum）

每个训练步骤分为两层循环：

**内层循环**（每个 micro-batch 调用一次 `update_direction`）：

```
输入: input_ids, target_ids
参数: η (inner_lr), α (so_ratio), β (momentum)

1. JVP 计算：
   logits, Jv = jvp(model, params, direction)  # J @ v_{t-1}
   
2. Hessian-vector 积：
   p = softmax(logits)
   H*Jv = diag(p)*Jv - p*(p^T * Jv)
   
3. 混合梯度：
   blended = J^T @ ((p - y) + η*α * H*Jv) / (B*S)
   # 其中 (p-y) 是一阶项，H*Jv 是二阶项
   # α 控制二阶信息的权重，α=0 退化为纯一阶
   
4. 累积到 grad_accum
```

**外层步骤**（每个 outer step 调用一次 `step`）：

```
1. 平均累积梯度: avg_grad = grad_accum / micro_batch_count

2. EMA 更新:
   m_t = β * m_{t-1} + (1-β) * avg_grad
   
3. LMO (Linear Minimization Oracle):
   - 2D 矩阵参数: Muon (Newton-Schulz 正交化 + Nesterov 动量)
   - 标量/embedding 参数: Adam
   得到方向 s_t
   
4. 凸组合更新:
   v_t = β * v_{t-1} + (1-β) * s_t
   
5. 参数更新:
   θ = θ + lr_scale * v_t
```

### 关键设计：JVP + VJP 的计算流程

为什么不直接计算完整的 `J^T H J` 矩阵？因为 J 的大小是 `(B×S×V) × num_params`，对于 5M 参数模型，
这个矩阵有 `~5M × B×S×V` 个元素，完全不可能存储。

解决方案：**JVP-VJP 组合**

```
1. JVP: Jv = J @ v        → 正向模式自动微分，O(forward) 时间
2. 手动: w = H @ (Jv)     → 利用 H 的特殊结构，O(B*S*V) 时间
3. VJP: g = J^T @ w       → 反向模式自动微分，O(backward) 时间
```

这避免了显式构造 J 或 H 矩阵，总计算量约为 2 倍标准 forward+backward。

### 对 logit_softcap 的处理

parameter-golf 使用 `logits = softcap * tanh(logits / softcap)` 进行 logit 裁剪。
这不影响算法正确性，因为：

- JVP 自动微分会正确地通过 tanh 传播导数
- softmax 是在裁剪后的 logits 上计算的
- Hessian `H = diag(p) - pp^T` 是关于最终 logits 的，所以是一致的

## 实现细节

### 文件结构

```
parameter-golf-AIStat-GN/
├── gauss_newton.py          # GN 优化器实现 (~230行)
├── train_gpt.py             # 修改: 新增 GN 训练路径
├── train_gn.sh              # 单 GPU GN 训练脚本
└── GN_NOTES.md              # 本文件
```

### 与原始 Inner_Solver.py 的差异

| 方面 | 原始实现 | 新实现 |
|------|---------|--------|
| 类设计 | 继承 `torch.optim.Optimizer` | 独立 Python 类 |
| 代码量 | ~740行 | ~230行 |
| 方向存储 | `nn.Parameter` 列表 | 普通 `Tensor` 列表 |
| 参数分组 | 硬编码 wte/wpe/lm_head 判断 | 由外部传入 param_info |
| separate 模式 | 支持分离一阶/二阶 | 仅支持混合模式（更简洁高效） |
| torch.compile | 不涉及 | 正确处理：GN 路径用 base_model |

### `torch.compile` 兼容性

`functional_call` + `jvp` 与 `torch.compile(fullgraph=True)` 不兼容。解决方案：

- **标准训练路径**: 使用 `compiled_model`（通过 DDP 包装后的 `model`）
- **GN 训练路径**: 使用未编译的 `base_model`（通过 `functional_call`）
- **验证**: 始终使用 `compiled_model`（GN 不影响 eval）

两者共享同一组参数，`compiled_model = torch.compile(base_model)` 是对同一对象的编译包装。

## 超参数说明

| 参数 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| GN_MODE | `GN_MODE` | 0 | 0=标准训练，1=GN 训练 |
| β (beta) | `GN_BETA` | 0.9 | EMA 动量系数和凸组合系数 |
| α (so_ratio) | `GN_SO_RATIO` | 0.1 | 二阶信息权重，α=0 退化为一阶 |
| η (inner_lr) | `GN_INNER_LR` | 1e-3 | 内层 LMO 学习率 |

### 超参数调优建议

1. **`GN_SO_RATIO`** 是最重要的超参数：
   - 太小（< 0.01）：退化为一阶方法，GN 无效果
   - 太大（> 1.0）：二阶项主导，可能不稳定
   - 推荐范围：0.05 ~ 0.3

2. **`GN_BETA`** 控制动量：
   - 0.9 是合理的默认值
   - 较高的值（0.95-0.99）更平滑但响应较慢
   - 较低的值（0.5-0.8）响应更快但可能振荡

3. **`GN_INNER_LR`** 控制方向更新步长：
   - 和模型的外层学习率（matrix_lr 等）是独立的
   - 原始实现使用 1e-3，在 parameter-golf 的小模型上可能需要调整

## 运行方式

### 快速冒烟测试

```bash
conda activate gn
GN_MODE=1 ITERATIONS=50 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=10 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### 完整训练

```bash
bash train_gn.sh
```

### 对比实验

```bash
# 基线 (标准 Muon+Adam)
GN_MODE=0 torchrun --standalone --nproc_per_node=1 train_gpt.py

# GN 模式
GN_MODE=1 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## 已知限制

1. **Flash Attention 不支持 forward-mode AD**: PyTorch 的 `F.scaled_dot_product_attention` 使用 Flash Attention CUDA kernel 时不支持 JVP。解决方案：在 `update_direction` 中临时切换到 `SDPBackend.MATH`（纯 PyTorch 实现）。这会导致 attention 计算变慢，但保证了 JVP 的正确性。
2. **速度**: GN 每步约为标准训练的 2-3 倍慢（额外的 JVP + VJP + math attention），但如果收敛更快则总体可能更优
3. **torch.compile**: GN 路径不使用 torch.compile，无法获得编译优化加速
4. **DDP**: 支持但未经充分测试，单 GPU 场景是主要目标

### 显存分析

GN 模式相比标准训练有显著额外的显存开销，来源于以下几个方面：

**额外的参数级别缓冲区**（与模型参数量成正比，约 17M 参数 × 4 bytes/param ≈ 65MB 每份）：

| 缓冲区 | 数量 | 说明 |
|--------|------|------|
| `direction` (v_t) | 1份 | 方向向量，跨步骤保持 momentum |
| `ema_m` (m_t) | 1份 | EMA 动量缓冲 |
| `grad_accum` | 1份 | 当前步的梯度累积 |
| `muon_buf` | ~1份 | Muon 动量缓冲（仅 matrix 参数） |
| `adam_state` | ~2份 | Adam 的 exp_avg + exp_avg_sq（仅非 matrix 参数） |

合计约 **6 份参数副本 ≈ 390MB**，这部分开销不大。

**真正的显存瓶颈是 JVP forward pass**：

- JVP 模式下，每个中间激活都需要同时存储 primal 值和 tangent 值，**激活内存大约翻倍**
- Math attention（替代 Flash Attention）的显存是 O(batch × heads × seq²)，不像 Flash Attention 是 O(batch × heads × seq)
- 对于 seq_len=1024, 8 heads, batch=64 sequences: Math attention 额外占用约 **2GB**

**实测对比**（A100 80GB，TRAIN_BATCH_TOKENS=131072，WARMUP_STEPS=0，5 iterations）：

| 指标 | Baseline (GN_MODE=0) | GN (GN_MODE=1) | GN/Baseline |
|------|---------------------|----------------|-------------|
| Peak memory allocated | 2,784 MiB | 40,219 MiB | **14.4x** |
| Peak memory reserved | 3,782 MiB | 40,386 MiB | 10.7x |
| 每步平均耗时 | ~13.7s | ~9.5s | 0.69x (GN更快*) |

> *注：baseline 的"每步耗时"包含了首步 torch.compile 编译时间（~67s），后续步仅 ~0.3s；GN 不使用 compile 所以每步恒定 ~2s。实际吞吐量 baseline 远优于 GN。

**显存差异来源分析**（约 37.4 GB 额外开销）：

| 来源 | 估计大小 | 说明 |
|------|---------|------|
| 参数级缓冲区 | ~390 MB | direction + ema_m + grad_accum + muon_buf + adam_state (约 6 份参数副本) |
| JVP 激活翻倍 | ~15-20 GB | forward 中每个中间激活同时存储 primal 和 tangent |
| Math Attention | ~5-8 GB | O(batch × heads × seq²) vs Flash Attention 的 O(batch × heads × seq) |
| autograd.grad 反向图 | ~10-15 GB | VJP 计算需要保留完整的计算图 |

> 结论：GN 的显存瓶颈不在参数缓冲区（<1GB），而在 JVP 激活翻倍 + Math Attention + 反向图保留。
> 在 A100 80GB 上，TRAIN_BATCH_TOKENS 需要从默认的 524288 降到 ~131072 才能跑通 GN。

## 参考资料

- 原始实现: `nanogpt-gn-senmiao/Inner_Solver.py`
- Muon 优化器: https://kellerjordan.github.io/posts/muon/
- Frank-Wolfe 方法: 凸优化中的线性极小化预言机方法
- `torch.func.jvp`: PyTorch 函数化自动微分（正向模式）

## 进展记录

### 2026-04-05: 初始移植完成 (commit 1ee32f2)

完成 GN 优化器到 parameter-golf 的移植，主要改动：

**新增文件:**
- `gauss_newton.py`: GN 优化器实现 (~230行)，从 `nanogpt-gn-senmiao/Inner_Solver.py` 精简而来
- `train_gn.sh`: 单 GPU GN 训练启动脚本
- `GN_NOTES.md`: 本笔记文件

**`train_gpt.py` 改动:**
- 导入 `jvp_flash_attention.JVPAttn`（可选依赖，import 失败时 fallback）
- `Hyperparameters` 新增 GN 相关配置：`GN_MODE`, `GN_BETA`, `GN_SO_RATIO`, `GN_INNER_LR`
- `CausalSelfAttention.forward` 新增 JVPAttn 分支（GN 的 forward-mode AD 需要可微分 attention）
- `GPT.forward` 新增 `return_logits` 参数（GN 需要拿到 logits 做 Hessian-vector 积）
- `main()` 中新增 GN 训练路径（与标准 backward 路径并列，由 `GN_MODE` 开关控制）

**修复:**
- `GN_INNER_LR` 默认值从误改的 `0.04` 恢复为 `1e-3`
- 恢复 `# TEST-TIME TRAINING (LoRA)` 段落下方被误删的 `# -----------------------------` 分隔线
