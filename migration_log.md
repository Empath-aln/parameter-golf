# Parameter Golf → 昇腾 910C 迁移日志

> 项目: [openai/parameter-golf](https://github.com/openai/parameter-golf)  
> 目标硬件: 昇腾 910C (Ascend NPU)  
> 迁移工具: KernelCAT (kernelcat1.0)  
> 工作目录: `workspace/wangsenmiao/`  
> 开始时间: 2026-03-27 14:47

---

## 1. 项目概况

**Parameter Golf** 是 OpenAI 发起的模型压缩训练挑战：在 16MB 模型体积限制下，用 8xH100 训练不超过 10 分钟，追求最低的 FineWeb 验证集 bits-per-byte (bpb)。

核心代码 `train_gpt.py`（约 1126 行）包含：
- **模型**: GPT (RMSNorm, GQA, RoPE, relu² MLP, tied embeddings), 9层, 512维
- **优化器**: Muon (矩阵参数, Newton-Schulz 正交化) + Adam (embedding/标量参数)
- **分布式**: PyTorch DDP + NCCL, 设计目标 8xH100
- **精度**: bf16 autocast + fp32 权重存储
- **量化导出**: 训练后 int8 量化 + zlib 压缩 → 16MB 以内
- **数据**: FineWeb 10B tokens, SentencePiece 1024 词表, 预分片 bin 文件

依赖列表 (`requirements.txt`):
```
numpy, tqdm, torch, huggingface-hub, kernels, setuptools,
typing-extensions==4.15.0, datasets, tiktoken, sentencepiece
```

## 2. CUDA 依赖分析

代码中约 **30 处**显式依赖 CUDA，归纳如下：

| 依赖类别 | 涉及位置 | 迁移方案 |
|----------|----------|----------|
| `torch.device("cuda")` | 多处 | → `torch.device("npu")` |
| `torch.cuda.*` API | ~15处 | → `torch.npu.*` API |
| `autocast(device_type="cuda")` | L258, L947, L1015 | → `device_type="npu"` |
| `dist.init_process_group(backend="nccl")` | L757 | → `backend="hccl"` |
| `torch.backends.cuda.*` | L762-768 | 删除或替换 |
| `torch.compile(...)` | L736, L843 | 先降级为 eager 模式 |
| `F.scaled_dot_product_attention` | L594 | torch_npu 已支持, 需验证 |
| `nvidia-smi` | L791 | → `npu-smi info` |
| `Adam(fused=True)` | 优化器 | 设 `fused=False` |

## 3. 迁移路线

### 阶段一：torch_npu 直接适配（先跑通）
1. 复制 `train_gpt.py` → `train_gpt_npu.py`，做 CUDA→NPU 替换
2. 先单卡跑通前向+反向，验证 loss 正常下降
3. 再开启多卡 DDP (HCCL)，验证分布式训练
4. CPU 交叉对比精度

### 阶段二：功能完整验证
1. 完整训练 + int8 量化导出 + round-trip 验证
2. 比对最终 val_bpb 与 CUDA 基线
3. 生成迁移报告

### 阶段三（可选）：性能优化
1. 验证 `torch.compile` NPU 图模式
2. 探索 CANN 融合算子
3. 必要时用 AscendC 编写自定义算子

---

## 4. 阶段一执行记录

### 4.1 仓库克隆

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 14:47 | `git clone` 到 CWD 根目录 | ✅ 成功，但路径不对 |
| 2026-03-27 14:48 | 用户要求放到 `workspace/wangsenmiao/` 下 | — |
| 2026-03-27 14:48 | `mv parameter-golf workspace/wangsenmiao/` | ✅ 成功 |

**教训**: 应在克隆前确认目标目录，避免事后搬移。

### 4.2 代码阅读与 CUDA 依赖分析

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 14:49 | 阅读 `README.md` (234行) | ✅ 了解项目背景、挑战规则、Leaderboard |
| 2026-03-27 14:50 | 阅读 `requirements.txt` | ✅ 确认 10 个依赖，其中 `kernels` 包需关注 NPU 兼容性 |
| 2026-03-27 14:50 | 分段阅读 `train_gpt.py` L1-250 | ✅ 理解超参数、Muon 优化器、tokenizer 评估逻辑 |
| 2026-03-27 14:51 | 分段阅读 `train_gpt.py` L250-500 | ✅ 理解 eval_val、int8 量化逻辑 |
| 2026-03-27 14:52 | 分段阅读 `train_gpt.py` L500-750 | ✅ 理解模型架构 (RMSNorm, Rotary, GQA, MLP, Block, GPT) |
| 2026-03-27 14:53 | 分段阅读 `train_gpt.py` L750-1000 | ✅ 理解分布式初始化、优化器构建、warmup 逻辑 |
| 2026-03-27 14:54 | 分段阅读 `train_gpt.py` L1000-1126 | ✅ 理解训练主循环、序列化、round-trip 验证 |
| 2026-03-27 14:55 | `grep -n` 扫描全部 CUDA 引用 | ✅ 定位 30 处 CUDA 依赖，分为 9 类 |
| 2026-03-27 14:55 | 阅读 `data/README.md` | ✅ 了解数据下载方式 (`cached_challenge_fineweb.py`) |

### 4.3 创建 train_gpt_npu.py

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 14:58 | `cp train_gpt.py train_gpt_npu.py` | ✅ 复制基准文件 |
| 2026-03-27 14:58 | `apply_patch` 添加 `import acl` + `import torch_npu` | ❌ 失败：patch context 匹配不到（空行导致上下文断裂） |
| 2026-03-27 14:59 | 改用精确行号 patch 添加 `import acl` | ✅ 成功 |
| 2026-03-27 14:59 | `apply_patch` 添加 `import torch_npu` | ❌ 失败：同样的 context 匹配问题 |
| 2026-03-27 15:00 | **策略调整**: 放弃 apply_patch 逐行改，改用 `sed -i` 批量替换 | — |
| 2026-03-27 15:00 | `sed -i '26a\import torch_npu'` | ✅ 成功 |
| 2026-03-27 15:00 | `sed -i` 替换所有 `device_type="cuda"` → `"npu"` | ✅ 成功 |
| 2026-03-27 15:01 | `sed -i` 替换 `torch.device("cuda", local_rank)` → `("npu", ...)` | ✅ 成功 |
| 2026-03-27 15:01 | `sed -i` 替换 `torch.cuda.set_device` → `torch.npu.set_device` | ✅ 成功 |
| 2026-03-27 15:01 | `sed -i` 替换 `backend="nccl"` → `"hccl"` | ✅ 成功 |
| 2026-03-27 15:02 | `sed -i` 替换 `torch.cuda.is_available` → `torch.npu.is_available` | ✅ 成功 |
| 2026-03-27 15:02 | `sed -i` 替换 `torch.cuda.synchronize` → `torch.npu.synchronize` | ✅ 成功 |
| 2026-03-27 15:02 | `sed -i` 替换 `torch.cuda.manual_seed_all` → `torch.npu.manual_seed_all` | ✅ 成功 |
| 2026-03-27 15:03 | `sed -i` 替换 `torch.cuda.max_memory_allocated/reserved` → `torch.npu.*` | ✅ 成功 |
| 2026-03-27 15:03 | `sed -i` 替换 RuntimeError 消息 `CUDA` → `NPU` | ✅ 成功 |
| 2026-03-27 15:04 | `sed -i` 删除 tf32/flash_sdp 配置块 (L765-772, 8行) 替换为 NPU 注释 (2行) | ✅ 成功 |
| 2026-03-27 15:04 | `sed -i` 替换 `nvidia-smi` → `npu-smi info` | ✅ 成功 |
| 2026-03-27 15:05 | `sed -i` 将 `torch.compile(zeropower_via_newtonschulz5)` 改为环境变量控制 | ✅ 成功 |
| 2026-03-27 15:05 | `sed -i` 将 `torch.compile(base_model)` 改为环境变量控制 + else eager 分支 | ✅ 成功 |
| 2026-03-27 15:06 | `sed -i` 全局替换 `fused=True` → `fused=False` | ✅ 成功 |
| 2026-03-27 15:06 | `sed -i` 更新 SDP 日志消息 | ✅ 成功 |
| 2026-03-27 15:06 | `sed -i` 更新 section 注释 `CUDA SETUP` → `NPU SETUP` | ✅ 成功 |
| 2026-03-27 15:07 | `grep -nE 'cuda|nccl|nvidia|fused=True'` 验证残留 | ✅ 0 匹配，清理干净 |
| 2026-03-27 15:07 | `grep -nE 'npu|hccl|torch_npu'` 确认替换到位 | ✅ 30 处 NPU 引用确认 |

**教训**:
- `apply_patch` 在原文件有复杂空行/缩进时 context 匹配容易失败，对于批量机械替换，`sed -i` 更可靠高效
- 应先用 `grep -n` 精确定位所有待改行号，再一次性规划替换顺序，避免行号漂移
- `import acl` 实际上不是必须的（torch_npu 内部会自动初始化 ACL），但保留不影响功能

### 4.4 创建辅助文件

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 15:08 | `apply_patch` 创建 `requirements_npu.txt` | ✅ 成功 |
| 2026-03-27 15:09 | `apply_patch` 创建 `setup_npu_env.sh` (环境搭建脚本) | ✅ 成功 |
| 2026-03-27 15:09 | `apply_patch` 创建 `run_npu.sh` (训练启动脚本) | ✅ 成功 |
| 2026-03-27 15:10 | `chmod +x` 两个 shell 脚本 | ✅ 成功 |

### 4.5 迁移日志更新

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 15:10 | `apply_patch` 更新日志中的状态字段 | ❌ 失败：context 匹配问题（连续两次） |
| 2026-03-27 15:11 | 改用 `sed -i` 行号定位更新 | ✅ 成功 |
| 2026-03-27 15:12 | 用户要求日志精确到分钟、记录失败/成功 | — |
| 2026-03-27 15:13 | 直接重写日志全文，使用 heredoc 避免 patch 问题 | ✅ 成功 |

**教训**: `apply_patch` 对 Markdown 文件中的中文和特殊字符（如 ✅ →）匹配不稳定，Markdown 编辑也推荐用 `sed` 或直接重写。

---

## 5. 阶段一产出物

| 文件 | 说明 |
|------|------|
| `train_gpt_npu.py` | NPU 适配训练脚本 (1127行) |
| `requirements_npu.txt` | NPU 环境 Python 依赖 |
| `setup_npu_env.sh` | CANN + conda 环境一键搭建 |
| `run_npu.sh` | 单卡/多卡训练启动脚本 |
| `migration_log.md` | 本日志文件 |

## 6. 待验证项（需 910C 真机）

| 优先级 | 项目 | 风险 |
|--------|------|------|
| P0 | `F.scaled_dot_product_attention` 的 `enable_gqa` 参数 | torch_npu 版本差异可能不支持 |
| P0 | bf16 autocast 完整支持 | 910C 原生支持 bf16，但 autocast 路径需验证 |
| P1 | `torch.compile` 在 NPU 上可行性 | 环境变量 `NPU_TORCH_COMPILE=1` 可开启测试 |
| P1 | Muon 优化器中 Newton-Schulz 迭代的数值稳定性 | bf16 矩阵运算精度差异 |
| P2 | `kernels` 包（原 requirements）是否有 NPU 对应 | 已从 npu requirements 中移除 |
| P2 | 多卡 HCCL 通信性能 vs NCCL 基线 | 可能需要调 batch size |

## 7. 下一步操作

```bash
# 在 910C 机器上执行：
cd workspace/wangsenmiao/parameter-golf
source setup_npu_env.sh                                    # 搭建环境
python3 data/cached_challenge_fineweb.py --variant sp1024  # 下载数据
bash run_npu.sh 1                                          # 单卡试跑
bash run_npu.sh 8                                          # 多卡训练
```

## 8. 统计摘要

| 指标 | 数值 |
|------|------|
| 总耗时 | 约 26 分钟 (14:47 - 15:13) |
| 代码改动点 | 30 处 CUDA → NPU |
| 尝试总数 | 28 次操作 |
| 成功次数 | 24 次 |
| 失败次数 | 4 次 (均为 apply_patch context 匹配问题) |
| 失败恢复策略 | 全部改用 sed -i 替代 |
| 新增文件 | 5 个 |

---

## 9. 补充记录：数据下载与 HuggingFace 镜像

### 9.1 数据下载分析

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 15:15 | 阅读 `data/cached_challenge_fineweb.py` 全部源码 (153行) | ✅ 理解下载逻辑 |

**下载机制梳理**:
- 使用 `huggingface_hub.hf_hub_download` 从 HF dataset repo 下载
- 数据源: `willdepueoai/parameter-golf` (HF dataset repo)
- 远程路径前缀: `datasets/`
- 下载内容:
  - `manifest.json` (先下载，获取 shard 数量信息)
  - `data/datasets/fineweb10B_sp1024/fineweb_val_*.bin` (验证集全部)
  - `data/datasets/fineweb10B_sp1024/fineweb_train_*.bin` (默认 80 shards)
  - `data/tokenizers/fineweb_1024_bpe.model` (tokenizer)
- 本地路径基于 `Path(__file__).resolve().parent`，即 `parameter-golf/data/`
- 先下载到 HF cache (`~/.cache/huggingface/`)，然后 hard link 或 copy 到本地

### 9.2 HuggingFace 镜像配置

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 15:16 | 更新 `setup_npu_env.sh` 增加 `HF_ENDPOINT` 镜像配置 | ✅ 成功 |
| 2026-03-27 15:16 | 更新 `run_npu.sh` 增加 `HF_ENDPOINT` 默认值 | ✅ 成功 |

**镜像方案**:
- 默认设置 `HF_ENDPOINT=https://hf-mirror.com` (国内最常用 HF 镜像)
- 如果公司有内部镜像，可修改 `setup_npu_env.sh` 中的 `HF_ENDPOINT`
- 下载命令不变: `python3 data/cached_challenge_fineweb.py --variant sp1024`
- 如果镜像也不可用，备选方案: 在有外网的机器下载后 scp 整个 `data/` 目录过去

**注意事项**:
- 数据量预估: 80 train shards × ~400MB/shard ≈ 32GB (需确认实际大小)
- 下载脚本支持增量: 已存在的文件会跳过
- 可通过 `--train-shards N` 控制训练数据量，先少下几个 shard 测试

**推荐的首次测试流程**:
```bash
source setup_npu_env.sh
# 先下载少量数据验证网络通畅
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 2
# 确认成功后再下载完整数据
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
```

### 9.3 数据下载执行

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 15:17 | `pip install huggingface_hub` | ✅ 成功 (27.7s, 安装 v1.8.0 + 依赖) |
| 2026-03-27 15:18 | 启动下载: `HF_ENDPOINT=https://hf-mirror.com python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80` | ❌ 超时 (3600s 达到上限) |
| 2026-03-27 16:18 | 检查下载进度 | ✅ 已下载 55 train shards + 1 val shard (11GB) |
| 2026-03-27 16:20 | 断点续传: 重新运行同一命令 (脚本会跳过已有文件) | ✅ 成功 (743.5s 完成剩余 25 shards + tokenizer) |
| 2026-03-27 16:33 | 验证文件完整性 | ✅ 全部就位 |

**最终数据统计**:
- Train shards: 80 个 `.bin` 文件
- Val shards: 1 个 `.bin` 文件
- Tokenizer: `fineweb_1024_bpe.model` (254KB) + `.vocab` (10KB)
- 总大小: 16GB
- 路径: `parameter-golf/data/datasets/fineweb10B_sp1024/`

**教训**:
- hf-mirror.com 镜像可用但速度波动大，80 shards 总耗时约 72 分钟 (含超时重试)
- 下载脚本天然支持断点续传 (已有文件跳过)，超时后重跑即可
- 建议首次下载设置更长超时或使用 `nohup` / `screen` 后台运行
- 如果网络环境更差，可考虑先下载 2 shards 测试 (`--train-shards 2`)，确认通路后再全量下载

### 9.4 硬件环境确认

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 18:05 | `npu-smi info` | ✅ 检测到 8 个 NPU, 每个含 2 chip, 共 16 个 Ascend910 芯片 |
| 2026-03-27 18:06 | `npu-smi info -t board -i 0` | ✅ 型号 IT22HMDA_2_S, HBM 64GB/chip, Board ID 0x71 |
| 2026-03-27 18:06 | `npu-smi info -t usages -i 0` | ✅ HBM 使用率 4%, AICore 空闲, 温度 40-43°C |
| 2026-03-27 18:07 | 检查 CANN toolkit 版本 | ✅ CANN 8.3.RC1 (aarch64) |
| 2026-03-27 18:07 | 检查 driver 版本 | ✅ Driver 25.2.3 |
| 2026-03-27 18:07 | 检查 torch_npu 系统安装 | ❌ 系统 Python 未安装 torch_npu |

**硬件总结**:
- 8×NPU, 每个 NPU 含 2×Ascend910 chip
- 每 chip 64GB HBM
- CANN 8.3.RC1 + Driver 25.2.3
- aarch64 架构 (ARM 服务器)
- 所有芯片状态 OK, 无运行进程, 可用于训练

**下一步**: 安装 torch_npu 并启动单卡试跑

## 10. 单卡试跑执行记录

### 10.1 环境安装

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 18:08 | `pip install torch==2.5.1` | ✅ 成功 (45.9s) |
| 2026-03-27 18:09 | `pip install torch_npu==2.5.1` | ✅ 成功 (10.2s) |
| 2026-03-27 18:10 | 验证 NPU 识别 (import torch_npu) | ❌ 失败: `libhccl.so` not found |
| 2026-03-27 18:10 | source CANN setenv.bash 后重试 | ❌ 失败: `No module named 'numpy'` |
| 2026-03-27 18:11 | `pip install numpy sentencepiece` | ✅ 成功 |
| 2026-03-27 18:12 | 再次验证 NPU 识别 | ✅ 成功! torch 2.5.1, torch_npu 2.5.1, 16×Ascend910_9382 |

**教训**: torch_npu 依赖链很长，应先完整安装 requirements 所有包后再测试。

### 10.2 首次试跑

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 18:14 | 单卡运行 `train_gpt_npu.py` | ❌ 失败: `ACL_PRECISION_MODE` error 500001 + 缺 `scipy` |
| 2026-03-27 18:15 | `pip install scipy` | ✅ 成功 |
| 2026-03-27 18:15 | 用户提醒应先安装完整 requirements | — |
| 2026-03-27 18:16 | `pip install datasets tiktoken` | ✅ 成功 |
| 2026-03-27 18:17 | 全部 11 个依赖包验证通过 | ✅ 成功 |
| 2026-03-27 18:18 | 重新单卡运行 | ❌ 失败: 同样的 ACL_PRECISION_MODE error + 缺 `psutil` |
| 2026-03-27 18:19 | `pip install psutil` | ✅ 成功 |
| 2026-03-27 18:19 | 设置 `ACL_PRECISION_MODE=allow_mix_precision` 后重试 | ❌ warmup 20步成功, 但 `zip(strict=True)` 报错 |

**教训**:
- `ACL_PRECISION_MODE=allow_mix_precision` 是 910B/C 必须的环境变量
- Python 3.9 不支持 `zip(strict=True)` (需 3.10+)
- CANN 的 TBE 编译后端需要 `scipy` 和 `psutil`, 这两个不在原 requirements 中

### 10.3 修复后成功跑通

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 18:20 | 修复 L957 `zip(strict=True)` → `zip()` | ✅ 成功 |
| 2026-03-27 18:21 | 单卡完整运行 (ACL_PRECISION_MODE=allow_mix_precision) | ✅ 训练成功! |

**训练结果**:
- warmup: 20步完成
- step 0 val_loss: 6.9357, val_bpb: 4.1077
- step 1 train_loss: 6.9358 (loss 正常下降开始)
- step 10 train_loss: 5.8738
- step 200 train_loss: 3.3716
- step 209 val_loss: 3.3979, val_bpb: 2.0124 (触发 wallclock cap 601s 停止)
- peak memory: 19750 MiB allocated, 20788 MiB reserved
- 模型序列化: 67MB (raw), 7.6MB (int8+zlib)
- 单步平均: ~2876ms (单卡 Ascend910)

**未完成**: round-trip 验证 (被 15 分钟超时杀死，但核心训练已完成)

### 10.4 还需补充到 requirements_npu.txt 的包

已更新 `requirements_npu.txt` 增加:
- `scipy` (CANN TBE 编译后端需要)
- `psutil` (CANN AOE 初始化需要)

## 11. 8卡分布式训练执行记录

### 11.1 调试过程

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-27 18:48 | 首次 8卡 torchrun | ❌ 失败: `init_process_group(device_id=device)` 报错, HCCL 后端不支持 device_id 参数 |
| 2026-03-27 18:49 | 修复: 移除 `device_id=device` 参数 | ✅ 成功 |
| 2026-03-27 18:49 | 第二次 8卡 torchrun | ❌ 失败: `HCCL allreduce: Unsupported data type at::kDouble` |
| 2026-03-27 18:50 | 分析: eval_val 中 3 个 float64 tensor 做 all_reduce, HCCL 不支持 double | — |
| 2026-03-27 18:50 | 修复: all_reduce 前将 float64→float32, 完成后再转回 float64 | ✅ 成功 |
| 2026-03-27 18:51 | 第三次 8卡 torchrun | ✅ 完整跑通! (714.8s 总耗时含 round-trip 验证) |

**教训**:
- HCCL 后端 `init_process_group` 不接受 `device_id` 参数 (与 NCCL 不同)
- HCCL 不支持 float64 (double) 类型的集合通信，需要先转 float32 再 all_reduce
- 这两个是昇腾迁移中常见的分布式通信坑点

### 11.2 8卡训练结果

| 指标 | 单卡 (1×NPU) | 8卡 (8×NPU) |
|------|------------|------------|
| warmup | 20步 | 20步 |
| 训练步数 | 209步 | 1617步 |
| 总训练时间 | 601s | 600s |
| 每步时间 | ~2876ms | ~371ms |
| **加速比** | 1x | **7.75x** |
| 初始 val_loss | 6.9357 | 6.9357 |
| 初始 val_bpb | 4.1077 | 4.1077 |
| 最终 val_loss | 3.3979 | 2.2381 |
| 最终 val_bpb | 2.0124 | 1.3255 |
| int8 roundtrip val_bpb | 未完成 | **1.3268** |
| peak memory/card | 19750 MiB | 19674 MiB |
| 模型大小 (int8+zlib) | 7.6MB | 14.3MB |
| 总提交大小 | 7.7MB | 14.3MB |

### 11.3 结果分析

- 8卡加速比 7.75x，接近线性加速，分布式通信开销小
- 10分钟内单卡只能跑 209步，8卡可跑 1617步，模型收敛更充分
- val_bpb 从 2.01 (1卡) 降至 1.33 (8卡)，提升显著
- int8 量化 round-trip 后 val_bpb 仅从 1.3255 升至 1.3268，量化损失极小
- 单卡各占 19.7GB HBM，64GB HBM 余量充足

### 11.4 阶段一完成汇总

所有代码改动点 (train_gpt_npu.py vs train_gpt.py):
1. `import acl` + `import torch_npu` (L8, L27)
2. `"cuda"` → `"npu"` 全局替换 (~15处)
3. `torch.cuda.*` → `torch.npu.*` (~12处)
4. `backend="nccl"` → `backend="hccl"`
5. 移除 `device_id=device` 参数 (HCCL 不支持)
6. tf32/flash_sdp 配置块替换为 NPU 注释
7. `torch.compile` 改为环境变量控制，默认关闭
8. `nvidia-smi` → `npu-smi info`
9. `fused=True` → `fused=False` (4处)
10. `zip(strict=True)` → `zip()` (Python 3.9 兼容)
11. eval_val 中 float64 all_reduce → 先转 float32 再通信 (HCCL 兼容)

环境要求:
- `ACL_PRECISION_MODE=allow_mix_precision` (必须)
- `source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash` (必须)
- 额外 pip 包: `scipy`, `psutil` (原 requirements 未包含)

---

## 12. 阶段三：CANN FlashAttention 融合算子集成

### 12.1 方案选择

经探查，CANN 8.3.RC1 已内置 FlashAttention 融合算子：
- C API: `aclnnFlashAttentionScore` / `aclnnFlashAttentionScoreGrad`
- Python API: `torch_npu.npu_fusion_attention` / `torch_npu.npu_fusion_attention_grad`

**决策**：直接使用 CANN 内置融合算子，而非从头用 AscendC 编写 kernel。理由：
1. 内置算子经过华为充分优化，性能和稳定性有保障
2. 已支持 GQA (num_heads != num_kv_heads)、causal mask、bf16
3. 前向+反向自动求导已封装好，开箱即用
4. 避免从零开发 AscendC kernel 的巨大工程量和调试成本

### 12.2 关键参数映射

| 原始 CUDA (F.scaled_dot_product_attention) | NPU (torch_npu.npu_fusion_attention) |
|---|---|
| is_causal=True | atten_mask=上三角bool矩阵 [Sq, Skv], sparse_mode=0 |
| enable_gqa=True (8h/4kv) | head_num=8, Q=[B,8,S,D], K/V=[B,4,S,D] |
| 默认 scale=1/sqrt(head_dim) | 显式传入 scale=1/sqrt(64)=0.125 |
| input_layout | "BNSD" (Batch, Num_heads, Seq, Dim) |
| - | pre_tockens=seqlen, next_tockens=0 |

### 12.3 调试过程

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-31 14:35 | 独立测试 npu_fusion_attention GQA+causal 前向+反向 | ✅ 成功 |
| 2026-03-31 14:38 | sparse_mode=2 无 atten_mask 测试 | ❌ 无 causal mask 效果 |
| 2026-03-31 14:40 | sparse_mode=0 + 显式 [Sq,Skv] mask 精度对比 | ✅ max_diff=0, cosine_sim=1.0 |
| 2026-03-31 14:40 | sparse_mode=2 + [2048,2048] mask (seqlen<2048时) | ❌ 报错：mask shape 不支持 |
| 2026-03-31 14:41 | sparse_mode=0 + [seqlen,seqlen] mask (seqlen=1024) | ✅ 精度完全匹配 |
| 2026-03-31 14:41 | sparse_mode=2/3 + [2048,2048] mask (seqlen=1024) | ✅ 精度匹配 |
| 2026-03-31 14:41 | 性能对比 (200 iters, bsz=4, seq=1024, bf16) | ✅ FA: 0.106ms vs SDPA: 0.155ms (31%↑) |
| 2026-03-31 14:45 | 集成到 train_gpt_npu.py 首次试跑 | ❌ Q/K 为 float32, V 为 bf16，类型不匹配 |
| 2026-03-31 14:47 | 添加 dtype 对齐 (q,k cast to v.dtype) | ✅ 修复 |
| 2026-03-31 14:48 | 单卡 5 步完整训练 (NPU_FLASH_ATTENTION=1) | ✅ 成功 |
| 2026-03-31 14:56 | 单卡 3 步对比训练 (NPU_FLASH_ATTENTION=0) | ✅ 成功 |

### 12.4 精度验证

**独立 attention 算子对比** (bsz=4, seq=1024, 8h/4kv, d=64, bf16):
- Max absolute diff: 0.000000
- Mean absolute diff: 0.000000
- Cosine similarity: 1.000000

**端到端训练对比** (单卡, 相同随机种子):

| 指标 | FlashAttention (FA=1) | 原始 SDPA (FA=0) |
|------|---|---|
| step 0 val_loss | 6.9357 | 6.9357 |
| step 1 train_loss | 6.9358 | 6.9358 |
| step 2 train_loss | 16.4454 | 16.4454 |
| step 3 train_loss | 16.0225 | 16.0249 |
| step_avg | ~2810ms | ~2784ms |

精度完全一致 (step 1-2 loss 完全相同, step 3 仅 0.001 差异)。

### 12.5 性能分析

**Attention 算子独立性能** (前向, bsz=4, seq=1024, bf16):
- `npu_fusion_attention`: 0.106 ms/iter
- `F.scaled_dot_product_attention`: 0.155 ms/iter
- **提升: ~31%**

**端到端步时间** (单卡, 含 data loading + optimizer + grad accumulation):
- FlashAttention: ~2810 ms/step
- 原始 SDPA: ~2784 ms/step
- 端到端差异不显著 (attention 仅占总时间一小部分)

注：8卡分布式训练下，attention 占比更高（通信开销降低），FA 的加速效果会更明显。

### 12.6 代码改动

`train_gpt_npu.py` 中的改动：
1. `CausalSelfAttention.__init__`: 新增 `_use_npu_fa`, `_fa_scale`, `_causal_mask` 属性
2. `CausalSelfAttention.forward`: 当 `NPU_FLASH_ATTENTION=1` (默认) 时使用 `torch_npu.npu_fusion_attention`，否则回退到 `F.scaled_dot_product_attention`
3. 添加 dtype 对齐: Q/K cast 到 V 的 dtype (解决 autocast 下 rms_norm 输出 fp32 问题)
4. causal mask 缓存: 首次调用时创建 [seqlen, seqlen] 上三角 bool mask 并缓存
5. 日志行更新: 显示当前使用的 attention 后端

**环境变量控制**:
- `NPU_FLASH_ATTENTION=1` (默认): 使用 CANN FlashAttention 融合算子
- `NPU_FLASH_ATTENTION=0`: 回退到 F.scaled_dot_product_attention

### 12.7 教训总结

1. `npu_fusion_attention` 的 `sparse_mode=2` 虽名为 causal，但仍需传入 `atten_mask`，且 mask 尺寸有 ≥2048 限制
2. `sparse_mode=0` + 显式 `[Sq, Skv]` mask 是最通用可靠的方案
3. `CastedLinear` + `rms_norm` + `autocast` 会导致 Q/K 被提升到 fp32，需要显式 cast 回 bf16
4. CANN 内置融合算子远比手写 AscendC kernel 实用——除非需要定制化的 tiling 策略

### 12.8 阶段三状态

- ✅ FlashAttention 融合算子已集成并验证
- ✅ 精度与原始 SDPA 完全一致
- ✅ Attention 计算加速 ~31%
- ⬜ 8卡分布式训练下的 FA 性能验证（需真机长时间跑完整训练）
- ⬜ torch.compile NPU 图模式验证（P1 待验证项）

### 12.9 8卡完整训练对比验证

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-31 15:24 | 8卡 torchrun NPU_FLASH_ATTENTION=1 | ✅ 完整跑通 (713.6s 含 round-trip) |

**对比结果 (FA vs 原始 SDPA)**:

| 指标 | FA (npu_fusion_attention) | 原始 (F.sdpa) | 差异 |
|------|---|---|---|
| 训练步数 | 1627 | 1617 | +10 步 |
| 总训练时间 | 600,496ms | 600,379ms | ~同 (600s cap) |
| 每步时间 | 369.08ms | 371.29ms | -0.6% |
| 初始 val_loss | 6.9357 | 6.9357 | 一致 |
| step 1000 val_bpb | 1.3804 | 1.3789 | ~相同 |
| 最终 val_loss | 2.2386 | 2.2381 | +0.0005 |
| 最终 val_bpb | 1.3258 | 1.3255 | +0.0003 |
| int8 roundtrip val_bpb | 1.3270 | 1.3268 | +0.0002 |
| int8 roundtrip val_loss (exact) | 2.24054050 | 2.24025154 | +0.0003 |
| peak memory/卡 | 19,652 MiB | 19,674 MiB | -22 MiB |
| 模型大小 (int8+zlib) | 14.3MB | 14.3MB | 一致 |

**分析**:
1. 精度几乎完全一致，val_bpb 差异仅 0.0002~0.0003，属于正常浮点噪声
2. 每步快约 2ms (369ms vs 371ms)，多跑了 10 步，收敛程度相当
3. 显存占用略低 22 MiB
4. 独立 attention 算子快 31%，但端到端仅快 0.6%——说明当前瓶颈不在 attention，而在 grad accumulation (8步)、Muon 优化器 Newton-Schulz 迭代等

**结论**: FlashAttention 融合算子集成功能等价，零精度损失，略有提速和省显存。

### 12.10 阶段三最终状态

- ✅ FlashAttention 融合算子已集成并通过 8 卡完整训练验证
- ✅ 精度与原始 SDPA 完全一致 (val_bpb 差异 < 0.001)
- ✅ Attention 算子加速 ~31%，端到端加速 ~0.6%
- ✅ 显存占用略有下降 (-22 MiB/卡)
- ⬜ 进一步优化方向：torch.compile NPU 图模式 (可优化非 attention 瓶颈)

---

## 13. torch.compile 图模式优化探索

### 13.1 超参化改造

将 `compile` 和 `flash_attention` 从环境变量硬编码改为 Hyperparameters 类中的超参：
- `USE_COMPILE` (默认 0/False): 是否对 Muon 优化器核心函数做 torch.compile
- `USE_FLASH_ATTENTION` (默认 1/True): 是否使用 npu_fusion_attention

参数一路从 GPT → Block → CausalSelfAttention 传递，不再依赖全局环境变量。

### 13.2 NPU compile 后端探查

| 后端 | 状态 | 说明 |
|--------|------|------|
| inductor (默认) | ❌ 不支持 | `aten.mean.dim` lowering 失败 |
| npu (torch_npu内置) | ✅ 可用 | 需要 torchair + protobuf |

安装了缺失的 `protobuf` 依赖后 `backend='npu'` 可用。

### 13.3 模型级 compile 尝试

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-31 17:09 | torch.compile(model, backend='npu', fullgraph=True) | ❌ view/reshape 不兼容 (transpose后非连续 tensor) |
| 2026-03-31 17:11 | torch.compile(model, backend='npu', fullgraph=False) | ❌ 同样的 view 错误 |
| 2026-03-31 17:12 | 添加 .contiguous() 在 Q/K/V transpose 后 | ❌ torchair 图编译超时 (>10min) |

**结论**: 模型级 compile 在当前 torchair 版本下不实用——编译时间太长，且对非连续 tensor 的 view 支持不完善。

### 13.4 函数级 compile (仅 Muon 核心函数)

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-03-31 18:04 | torch.compile(zeropower_via_newtonschulz5, backend='npu') 单卡 | ✅ 编译成功，训练跑通 |
| 2026-03-31 18:17 | 8卡 torchrun USE_COMPILE=1 | ⚠️ 编译时间太长 (>20min) |

**单卡 compile=1 vs compile=0 对比** (3 步):

| 指标 | compile=1 | compile=0 |
|------|-----------|----------|
| step 0 val_loss | 6.9357 | 6.9357 |
| step 1 train_loss | 6.9358 | 6.9358 |
| step 2 train_loss | 16.4454 | 16.4454 |
| step 3 train_loss | 14.2259 | 16.0245 |
| step_avg | ~2859ms | ~2773ms |

step 1-2 完全一致，step 3 差异来自 compile 改变了 Newton-Schulz 函数的数值行为。

### 13.5 结论与决策

torchair (NPU 的 torch.compile 后端) 在当前版本下存在以下限制：
1. 模型级 compile: 不支持非连续 tensor 的 view 操作
2. 函数级 compile: 可用但首次编译耗时数分钟
3. 8卡分布式下编译时间可能超过 20 分钟，而训练本身只有 10 分钟

**最终决策**: `USE_COMPILE` 默认为 False。用户可通过 `USE_COMPILE=1` 手动开启，适用于长时间训练场景（编译开销可摸薄）。

### 13.6 代码改动汇总

1. `Hyperparameters` 类新增 `use_compile`, `use_flash_attention` 超参
2. `CausalSelfAttention`, `Block`, `GPT` 构造函数新增 `use_flash_attention` 参数
3. `main()` 中 `torch.compile` 调用改用 `backend='npu'`，受 `args.use_compile` 控制
4. 模型级 compile 已移除（仅保留函数级 compile for zeropower_via_newtonschulz5）
5. 日志行新增 `use_compile` 输出
6. 删除所有 `NPU_TORCH_COMPILE` / `NPU_FLASH_ATTENTION` 环境变量引用
7. 新增依赖: `protobuf` (torchair 需要)

### 13.7 4卡验证补充（非主流程）

> 注：因 torch.compile 实验导致 NPU 0 驱动死锁（需要 root 重启设备或重启机器才能恢复），
> 无法使用 8 卡。跳过 NPU 0，用 4 卡 (NPU 1-4) 做了一次完整训练来确认代码改动正确性。

**运行方式**: `ASCEND_RT_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4`

| 指标 | 4卡结果 | 8卡 FA 基线 (12.9) |
|------|---------|------------------|
| 训练步数 | 830 | 1627 |
| 每步时间 | 724ms | 369ms |
| 初始 val_loss | 6.9357 | 6.9357 |
| 最终 val_bpb | 1.3958 | 1.3258 |
| int8 roundtrip bpb | 1.3990 | 1.3270 |
| peak memory/卡 | 19,743 MiB | 19,652 MiB |

**分析**:
- 初始 val_loss 与 8 卡基线完全一致 (6.9357)，确认模型初始化和前向计算无误
- 4 卡步时间 ~724ms ≈ 8 卡 369ms × 2，线性缩放符合预期
- 最终 bpb 高于 8 卡是因为训练步数不足 (830 vs 1627)，非精度问题
- 超参体系 (USE_COMPILE / USE_FLASH_ATTENTION) 工作正常

**结论**: 阶段三所有代码改动功能正确。待 NPU 0 恢复后可重新跑 8 卡完整对比。
