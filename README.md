# MOSS-Image DiT 适配器训练器

MOSS-Image 5B 单流 DiT 的适配器训练器，支持 LoRA / LoKr（走 LyCORIS）。TE、VAE、DiT 基座全部冻结，只训练挂在 DiT `_Block` 上的适配器；RTX 5080 16GB 单卡可跑。

详细配置清单见 [TUTORIAL.md](TUTORIAL.md)，现成配置见 [configs/](configs/)。

## 环境

- Python 3.11；CUDA 12.8+ 的 PyTorch。Blackwell / RTX 5080 不要装 cu121/cu124。

## 安装

```powershell
pip install -r requirements.txt
git clone https://github.com/KohakuBlueleaf/LyCORIS
pip install ./LyCORIS
```

官方预设默认开启 `training.compile_pointwise`：Triton 已由上面的 `requirements.txt` 按平台安装（Linux `triton`，Windows `triton-windows`，实测 `3.8.0.post28`），另需 PyTorch 暴露 Inductor 的 `emulate_precision_casts`（实测 `2.14.0+cu130`）；缺任一项启动即显式报错。PyTorch 版本不满足时，把配置里的 `compile_pointwise` 设为 `false`，cuDNN 收益仍在（见「训练性能」）。

## 模型

官方 BF16 SFT 权重：

- Hugging Face：<https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT>
- ModelScope：<https://modelscope.cn/models/openmoss/MOSS-Image-SFT>

下载后把配置里的 `model_dir` 指向本地目录，例如 `G:/ComfyUI/models/moss_image_sft_bf16`。

## 数据

`dataset.image_dir` 指向图片目录。caption 两种写法：

- **同名 `.txt`**（`caption_format: "plain"`）：图片同目录同 stem 的文本文件；没有该文件时用配置 `dataset.caption` 字段兜底。
- **官方打标 JSON**（`caption_format: "moss_official"`）：图片同 stem 的 `.json`，含 `caption.long/medium/short` 三级文本，训练时按 `caption_weights` 加权抽取一级。

## 训练流程

子命令按 cache → train 顺序执行：

```powershell
# 1. 编码 latent 与 caption embedding
python launch.py cache --config configs/moss_official.json --cache-dir temp/my-cache

# 2. 训练
python launch.py train --config configs/moss_official.json --cache-dir temp/my-cache --output-dir runs/my-run
```

`--output-dir` 必须是不存在的目录。换了图片、caption 文字或分桶设置，要换新的 `--cache-dir`。

## 训练性能（实测）

实测机器与工作负载：RTX 5080 16GB 单卡；官方 1024 面积档 + 多宽高比桶；289 图数据集；与 `moss_official.json` 同款训练参数（seed42、LoRA rank4/alpha4、Muon、`batch_size=1`、`gradient_accumulation=8`、gradient checkpointing）；12 步 A/B 与生产 run 复用同一 cache。bench 只把 `max_steps` 限制到 12，constant scheduler 不受影响；warmup 的 step1 与 epoch 出图（原 cadence 落在 12 步之外）都排除；没有为提速降低分辨率、采样步数或样本数。

| 配置 | median s/update（step2–12） | s/sample（accum8） | 相对原始 |
|---|---|---|---|
| 原始（stock SDPA、未编译） | 30.087 | 3.761 | 1.000× |
| 仅 cuDNN attention 优先 | 22.521 | 2.815 | 1.336×（−25.15%） |
| cuDNN + pointwise 编译（候选 A/B） | 17.743 | 2.218 | 1.696×（−41.03%） |
| 生产 run（无插桩，官方配置，前 12 步） | 17.719 | 2.215 | 1.698×（−41.11%） |

- 上表前三行是独立的单变量 A/B（原始 / 只加 cuDNN / 再加 pointwise 编译），同一 cache、同一 workload、12 步同索引对比；第四行是官方配置的生产 run（`production-plain-12`，同一 cache、`max_steps=12` 截断、无 profiler）的前 12 步。生产全 epoch 复测见下，覆盖全部 289 张图、26 个桶、epoch 尾批、preview 与 resume。
- 基线重复跑 median 29.964s（相对 30.087s 漂移 <0.5%）。首 12 步 loss 相对初始基线的最大差：候选 A/B 2.21e-4（均值 8.9e-5）、生产 run 无插桩 6.89e-4、生产 run 开 profiler 6.04e-4；两次基线自身差 2.37e-4。loss 量级约 0.94，三组差异都 <0.1%，但都不逐 bit 复现。
- 显存：原始峰值 12.076 GiB；候选首次 11.737 GiB、之后 warm 稳定 ≤11.44 GiB，编译不额外抬高峰值。
- 瓶颈定位（本机 profile，`temp/profile_training.baseline-profile-4b.jsonl`，本地临时产物不入库）：一个 update 里前向 ≈7.6s、反向+重算 ≈22.3s（GPU），optimizer ≈0.13s GPU/0.33s host，data ≈0.001s host，H2D/加噪准备 ≈0.0025s，末尾 item 同步 0.00036s。单个 micro 的 CUDA trace 里 attention 内核占大头：efficient 内核 forward 86 次调用总计 0.512s、backward 42 次调用总计 1.040s；换 cuDNN 的 trace 样本为 0.194s / 0.341s（同为该 micro 的合计值，形状相近，但不同 trace 样本、非逐算子受控计时）。本机 Windows 版 torch 的 flash attention 未编译（built false），stock 选到 cutlass efficient 内核，因此默认改为 cuDNN 优先。同一 micro 的 trace 里 H2D 只有 3 次共 5.10MB（0.180ms）、DtoH 2 次共 5 字节（标量，0.00096ms）、DtoD 169 次 9.20GB（24.20ms），说明稳态循环没有模型 CPU offload（按传输量而非推测）；trace 里的 `cudaDeviceSynchronize` 235.8ms 是 profiler 关闭开销，不是训练开销。

### 生产全 epoch 复测（官方配置，profiler 开启）

`temp/train-speed-opt/production-epoch-38`（本地临时产物不入库）：官方 `moss_official.json`（含 `cudnn` + `compile_pointwise`）原调度器 `num_training_steps=740`（20 epoch × 37 步，不设 cap），外部在第 38 步停止。第一个 epoch 跑满 37 步 = 289 个唯一索引各出现一次、26 个桶全部命中；第 37 步只剩 1 个 microbatch（2.461s），第 38 步进入下一 epoch 处理 8 个样本后停止，共 297 个样本。第 37 步同时保存了 state 与 preview PNG。

| 生产 run（profiler 开启） | 数值 |
|---|---|
| median s/update（step2–12） | 18.345（2.293 s/sample） |
| median s/update（step2–36） | 18.321 |
| 第 1 步 / warm 峰值显存 | 26.158s / 11.737 GiB → warm ≤11.509 GiB |
| 平均 GPU 前向 / 反向+重算 | 5.481s / 12.657s |
| 平均 optimizer / prepare | 0.128s / 0.00248s |
| 第 37 步保存 / 该步 preview | 0.224s / 29.211s |

- 全 26 个桶都真实跑过：无重编译溢出、无 OOM、无插桩报错。preview 29.211s 只与基线启动时那次 34.749s 作运行开销对比（适配器状态不同，不是同权重对照）。
- 首 12 步 loss 相对初始基线的最大差：生产 run 开 profiler 6.04e-4、无插桩 6.89e-4，候选 A/B 2.21e-4；loss 量级约 0.94，三者都 <0.1%，只记录“非 bitexact”，不作为质量结论。
- resume 验证（`production-resume-38`）：从第 37 步的 `step000037.state.pt` 用 `--resume`（不传 `--config`，配置与 cache 路径都来自 state）续跑到第 38 步并保存。state 里的 `attention_backend="cudnn"` 与 `compile_pointwise=true` 被原样恢复（`train.jsonl` 的 `start` 事件同为 `cudnn`/`true`），preview 文本在启动阶段重新编码 11.484s。续跑与不中断的第 38 步对齐：`data_indices`/`buckets`/`caption_stats`/LR/`next_epoch`/`next_batch`/microbatch 数完全一致，loss 同为 0.8431429266929626；grad_norm 0.0387445204 vs 0.0387395062；两次保存的 scheduler state 一致，CPU/CUDA/噪声生成器/Python/numpy 的 RNG state 逐位相等。
- 权重不逐 bit 复现：两次第 38 步的适配器 1260 个张量同名同形，maxabs 1.9446e-4、全张量（含 buffer）相对 L2 7.36e-5。即游标/优化器/RNG 的 resume 是精确的，权重层面存在已记录的数值非确定性，不宣称 bitexact。
- 无插桩对照（`production-plain-12`，官方配置、同一 cache、`max_steps=12`）：median 17.719s/update（2.215 s/sample），相对原始基线 1.698×（−41.11%），第 1 步 24.227s，首次 11.737 GiB 后 warm ≤11.44 GiB。开 profiler 的同配置 run 实测 18.345s，比无插桩慢约 3.4%（不同 run，不作因果结论）；结论性对比一律用无插桩结果，上表第四行即该结果。

### 复现与验证

上表的对照做法：把预设复制成两份临时配置（例如 `temp/bench-auto.json` / `temp/bench-cudnn.json`），只改 `training.attention_backend`（`auto` / `cudnn`）与 `training.compile_pointwise`（`false` / `true`），把 `training.max_steps` 限制到 `12`（`constant` scheduler 与 warmup 不受影响），用同一个 `--cache-dir` 各跑一次，再比较两个 `train.jsonl` 里 `step_seconds` 在 step2–12 的中位数（step1 是 warmup；epoch 出图步骤另算）。

```powershell
python launch.py train --config temp/bench-auto.json  --cache-dir temp/my-cache --output-dir temp/bench-auto
python launch.py train --config temp/bench-cudnn.json --cache-dir temp/my-cache --output-dir temp/bench-cudnn
```

固定内核契约的用例（含 CUDA 梯度检查点场景）：

```powershell
venv/Scripts/python.exe -B -m unittest discover -s moss_trainer/tests -p test_training_kernels.py -v
# 6 tests passed, 0 skipped（含 CUDA checkpointed-grad），10.829s
```

数值对照（111 项）按显式容差判定：全部有限，前向/loss 用相对 L2 ≤1%，梯度可用绝对上限 ≤1e-5 兜底，全部通过。以上只覆盖 12 步 A/B 与一个 epoch（37 步）的实测；**全部 20 epoch 的完整训练与收敛质量没有跑**，本页不给出收敛结论。

### 两个性能选项

- **`training.attention_backend`** — `cudnn`（新 run 默认）或 `auto`（完全沿用 stock 选择）；实测收益见上表。
- **`training.compile_pointwise`** — 默认 `false`；官方预设 `configs/moss_official.json` 显式开 `true`。开启需要已安装的 PyTorch 暴露 Inductor 的 `emulate_precision_casts`（实测 2.14.0+cu130）、可用的 Triton（Windows 实测 `triton-windows 3.8.0.post28`）与 CUDA；缺任一项会在加载模型前显式报错，不会静默退回 eager。`false` 时完全不引入 `torch.compile`，其他环境照常可跑，且仍保留 cuDNN 收益。

两者只改训练进程内的内核选择与点算子实现，不改冻结基座权重、分桶、数据或步数；详细默认值、作用域与回退规则见 [TUTORIAL.md](TUTORIAL.md) 第 5 节。

### 等价性边界

原生 cuDNN 与 stock SDPA 是同一 attention 数学，但 tiled reduction 顺序不同，不是逐 bit 相同；pointwise 编译保持 eager 的 BF16 逐算子舍入，不做精度降级，同样不保证逐 bit 一致。数值对照（111 项，本地临时记录 `temp/train-speed-opt/combined-math-v2.json`）：全部有限；前向/loss 按相对 L2 ≤1% 判定，个别 LoKr 小梯度改用绝对上限（≤1e-5）判定，因此不宣称"全部相对误差 <1%"；LoRA/LoKr 小模型下冻结基座无梯度、权重不变，适配器梯度有限非零。以上都是短程证据（最多一个 epoch），不是收敛质量证明。

### 已评估但未采用

LyCORIS 的 triton 后端在 cuDNN 下 step2–4 median 22.666s vs 22.838s，同索引对比仅约 0.75%，落在运行间波动内，不构成收益；训练继续用 torch 后端。

## 独立采样 `samples.py`

`launch.py sample` 之外的独立入口：一次加载模型，可对**任意 prompt**（不需要预先写进 cache）和**多个 checkpoint** 出图：

```powershell
python samples.py --config runs/my-run/config.json `
  --adapter runs/my-run/step000629.safetensors --multiplier 0 1 `
  --prompts "..." --seed 42 --width 832 --height 1216 --steps 20 --cfg 3 `
  --output-dir temp/new-dir
```

`--adapter` 可传多个文件，`--multiplier 0` 只出基座图。`--output-dir` 同样必须是新目录。

## 配置预设

| 预设 | 用途 |
|---|---|
| `configs/moss_official.json` | 官方 JSON caption，显式开 `attention_backend="cudnn"` + `compile_pointwise=true` |
| `configs/plain_txt.json` | 同名 `.txt` caption |
| `configs/res768.json` | 768 面积档 |
| `configs/lokr_factor16.json` | LoKr，factor=16 |

其余预设不写这两个性能项：新 run 未写 `attention_backend` 时同样解析为 `cudnn`，`compile_pointwise` 取默认 `false`。
