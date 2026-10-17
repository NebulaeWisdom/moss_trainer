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

预设默认开启 `training.compile_pointwise`：Triton 已由上面的 `requirements.txt` 按平台安装（Linux `triton`，Windows `triton-windows`，实测 `3.8.0.post28`），另需 PyTorch 暴露 Inductor 的 `emulate_precision_casts`（实测 `2.14.0+cu130`）；缺任一项启动即显式报错。PyTorch 版本不满足时，把配置里的 `compile_pointwise` 设为 `false`，cuDNN 收益仍在（见「训练性能」）。

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
# 13 tests passed（含真实 CUDA、block/dropout/RNG 与重算线程契约）
```

数值对照（111 项）按显式容差判定：全部有限，前向/loss 用相对 L2 ≤1%，梯度可用绝对上限 ≤1e-5 兜底，全部通过。以上只覆盖 12 步 A/B 与一个 epoch（37 步）的实测；**全部 20 epoch 的完整训练与收敛质量没有跑**，本页不给出收敛结论。

### 性能选项

- **`training.attention_backend`** — `cudnn`（新 run 默认）或 `auto`（完全沿用 stock 选择）；实测收益见上表。
- **`training.compile_pointwise`** — 默认 `false`；`configs/` 下四个预设都显式开 `true`。开启需要已安装的 PyTorch 暴露 Inductor 的 `emulate_precision_casts`（实测 2.14.0+cu130）、可用的 Triton（Windows 实测 `triton-windows 3.8.0.post28`）与 CUDA；缺任一项会在加载模型前显式报错，不会静默退回 eager。`false` 时完全不引入 `torch.compile`，其他环境照常可跑，且仍保留 cuDNN 收益。
- **`training.compile_blocks`** — 默认 `false`，四个预设均保持未开启。只支持 `adapter.algo="lora"`，且必须显式满足 `compile_pointwise=true`；不满足时启动报错，不会替你改配置。开启后在 LyCORIS 挂载、resume 恢复之后编译全部 42 个主 block 的原始 forward，包含原有 LoRA 调用；不编译 TE/VAE，不替换模块或参数，不改分桶。使用动态形状、fullgraph 与 BF16 精度转换模拟，不启用 CUDA graphs；随机/dropout 算子保留 ATen RNG 路径。编译失败直接报错。LoKr 尚未通过相同数值契约，开启此项会明确拒绝；`compile_blocks=false` 的 LoKr 训练不变。
- **`training.compile_blocks_mode`** — 仅在 `compile_blocks=true` 时校验、补默认值并生效；取 `default`（默认，不做 GEMM autotune）或 `max-autotune-no-cudagraphs`（显式启用 Inductor autotune，可能增加首次编译/调优开销）。使用 PyTorch 自带 mode 选项映射，再合并相同的精度/RNG约束与禁用 CUDA graphs 设置；不是改变基座精度的选项。启用时解析后的 mode 随配置/state 保存，并写入 start 日志；关闭 block 编译时不新增默认 mode、不校验该项，start 日志记为 `null`。

这些选项只作用于训练进程中的训练、反向重算与 live preview，退出时还原函数、实例 forward 和拥有的编译器环境；独立 `sample.py`/`samples.py` 不因训练配置而开启 block 编译。`compile_blocks` 随 run 配置/state 保存，旧 state 缺该项时保持 `false`；详细行为见 [TUTORIAL.md](TUTORIAL.md) 第 5 节。

**block 编译的现有证据**：临时原型的同口径无插桩 12-step A/B，未开启 block 编译的 warm median 为 **2.195143 s/sample**：

| block 模式 | warm median s/sample | 相对吞吐 | 本次进程首步秒数（含编译/autotune，部分缓存已存在） | 对照 step loss 最大绝对差 |
|---|---:|---:|---:|---:|
| `default` | 2.026771 | +8.3% | 35.68 | 0.001108 |
| `max-autotune-no-cudagraphs` | 1.944467 | +12.89% | 129.14 | 0.0002687 |

`default`记录为`temp/train-next-opt/block-default-thread-12`。35.68s与129.14s是**本次进程首步**：复用了此前探测留下的部分共享Triton/Inductor缓存，**不是清空缓存后的冷启动完整编译时间**。两种模式均**不是 bitexact**；这些额外首步成本不能从 warm median 中消失，短任务可能不划算。BF16 冻结基座和 FP32 适配器精度保持不变。

**发布集成实跑（autotune）**：直接使用生产`launch.py`，显式设置`compile_blocks=true`与`compile_blocks_mode="max-autotune-no-cudagraphs"`，已完成**38个update、297张样本**（完整289图epoch，再训练下一epoch的8张）。第37步保存adapter与完整state，完成**20步、832×1216 live preview**，随后成功回到训练并完成第38步。

- 同口径前12步的warm median：**1.93800625 vs 2.195143125 s/sample**，吞吐 **+13.268%**；对应step loss最大绝对差 **4.17768955e-4**，不宣称bitexact。整个38步run的warm样本加权耗时为**1.944404 s/sample**；这一全run指标不直接套用前12步的相对提升比例。
- 训练阶段torch记录的峰值显存为**11.522 GiB**；这不是全进程或WDDM dedicated显存口径。preview耗时**46.699s**，其中首次preview图入口约**24.68s**（含惰性编译）。本次进程首步为**23.24s**，已有编译缓存较暖，仍不是清空缓存后的冷启动测量。
- 集成内核契约测试**13/13通过**，含真实CUDA。
- **跨进程resume已实跑**：生产`launch.py`从第37步state续到第38步；与不中断run相比，step38 loss **0.8430308103561401完全相同**，config、scheduler、游标及CPU/CUDA/噪声生成器/Python/NumPy RNG均完全一致。记录：`temp/train-next-opt/production-block-resume-comparison.json`。
- **下一次update后的数值并非bitexact**：FP32 adapter状态494/1260个张量不同，maxabs **1.6793608665e-4**、最大单张量relative L2约**0.002355**；optimizer状态494/504个张量不同，maxabs **9.0394e-7**；BF16导出494/756个张量不同，maxabs **4.8828125e-4**。这是各自完成第38次update后的对照，不是刚加载第37步state时的比较；**保存边界的状态恢复与后续计算轨迹逐bit复现是不同契约**。

autotune选择可能随首个动态shape hint、实测计时和缓存状态变化，checkpoint不保存编译图/调优选择；这是跨进程数值风险，不是本次差异已被归因于某一种kernel。既有未开启block编译的resume实跑也有已记录的浮点漂移，以上不新增bitexact保证。**完整20 epoch收敛与质量未验证**。

### 等价性边界

原生 cuDNN 与 stock SDPA 是同一 attention 数学，但 tiled reduction 顺序不同，不是逐 bit 相同；pointwise 编译保持 eager 的 BF16 逐算子舍入，不做精度降级，同样不保证逐 bit 一致。数值对照（111 项，本地临时记录 `temp/train-speed-opt/combined-math-v2.json`）：全部有限；前向/loss 按相对 L2 ≤1% 判定，个别 LoKr 小梯度改用绝对上限（≤1e-5）判定，因此不宣称"全部相对误差 <1%"；LoRA/LoKr 小模型下冻结基座无梯度、权重不变，适配器梯度有限非零。以上都是短程证据（最多一个 epoch），不是收敛质量证明。

### 已评估但未采用

LyCORIS 的 triton 后端在 cuDNN 下 step2–4 median 22.666s vs 22.838s，同索引对比仅约 0.75%，落在运行间波动内，不构成收益；训练继续用 torch 后端。

**向量化前向原型（未采用）**：把vendor的逐样本循环改成masked batched forward的本地原型（工作区根目录`temp/train-batch-opt/vectorized_candidate.py`，即`../temp/...`，不入库）：`bs3`+offload实测超出WDDM上限（reserved 17.777GiB），弃用；BF16下LoKr的2/1080项比较超出预设容差（同一配置的FP32结构诊断全部通过），因此不纳入生产。没有测过`bs2`的向量路径，也不做质量等价推断。

**显式CPU offload（未采用）**：把checkpoint激活用`save_on_cpu`钩子搬到CPU的本地实验（`group1`+offload+`expandable_segments`，实验性13.5GiB allocator上限）。step2的8个microbatch/24样本里每个方向搬27,417,933,312B（25.535GiB）、3024次copy调用；纯拷贝H2D 957.194ms（还多含98,854,681B输入传输）、D2H 961.120ms（多112B），合计1.918314s，**已包含在前向/反向的阶段时间里**，不是在step外额外叠加；hook的CUDA event区间988.019/995.572ms含launch/host间隙，不能当纯拷贝耗时。该run里pinned池4.295GiB、PDH shared 82→4480MiB（+4398由该池解释）；torch峰值`allocated` 10.484GiB/`reserved` 11.053GiB，设备峰12631MiB，allocator retries/OOM为0。无profiler的配对前缀实跑（同`data_indices`顺序、前3个update共68样本）：`group2`无offload 161.220262s（2.370886 s/sample，warm 2.284373）vs `group1`+CPU offload 167.791281s（2.467519，warm 2.376396），本短对照慢4.08%——但它比的是两种checkpoint策略（group数不同），不是offload单因素A/B，**不外推**这个比例（该实跑仍带轻量的hook字节/CUDA event计数，只是没开`torch.profiler`）。每个update约54.84GB双向搬运、pinned池4.295GiB，未带来吞吐收益，所以这是**显式CPU offload**、不纳入生产，也绝不进入上面无offload的容量主表。原始：工作区根目录`temp/train-batch-opt/offload-transfer-summary.json`（即`../temp/...`）。

## batch、梯度累积与显存口径（16GB 实测）

**复测口径**：289图/26桶/seed42/1024档/冻结BF16基座/FP32 LoRA rank4/Muon 5e-4；`checkpoint_group_size=2`、`loader.pin_memory=true`、显式`expandable_segments`、无CPU offload、无allocator上限；跑完整1个epoch（含该epoch的保存state与preview）。三行的batch计划不同、update数也不同，**名义样本数相同不等于质量等价**；`s/sample`=Σ`step_seconds`/289。原始记录在**工作区根目录**`temp/train-batch-opt/resident-matrix-results.json`（从仓库`moss_trainer/`出发即`../temp/train-batch-opt/...`；本地git-ignored测量产物，不随仓库提交）与各run的`train.jsonl`/显存采样JSONL。复现只需用原config改`batch_size`/`gradient_accumulation`/`checkpoint_group_size=2`并加allocator环境变量，cache仍在仓库`temp/new-cache`。

| 配置 | updates / microbatches | Σ step_seconds | s/sample | 相对 bs1 | epoch_loop | 全命令 |
|---|---|---|---|---|---|---|
| `bs1 × acc24` | 13 / 289 | 655.484s | 2.2681 | 1.000× | 681.909s | 703.610s |
| `bs2 × acc12` | 13 / 153 | 659.892s | 2.2834 | 0.993×（−0.67%） | 687.048s | 710.469s |
| `bs3 × acc8` | 14 / 107 | 662.718s | 2.2931 | 0.989×（−1.10%） | 689.120s | 711.094s |

`epoch_loop`是完整一个epoch的训练循环（含该epoch的保存与preview）；`全命令`是整个进程的墙钟（含加载TE等启动开销）。三行的真实样本数都是289个唯一索引、26个桶，caption分布一致（long 106/medium 99/short 84）。

| 配置 | 峰值 allocated（GiB，冷/暖） | 峰值 reserved（GiB） | 训练进程 dedicated 峰值（MiB） | 训练时段设备峰值（MiB） | 全命令时段设备峰值（MiB） | shared 增长 / pinned 池峰值（MiB） | retries / OOM |
|---|---|---|---|---|---|---|---|
| `bs1 × acc24` | 11.688 / 11.426 | 11.896 | 14121.531 | 13216 | 14732 | 60 / 61.255 | 0 / 0 |
| `bs2 × acc12` | 12.229 / 11.960 | 12.479 | 14121.637 | 14022 | 14973 | 114 / 115.009 | 0 / 0 |
| `bs3 × acc8` | 12.762 / 12.468 | 13.025 | 14121.754 | 14650 | 14999 | 174 / 174.514 | 0 / 0 |

`allocated`/`reserved`是torch分配器口径（GiB）。`训练进程 dedicated 峰值`是PDH `\GPU Process Memory(*)\Dedicated Usage`在整个run内的峰值（preview主导）。`训练时段/全命令时段设备峰值`是nvidia-smi的1Hz采样：分别是训练循环期间与整个进程期间的**设备整体**dedicated使用量，**含桌面与其他GPU进程，不是本训练进程的峰值**。两套单位不混用。

- **吞吐结论**：bs2/bs3相对bs1没有实质收益——`s/sample`分别慢0.67%/1.10%。这与“原生前向仍逐样本”一致（见下）：batch>1只把多个样本串进同一个update，不提高单样本内部的并行度。
- **容量结论**：同一套生产口径下，bs1/bs2/bs3都跑完了完整289图/26桶一个epoch：1Hz PDH未观察到超出pinned池的shared增长（60/114/174MiB vs 池峰值61.255/115.009/174.514MiB），训练进程dedicated峰值14121.531/14121.637/14121.754MiB（preview主导）仍留有余量，allocator retries与OOM都是0——结合这三点，结论**不依赖WDDM spill**。设备整体口径（含桌面与其他GPU进程）在训练时段峰值13216/14022/14650MiB、全命令时段峰值14732/14973/14999MiB（均为1Hz采样）。
- **边界**：以上只覆盖这一套数据（289图/26桶/1024档/rank4 LoRA/seed42/单epoch，`group2`+显式`expandable_segments`），不推广到其他数据集、更长的文本或别的rank。分组与allocator的显存消融见本节末「显存贡献」，分组等价性见「分组的等价性实跑」，当前生产配置的kernel构成见「kernel 成本构成」。
- 官方四个预设保持`batch_size=1`+`gradient_accumulation=8`+默认`checkpoint_group_size=1`，本次复测不改预设。

下面是选项的行为与改配置方式。

- **`training.checkpoint_group_size`**（默认`1`，正整数；只在`training.gradient_checkpointing=true`时改变计算）：`1`是原来的行为——42个主block各自一个checkpoint；`2`把2个**相邻的激活block**合成一组、每组一个checkpoint（`skip_layers`先排除，再按激活block的索引顺序切分）。分组只改变“保留哪些中间激活、反向时重算哪些”：前向/反向/重算的数学与adapter dropout的RNG消耗顺序都不变，输出与梯度与逐block checkpoint一致（CPU与CUDA用例逐张量比对，含`skip_layers`与dropout）。组内保留的激活更少，但反向重算时要同时驻留的live set更大，**不能假定组越大越省或越快**，要按实测选。只作用于主block，`text_embedder`仍按逐block checkpoint；四个预设都不写这项（默认`1`）。本次复测三行都用`group_size=2`，分组自身的单变量消融只到`1→2`且只报torch allocated/reserved（见本节末「显存贡献」）——这里不断言`2`优于`1`。
- **累积loss按样本加权**：`Σ(microbatch_mean × actual_batch_size) / window_samples`，窗口内每个样本等权。`batch_size=1`时`window_samples`等于microbatch数，数学与旧实现完全相同——本页此前的实测（含1.70×那组）都在`batch_size=1`下测得，结论继续有效。旧实现`batch_size>1`时用mean-of-batch-means，窗口内含不等大小批（桶尾）会高估小批里的样本，现已修正；旧`batch_size>1`的state会按新口径续跑，在桶尾窗口上不与旧轨迹逐bit一致，没有legacy开关或兼容层。
- **样本数只是上界**：只把同桶（同宽高）图片stack成一批，桶尾批更小，所以`batch_size × gradient_accumulation`只是每update样本数的上界；每step记录里的`sample_count`（窗口实际样本数）与`batch_sizes`（各microbatch大小）才是实际值。
- **容量口径（Windows 16GB）**：判断“装得下”不能只看一个数。`torch.cuda.memory_allocated`/`memory_reserved`只是PyTorch分配器视角，`nvidia-smi`是设备视角，WDDM下还要看进程级`\GPU Process Memory(*)\Dedicated Usage`/`Shared Usage`，并同时满足dedicated仍有剩余余量、allocator retries/OOM为0。**shared增长本身不是溢出**：`loader.pin_memory=true`（预设默认、生产口径）时CUDA pinned主机缓冲就计入shared，属正常，增量应与`torch.cuda.host_memory_stats()`的registered/cached pinned pool对得上；只有**未被pinned pool解释的大额shared增长**才是WDDM spill风险。反过来没有OOM也不等于激活都常驻显存。`pin_memory=false`只作为附录对照。显式CPU offload（checkpoint激活搬CPU）只在本地实验里评估过，不在发布代码里、也没有配置项，**不进本节的容量表**（见「已评估但未采用」）。
- **显存分配器（可选，显式opt-in）**：`expandable_segments`只在启动前显式设环境变量才生效：`$env:PYTORCH_ALLOC_CONF = 'expandable_segments:True'`（PowerShell，之后照常跑`launch.py train`）。它只改CUDA caching allocator的段管理，不改训练数学、数据、步数或任何配置项；本机torch 2.14.0+cu130/Windows实测支持，旧torch或其他平台不保证。默认不启用：`launch.py`/配置/`requirements.txt`都不设它，也不覆盖你已有的allocator设置（本机torch先读旧名`PYTORCH_CUDA_ALLOC_CONF`，未设置时才读`PYTORCH_ALLOC_CONF`）。
- **原生前向仍逐样本**：vendor的DiT `forward`按batch维逐样本循环，`batch_size>1`不会自动把单样本内部的计算并行化，只是把多个样本串进同一个update。本地另有一个把这些样本合并成masked batched forward的向量化原型，已弃用、不入库、**不是生产路径**（弃用依据见「已评估但未采用」）。
- **改配置的名义对齐**：想在不改每update名义样本数的情况下试更大batch，写`batch_size=2, gradient_accumulation=4`（名义8，与预设的`1×8`对齐）；做等样本数对照可用`1×24`/`2×12`/`3×8`（名义都是24）。这些只是**名义样本数对齐**，不是质量等价：本次复测的`1×24`/`2×12`/`3×8`三行在16GB、生产口径下都跑完了一个epoch（见上表），但该结论只限这套数据。两个选项的完整默认值、校验与回退规则见 [TUTORIAL.md](TUTORIAL.md) 第 5 节 `training`。

### 显存贡献：group 与 allocator（早期消融，3 updates）

口径：`bs3 × acc8`、同前3个update、同一native allocator（未开`expandable_segments`）；只报torch `allocated`/`reserved`。该早期消融**没有采PDH shared**，也不能用旧时间宣称可靠加速。

- `group1` → `group2`：warm峰值`allocated` 13.533 → 12.436 GiB，省1.097 GiB（1123 MiB）。
- `group2`再加`expandable_segments`：warm峰值`reserved` 14.258 → 13.025 GiB，省1.233 GiB（1263 MiB），是分配器缓存/碎片层面的；同期`allocated` 12.436 → 12.426 GiB近似不变。
- **两个数不能相加**成“总显存节省”：一个降的是保留激活，一个降的是分配器缓存/碎片。组大小只测了`1→2`、只在这一套配置与数据上，不推断更大组或别的rank/分辨率。

### 分组的等价性实跑（BF16 + 生产kernel作用域）

`group2`在BF16实跑（`attention_backend=cudnn`、`compile_pointwise=true`，与逐block checkpoint基准同作用域）逐项比较：726项（54项前向/loss + 672项梯度）的`maxabs`/`meanabs`/`relative_l2`/`mismatch_pct`全为0，覆盖LoRA rank4、LoKr rank4+`factor=16`+`use_scalar`、LoRA rank4+dropout0.1，`batch_size`2/3、`skip_layers`[]/[1]；backward后调用方RNG逐位不变（`caller_rng_restored=true`），冻结基座无梯度且未被改写。另有unit用例23项CPU/CUDA通过（`tests/test_checkpoint_groups.py`8项 + `tests/test_accumulation.py`9项 + `tests/test_training_kernels.py`6项）。这是helper层面的小模型实跑（`dim=128`/4层），不是5B全模型长跑，也不构成质量结论。原始：工作区根目录`temp/train-batch-opt/production-group2-math.json`（即`../temp/...`）。

### 当前生产配置的 kernel 成本构成

口径：真实当前生产配置（`checkpoint_group_size=2`、`attention_backend=cudnn`、`compile_pointwise=true`、`expandable_segments`，`batch_size=1`/`gradient_accumulation=8`）的一次CUDA trace，step2的microbatch2；只聚合kernel原始事件（`cat=kernel`，不重复叠加operator key_averages），是**构成分解，不是受控加速A/B**。

| 类别 | 时间 | 占比 | kernel调用 |
|---|---|---|---|
| GEMM | 1300.734ms | 59.1% | 3204 |
| attention | 530.880ms | 24.1% | 212 |
| eager elementwise | 259.446ms | 11.8% | 3949 |
| compiled pointwise | 91.925ms | 4.2% | 897 |
| copy/cast/cat | 18.271ms | 0.8% | 2080 |
| 其他 | 0.012ms | ~0% | 6 |
| kernel总计 | 2201.268ms | — | — |

- GPU活动union/span = 2203678/2283879µs = 96.49%，该trace未观察到kernel重叠；传输只有H2D 3次4.512MB、DtoH 2次5B、DtoD 1次25.240MB（memset 852次2.275GB）。
- 结论范围：只说明主要device成本在GEMM与attention、host不是主瓶颈；**不说**backend已最优，也不说没有kernel优化空间（没有roofline/SM计数器）。
- 该profiler的step墙钟被CUPTI额外约6.16s污染，**吞吐一律用上面的resident主表**。原始：工作区根目录`temp/train-batch-opt/production-kernel-summary.json`、`temp/train-batch-opt/production-kernel-profile.jsonl`、`temp/production-kernel-profile.trace-step2.microbatch2.json`（即`../temp/...`，本地测量产物，不随仓库提交）。

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
| `configs/moss_official.json` | 官方 JSON caption |
| `configs/plain_txt.json` | 同名 `.txt` caption |
| `configs/res768.json` | 768 面积档 |
| `configs/lokr_factor16.json` | LoKr，factor=16 |

四个预设都显式写 `attention_backend="cudnn"` + `compile_pointwise=true`。
