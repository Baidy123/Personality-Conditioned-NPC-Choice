# RQ2 · 2A 实验操作手册

这份手册给实验执行者（你自己）看：按顺序敲命令即可，不需要读代码。
设计文档：`docs/specs/2026-07-10-rq2-2a-pipeline-design.md`。

## 0. 环境准备（一次性）

需要 Python 3.11+，依赖 numpy、matplotlib、torch：

```powershell
pip install numpy matplotlib
# CPU 版 torch（本机没独立显卡时）：
pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU 版 torch（服务器 / 有 NVIDIA 显卡，CUDA 12.x）：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

所有命令都在 `code/` 目录下执行。装好后先跑一遍单元测试确认环境没问题：

```powershell
python -m pytest tests/ -q
```

预期最后一行是全部通过（无 failed）。

## 1. 四条命令

| 步骤 | 命令 | 作用 | 预计耗时 |
|---|---|---|---|
| 1 | `python -m experiments.rq2.gen_controlled` | 生成数据集 | 几分钟 |
| 2 | `python -m experiments.rq2.train` | 130 次训练（可断点续跑） | CPU 数小时 / GPU 更快 |
| 3 | `python -m experiments.rq2.run_2a` | 汇总指标、出表出图 | 几分钟到几十分钟 |
| 4 | `python -m experiments.rq2.run_e_diag` | E1–E4 结构诊断 | 几十分钟 |

规则：**一条跑完（回到提示符）再敲下一条**，顺序不能乱。
懒人写法（自动按顺序执行）：

```powershell
python -m experiments.rq2.gen_controlled; python -m experiments.rq2.train; python -m experiments.rq2.run_2a; python -m experiments.rq2.run_e_diag
```

## 2. 先冒烟，再全量

第一次一定先加 `--smoke` 跑一轮小规模流程（总共约 10 分钟），确认四步都能走通：

```powershell
python -m experiments.rq2.gen_controlled --smoke
python -m experiments.rq2.train --smoke
python -m experiments.rq2.run_2a --smoke
python -m experiments.rq2.run_e_diag --smoke
```

冒烟的输出在 `data/rq2_controlled_smoke/` 和 `results/rq2_smoke/`，和全量完全隔离，
确认没问题后可以整目录删掉。之后跑不带 `--smoke` 的全量四条。

## 3. 每步跑完应该看到什么

**第 1 步 gen_controlled** → `data/rq2_controlled/`：
- `pool.jsonl`（主数据池，约 21 万行，几百 MB）
- `test_S0.jsonl` … `test_G6.jsonl`（7 个测试集）
- `splits.json`、`meta.json`（划分清单和元信息）
- 屏幕上每个划分打印 `train 100000 / val 5000 / test 5000`

**第 2 步 train** → `results/rq2/`：
- `runs/*.json`（每次训练一个结果文件，全量共 130 个）
- `models/*.pt`（对应权重）
- 屏幕上每完成一次打印 `[k/130] 运行名: val KL …`
- **中断了怎么办**：直接重新敲同一条命令，开头会打印
  `130 runs, N already done, M to go`，自动接着跑。

**第 3 步 run_2a** → `results/rq2/`：
- `main_table.csv`（主表：每划分×每模型的 KL / JSD / top-1）
- `gap_by_split.png`（简单 vs 非线性差距图，含 S0 主模型参照条）
- `data_size_curve.png`（数据量曲线）
- `diagnostics.csv`（c_L⊙o 权重范数、零人格对照差距、消融差值）

**第 4 步 run_e_diag** → `results/rq2/e_diag/`：
- `e1_overlay_simple.png`、`e1_overlay_nonlinear.png`（教师/学生曲线叠加，
  右下角 N 温度熵曲线就是预注册检验）
- `e1_n_entropy.csv`、`e2_correlation.csv`、`e3_e4_traj_stats.csv`、
  `e3_visit_entropy.png`

四步全部跑完后告诉 Claude，把 `results/rq2/` 交给它分析。

## 4. 在服务器 / GPU 上跑（可选）

1. 把代码弄上去：`git clone <你的私有仓库>` 或直接把 `code/` 打包上传。
2. 按第 0 节装依赖（GPU 机器装 cu121 版 torch）。
3. 训练命令默认自动检测 GPU（`--device auto`）；强制指定用
   `python -m experiments.rq2.train --device cuda`。
4. 断开 SSH 也继续跑：

   ```bash
   nohup python -m experiments.rq2.train > train.log 2>&1 &
   tail -f train.log        # 随时看进度
   ```

5. 跑完把 `results/rq2/` 和 `data/rq2_controlled/` 整目录拷回本地
   （scp / 网盘均可），Claude 在本地读文件分析。

技术说明：GPU 上自动用 float32 训练（更快），权重保存时统一转回 float64，
第 3、4 步在哪台机器上跑结果都一样。

## 5. 常见问题

- **`dataset missing: … - run gen_controlled first`**：第 2 步在第 1 步之前跑了，
  或 `--smoke` 和全量混用了。冒烟四条全带 `--smoke`，全量四条全不带。
- **第 2 步某一次训练报错退出**：重新敲同一条命令即可（已完成的会跳过）。
  反复在同一个运行上报错就把屏幕输出发给 Claude。
- **只想重跑某几个运行**：删掉 `results/rq2/runs/` 里对应的 `.json`（和
  `models/` 里同名 `.pt`），再敲训练命令；或用
  `python -m experiments.rq2.train --only S0__simple` 只跑指定前缀。
- **第 4 步提示缺 `data/rq1_cases/cases.json`**：先跑一次
  `python -m experiments.rq1.gen_cases`。
- **内存**：第 2 步会把主数据池整个读进内存，需要约 3–5 GB 空闲内存。
