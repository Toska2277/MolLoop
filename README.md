# MolLoop · QM9 分子能隙预测的可复现 AI4S 流程

面向第五届「未来杯」技术大赛**赛题二（科学智能 AI4S 流程与应用创新）**的科研工程仓库。

研究对象：QM9 公开数据集的 13.4 万个小分子，目标量是 **HOMO–LUMO 能隙（gap, eV）**。
整条流程按「文献 → 假设 → 数据 → 计算 → 验证 → 结论」六个环节推进，每一步都留痕、可复现、可核验。

---

## 一、当前真实结果（本机实测，5 折交叉验证，seed=42）

数据规模：原始 133,885 条 → 清洗后 **133,802 条**（剔除 83 条空值/重复/异常），1–9 个重原子（C/H/N/O/F）。

| 模型 | MAE (eV) | RMSE | R² | MedAE | 说明 |
|---|---|---|---|---|---|
| 随机森林 RF（基线） | **0.1622** | 0.2495 | 0.9627 | 0.1128 | 222 维 RDKit 2D 描述符，200 棵，max_features=0.5 |
| XGBoost（对照） | 0.2057 | 0.2885 | 0.9502 | 0.1584 | 同特征 |
| 堆叠集成（RF+ET+XGB+岭） | 0.1555 | — | 0.9646 | — | 岭回归元学习器 |
| 混合特征 XGB（指纹+3D） | 0.1669 | 0.2425 | 0.9648 | — | Morgan(r=2,2048)+MACCS(167)+库仑矩阵特征 31 维 |
| **融合（0.7 堆叠 + 0.3 混合）** | **0.1522** | 0.2342 | 0.9672 | 0.1077 | **当前最优，演示页即用此模型** |

> 页面 `demo/molloop_工具版.html` 的口径：能隙取融合预测（MAE 0.1522、R² 0.9672），90% 区间用融合残差的 split conformal（±0.3257，实测覆盖 0.9000）；其余 5 个性质仍为 RF 多目标预测。页面上所有数字均为 5 折折外预测，不是训练集拟合值。

对照实验（`eval/extra_report.md`）：

| 实验 | MAE (eV) | 结论 |
|---|---|---|
| 骨架划分（Bemis-Murcko，15,989 骨架） | 0.2339 | 换骨架后误差涨 44%，说明分子级随机划分偏乐观，这是必须如实报告的 |
| 仅元素组成（5 维） | 0.9592 | 描述符信息量是核心 |
| 去掉 85 个 fr_* 官能团片段 | 0.1656 | 片段特征贡献约 0.0034 eV |
| 均值基线 / 岭回归 | 1.0788 / 0.3492 | 基线参照 |

不确定度与适用域（`eval/uncertainty.json`、`eval/uncertainty_blend.json`）：

- 树间标准差与绝对误差的 Spearman 相关系数 **0.445**，五个 std 分箱的 MAE 从 0.0718 单调升到 0.2919（RF 树间分歧，页面标注为「模型分歧度」）；
- RF 的 conformal 90% 区间 ±0.3519（实测覆盖 0.9000），95% 区间 ±0.4597（覆盖 0.9499）；
- **融合模型的 conformal（页面采用）**：90% 区间 **±0.3257**（覆盖 0.9000），95% ±0.4237（覆盖 0.9500）；
- 适用域阈值 **29.10**（PCA20 + kNN），域内样本占 94.99%，域内 MAE 0.1592、域外 0.2204。

多目标与物理一致性（`eval/metrics_multi.json`）：gap 0.1622 / homo 0.1237 / lumo 0.1321 / mu 0.5347 / alpha 0.0923 / cv 0.2398。
直接预测 gap（0.1622）优于 lumo−homo 相减（0.1759），标签自身残差仅 0.0007 eV。

逆向筛选（`eval/screen_meta.json`）：在 5.5–6.5 eV 窗口内，39,855 条候选 → 适用域内 38,097 → 去同骨架后 **40 个候选 / 40 个骨架**，中位绝对误差 0.021 eV。

演示页 `demo/molloop_工具版.html`（单文件、离线可用、无需服务端）已实现：全库 133,802 条单框检索（中文名 / 常用名 / 分子式 / 结构式片段）、库外分子同模型现算、6 性质切换、90% 区间与适用域徽标、结构特征中文描述、系统命名、**结构相似度检索（Morgan 半径 2、256 位指纹，Tanimoto Top-5，随选中分子实时计算）**、逆向筛选与 CSV 导出。页面内嵌全库指纹（约 4.3 MB 二进制），因此文件体积 11.9 MB，双击即可打开。

---

## 二、目录结构

```
MolLoop/
├─ data/            prepare 脚本、清洗后 parquet、三维构型分片（大文件不入库）
├─ features/        222 维描述符表（大文件不入库）
├─ train/           fit_final / multi_target / stack / hybrid / blend / screen / schnet / make_geom
├─ eval/            评估脚本 + 全部指标 JSON + 四份报告 markdown
├─ demo/            单文件离线演示页 build_demo3.py 与产物 molloop_工具版.html
├─ tools/           predict_one.py（任意分子现算）、make_repo.py
├─ agent/           LLM 驱动的自动实验循环（loop / prompts / trace / report_writer）
├─ figures/         散点、残差、特征重要性、按原子数误差图
├─ configs/         base.json（超参）、llm.json（密钥，gitignore）
├─ run_all.bat / run_agent.bat / quick_run.bat
└─ README.md
```

---

## 三、复现步骤

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

python data\prepare.py            # 清洗 + 5 折划分，约 10 s
python features\featurize.py      # 222 维描述符，约 13 min
python train\fit_final.py         # 主模型，5 折 OOF，约 82 s
python train\multi_target.py      # 六个性质，约 20 min
python eval\uncertainty.py        # 不确定度 / conformal / 适用域
python train\screen.py            # 逆向筛选
python train\stack.py             # 堆叠集成
python train\hybrid.py            # 指纹 + 三维构型特征
python train\blend.py             # 融合权重搜索
python demo\build_demo3.py        # 生成单文件演示页
```

任意分子现算：

```powershell
python tools\predict_one.py "CC(=O)Oc1ccccc1C(=O)O"
```

---

## 四、一个必须写进文档的口径问题：单位

QM9 原始 CSV 的 `gap` 列单位是 **Hartree**，直接当 eV 会得到「平均 0.25 eV 的能隙」这种明显违背物理的结果。
本流程在 `data/prepare.py` 中统一换算为 eV（×27.211386），换算后均值 6.8338 eV、标准差 1.2927 eV、范围 0.6694–16.9282 eV。
用苯交叉验证：0.2503 Ha → 6.81 eV，与 B3LYP/6-31G(2df,p) 文献值一致。

---

## 五、AI 使用方式（对应评分项「AI 使用充分性」）

1. **LLM 驱动的自动实验循环**（`agent/loop.py`）：模型读取历史指标与失败原因，自主提出下一组超参，改 `configs/base.json` 后触发重训；指标变好才采纳，否则从 `logs/before_agent_backup/` 回滚。
   实跑两轮：`max_features 0.3→0.5` 使 MAE 0.1661→0.1629；`n_estimators 100→200` 使 0.1629→0.1622（5 折全部下降）。
   全过程写入 `logs/trace_*.jsonl`，配置快照存于 `logs/config_snapshots/`。
2. **LLM 结构化命名**（`demo/gen_names.py`）：为展示分子生成中文系统命名，要求模型同时回填 SMILES，用 InChIKey 与分子式双重校验，对不上直接丢弃、不入页面。
3. **AI 自动结论**（`agent/report_writer.py`）：把指标表交给模型生成 `eval/ai_conclusion.md` 初稿，再由人工核对数字。
4. **失败实验如实归档**（`eval/gnn_route_report.md`）：GNN 路线受算力与数据源限制未达预期，完整记录实测带宽、逐轮指标、放弃依据与后续条件。

---

## 六、留痕与可复现

- `logs/trace_*.jsonl`：每步一行，记录时间、阶段、输入、决策、配置快照、耗时；
- `logs/config_snapshots/`：每轮改动前快照，可回滚；
- 所有随机过程固定 `seed=42`；指标均由脚本重新计算并落盘，不手工填数。

---

## 七、已知边界（如实披露）

1. **适用域限制**：模型只见过 ≤9 个重原子的分子。阿司匹林（21 重原子）会被标注「超出适用域」，且预测有饱和倾向（萘 4.96、蒽 4.96、芘 4.87，文献约 4.3 / 3.4 / 3.0），说明树模型不外推。
2. **命名覆盖**：演示页 117 个精选分子全部有中文系统命名；库内其余约 13 万条只显示结构特征描述，不逐条命名（单条 LLM 调用 8–14 秒，全库需数百小时）。
3. **GNN 未落地**：本机无法下载 2.6 GB CUDA 版 PyTorch（实测 0.02–1 MB/s，多镜像均无对应轮子），QM9 原始 DFT 几何的公开源全部不可达（Figshare 202/0 字节、deepchem 403、Zenodo DNS 失败）；CPU 版 SchNet 训练 201 s/轮，第 7 轮验证 MAE 0.2628 未反超现有模型而终止。
4. **误差最大子群**：≤5 个原子（178 条）MAE 0.2868 eV，样本占比 0.14%，是后续改进的靶点。

---

## 八、团队分工

| 角色 | 分工 |
|---|---|
| A（队长） | 科研流程设计、指标口径、方案书统稿 |
| B | 数据清洗、特征工程、模型训练与评估 |
| C | Agent 工程、演示系统、复现脚本 |
| D | PPT、演示视频、答辩材料 |

## 九、引用

数据：QM9（Ramakrishnan et al., *Scientific Data*, 2014）。代码仅用于参赛与教学。
