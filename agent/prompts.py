# -*- coding: utf-8 -*-
"""六环节科研提示词模板。所有模板集中管理，保证每轮自循环的输入可审查。"""

SYSTEM = """你是一名计算化学方向的研究助理。你的任务是分析机器学习模型的交叉验证结果，
提出**可执行、可验证**的改进假设，并能明确指出该假设预期改善哪一类误差。

约束：
1. 只能修改我给你的 JSON 配置中的数值，不要引入新字段；
2. 每次最多调整 2 个超参数，避免多变量同时变动导致无法归因；
3. 必须给出量化的预期与失败判据；
4. 只输出 JSON，不要任何解释性文字。"""


def literature_prompt(topic: str) -> str:
    return (f"针对研究问题「{topic}」，列出 3-5 个关键方法脉络、代表性工作及其报告的指标，"
            "按时间排序，并指出当前工作的空白点。输出为表格。")


def hypothesis_prompt(metrics: dict, top_features: list[str]) -> str:
    return (f"当前 5 折交叉验证结果为：{metrics}。\n"
            f"模型最依赖的描述符为：{top_features}。\n"
            "请判断误差的主要来源（描述符覆盖不足 / 模型容量 / 分布外样本 / 标签噪声），"
            "给出 1 个最可能有效的改进假设，并说明如何判定改进是否成立。")


def loop_prompt(metrics: dict, cfg: dict, history: list[dict]) -> str:
    return f"""当前配置：{cfg}
当前指标（5 折 OOF）：{metrics}
历史轮次：{history}

请输出 JSON：
{{"hypothesis": "一句话假设",
  "changes": {{"rf.n_estimators": 200}},  // 只允许修改 configs/base.json 中的数值字段
  "expect": "预期 MAE 下降幅度及判据",
  "rollback_if": "什么情况下应判定该轮无效并回滚"}}"""


def analyse_prompt(slices: str) -> str:
    return ("以下是按分子大小和环数分组的误差统计，请判断模型在哪个子群上系统性偏差最大，"
            "并从化学角度给出可能原因（如共轭体系、杂原子、环张力等）。\n" + slices)


def conclusion_prompt(metrics: dict, literature: str) -> str:
    return (f"给定本研究结果 {metrics} 与文献基准 {literature}，"
            "请写出结论与局限：明确本方法在什么范围内可用、在什么范围内不适用，"
            "不夸大，不掩盖差距。")
