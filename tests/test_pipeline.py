# -*- coding: utf-8 -*-
"""冒烟测试：验证数据、特征、指标文件之间的契约关系。"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "qm9_clean.parquet"
FEAT = ROOT / "features" / "features.parquet"


def test_clean_dataset_exists():
    assert CLEAN.exists(), "缺少清洗后的数据集，请先运行 data/prepare.py"
    df = pd.read_parquet(CLEAN)
    assert len(df) > 100000
    assert set(["mol_id", "smiles", "gap", "fold"]).issubset(df.columns)


def test_target_range():
    df = pd.read_parquet(CLEAN)
    assert df["gap"].between(0, 20).all(), "存在超出物理范围的能隙值"
    assert df["fold"].between(0, 4).all()


def test_features_alignment():
    if not FEAT.exists():
        pytest.skip("特征文件尚未生成")
    feat = pd.read_parquet(FEAT)
    clean = pd.read_parquet(CLEAN)
    assert len(feat) <= len(clean)
    assert feat["gap"].notna().all()
    assert feat["n_atoms"].min() >= 1


def test_predictions_length():
    f = ROOT / "train" / "preds_rf.csv"
    if not f.exists():
        pytest.skip("尚未训练")
    p = pd.read_csv(f)
    assert p["y_pred"].notna().all(), "存在未预测到的样本"
    assert len(p) > 100000
