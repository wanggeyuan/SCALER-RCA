from pathlib import Path

import pandas as pd

from scaler.data.rcaeval import RCAEvalDataset


def _write_case(case_dir: Path, metrics_name: str, include_logs: bool, include_traces: bool) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": [1, 2], "a": [0.1, 0.2], "b": [0.3, 0.4]}).to_csv(case_dir / metrics_name, index=False)
    if include_logs:
        pd.DataFrame({"time": [1, 2], "msg": ["error", "warn"], "value": [1, 2]}).to_csv(case_dir / "logts.csv", index=False)
    if include_traces:
        pd.DataFrame({"time": [1, 2], "lat": [3, 4]}).to_csv(case_dir / "tracets_lat.csv", index=False)
        pd.DataFrame({"time": [1, 2], "err": [0, 1]}).to_csv(case_dir / "tracets_err.csv", index=False)


def test_dataset_loads_minimal_rcaeval_tree(tmp_path: Path):
    _write_case(tmp_path / "RE1" / "RE1-SS" / "cart_cpu" / "0", "data.csv", False, False)
    _write_case(tmp_path / "RE2" / "RE2-SS" / "cart_delay" / "0", "metrics.csv", True, True)
    _write_case(tmp_path / "RE2" / "RE2-OB" / "payment_mem" / "0", "metrics.csv", True, True)
    _write_case(tmp_path / "RE2" / "RE2-TT" / "catalog_loss" / "0", "metrics.csv", True, True)

    dataset = RCAEvalDataset(data_root=tmp_path, stages=["RE1", "RE2"], max_cases_per_system=1)
    assert len(dataset) >= 2
    sample = dataset[0]
    assert "service_label" in sample
    assert "cpu" in sample["fault_text"].lower() or "delay" in sample["fault_text"].lower() or "mem" in sample["fault_text"].lower() or "loss" in sample["fault_text"].lower()


def test_dataset_ignores_empty_modalities(tmp_path: Path):
    case_dir = tmp_path / "RE2" / "RE2-SS" / "orders_f1" / "0"
    case_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time": [1, 2], "a": [0.1, 0.2]}).to_csv(case_dir / "metrics.csv", index=False)
    pd.DataFrame({"time": [], "msg": [], "value": []}).to_csv(case_dir / "logts.csv", index=False)
    pd.DataFrame({"time": [], "lat": []}).to_csv(case_dir / "tracets_lat.csv", index=False)
    pd.DataFrame({"time": [], "err": []}).to_csv(case_dir / "tracets_err.csv", index=False)

    dataset = RCAEvalDataset(data_root=tmp_path, stages=["RE2"], max_cases_per_system=1)
    sample = dataset[0]

    assert "metrics" in sample["data"]
    assert "logs" not in sample["data"]
    assert "traces" not in sample["data"]


def test_dataset_ignores_numeric_suffix_on_re3_fault_directory(tmp_path: Path):
    (tmp_path / "RE2" / "RE2-TT").mkdir(parents=True)
    _write_case(tmp_path / "RE3" / "RE3-TT" / "ts-route-service_f3_1" / "0", "metrics.csv", False, False)

    dataset = RCAEvalDataset(data_root=tmp_path, stages=["RE3"])

    assert dataset.cases[0].root_cause_service == "ts-route-service"
    assert dataset.cases[0].fault_type == "f3"
