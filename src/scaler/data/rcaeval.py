from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from scaler.utils.paths import resolve_data_root


@dataclass
class RCACase:
    case_id: str
    service: str
    fault_type: str
    stage: str
    system: str
    root_cause_service: str
    fault_text: str
    metrics: Optional[np.ndarray] = None
    logs: Optional[np.ndarray] = None
    traces: Optional[np.ndarray] = None
    complexity_score: float = 0.0


class RCAEvalDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path | None = None,
        stages: Sequence[str] = ("RE1", "RE2", "RE3"),
        include_modalities: Sequence[str] = ("metrics", "logs", "traces"),
        systems: Sequence[str] | None = None,
        max_cases_per_system: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.data_root = resolve_data_root(str(data_root) if data_root else None)
        self.stages = list(stages)
        self.include_modalities = list(include_modalities)
        self.systems = {name.upper() for name in systems} if systems else None
        self.max_cases_per_system = max_cases_per_system
        self.normalize = normalize
        self.service_encoder = LabelEncoder()
        self.fault_encoder = LabelEncoder()
        self.metrics_scaler = StandardScaler()
        self.logs_scaler = StandardScaler()
        self.traces_scaler = StandardScaler()
        self.cases: List[RCACase] = []
        self.input_dims: Dict[str, int] = {"metrics": 1, "logs": 1, "traces": 1}
        self._load_cases()
        self._fit_label_encoders()
        if self.normalize:
            self._normalize_modalities()

    def _load_cases(self) -> None:
        systems_seen = {}
        for stage in self.stages:
            stage_root = self.data_root / stage
            if not stage_root.exists():
                continue
            for system_root in sorted(stage_root.iterdir()):
                system_name = system_root.name.upper()
                if self.systems and system_name not in self.systems:
                    continue
                if not system_root.is_dir():
                    continue
                systems_seen.setdefault(system_name, 0)
                for service_fault_dir in sorted(system_root.iterdir()):
                    if not service_fault_dir.is_dir():
                        continue
                    parsed = self._parse_service_fault(service_fault_dir.name)
                    if parsed is None:
                        continue
                    service, fault_type = parsed
                    for exp_dir in sorted(service_fault_dir.iterdir()):
                        if not exp_dir.is_dir():
                            continue
                        if self.max_cases_per_system is not None and systems_seen[system_name] >= self.max_cases_per_system:
                            break
                        case = RCACase(
                            case_id=f"{stage}_{system_name}_{service}_{fault_type}_{exp_dir.name}",
                            service=service,
                            fault_type=fault_type,
                            stage=stage,
                            system=system_name,
                            root_cause_service=service,
                            fault_text=self._build_fault_text(system_name, fault_type),
                        )
                        self._load_modalities(case, exp_dir)
                        if any(getattr(case, name) is not None for name in ("metrics", "logs", "traces")):
                            case.complexity_score = self._compute_complexity(case)
                            self.cases.append(case)
                            systems_seen[system_name] += 1
        if not self.cases:
            raise RuntimeError(f"No valid RCAEval cases found under {self.data_root}.")

    @staticmethod
    def _parse_service_fault(directory_name: str) -> tuple[str, str] | None:
        parts = directory_name.split("_")
        if len(parts) < 2:
            return None
        known_faults = {"cpu", "mem", "delay", "loss", "disk", "socket", "f1", "f2", "f3", "f4", "f5"}
        for index in range(len(parts) - 1, 0, -1):
            if parts[index] in known_faults:
                return "_".join(parts[:index]), parts[index]
        return "_".join(parts[:-1]), parts[-1]

    def _load_modalities(self, case: RCACase, exp_dir: Path) -> None:
        if "metrics" in self.include_modalities:
            metrics_file = exp_dir / ("data.csv" if case.stage == "RE1" else "metrics.csv")
            if metrics_file.exists():
                case.metrics = self._sanitize_array(self._load_numeric_csv(metrics_file))

        if "logs" in self.include_modalities and case.stage in {"RE2", "RE3"}:
            logs_file = exp_dir / ("logts.csv" if case.stage == "RE2" else "logs.csv")
            if logs_file.exists():
                case.logs = self._sanitize_array(self._load_mixed_csv(logs_file))

        if "traces" in self.include_modalities and case.stage in {"RE2", "RE3"}:
            if case.stage == "RE2":
                lat_file = exp_dir / "tracets_lat.csv"
                err_file = exp_dir / "tracets_err.csv"
                if lat_file.exists():
                    case.traces = self._sanitize_array(self._load_re2_traces(lat_file, err_file))
            else:
                traces_file = exp_dir / "traces.csv"
                if traces_file.exists():
                    case.traces = self._sanitize_array(self._load_mixed_csv(traces_file))

    @staticmethod
    def _build_fault_text(system: str, fault_type: str) -> str:
        fault_descriptions = {
            "cpu": "CPU overload causing high utilization and potential throttling of service requests",
            "mem": "memory pressure leading to out-of-memory errors and increased garbage collection latency",
            "delay": "network latency injection causing delayed response times and timeout errors between services",
            "loss": "packet loss in the network causing intermittent connection failures and retry storms",
            "disk": "disk I/O bottleneck causing slow read and write operations on persistent storage",
            "socket": "socket connection errors leading to failed inter-service communication and dropped requests",
        }
        desc = fault_descriptions.get(fault_type, f"fault type {fault_type} affecting system performance")
        return f"{system}: {desc}"

    @staticmethod
    def _sanitize_array(array: np.ndarray) -> Optional[np.ndarray]:
        if array.ndim != 2:
            return array
        if array.shape[0] == 0 or array.shape[1] == 0:
            return None
        return array

    @staticmethod
    def _load_numeric_csv(path: Path) -> np.ndarray:
        frame = pd.read_csv(path).fillna(0)
        if "time" in frame.columns:
            frame = frame.drop(columns=["time"])
        return frame.values.astype(np.float32)

    @staticmethod
    def _load_mixed_csv(path: Path, max_rows: int = 1000) -> np.ndarray:
        frame = pd.read_csv(path, nrows=max_rows).fillna(0)
        if "time" in frame.columns:
            frame = frame.drop(columns=["time"])
        features: List[np.ndarray] = []
        for column in frame.columns:
            series = frame[column]
            if is_numeric_dtype(series.dtype):
                features.append(series.astype(np.float32).to_numpy())
            else:
                features.append(series.astype(str).str.len().astype(np.float32).to_numpy())
        if not features:
            return np.zeros((1, 1), dtype=np.float32)
        return np.column_stack(features).astype(np.float32)

    @classmethod
    def _load_re2_traces(cls, lat_file: Path, err_file: Path) -> np.ndarray:
        lat = cls._load_numeric_csv(lat_file)
        if err_file.exists():
            err = cls._load_numeric_csv(err_file)
            if err.shape[0] == lat.shape[0]:
                return np.concatenate([lat, err], axis=1).astype(np.float32)
        return lat.astype(np.float32)

    @staticmethod
    def _compute_complexity(case: RCACase) -> float:
        fault_weight = {
            "cpu": 0.25,
            "mem": 0.25,
            "delay": 0.5,
            "loss": 0.5,
            "disk": 0.65,
            "socket": 0.65,
        }.get(case.fault_type, 0.75)
        modality_count = sum(getattr(case, name) is not None for name in ("metrics", "logs", "traces"))
        modal_weight = modality_count / 3.0
        temporal_lengths = [arr.shape[0] for arr in (case.metrics, case.logs, case.traces) if arr is not None]
        temporal_weight = min((max(temporal_lengths) if temporal_lengths else 1) / 1000.0, 1.0)
        scope_weight = min(max((arr.shape[1] for arr in (case.metrics, case.logs, case.traces) if arr is not None), default=1) / 100.0, 1.0)
        return float((fault_weight + modal_weight + temporal_weight + scope_weight) / 4.0)

    def _fit_label_encoders(self) -> None:
        self.service_encoder.fit([case.root_cause_service for case in self.cases])
        self.fault_encoder.fit([case.fault_type for case in self.cases])

    def _normalize_modalities(self) -> None:
        for name, scaler in (("metrics", self.metrics_scaler), ("logs", self.logs_scaler), ("traces", self.traces_scaler)):
            arrays = [getattr(case, name) for case in self.cases if getattr(case, name) is not None and getattr(case, name).shape[0] > 0]
            if not arrays:
                continue
            max_dim = max(array.shape[1] for array in arrays)
            stacked = []
            for case in self.cases:
                array = getattr(case, name)
                if array is None or array.shape[0] == 0:
                    continue
                if array.shape[1] < max_dim:
                    pad = np.zeros((array.shape[0], max_dim - array.shape[1]), dtype=np.float32)
                    array = np.concatenate([array, pad], axis=1)
                    setattr(case, name, array)
                stacked.append(array)
            scaler.fit(np.concatenate(stacked, axis=0))
            for case in self.cases:
                array = getattr(case, name)
                if array is not None and array.shape[0] > 0:
                    normalized = scaler.transform(array).astype(np.float32)
                    setattr(case, name, normalized)
                    self.input_dims[name] = normalized.shape[1]

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> Dict[str, object]:
        case = self.cases[index]
        data: Dict[str, torch.Tensor] = {}
        for name in ("metrics", "logs", "traces"):
            array = getattr(case, name)
            if array is not None:
                data[name] = torch.as_tensor(array, dtype=torch.float32)
        return {
            "data": data,
            "service_label": int(self.service_encoder.transform([case.root_cause_service])[0]),
            "fault_label": int(self.fault_encoder.transform([case.fault_type])[0]),
            "complexity_score": case.complexity_score,
            "case_id": case.case_id,
            "stage": case.stage,
            "fault_text": case.fault_text,
            "fault_type": case.fault_type,
            "service_name": case.root_cause_service,
        }


def _pad_sequences(sequences: List[Optional[torch.Tensor]], max_len: int = 1000) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    valid = [seq for seq in sequences if seq is not None]
    if not valid:
        return None, None, None
    seq_len = min(max(seq.shape[0] for seq in valid), max_len)
    feature_dim = valid[0].shape[1]
    padded, masks, lengths = [], [], []
    for seq in sequences:
        if seq is None:
            padded.append(torch.zeros(seq_len, feature_dim))
            masks.append(0)
            lengths.append(0)
            continue
        current = seq[-seq_len:]
        if current.shape[0] < seq_len:
            pad = torch.zeros(seq_len - current.shape[0], feature_dim)
            current = torch.cat([pad, current], dim=0)
        padded.append(current)
        masks.append(1)
        lengths.append(min(int(seq.shape[0]), seq_len))
    return torch.stack(padded), torch.tensor(masks, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def collate_rca_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "service_labels": torch.tensor([item["service_label"] for item in batch], dtype=torch.long),
        "fault_labels": torch.tensor([item["fault_label"] for item in batch], dtype=torch.long),
        "complexity_scores": torch.tensor([item["complexity_score"] for item in batch], dtype=torch.float32),
        "case_ids": [item["case_id"] for item in batch],
        "stages": [item["stage"] for item in batch],
        "fault_texts": [item["fault_text"] for item in batch],
        "fault_types": [item["fault_type"] for item in batch],
        "service_names": [item["service_name"] for item in batch],
    }
    for name in ("metrics", "logs", "traces"):
        padded, mask, lengths = _pad_sequences([item["data"].get(name) for item in batch])
        if padded is not None:
            result[name] = padded
            result[f"{name}_mask"] = mask
            result[f"{name}_lengths"] = lengths
    return result


def create_splits(
    dataset: RCAEvalDataset,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    stratify_by: str = "fault_type",
) -> tuple[List[int], List[int], List[int]]:
    indices = np.arange(len(dataset))
    total = len(indices)
    train_end = max(1, int(total * train_fraction)) if total >= 1 else 0
    val_size = max(1, int(total * val_fraction)) if total >= 3 else max(0, total - train_end - 1)
    if stratify_by not in {"fault_type", "service"}:
        raise ValueError(f"Unsupported split stratification: {stratify_by}")
    attribute = "fault_type" if stratify_by == "fault_type" else "root_cause_service"
    strata = np.asarray([getattr(case, attribute) for case in dataset.cases])

    if strata.size and min(Counter(strata).values()) >= 6:
        train_idx, remainder_idx = train_test_split(
            indices,
            train_size=train_end,
            random_state=seed,
            stratify=strata,
        )
        remainder_strata = strata[remainder_idx]
        val_idx, test_idx = train_test_split(
            remainder_idx,
            train_size=val_size,
            random_state=seed,
            stratify=remainder_strata,
        )
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_end = min(total - 1, train_end + val_size) if total >= 2 else train_end
    return indices[:train_end].tolist(), indices[train_end:val_end].tolist(), indices[val_end:].tolist()
