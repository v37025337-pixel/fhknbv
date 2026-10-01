
"""
v0.17_data_world.py

Read-only structured-data world model for CSV/TSV/JSON and already structured
Python dict/list payloads.

It infers a conservative schema:
- column / field names
- scalar types
- nullability
- row count
- simple uniqueness ratios / key candidates

It does not mutate source data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import hashlib
import json
import math


@dataclass(slots=True)
class FieldSchema:
    name: str
    inferred_type: str
    nullable: bool
    unique_ratio: float


@dataclass(slots=True)
class Dataset:
    dataset_id: str
    source: str
    format: str
    row_count: int
    fields: List[FieldSchema]
    raw_sha256: Optional[str]
    key_candidates: List[str] = field(default_factory=list)


class DataWorldModel:
    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}

    @staticmethod
    def _scalar_type(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, (dict, list)):
            return "object" if isinstance(value, dict) else "array"
        s = str(value).strip()
        if s == "":
            return "null"
        low = s.lower()
        if low in {"true", "false"}:
            return "bool"
        try:
            int(s)
            return "int"
        except Exception:
            pass
        try:
            float(s)
            return "float"
        except Exception:
            return "string"

    @staticmethod
    def _combine(types: List[str]) -> str:
        t = {x for x in types if x != "null"}
        if not t:
            return "null"
        if t <= {"int"}:
            return "int"
        if t <= {"int", "float"}:
            return "float"
        if len(t) == 1:
            return next(iter(t))
        if t <= {"object"}:
            return "object"
        if t <= {"array"}:
            return "array"
        return "mixed"

    @staticmethod
    def _sha_bytes(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _dataset_id(source: str, digest: str) -> str:
        return "dataset:" + hashlib.sha256(
            (source + "|" + digest).encode("utf-8")
        ).hexdigest()[:20]

    def _infer_records(
        self,
        records: List[Dict[str, Any]],
        *,
        source: str,
        fmt: str,
        raw_sha256: Optional[str],
    ) -> Dataset:
        names = []
        seen = set()
        for row in records:
            for k in row.keys():
                k = str(k)
                if k not in seen:
                    seen.add(k)
                    names.append(k)

        fields = []
        keys = []
        n = len(records)

        for name in names:
            values = [row.get(name) for row in records]
            types = [self._scalar_type(v) for v in values]
            nullable = any(t == "null" for t in types)

            normalized = [
                json.dumps(v, sort_keys=True, ensure_ascii=False)
                if isinstance(v, (dict, list))
                else repr(v)
                for v in values
                if v is not None and str(v).strip() != ""
            ]
            unique_ratio = (
                len(set(normalized)) / len(normalized)
                if normalized else 0.0
            )
            inferred = self._combine(types)
            fields.append(FieldSchema(name, inferred, nullable, unique_ratio))

            if n >= 2 and not nullable and unique_ratio == 1.0:
                keys.append(name)

        digest = raw_sha256 or hashlib.sha256(
            json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        ds = Dataset(
            dataset_id=self._dataset_id(source, digest),
            source=source,
            format=fmt,
            row_count=n,
            fields=fields,
            raw_sha256=raw_sha256,
            key_candidates=keys,
        )
        self.datasets[ds.dataset_id] = ds
        return ds

    def ingest_delimited(self, path: str, delimiter: str = ",") -> Dataset:
        p = Path(path)
        raw = p.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        records = [dict(row) for row in reader]
        return self._infer_records(
            records,
            source=str(p),
            fmt="csv" if delimiter == "," else "tsv",
            raw_sha256=self._sha_bytes(raw),
        )

    def ingest_json_file(self, path: str) -> Dataset:
        p = Path(path)
        raw = p.read_bytes()
        payload = json.loads(raw.decode("utf-8-sig"))
        return self.ingest_payload(
            payload,
            source=str(p),
            fmt="json",
            raw_sha256=self._sha_bytes(raw),
        )

    def ingest_payload(
        self,
        payload: Any,
        *,
        source: str = "<structured>",
        fmt: str = "structured",
        raw_sha256: Optional[str] = None,
    ) -> Dataset:
        if isinstance(payload, dict):
            records = [payload]
        elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
            records = list(payload)
        elif isinstance(payload, list):
            records = [{"value": x} for x in payload]
        else:
            records = [{"value": payload}]

        return self._infer_records(
            records,
            source=source,
            fmt=fmt,
            raw_sha256=raw_sha256,
        )

    @staticmethod
    def schema_dict(dataset: Dataset) -> dict:
        return {
            "dataset_id": dataset.dataset_id,
            "source": dataset.source,
            "format": dataset.format,
            "row_count": dataset.row_count,
            "fields": [
                {
                    "name": f.name,
                    "type": f.inferred_type,
                    "nullable": f.nullable,
                    "unique_ratio": round(f.unique_ratio, 6),
                }
                for f in dataset.fields
            ],
            "key_candidates": list(dataset.key_candidates),
        }

    def stats(self) -> dict:
        return {
            "datasets": len(self.datasets),
            "rows": sum(d.row_count for d in self.datasets.values()),
            "formats": sorted({d.format for d in self.datasets.values()}),
        }
