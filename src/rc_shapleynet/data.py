"""Auditable ACN session replay, chronological day selection and load calibration.

These readers compute counts from supplied records. They contain no reconstructed
author manifest or benchmark observations. ACN fields follow the public schema:
https://ev.caltech.edu/dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, date, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from collections import Counter
from typing import Mapping, Callable, Any
from zoneinfo import ZoneInfo
import csv
import json
import math
import numpy as np


def parse_timestamp(value: str | datetime) -> datetime:
    """Accept aware RFC3339/ISO or ACN RFC1123/GMT times and return UTC."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            parsed = parsedate_to_datetime(value.strip())
    else:
        raise ValueError("A timestamp is required")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Naive timestamps are not assigned an invented timezone")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ACNParserConfig:
    timezone: str = "America/Los_Angeles"
    site_label: str = "caltech"
    accepted_site_ids: tuple[str, ...] = ("0002", "caltech")
    cutoff_local_date: str = "2021-09-13"
    train_end_local_date: str = "2020-07-01"
    validation_end_local_date: str = "2021-02-01"
    min_duration_seconds: float = 0.0
    max_duration_seconds: float | None = None
    max_demand_kwh: float | None = None
    cap_declared_by_disconnect: bool = False

    def __post_init__(self):
        ZoneInfo(self.timezone)
        for value in (self.min_duration_seconds, self.max_duration_seconds, self.max_demand_kwh):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Cleaning thresholds must be finite and nonnegative")
        for value in (self.cutoff_local_date, self.train_end_local_date, self.validation_end_local_date):
            date.fromisoformat(value)


@dataclass(frozen=True)
class ACNSession:
    session_id: str
    location_id: str
    arrival: datetime
    disconnect: datetime
    requested_departure: datetime | None
    departure: datetime
    deadline_source: str
    demand_kwh: float
    delivered_kwh: float | None
    source_proxy: bool
    local_date: date
    split: str


@dataclass(frozen=True)
class ACNParseResult:
    sessions: tuple[ACNSession, ...]
    raw_count: int
    rejected: Mapping[str, int]
    config: ACNParserConfig


def _energy(value) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not energy")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Energy must be finite and nonnegative")
    return result


def load_acn_export(source: str | Path | list | dict,
                    config: ACNParserConfig = ACNParserConfig()) -> ACNParseResult:
    """Read a JSON list or one exported ``_items`` collection.

    The latest valid explicit-energy declaration is preferred. Its departure is
    used when valid; otherwise the latest valid declared departure is used.
    Delivered energy supplies demand only if no valid explicit request exists.
    Duplicate IDs retain the first valid record in the supplied export order.
    Thresholds are caller configuration, not an asserted author cleaning recipe.
    """
    ZoneInfo(config.timezone)
    if config.min_duration_seconds < 0 or (config.max_duration_seconds is not None
            and config.max_duration_seconds <= config.min_duration_seconds):
        raise ValueError("Invalid explicit duration thresholds")
    if not config.train_end_local_date < config.validation_end_local_date:
        raise ValueError("Chronological split boundaries must increase")
    raw = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else source
    records = raw.get("_items") if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError("ACN input must be a JSON list or an object containing _items")
    rejected: Counter[str] = Counter()
    sessions, seen = [], set()
    for record in records:
        if not isinstance(record, dict) or not record:
            rejected["empty_or_nonobject"] += 1
            continue
        identifier = str(record.get("sessionID") or record.get("_id") or "").strip()
        if not identifier:
            rejected["missing_session_id"] += 1
            continue
        if identifier in seen:
            rejected["duplicate_session"] += 1
            continue
        if (record.get("siteID") is not None and config.accepted_site_ids
                and str(record["siteID"]) not in config.accepted_site_ids):
            rejected["site_not_selected"] += 1
            continue
        try:
            arrival = parse_timestamp(record.get("connectionTime"))
            disconnect = parse_timestamp(record.get("disconnectTime"))
        except (ValueError, TypeError, OverflowError):
            rejected["invalid_timestamp"] += 1
            continue
        duration = (disconnect-arrival).total_seconds()
        if duration <= config.min_duration_seconds or (config.max_duration_seconds is not None
                                                       and duration > config.max_duration_seconds):
            rejected["invalid_duration"] += 1
            continue
        local_date = arrival.astimezone(ZoneInfo(config.timezone)).date()
        if local_date > date.fromisoformat(config.cutoff_local_date):
            rejected["after_cutoff"] += 1
            continue
        delivered = None
        if record.get("kWhDelivered") is not None:
            try:
                delivered = _energy(record["kWhDelivered"])
            except (ValueError, TypeError):
                rejected["invalid_delivered_energy"] += 1
                continue
        inputs = record.get("userInputs") or []
        if not isinstance(inputs, list):
            inputs = []
        ordered = []
        for index, item in enumerate(inputs):
            if not isinstance(item, dict):
                continue
            try:
                modified = parse_timestamp(item.get("modifiedAt"))
            except (ValueError, TypeError, OverflowError):
                modified = datetime.min.replace(tzinfo=timezone.utc)
            ordered.append((modified, index, item))
        ordered.sort(key=lambda row: (row[0], row[1]), reverse=True)
        demand, energy_input = None, None
        for _, _, item in ordered:
            try:
                demand = _energy(item.get("kWhRequested"))
                energy_input = item
                break
            except (ValueError, TypeError):
                continue
        proxy = demand is None
        if proxy:
            demand = delivered
        if demand is None or (config.max_demand_kwh is not None and demand > config.max_demand_kwh):
            rejected["missing_or_excessive_demand"] += 1
            continue
        declaration = None
        departure_inputs = ([energy_input] if energy_input else []) + [v[2] for v in ordered]
        for item in departure_inputs:
            try:
                candidate = parse_timestamp(item.get("requestedDeparture"))
                if candidate > arrival:
                    declaration = candidate
                    break
            except (ValueError, TypeError, OverflowError):
                continue
        departure = declaration or disconnect
        deadline_source = "declared" if declaration else "disconnect_proxy"
        if config.cap_declared_by_disconnect:
            departure = min(departure, disconnect)
            deadline_source = "min_declared_disconnect" if declaration else "disconnect_proxy"
        location = str(record.get("spaceID") or record.get("stationID") or "").strip()
        if not location:
            rejected["missing_location_id"] += 1
            continue
        split = ("train" if local_date < date.fromisoformat(config.train_end_local_date)
                 else "validation" if local_date < date.fromisoformat(config.validation_end_local_date)
                 else "test")
        sessions.append(ACNSession(identifier, location, arrival, disconnect, declaration,
                                   departure, deadline_source, demand, delivered, proxy, local_date, split))
        seen.add(identifier)
    return ACNParseResult(tuple(sorted(sessions, key=lambda s: (s.arrival, s.session_id))),
                          len(records), dict(rejected), config)


@dataclass(frozen=True)
class SessionDay:
    scenario_id: str
    local_date: date
    split: str
    sessions: tuple[ACNSession, ...]
    timezone: str = "America/Los_Angeles"


def group_session_days(parsed: ACNParseResult) -> tuple[SessionDay, ...]:
    """One scenario is one local arrival day, including all that day's sessions."""
    groups: dict[date, list[ACNSession]] = {}
    for session in parsed.sessions:
        groups.setdefault(session.local_date, []).append(session)
    return tuple(SessionDay(f"{parsed.config.site_label}:{day.isoformat()}", day, values[0].split,
                             tuple(sorted(values, key=lambda s: (s.arrival, s.session_id))), parsed.config.timezone)
                 for day, values in sorted(groups.items()))


@dataclass(frozen=True)
class FrozenDaySelection:
    selection_seed: int
    scenario_ids: tuple[str, ...]
    simulation_seeds: tuple[int, ...]
    session_ids: tuple[tuple[str, ...], ...]
    training_seeds: tuple[int, ...] = (1, 2, 3, 4, 5)

    def __post_init__(self):
        size = len(self.scenario_ids)
        if (not size or len(self.simulation_seeds) != size or len(self.session_ids) != size
                or len(set(self.scenario_ids)) != size or len(set(self.simulation_seeds)) != size
                or any(not members for members in self.session_ids)):
            raise ValueError("A frozen selection needs unique scenarios/seeds and matching nonempty session membership")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrozenDaySelection":
        fields = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("scenario_ids", "simulation_seeds", "training_seeds"):
            fields[key] = tuple(fields[key])
        fields["session_ids"] = tuple(tuple(v) for v in fields["session_ids"])
        return cls(**fields)


def freeze_test_days(days: tuple[SessionDay, ...], count: int = 100,
                     selection_seed: int = 10001,
                     first_simulation_seed: int = 10001) -> FrozenDaySelection:
    """Select once from date strata; all training seeds share the selected days."""
    test = sorted((d for d in days if d.split == "test"), key=lambda d: d.local_date)
    if count <= 0 or len(test) < count or len({d.scenario_id for d in test}) != len(test):
        raise ValueError("Selection requires enough distinct held-out scenario days")
    rng = np.random.Generator(np.random.PCG64(selection_seed))
    strata = np.array_split(np.arange(len(test)), count)
    chosen = [test[int(rng.choice(stratum))] for stratum in strata]
    return FrozenDaySelection(selection_seed, tuple(d.scenario_id for d in chosen),
                               tuple(range(first_simulation_seed, first_simulation_seed+count)),
                               tuple(tuple(s.session_id for s in d.sessions) for d in chosen))


@dataclass(frozen=True)
class LocationVocabulary:
    """Training-only location categories; zero is the frozen unknown category."""
    known: tuple[str, ...]

    @classmethod
    def fit(cls, sessions: tuple[ACNSession, ...]) -> "LocationVocabulary":
        if not sessions or any(s.split != "train" for s in sessions):
            raise ValueError("Location vocabulary may only fit nonempty training sessions")
        return cls(tuple(sorted({s.location_id for s in sessions})))

    def encode(self, location_id: str) -> int:
        return self.known.index(location_id)+1 if location_id in self.known else 0

    @property
    def size(self) -> int:
        return len(self.known)+1

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"known": self.known}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LocationVocabulary":
        return cls(tuple(json.loads(Path(path).read_text(encoding="utf-8"))["known"]))


def fit_training_normalizer(days: tuple[SessionDay, ...], graph_builder: Callable):
    """Delegate to the graph normalizer after checking chronological membership."""
    if not days or any(day.split != "train" for day in days):
        raise ValueError("Feature moments must be fit exclusively on training days")
    from .graph import FeatureNormalizer
    return FeatureNormalizer.fit([graph_builder(day) for day in days])


@dataclass(frozen=True)
class EnergestCalibration:
    voltage_v: float
    ticks_per_second: float
    cpu_ma: float
    lpm_ma: float
    tx_ma: float
    rx_ma: float

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if not all(math.isfinite(v) and v >= 0 for v in values) or self.voltage_v <= 0 or self.ticks_per_second <= 0:
            raise ValueError("Calibration needs finite nonnegative mA currents and positive V/tick rate")


def read_energest_csv(path: str | Path, calibration: EnergestCalibration,
                      horizon: int, step_seconds: float, *, cumulative: bool = False,
                      counter_modulus: int | None = None) -> dict[str, tuple[float, ...]]:
    """Convert explicit mode ticks to received-node load energy in kWh per slot.

    Columns: node_id, slot, cpu_ticks, lpm_ticks, tx_ticks, rx_ticks. Delta rows
    cover slots 0..H-1. Cumulative rows cover boundaries 0..H. CPU/LPM currents
    describe the processor; TX/RX currents are incremental radio components.
    Every CPU/LPM observation must cover the complete slot. No missing trace is
    filled with invented consumption. Optional modulus handles counter rollover.
    """
    if horizon <= 0 or step_seconds <= 0 or (counter_modulus is not None and counter_modulus <= 0):
        raise ValueError("Invalid calibration horizon, step or counter modulus")
    columns = ("cpu_ticks", "lpm_ticks", "tx_ticks", "rx_ticks")
    groups: dict[str, dict[int, tuple[float, ...]]] = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"node_id", "slot", *columns} <= set(reader.fieldnames or ()):
            raise ValueError("Energest CSV lacks the documented tick columns")
        for row in reader:
            node, slot = row["node_id"].strip(), int(row["slot"])
            ticks = tuple(float(row[c]) for c in columns)
            if not node or any(not math.isfinite(v) or v < 0 or v != int(v) for v in ticks):
                raise ValueError("Node IDs and finite nonnegative integer mode ticks are required")
            if slot in groups.setdefault(node, {}):
                raise ValueError("Duplicate Energest node/slot observation")
            groups[node][slot] = ticks
    if not groups:
        raise ValueError("Energest CSV contains no observations")
    result = {}
    expected = set(range(horizon+1 if cumulative else horizon))
    currents = (calibration.cpu_ma, calibration.lpm_ma, calibration.tx_ma, calibration.rx_ma)
    for node, rows in groups.items():
        if set(rows) != expected:
            raise ValueError("Every calibration node needs all documented slot observations")
        energy = []
        for t in range(horizon):
            ticks = tuple(rows[t+1][i]-rows[t][i] for i in range(4)) if cumulative else rows[t]
            if cumulative and counter_modulus is not None:
                ticks = tuple(v % counter_modulus for v in ticks)
            if any(v < 0 for v in ticks):
                raise ValueError("A cumulative counter reset needs an explicit rollover modulus")
            window = calibration.ticks_per_second*step_seconds
            if abs(ticks[0]+ticks[1]-window) > 1e-6 or ticks[2]+ticks[3] > window+1e-6:
                raise ValueError("Mode ticks do not describe one complete, physically consistent slot")
            energy.append(calibration.voltage_v*sum(i*tick/calibration.ticks_per_second
                                                      for i, tick in zip(currents, ticks))/3_600_000_000)
        result[node] = tuple(energy)
    return result
