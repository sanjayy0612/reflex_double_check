from __future__ import annotations

from pathlib import Path

import yaml

from .schema import MachineProfile

DEFAULT_PATH = Path("config/machines.yaml")


def load_profiles(path: Path | str = DEFAULT_PATH) -> dict[str, MachineProfile]:
    data = yaml.safe_load(Path(path).read_text())
    profiles = [MachineProfile.model_validate(m) for m in data["machines"]]
    return {p.machine_id: p for p in profiles}
