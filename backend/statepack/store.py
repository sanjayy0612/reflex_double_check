"""Versioned storage for State Packs.

Layout:
    statepacks/drafts/M01.v1.json      editable, not used at runtime
    statepacks/approved/M01.v1.json    frozen (read-only, content-hashed)

Runtime loads the highest approved version per machine and refuses packs whose
hash no longer matches (tampered) or whose profile has changed since approval
(stale).
"""

from __future__ import annotations

import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path

from .schema import MachineProfile, StatePack
from .validator import validate

FILE_RE = re.compile(r"^(?P<machine>.+)\.v(?P<version>\d+)\.json$")


class StatePackError(Exception):
    pass


class StatePackStore:
    def __init__(self, root: Path | str = "statepacks"):
        self.root = Path(root)
        self.drafts = self.root / "drafts"
        self.approved = self.root / "approved"

    # ── paths & listing ──────────────────────────────────────────────────────

    def _path(self, folder: Path, machine_id: str, version: int) -> Path:
        return folder / f"{machine_id}.v{version}.json"

    def _versions(self, folder: Path, machine_id: str) -> list[int]:
        if not folder.exists():
            return []
        return sorted(
            int(m["version"])
            for f in folder.iterdir()
            if (m := FILE_RE.match(f.name)) and m["machine"] == machine_id
        )

    def next_version(self, machine_id: str) -> int:
        versions = self._versions(self.drafts, machine_id) + self._versions(self.approved, machine_id)
        return max(versions, default=0) + 1

    def list(self) -> list[tuple[str, int, str]]:
        """(machine_id, version, status) for every stored pack."""
        out = []
        for status, folder in (("draft", self.drafts), ("approved", self.approved)):
            if folder.exists():
                for f in folder.iterdir():
                    if m := FILE_RE.match(f.name):
                        out.append((m["machine"], int(m["version"]), status))
        return sorted(out)

    # ── drafts ───────────────────────────────────────────────────────────────

    def save_draft(self, pack: StatePack) -> Path:
        if pack.status != "draft":
            raise StatePackError("Only drafts can be saved as drafts.")
        self.drafts.mkdir(parents=True, exist_ok=True)
        path = self._path(self.drafts, pack.machine_id, pack.version)
        path.write_text(pack.model_dump_json(indent=2))
        return path

    def load_draft(self, machine_id: str, version: int | None = None) -> StatePack:
        versions = self._versions(self.drafts, machine_id)
        if not versions:
            raise StatePackError(f"No drafts for {machine_id}.")
        version = version or versions[-1]
        return StatePack.model_validate_json(self._path(self.drafts, machine_id, version).read_text())

    # ── approval ─────────────────────────────────────────────────────────────

    def approve(self, machine_id: str, profile: MachineProfile, approver: str, version: int | None = None) -> Path:
        pack = self.load_draft(machine_id, version)

        if pack.profile_fingerprint != profile.fingerprint():
            raise StatePackError("Machine profile changed since this draft was generated; regenerate it.")
        report = validate(pack.states, profile)
        if not report.ok:
            raise StatePackError("Draft fails validation:\n- " + "\n- ".join(report.errors))

        pack.status = "approved"
        pack.approved_by = approver
        pack.approved_at = datetime.now(timezone.utc)
        pack.content_hash = pack.compute_content_hash()

        self.approved.mkdir(parents=True, exist_ok=True)
        path = self._path(self.approved, machine_id, pack.version)
        if path.exists():
            raise StatePackError(f"{path.name} is already approved; approved packs are immutable.")
        path.write_text(pack.model_dump_json(indent=2))
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # freeze
        self._path(self.drafts, machine_id, pack.version).unlink()
        return path

    # ── runtime access ───────────────────────────────────────────────────────

    def load_active(self, machine_id: str, profile: MachineProfile | None = None) -> StatePack:
        """The pack JEV should use: latest approved, integrity-checked."""
        versions = self._versions(self.approved, machine_id)
        if not versions:
            raise StatePackError(f"No approved State Pack for {machine_id}.")
        pack = StatePack.model_validate_json(self._path(self.approved, machine_id, versions[-1]).read_text())

        if pack.content_hash != pack.compute_content_hash():
            raise StatePackError(f"{machine_id} v{pack.version}: content hash mismatch (file modified after approval).")
        if profile is not None and pack.profile_fingerprint != profile.fingerprint():
            raise StatePackError(f"{machine_id} v{pack.version}: stale; machine profile changed since approval.")
        return pack
