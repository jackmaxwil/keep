"""Deterministic terminal and JSON rendering for campaign observations."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import stat
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from mlx_vq.io.authenticated_artifacts import (
    AuthenticatedFile,
    publish_bytes_transactionally,
)

from .controller import plan_next_transition
from .ledger import LedgerError, evidence_from_event
from .models import (
    CampaignConfig,
    CampaignObservation,
    FrozenJsonDict,
    LedgerEvent,
)
from .verification import get_verification_profile, profile_fingerprint


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_SAFE_METRIC_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_QUALITY_FIELDS = {
    "campaign_config_sha256",
    "transition",
    "split",
    "metric",
    "value",
    "evidence_event_sha256",
    "manifest_sha256",
    "quality_row_sha256",
}
_QUALITY_SPLITS = {"selection", "report", "holdout"}
_PROTECTED_ROOT_HANDOFFS = {
    "glm52-community-wow-section-handoff-20260710.md",
    "glm52-recovery-campaign-handoff-20260710.md",
}
_HANDOFF_SEAL_KEY = secrets.token_bytes(32)
_PUBLICATION_SEAL_KEY = secrets.token_bytes(32)


class CampaignHandoffError(ValueError):
    """Handoff evidence, repository identity, or publication is unsafe."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise CampaignHandoffError(
            f"handoff value is not finite canonical JSON: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class RepositoryIdentity:
    branch: str
    head: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.branch, str)
            or _SAFE_BRANCH_RE.fullmatch(self.branch) is None
            or ".." in PurePosixPath(self.branch).parts
        ):
            raise CampaignHandoffError("repository branch identity is unsafe")
        if not isinstance(self.head, str) or _HEAD_RE.fullmatch(self.head) is None:
            raise CampaignHandoffError("repository HEAD must be exact 40-hex")


@dataclass(frozen=True, slots=True)
class CampaignHandoff:
    facts: FrozenJsonDict
    fingerprint: str
    _validation_seal: str = field(
        default="", init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.facts, FrozenJsonDict):
            raise CampaignHandoffError("handoff facts must be immutable")
        if not _is_sha256(self.fingerprint):
            raise CampaignHandoffError("handoff fingerprint is invalid")
        expected = hashlib.sha256(_canonical_json(self.facts)).hexdigest()
        if self.fingerprint != expected:
            raise CampaignHandoffError("handoff fingerprint does not match facts")


def _handoff_seal(handoff: CampaignHandoff) -> str:
    return hmac.new(
        _HANDOFF_SEAL_KEY,
        _canonical_json(
            {
                "facts": handoff.facts,
                "fingerprint": handoff.fingerprint,
            }
        ),
        hashlib.sha256,
    ).hexdigest()


def _require_built_handoff(handoff: CampaignHandoff) -> None:
    if not isinstance(handoff, CampaignHandoff) or not hmac.compare_digest(
        handoff._validation_seal, _handoff_seal(handoff)
    ):
        raise CampaignHandoffError("handoff value lacks builder authority")


@dataclass(frozen=True, slots=True)
class HandoffPublication:
    path: Path
    sha256: str
    size: int
    _publication_seal: str = field(
        default="", init=False, repr=False, compare=False
    )
    _authenticated: AuthenticatedFile | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not _is_sha256(self.sha256):
            raise CampaignHandoffError("handoff publication identity is invalid")
        if type(self.size) is not int or self.size < 0:
            raise CampaignHandoffError("handoff publication size is invalid")

    def close(self) -> None:
        if self._closed:
            return
        authenticated = self._authenticated
        object.__setattr__(self, "_closed", True)
        object.__setattr__(self, "_authenticated", None)
        if authenticated is not None:
            authenticated.close()

    def __enter__(self) -> HandoffPublication:
        if self._closed:
            raise CampaignHandoffError("handoff publication receipt is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


def _publication_seal(receipt: HandoffPublication) -> str:
    return hmac.new(
        _PUBLICATION_SEAL_KEY,
        _canonical_json(
            {
                "path": str(receipt.path),
                "sha256": receipt.sha256,
                "size": receipt.size,
            }
        ),
        hashlib.sha256,
    ).hexdigest()


def _require_published_receipt(receipt: HandoffPublication) -> None:
    if isinstance(receipt, HandoffPublication) and receipt._closed:
        raise CampaignHandoffError("handoff publication receipt is closed")
    if not isinstance(receipt, HandoffPublication) or not hmac.compare_digest(
        receipt._publication_seal, _publication_seal(receipt)
    ):
        raise CampaignHandoffError(
            "handoff publication receipt lacks publisher authority"
        )


def _reauthenticate_published_file(receipt: HandoffPublication) -> None:
    authenticated = receipt._authenticated
    if authenticated is None:
        raise CampaignHandoffError(
            "handoff publication receipt lacks retained authentication"
        )
    try:
        authenticated.verify_visible()
        if (
            authenticated.path != receipt.path
            or authenticated.sha256 != receipt.sha256
            or authenticated.size != receipt.size
        ):
            raise CampaignHandoffError(
                "published handoff changed before receipt authentication"
            )
    except (OSError, ValueError) as error:
        raise CampaignHandoffError(
            f"published handoff changed before receipt authentication: {error}"
        ) from error


def _ledger_body(event: LedgerEvent) -> dict[str, object]:
    return {
        "schema_version": event.schema_version,
        "sequence": event.sequence,
        "timestamp": event.timestamp,
        "event_kind": event.event_kind,
        "experiment": event.experiment,
        "payload": _thaw(event.payload),
        "previous_event_sha256": event.previous_event_sha256,
    }


def _validated_ledger(events: Sequence[LedgerEvent]) -> tuple[LedgerEvent, ...]:
    if isinstance(events, (str, bytes, bytearray)):
        raise CampaignHandoffError("ledger must be a verified event sequence")
    result = tuple(events)
    previous: str | None = None
    for index, event in enumerate(result, start=1):
        if not isinstance(event, LedgerEvent):
            raise CampaignHandoffError("ledger contains a non-event value")
        computed = hashlib.sha256(_canonical_json(_ledger_body(event))).hexdigest()
        if (
            type(event.schema_version) is not int
            or event.schema_version != 1
            or type(event.sequence) is not int
            or event.sequence != index
            or event.previous_event_sha256 != previous
            or event.event_sha256 != computed
        ):
            raise CampaignHandoffError("ledger chain or event identity is corrupt")
        previous = event.event_sha256
    return result


def _validate_ledger_anchor(
    events: tuple[LedgerEvent, ...],
    expected_head_sha256: str | None,
    expected_event_count: int | None,
) -> None:
    if not events:
        if expected_head_sha256 is not None or expected_event_count is not None:
            raise CampaignHandoffError(
                "empty ledger must not carry a nonempty ledger anchor"
            )
        return
    if expected_head_sha256 is None or expected_event_count is None:
        raise CampaignHandoffError(
            "nonempty handoff ledger requires exact head and event-count anchors"
        )
    if not _is_sha256(expected_head_sha256):
        raise CampaignHandoffError("ledger head anchor must be exact SHA-256")
    if type(expected_event_count) is not int or expected_event_count < 1:
        raise CampaignHandoffError(
            "ledger event-count anchor must be a positive integer"
        )
    if (
        events[-1].event_sha256 != expected_head_sha256
        or len(events) != expected_event_count
    ):
        raise CampaignHandoffError("ledger does not match external anchor")


def _quality_row_fingerprint(
    config: CampaignConfig,
    event: LedgerEvent,
    *,
    transition: str,
    split: str,
    metric: str,
    value: float,
    manifest_sha256: str,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "campaign_config_sha256": config.campaign_config_sha256,
                "experiment": event.experiment,
                "transition": transition,
                "split": split,
                "metric": metric,
                "value": value,
                "manifest_sha256": manifest_sha256,
            }
        )
    ).hexdigest()


def _physical_manifest(
    observation: CampaignObservation, experiment_name: str
) -> str | None:
    if observation.experiment_observations:
        try:
            snapshot = observation.snapshot(experiment_name)
        except KeyError:
            return None
        return snapshot.manifest_file_sha256 if snapshot.physically_complete else None
    if (
        observation.experiment_name == experiment_name
        and observation.manifest_state == "valid"
        and observation.manifest_file_sha256 is not None
        and observation.recovered_groups_state == "present"
        and observation.recovered_groups == observation.expected_groups
        and observation.artifact_links_state == "present"
        and observation.artifact_links == observation.expected_artifact_links
        and not observation.contradictions
    ):
        return observation.manifest_file_sha256
    return None


def _quality_projection(
    config: CampaignConfig,
    events: tuple[LedgerEvent, ...],
    observation: CampaignObservation,
) -> tuple[dict[str, object], tuple[str, ...]]:
    evidence_events: dict[str, LedgerEvent] = {}
    rows: dict[tuple[str, str, str], dict[str, object]] = {}
    conflicts: set[tuple[str, str, str]] = set()
    consumed_evidence: set[str] = set()
    contradictions: list[str] = []
    declared = {experiment.name: experiment for experiment in config.experiments}
    for event in events:
        if event.event_kind == "evidence_registered":
            evidence_events[event.event_sha256] = event
            continue
        if event.event_kind != "quality_observed":
            continue
        payload = event.payload
        if set(payload) != _QUALITY_FIELDS:
            contradictions.append(
                f"quality event {event.event_sha256} has partial or unknown fields"
            )
            continue
        experiment = declared.get(event.experiment)
        value = payload.get("value")
        split = payload.get("split")
        metric = payload.get("metric")
        transition = payload.get("transition")
        evidence_hash = payload.get("evidence_event_sha256")
        manifest_hash = payload.get("manifest_sha256")
        row_hash = payload.get("quality_row_sha256")
        if (
            experiment is None
            or payload.get("campaign_config_sha256")
            != config.campaign_config_sha256
            or transition != experiment.transition_name
            or split not in _QUALITY_SPLITS
            or not isinstance(metric, str)
            or _SAFE_METRIC_RE.fullmatch(metric) is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not _is_sha256(evidence_hash)
            or not _is_sha256(manifest_hash)
            or not _is_sha256(row_hash)
        ):
            continue
        current_manifest = _physical_manifest(observation, event.experiment)
        if current_manifest is None or manifest_hash != current_manifest:
            continue
        evidence_event = evidence_events.get(str(evidence_hash))
        if evidence_event is None or evidence_event.experiment != event.experiment:
            continue
        if str(evidence_hash) in consumed_evidence:
            contradictions.append(
                f"quality evidence event {evidence_hash} was reused"
            )
            continue
        try:
            evidence = evidence_from_event(evidence_event)
        except LedgerError as error:
            contradictions.append(f"quality evidence is invalid: {error}")
            continue
        if (
            evidence.evidence_class != "release"
            or not evidence.release_eligible
            or evidence.manifest_identity_sha256 != manifest_hash
            or not _is_sha256(evidence.candidate_identity_sha256)
            or not _is_sha256(evidence.baseline_identity_sha256)
            or len(evidence.verification_results) != 1
            # The handoff does not invent a verifier command. EvidenceRecord.argv
            # retains the producer's exact invocation; equality below binds the
            # sole named result to that invocation, while the two exact levers
            # bind its purpose and canonical row identity.
            or evidence.verification_results[0].name
            != "quality-row-verification"
            or evidence.verification_results[0].argv != evidence.argv
            or evidence.verification_results[0].exit_code != 0
        ):
            continue
        expected_row_hash = _quality_row_fingerprint(
            config,
            event,
            transition=transition,  # type: ignore[arg-type]
            split=str(split),
            metric=str(metric),
            value=float(value),
            manifest_sha256=str(manifest_hash),
        )
        if (
            row_hash != expected_row_hash
            or evidence.recovery_levers
            != (
                "quality-observation",
                f"quality-row-sha256:{expected_row_hash}",
            )
        ):
            continue
        consumed_evidence.add(str(evidence_hash))
        key = (event.experiment, str(split), str(metric))
        row = {
            "experiment": event.experiment,
            "transition": transition,
            "split": split,
            "metric": metric,
            "value": float(value),
            "evidence_sha256": evidence.content_sha256,
            "manifest_sha256": manifest_hash,
        }
        previous = rows.get(key)
        if previous is not None and previous != row:
            conflicts.add(key)
            rows.pop(key, None)
            continue
        if key not in conflicts:
            rows[key] = row
    if conflicts:
        contradictions.append(
            "conflicting quality rows: "
            + ", ".join("/".join(key) for key in sorted(conflicts))
        )
    ordered = [rows[key] for key in sorted(rows)]
    if contradictions:
        return {"state": "contradictory", "rows": []}, tuple(contradictions)
    return (
        {"state": "present", "rows": ordered}
        if ordered
        else {"state": "absent", "rows": []},
        (),
    )


def _approval_projection(
    config: CampaignConfig,
    events: tuple[LedgerEvent, ...],
) -> tuple[dict[str, object], tuple[str, ...]]:
    profile = get_verification_profile("glm52-recovery-campaign")
    profile_sha = profile_fingerprint(profile)
    contradictions: list[str] = []
    for event in events:
        if event.event_kind in {"verification_approved", "review_approved"}:
            contradictions.append(
                "ledger approval event lacks independently authenticated "
                "profile and review authority"
            )
    return (
        {
            "verification": {
                "state": "absent",
                "profile": profile.name,
                "profile_sha256": profile_sha,
                "evidence_sha256": None,
            },
            "review": {
                "state": "absent",
                "required_record": profile.required_review_record,
                "reviewer_authority": profile.reviewer_authority,
                "evidence_sha256": None,
            },
            "human_release": {
                "state": "required",
                "boundaries": [
                    "quality-slope investment",
                    "size/quality tradeoff",
                    "quiet-machine scheduling",
                    "external spend",
                    "publication",
                    "release approval",
                ],
            },
        },
        tuple(contradictions),
    )


def _decision_projection(decision, contradictions: tuple[str, ...]):
    if contradictions:
        blocker = {
            "state": "contradictory",
            "reason": "; ".join(contradictions),
        }
        action = {
            "action": "read-only-review",
            "reason": "resolve contradictory evidence before mutation",
            "command": None,
            "heavy": False,
        }
    elif decision.action == "wait":
        blocker = {"state": "in-flight", "reason": decision.reason}
        action = {
            "action": "wait",
            "reason": decision.reason,
            "command": list(decision.argv),
            "heavy": decision.heavy,
        }
    elif decision.action == "blocked":
        blocker = {"state": "blocked", "reason": decision.reason}
        action = {
            "action": "resolve-blocker",
            "reason": decision.reason,
            "command": None,
            "heavy": False,
        }
    else:
        blocker = {
            "state": "absent" if decision.action == "complete" else "incomplete",
            "reason": decision.reason,
        }
        action = {
            "action": decision.action,
            "reason": decision.reason,
            "command": list(decision.argv) if decision.argv else None,
            "heavy": decision.heavy,
        }
    return blocker, action


def build_campaign_handoff(
    config: CampaignConfig,
    ledger: Sequence[LedgerEvent],
    observation: CampaignObservation,
    repository: RepositoryIdentity,
    *,
    ledger_head_sha256: str | None = None,
    ledger_event_count: int | None = None,
) -> CampaignHandoff:
    if not isinstance(config, CampaignConfig):
        raise CampaignHandoffError("handoff config must be parsed")
    if not isinstance(observation, CampaignObservation):
        raise CampaignHandoffError("handoff observation is invalid")
    if observation.campaign != config.campaign:
        raise CampaignHandoffError(
            "handoff observation campaign does not match configuration"
        )
    if not isinstance(repository, RepositoryIdentity):
        raise CampaignHandoffError("handoff repository identity is invalid")
    events = _validated_ledger(ledger)
    _validate_ledger_anchor(
        events, ledger_head_sha256, ledger_event_count
    )
    contradictions = list(observation.contradictions)
    throughput = observation.throughput_groups_per_second
    eta = observation.eta_seconds
    for label, value in (
        ("throughput", throughput),
        ("ETA", eta),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            contradictions.append(f"{label} is non-finite and was rendered absent")
            if label == "throughput":
                throughput = None
            else:
                eta = None
    quality, quality_contradictions = _quality_projection(
        config, events, observation
    )
    approvals, approval_contradictions = _approval_projection(config, events)
    contradictions.extend(quality_contradictions)
    contradictions.extend(approval_contradictions)
    contradictions_tuple = tuple(dict.fromkeys(contradictions))
    if contradictions_tuple:
        quality = {"state": "contradictory", "rows": []}
    decision = plan_next_transition(config, observation, events)
    blocker, next_action = _decision_projection(decision, contradictions_tuple)
    profile = get_verification_profile("glm52-recovery-campaign")
    protected_paths = tuple(
        dict.fromkeys(
            (
                *profile.protected_paths,
                config.campaign_config_path,
                profile.required_review_record,
            )
        )
    )
    ledger_head = events[-1].event_sha256 if events else None
    ledger_chain = (
        hashlib.sha256(
            _canonical_json([event.event_sha256 for event in events])
        ).hexdigest()
        if events
        else None
    )
    body = {
        "schema_version": 1,
        "campaign": config.campaign,
        "model": {
            "id": config.model_id,
            "revision": config.model_revision,
        },
        "experiments": [
            {
                "name": experiment.name,
                "kind": experiment.kind,
                "transition": experiment.transition_name,
                "depends_on": list(experiment.depends_on),
                "output_path": experiment.output_path,
                "expected_payload_bytes": experiment.expected_payload_bytes,
            }
            for experiment in config.experiments
        ],
        "repository": {
            "branch": repository.branch,
            "head": repository.head,
        },
        "active_jobs": [
            {
                "pid": process.pid,
                "elapsed_seconds": process.elapsed_seconds,
                "experiment": process.experiment_name,
                "transition": process.transition_name,
                "command": list(process.command),
            }
            for process in observation.active_processes
        ],
        "groups": {
            "state": observation.recovered_groups_state,
            "observed": (
                observation.recovered_groups
                if observation.recovered_groups_state == "present"
                else None
            ),
            "expected": observation.expected_groups,
        },
        "artifact_links": {
            "state": observation.artifact_links_state,
            "observed": (
                observation.artifact_links
                if observation.artifact_links_state == "present"
                else None
            ),
            "expected": observation.expected_artifact_links,
        },
        "throughput_groups_per_second": throughput,
        "eta_seconds": eta,
        "evidence": {
            "campaign_matrix_sha256": config.campaign_config_sha256,
            "verification_profile_sha256": profile_fingerprint(profile),
            "model_config_sha256": config.config_sha256,
            "model_index_sha256": config.index_sha256,
            "full_source_inventory_sha256": config.full_source_blob_inventory_sha256,
            "routed_source_inventory_sha256": config.routed_source_blob_inventory_sha256,
            "authorities": [
                {
                    "name": authority.name,
                    "path": authority.path,
                    "sha256": authority.sha256,
                    "role": authority.role,
                    "split": authority.split,
                }
                for authority in config.authorities
            ],
            "manifest": {
                "state": observation.manifest_state,
                "file_sha256": observation.manifest_file_sha256,
                "body_sha256": observation.manifest_body_sha256,
            },
            "ledger": {
                "state": "anchored" if events else "absent",
                "event_count": len(events),
                "head_sha256": ledger_head,
                "chain_sha256": ledger_chain,
                "anchor_head_sha256": ledger_head_sha256,
                "anchor_event_count": ledger_event_count,
            },
        },
        "quality_curve": quality,
        "current_blocker": blocker,
        "next_action": next_action,
        "protected_paths": list(protected_paths),
        "approvals": approvals,
        "contradictions": list(contradictions_tuple),
        "release_status": "not-human-approved",
    }
    facts = FrozenJsonDict(body)
    fingerprint = hashlib.sha256(_canonical_json(facts)).hexdigest()
    handoff = CampaignHandoff(facts=facts, fingerprint=fingerprint)
    object.__setattr__(handoff, "_validation_seal", _handoff_seal(handoff))
    return handoff


def handoff_fingerprint(handoff: CampaignHandoff) -> str:
    _require_built_handoff(handoff)
    return handoff.fingerprint


def handoff_payload(handoff: CampaignHandoff) -> dict[str, object]:
    _require_built_handoff(handoff)
    return {**_thaw(handoff.facts), "handoff_sha256": handoff.fingerprint}  # type: ignore[arg-type]


def render_handoff_json(handoff: CampaignHandoff) -> str:
    return json.dumps(
        handoff_payload(handoff),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _display(value: object) -> str:
    return "absent" if value is None else str(value)


def render_handoff_terminal(handoff: CampaignHandoff) -> str:
    payload = handoff_payload(handoff)
    repository = payload["repository"]
    groups = payload["groups"]
    evidence = payload["evidence"]
    quality = payload["quality_curve"]
    approvals = payload["approvals"]
    lines = [
        f"Campaign: {payload['campaign']}",
        f"Model: {payload['model']['id']}@{payload['model']['revision']}",
        f"Repository: {repository['branch']} @ {repository['head']}",
        f"Handoff SHA-256: {payload['handoff_sha256']}",
        f"Recovered groups: {_display(groups['observed'])}/{groups['expected']} ({groups['state']})",
        f"Throughput: {_display(payload['throughput_groups_per_second'])}",
        f"ETA seconds: {_display(payload['eta_seconds'])}",
        "Active jobs:",
    ]
    jobs = payload["active_jobs"]
    if jobs:
        lines.extend(
            f"  - pid={job['pid']} elapsed_seconds={_display(job['elapsed_seconds'])} "
            f"experiment={_display(job['experiment'])} transition={_display(job['transition'])}"
            for job in jobs
        )
    else:
        lines.append("  absent")
    lines.extend(
        [
            f"Matrix SHA-256: {evidence['campaign_matrix_sha256']}",
            f"Manifest: {evidence['manifest']['state']} "
            f"({_display(evidence['manifest']['file_sha256'])})",
            f"Ledger: {evidence['ledger']['state']} "
            f"({_display(evidence['ledger']['head_sha256'])})",
            f"Quality curve: {quality['state']}",
        ]
    )
    lines.extend(
        f"  - {row['experiment']} {row['split']} {row['metric']}={row['value']}"
        for row in quality["rows"]
    )
    lines.extend(
        [
            f"Current blocker: {payload['current_blocker']['state']} — "
            f"{payload['current_blocker']['reason']}",
            f"Next action: {payload['next_action']['action']} — "
            f"{payload['next_action']['reason']}",
            f"Verification: {approvals['verification']['state']}",
            f"Review: {approvals['review']['state']}",
            f"Human release approval: {approvals['human_release']['state']}",
            "Protected paths:",
        ]
    )
    lines.extend(f"  - {path}" for path in payload["protected_paths"])
    lines.append("Contradictions:")
    lines.extend(
        (f"  - {item}" for item in payload["contradictions"])
        if payload["contradictions"]
        else ("  none",)
    )
    lines.extend(
        [
            "Canonical facts JSON:",
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        ]
    )
    return "\n".join(lines) + "\n"


def render_handoff_markdown(handoff: CampaignHandoff) -> str:
    payload = handoff_payload(handoff)
    evidence = payload["evidence"]
    quality = payload["quality_curve"]
    lines = [
        "# GLM-5.2 Recovery Campaign Handoff",
        "",
        f"- Campaign: `{payload['campaign']}`",
        f"- Model: `{payload['model']['id']}` at `{payload['model']['revision']}`",
        f"- Repository: `{payload['repository']['branch']}` at `{payload['repository']['head']}`",
        f"- Handoff SHA-256: `{payload['handoff_sha256']}`",
        f"- Recovered groups: `{_display(payload['groups']['observed'])}/{payload['groups']['expected']}` (`{payload['groups']['state']}`)",
        f"- Throughput: `{_display(payload['throughput_groups_per_second'])}` groups/second",
        f"- ETA: `{_display(payload['eta_seconds'])}` seconds",
        "",
        "## Active jobs",
        "",
    ]
    if payload["active_jobs"]:
        lines.extend(
            f"- PID `{job['pid']}`; elapsed `{_display(job['elapsed_seconds'])}` seconds; "
            f"experiment `{_display(job['experiment'])}`; transition `{_display(job['transition'])}`"
            for job in payload["active_jobs"]
        )
    else:
        lines.append("- absent")
    lines.extend(
        [
            "",
            "## Evidence identities",
            "",
            f"- Campaign matrix: `{evidence['campaign_matrix_sha256']}`",
            f"- Verification profile: `{evidence['verification_profile_sha256']}`",
            f"- Model config: `{evidence['model_config_sha256']}`",
            f"- Model index: `{evidence['model_index_sha256']}`",
            f"- Manifest: `{evidence['manifest']['state']}` / `{_display(evidence['manifest']['file_sha256'])}`",
            f"- Ledger: `{evidence['ledger']['state']}` / `{_display(evidence['ledger']['head_sha256'])}`",
            "",
            "## Experiment quality curve",
            "",
            f"State: **{quality['state']}**",
            "",
        ]
    )
    if quality["rows"]:
        lines.extend(
            [
                "| Experiment | Transition | Split | Metric | Value | Evidence | Manifest |",
                "|---|---|---|---|---:|---|---|",
            ]
        )
        lines.extend(
            f"| {row['experiment']} | {_display(row['transition'])} | {row['split']} | "
            f"{row['metric']} | {row['value']} | `{row['evidence_sha256']}` | "
            f"`{row['manifest_sha256']}` |"
            for row in quality["rows"]
        )
    else:
        lines.append("absent")
    lines.extend(
        [
            "",
            "## Blocker and next permissible action",
            "",
            f"- Current blocker: **{payload['current_blocker']['state']}** — {payload['current_blocker']['reason']}",
            f"- Next action: **{payload['next_action']['action']}** — {payload['next_action']['reason']}",
            "",
            "## Approval boundaries",
            "",
            f"- Machine verification: **{payload['approvals']['verification']['state']}**",
            f"- Independent review: **{payload['approvals']['review']['state']}**",
            f"- Human release approval: **{payload['approvals']['human_release']['state']}**",
            "- Generation of this handoff is not release approval and makes no RC quality claim.",
            "",
            "## Protected paths",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in payload["protected_paths"])
    lines.extend(["", "## Contradictions", ""])
    lines.extend(
        (f"- {item}" for item in payload["contradictions"])
        if payload["contradictions"]
        else ("none",)
    )
    lines.extend(
        [
            "",
            "## Canonical facts JSON",
            "",
            "```json",
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def observe_repository_identity(
    repo_root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> RepositoryIdentity:
    root = Path(repo_root)
    commands = (
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        ("rev-parse", "--verify", "HEAD^{commit}"),
    )
    outputs: list[str] = []
    for command in commands:
        try:
            completed = runner(
                ["/usr/bin/git", "-C", str(root), *command],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CampaignHandoffError(f"Git identity observation failed: {error}") from error
        if completed.returncode != 0:
            stderr = completed.stderr.decode(errors="replace").strip()
            raise CampaignHandoffError(
                f"Git identity observation failed: {stderr or completed.returncode}"
            )
        try:
            output = completed.stdout.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise CampaignHandoffError("Git identity output is not UTF-8") from error
        outputs.append(output)
    return RepositoryIdentity(branch=outputs[0], head=outputs[1])


def _publication_target(
    destination: str | Path,
    repo_root: Path,
) -> Path:
    root = Path(repo_root).absolute()
    requested = Path(destination)
    target = requested if requested.is_absolute() else root / requested
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise CampaignHandoffError("handoff output must remain inside the repository") from error
    if (
        relative.as_posix() in _PROTECTED_ROOT_HANDOFFS
        or len(relative.parts) < 3
        or relative.parts[:2] != ("artifacts", "quality")
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        label = "protected" if relative.as_posix() in _PROTECTED_ROOT_HANDOFFS else "unsafe"
        raise CampaignHandoffError(f"handoff output is {label}: {relative}")
    try:
        visible = target.lstat()
    except FileNotFoundError:
        visible = None
    except OSError as error:
        raise CampaignHandoffError(f"handoff destination inspection failed: {error}") from error
    if visible is not None:
        kind = "symlink/non-regular" if not stat.S_ISREG(visible.st_mode) else "existing"
        raise CampaignHandoffError(f"handoff destination is {kind}; refusing overwrite")
    return target


def publish_handoff(
    destination: str | Path,
    markdown: str,
    *,
    repo_root: Path,
) -> HandoffPublication:
    if not isinstance(markdown, str) or not markdown:
        raise CampaignHandoffError("handoff Markdown must be nonempty text")
    payload = markdown.encode("utf-8")
    target = _publication_target(destination, repo_root)
    try:
        published = publish_bytes_transactionally(target, payload, mode=0o600)
    except BaseException as error:
        raise CampaignHandoffError(f"handoff publication failed: {error}") from error
    if published.sha256 != hashlib.sha256(payload).hexdigest() or published.size != len(
        payload
    ):
        raise CampaignHandoffError("handoff publication identity mismatch")
    authenticated: AuthenticatedFile | None = None
    try:
        authenticated = AuthenticatedFile.open(
            target,
            label="published recovery campaign handoff",
            allow_resolved_symlink=False,
        )
        if (
            authenticated.sha256 != published.sha256
            or authenticated.size != published.size
        ):
            raise CampaignHandoffError(
                "handoff publication identity mismatch"
            )
        receipt = HandoffPublication(
            path=target,
            sha256=published.sha256,
            size=published.size,
        )
        object.__setattr__(receipt, "_authenticated", authenticated)
        object.__setattr__(
            receipt, "_publication_seal", _publication_seal(receipt)
        )
        authenticated = None
        return receipt
    except (OSError, ValueError) as error:
        raise CampaignHandoffError(
            f"handoff publication authentication failed: {error}"
        ) from error
    finally:
        if authenticated is not None:
            authenticated.close()


def render_publication_receipt(
    receipt: HandoffPublication,
    *,
    as_json: bool,
) -> str:
    _require_published_receipt(receipt)
    _reauthenticate_published_file(receipt)
    payload = {
        "path": str(receipt.path),
        "sha256": receipt.sha256,
        "size": receipt.size,
    }
    if as_json:
        return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    return (
        f"Handoff: {receipt.path}\n"
        f"SHA-256: {receipt.sha256}\n"
        f"Bytes: {receipt.size}\n"
    )


def status_payload(observation: CampaignObservation) -> dict[str, Any]:
    """Return the stable public JSON shape for one status observation."""

    return {
        "campaign": observation.campaign,
        "experiment": observation.experiment_name,
        "lock": {
            "state": observation.lock_state,
            "owner_known": observation.lock_owner_known,
        },
        "active_processes": [
            {
                "pid": process.pid,
                "elapsed_seconds": process.elapsed_seconds,
                "command": list(process.command),
                "experiment": process.experiment_name,
                "transition": process.transition_name,
            }
            for process in observation.active_processes
        ],
        "groups": {
            "state": observation.recovered_groups_state,
            "observed": (
                observation.recovered_groups
                if observation.recovered_groups_state == "present"
                else None
            ),
            "expected": observation.expected_groups,
        },
        "artifact_links": {
            "state": observation.artifact_links_state,
            "observed": (
                observation.artifact_links
                if observation.artifact_links_state == "present"
                else None
            ),
            "expected": observation.expected_artifact_links,
        },
        "manifest": {
            "state": observation.manifest_state,
            "file_sha256": observation.manifest_file_sha256,
            "body_sha256": observation.manifest_body_sha256,
        },
        "throughput_groups_per_second": observation.throughput_groups_per_second,
        "eta_seconds": observation.eta_seconds,
        "contradictions": list(observation.contradictions),
    }


def render_status(observation: CampaignObservation, *, as_json: bool = False) -> str:
    """Render a complete status without inventing values for missing evidence."""

    if as_json:
        return json.dumps(
            status_payload(observation),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    owner = "yes" if observation.lock_owner_known else "no"
    lines = [
        f"Campaign: {observation.campaign}",
        f"Experiment: {observation.experiment_name or 'absent'}",
        f"Lock: {observation.lock_state} (owner known: {owner})",
        "Processes:",
    ]
    if observation.active_processes:
        lines.extend(
            "  "
            + f"pid={process.pid} elapsed_seconds="
            + (
                str(process.elapsed_seconds)
                if process.elapsed_seconds is not None
                else "absent"
            )
            + " command="
            + shlex_join(process.command)
            for process in observation.active_processes
        )
    else:
        lines.append("  absent")
    lines.extend(
        [
            "Recovered groups: "
            + (
                f"{observation.recovered_groups}/{observation.expected_groups}"
                if observation.recovered_groups_state == "present"
                else (
                    f"{observation.recovered_groups_state} "
                    f"(expected {observation.expected_groups})"
                )
            ),
            "Artifact links: "
            + (
                f"{observation.artifact_links}/{observation.expected_artifact_links}"
                if observation.artifact_links_state == "present"
                else (
                    f"{observation.artifact_links_state} "
                    f"(expected {observation.expected_artifact_links})"
                )
            ),
            f"Manifest: {observation.manifest_state}",
            f"Manifest file SHA-256: {observation.manifest_file_sha256 or 'absent'}",
            f"Manifest body SHA-256: {observation.manifest_body_sha256 or 'absent'}",
            "Throughput: "
            + (
                f"{observation.throughput_groups_per_second:.12g} groups/second"
                if observation.throughput_groups_per_second is not None
                else "absent"
            ),
            "ETA: "
            + (
                f"{observation.eta_seconds:.3f} seconds"
                if observation.eta_seconds is not None
                else "absent"
            ),
            "Contradictions:",
        ]
    )
    lines.extend(
        (f"  - {item}" for item in observation.contradictions)
        if observation.contradictions
        else ("  none",)
    )
    return "\n".join(lines) + "\n"


def shlex_join(argv: tuple[str, ...]) -> str:
    """Use the standard lossless command renderer without shell execution."""

    import shlex

    return shlex.join(argv)


__all__ = [
    "CampaignHandoff",
    "CampaignHandoffError",
    "HandoffPublication",
    "RepositoryIdentity",
    "build_campaign_handoff",
    "handoff_fingerprint",
    "handoff_payload",
    "observe_repository_identity",
    "publish_handoff",
    "render_handoff_json",
    "render_handoff_markdown",
    "render_handoff_terminal",
    "render_publication_receipt",
    "render_status",
    "status_payload",
]
