import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Finding:
    severity: str  # "high" | "medium" | "low"
    title: str
    evidence: Tuple[str, ...]
    recommendation: Tuple[str, ...]


# Launch-storm thresholds (tunable). These are intentionally simple heuristics:
# a storm is "many launches/sec" with "tiny median kernel".
LAUNCH_STORM_THRESHOLDS: Mapping[str, float] = {
    "launches_per_s_threshold_1": 50_000.0,
    "p50_kernel_us_threshold_1": 10.0,
    "launches_per_s_threshold_2": 100_000.0,
    "p50_kernel_us_threshold_2": 20.0,
}


def classify_launch_storm(launches_per_s: float, p50_kernel_us: float) -> bool:
    t = LAUNCH_STORM_THRESHOLDS
    return bool(
        (float(launches_per_s) >= float(t["launches_per_s_threshold_1"]) and float(p50_kernel_us) <= float(t["p50_kernel_us_threshold_1"]))
        or (float(launches_per_s) >= float(t["launches_per_s_threshold_2"]) and float(p50_kernel_us) <= float(t["p50_kernel_us_threshold_2"]))
    )


def load_phase_map(path: Optional[str]) -> Mapping[str, Sequence[str]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("phase map must be a JSON object: {phase: [patterns...]}")
    out: Dict[str, List[str]] = {}
    for phase, patterns in data.items():
        if not isinstance(phase, str):
            continue
        if isinstance(patterns, str):
            out[phase] = [patterns]
        elif isinstance(patterns, list):
            out[phase] = [str(p) for p in patterns]
        else:
            out[phase] = [str(patterns)]
    return out


def map_range_to_phase(range_name: str, phase_map: Mapping[str, Sequence[str]]) -> Optional[str]:
    if not phase_map:
        return None
    s = (range_name or "").strip()
    s_low = s.lower()
    for phase, patterns in phase_map.items():
        for p in patterns:
            p = str(p)
            if p.startswith("re:"):
                if re.search(p[3:], s):
                    return phase
            else:
                if p.lower() in s_low:
                    return phase
    return None


def nvtx_phase_breakdown(nvtx: Mapping[str, Any], *, phase_map: Mapping[str, Sequence[str]]) -> Mapping[str, Any]:
    ranges = nvtx.get("ranges") or []
    phase_totals: Dict[str, float] = {}
    unknown_total = 0.0
    total = 0.0
    for r in ranges:
        name = str(r.get("range_name") or "")
        ms = float(r.get("total_time_ms") or 0.0)
        total += ms
        phase = map_range_to_phase(name, phase_map)
        if phase is None:
            unknown_total += ms
        else:
            phase_totals[phase] = phase_totals.get(phase, 0.0) + ms

    phases: List[Dict[str, Any]] = []
    for phase, ms in sorted(phase_totals.items(), key=lambda kv: kv[1], reverse=True):
        phases.append({"phase": phase, "total_time_ms": ms, "pct_of_nvtx_total": (ms / total * 100.0) if total else 0.0})

    if unknown_total:
        phases.append(
            {"phase": "unmapped", "total_time_ms": unknown_total, "pct_of_nvtx_total": (unknown_total / total * 100.0) if total else 0.0}
        )

    return {"present": bool(ranges), "nvtx_total_time_ms": total, "phases": phases}


def nvtx_kernel_phase_breakdown(
    nvtx_kernel: Mapping[str, Any], *, phase_map: Mapping[str, Sequence[str]]
) -> Mapping[str, Any]:
    """Map NVTX range names into phases, summing *kernel time* attributed to those ranges."""

    if not phase_map:
        return {"present": False, "phases": [], "notes": ["No phase_map provided."]}
    if not nvtx_kernel.get("present"):
        return {"present": False, "phases": [], "notes": ["NVTX→kernel attribution not present."]}

    ranges = nvtx_kernel.get("ranges") or []
    if not ranges:
        return {"present": False, "phases": [], "notes": ["No NVTX-attributed kernel ranges."]}

    phase_totals: Dict[str, float] = {}
    unknown_total = 0.0
    total = 0.0
    for r in ranges:
        name = str(r.get("range_name") or "")
        ms = float(r.get("total_kernel_time_ms") or 0.0)
        total += ms
        phase = map_range_to_phase(name, phase_map)
        if phase is None:
            unknown_total += ms
        else:
            phase_totals[phase] = phase_totals.get(phase, 0.0) + ms

    phases: List[Dict[str, Any]] = []
    for phase, ms in sorted(phase_totals.items(), key=lambda kv: kv[1], reverse=True):
        phases.append(
            {"phase": phase, "total_kernel_time_ms": ms, "pct_of_attributed_kernel_time": (ms / total * 100.0) if total else 0.0}
        )
    if unknown_total:
        phases.append(
            {
                "phase": "unmapped",
                "total_kernel_time_ms": unknown_total,
                "pct_of_attributed_kernel_time": (unknown_total / total * 100.0) if total else 0.0,
            }
        )

    return {"present": True, "attributed_kernel_time_ms": total, "phases": phases}


def generate_findings(
    *,
    top_kernels: Mapping[str, Any],
    launch_storm: Mapping[str, Any],
    sync: Mapping[str, Any],
    gpu_idle: Mapping[str, Any],
    nvtx: Mapping[str, Any],
    nvtx_phases: Optional[Mapping[str, Any]] = None,
    nvtx_kernel_phases: Optional[Mapping[str, Any]] = None,
) -> List[Finding]:
    findings: List[Finding] = []

    kernels = top_kernels.get("kernels") or []
    if kernels:
        top = kernels[0]
        pct = float(top.get("pct_total_kernel_time") or 0.0)
        if pct >= 50.0:
            findings.append(
                Finding(
                    severity="high",
                    title="Top kernel dominates GPU time",
                    evidence=(
                        "Top kernel `{}` is {:.1f}% of total kernel time.".format(top.get("kernel_name"), pct),
                        "Total {:.1f} ms across {} calls; avg {:.2f} us.".format(
                            float(top.get("total_time_ms") or 0.0),
                            int(top.get("call_count") or 0),
                            float(top.get("avg_duration_us") or 0.0),
                        ),
                    ),
                    recommendation=(
                        "Inspect this kernel in Nsight Compute and verify occupancy/memory bottlenecks.",
                        "Consider kernel fusion, alternative algorithms, or reducing launch count for this hotspot.",
                    ),
                )
            )
        elif pct >= 25.0:
            findings.append(
                Finding(
                    severity="medium",
                    title="Single kernel is a large share of GPU time",
                    evidence=("Top kernel `{}` is {:.1f}% of total kernel time.".format(top.get("kernel_name"), pct),),
                    recommendation=("Focus optimization effort on this kernel first.",),
                )
            )

    lps = float(launch_storm.get("launches_per_s") or 0.0)
    p50_us = launch_storm.get("p50_kernel_us") or launch_storm.get("median_kernel_us")
    if p50_us is not None:
        p50_us_f = float(p50_us)
        is_storm = launch_storm.get("is_launch_storm")
        if bool(is_storm) or (lps >= 50_000.0 and p50_us_f <= 10.0):
            findings.append(
                Finding(
                    severity="high",
                    title="Kernel launch storm detected",
                    evidence=(
                        "{:.0f} launches/s over {:.3f}s window; median kernel {:.2f} us.".format(
                            lps, float(launch_storm.get("window_s") or 0.0), p50_us_f
                        ),
                    ),
                    recommendation=(
                        "Reduce per-token micro-kernels: fuse pointwise ops, increase work per launch, or use persistent kernels.",
                        "For inference, verify attention/MLP paths are using efficient fused kernels for your dtype/shape.",
                    ),
                )
            )
        elif lps >= 100_000.0:
            findings.append(
                Finding(
                    severity="medium",
                    title="Very high kernel launch rate",
                    evidence=("{:.0f} launches/s; p50 kernel {:.2f} us.".format(lps, p50_us_f),),
                    recommendation=("Investigate tiny-kernel suspects and operator fusion opportunities.",),
                )
            )

    sync_calls = sync.get("sync_calls") or []
    if sync_calls:
        top_sync = sync_calls[0]
        total_ms = sum(float(c.get("total_time_ms") or 0.0) for c in sync_calls)
        findings.append(
            Finding(
                severity="medium" if total_ms >= 1.0 else "low",
                title="CPU↔GPU synchronization detected (runtime API)",
                evidence=(
                    "Top sync-like call `{}` total {:.2f} ms across {} calls.".format(
                        top_sync.get("api_name"), float(top_sync.get("total_time_ms") or 0.0), int(top_sync.get("call_count") or 0)
                    ),
                    "All sync-like calls total {:.2f} ms.".format(total_ms),
                ),
                recommendation=(
                    "Look for `cudaDeviceSynchronize` / stream waits in your serving loop and remove unnecessary barriers.",
                    "Prefer async launches and overlap CPU work with GPU execution; avoid per-token synchronization.",
                ),
            )
        )

    devices = gpu_idle.get("devices") or []
    if devices:
        worst = max(devices, key=lambda d: float(d.get("idle_pct_of_window") or 0.0))
        idle_pct = float(worst.get("idle_pct_of_window") or 0.0)
        if idle_pct >= 20.0:
            findings.append(
                Finding(
                    severity="high" if idle_pct >= 40.0 else "medium",
                    title="Significant GPU idle gaps",
                    evidence=(
                        "GPU {} idle {:.1f}% of observed window ({:.1f} ms / {:.1f} ms).".format(
                            worst.get("device_id"),
                            idle_pct,
                            float(worst.get("idle_ms") or 0.0),
                            float(worst.get("window_ms") or 0.0),
                        ),
                    ),
                    recommendation=(
                        "If kernels are short, CPU scheduling/launch overhead can create gaps; batch more work per launch.",
                        "Check for sync calls and data dependencies that serialize work; increase overlap across streams.",
                    ),
                )
            )

    if nvtx.get("ranges"):
        findings.append(
            Finding(
                severity="low",
                title="NVTX ranges present",
                evidence=("Found {} NVTX range names.".format(len(nvtx.get("ranges") or [])),),
                recommendation=("Use NVTX phase totals in the report to focus prefill vs decode vs sampling.",),
            )
        )
        if nvtx_phases and nvtx_phases.get("phases"):
            phases = nvtx_phases["phases"]
            top_phase = phases[0]
            pct = float(top_phase.get("pct_of_nvtx_total") or 0.0)
            if pct >= 70.0:
                findings.append(
                    Finding(
                        severity="medium",
                        title="One NVTX phase dominates",
                        evidence=("Phase `{}` is {:.1f}% of NVTX time.".format(top_phase.get("phase"), pct),),
                        recommendation=("Prioritize optimization and instrumentation in this phase first.",),
                    )
                )

        if nvtx_kernel_phases and nvtx_kernel_phases.get("phases"):
            phases2 = nvtx_kernel_phases["phases"]
            top2 = phases2[0]
            pct2 = float(top2.get("pct_of_attributed_kernel_time") or 0.0)
            if pct2 >= 70.0:
                findings.append(
                    Finding(
                        severity="medium",
                        title="One phase dominates attributed GPU kernel time",
                        evidence=("Phase `{}` is {:.1f}% of NVTX-attributed kernel time.".format(top2.get("phase"), pct2),),
                        recommendation=("Optimize the dominant phase first; if this is decode, focus on per-token overhead and fusion.",),
                    )
                )

    return findings


def findings_to_dict(findings: Sequence[Finding]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for f in findings:
        out.append(
            {
                "severity": f.severity,
                "title": f.title,
                "evidence": list(f.evidence),
                "recommendation": list(f.recommendation),
            }
        )
    return out

