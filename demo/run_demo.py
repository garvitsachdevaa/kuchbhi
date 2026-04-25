"""Interactive demo runner — displays pre-computed demo assets for the pitch."""

from __future__ import annotations
import json
from pathlib import Path


def run_demo():
    assets_dir = Path("demo/assets")

    print("\n" + "="*70)
    print("SPINDLEFLOW RL -- HACKATHON DEMO")
    print("="*70)
    print()

    # Demo Moment 1
    m1_path = assets_dir / "demo_moment_1.json"
    if m1_path.exists():
        with open(m1_path) as f:
            m1 = json.load(f)
        print("DEMO MOMENT 1: Before/After Quality Gap")
        print("-"*70)
        print(f"Task: {m1['task']}\n")
        print("--- GENERALIST OUTPUT (no delegation) ---")
        print(m1["generalist_output"][:600])
        print("\n--- SPECIALIST-ROUTED OUTPUT ---")
        print(m1["specialist_output"][:1200])
        print()
        print("PITCH SCRIPT:")
        print(m1["demo_script"])
    else:
        print("[Run precompute_demo.py first to generate assets]")

    print("\n" + "="*70)
    print()

    # Demo Moment 2
    m2_path = assets_dir / "demo_moment_2.json"
    if m2_path.exists():
        with open(m2_path) as f:
            m2 = json.load(f)
        print("DEMO MOMENT 2: Policy Comparison (Quality vs Latency)")
        print("-"*70)
        qp = m2["quality_policy"]
        lp = m2["latency_policy"]
        print(f"Quality-Optimized Policy (latency_weight={qp['latency_weight']}):")
        print(f"  Specialists: {', '.join(qp['specialists_called'])}")
        print(f"  Mode: {qp['mode']}")
        print(f"  Estimated time: {qp['estimated_time_s']}s")
        print(f"  Path: {qp['delegation_path']}")
        print()
        print(f"Latency-Optimized Policy (latency_weight={lp['latency_weight']}):")
        print(f"  Specialists: {', '.join(lp['specialists_called'])}")
        print(f"  Mode: {lp['mode']}")
        print(f"  Estimated time: {lp['estimated_time_s']}s")
        print(f"  Path: {lp['delegation_path']}")
        print()
        print("PITCH SCRIPT:")
        print(m2["demo_script"])

    print("\n" + "="*70)


if __name__ == "__main__":
    run_demo()
