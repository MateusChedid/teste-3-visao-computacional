#!/usr/bin/env python3
"""
result_reader.py — Interpreta detecções YOLOv8 → resultado da jogada.

NOTA: d10_0 representa o valor ZERO (não dez).
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class DieResult:
    die_type:   str
    face:       int       # valor numérico literal (d10: 0 é zero)
    confidence: float
    bbox:       Tuple

    @property
    def display_value(self) -> str:
        return str(self.face)

    @property
    def numeric_value(self) -> int:
        return self.face

    def __str__(self):
        return f"{self.die_type}={self.display_value} ({self.confidence:.0%})"


@dataclass
class RollResult:
    dice:    List[DieResult] = field(default_factory=list)
    unknown: int             = 0

    def by_type(self) -> Dict[str, List[DieResult]]:
        groups: Dict[str, List[DieResult]] = {}
        for die in self.dice:
            groups.setdefault(die.die_type, []).append(die)
        return groups

    def total(self) -> int:
        return sum(d.numeric_value for d in self.dice)

    def summary(self) -> str:
        parts = []
        for dtype, results in sorted(self.by_type().items()):
            values = [r.display_value for r in results]
            parts.append(f"{len(results)}{dtype}: [{', '.join(values)}]")
        if self.unknown:
            parts.append(f"{self.unknown}× ilegível")
        total_str = f"  →  total = {self.total()}" if self.dice else ""
        return "  |  ".join(parts) + total_str

    def to_dict(self) -> dict:
        return {
            "dice": [{"type": d.die_type, "face": d.numeric_value,
                      "confidence": round(d.confidence, 4), "bbox": d.bbox}
                     for d in self.dice],
            "by_type": {dt: [r.numeric_value for r in rs]
                        for dt, rs in self.by_type().items()},
            "total":   self.total(),
            "unknown": self.unknown,
        }


_CLASS_RE = re.compile(r"^(d\d+)_(\d+)$")


def parse_class_name(class_name: str) -> Optional[Tuple[str, int]]:
    m = _CLASS_RE.match(class_name.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union > 0 else 0.0


def interpret_detections(detections: List[Tuple],
                         conf_threshold: float = 0.45,
                         iou_merge_threshold: float = 0.3) -> RollResult:
    filtered = [(cls, conf, bbox) for cls, conf, bbox in detections
                if conf >= conf_threshold]
    filtered.sort(key=lambda x: x[1], reverse=True)

    kept = []
    for det in filtered:
        if not any(_iou(det[2], k[2]) > iou_merge_threshold for k in kept):
            kept.append(det)

    roll = RollResult()
    for cls_name, conf, bbox in kept:
        if cls_name == "unknown":
            roll.unknown += 1
            continue
        parsed = parse_class_name(cls_name)
        if parsed is None:
            roll.unknown += 1
            continue
        die_type, face = parsed
        roll.dice.append(DieResult(die_type=die_type, face=face,
                                   confidence=conf, bbox=bbox))

    roll.dice.sort(key=lambda d: d.bbox[0])
    return roll


if __name__ == "__main__":
    # Teste rápido incluindo d12 e d10_0=0
    mock = [
        ("d6_4",   0.92, (50,  80, 150, 180)),
        ("d8_7",   0.85, (170, 75, 290, 195)),
        ("d10_0",  0.88, (310, 70, 420, 180)),   # 0 é zero
        ("d12_11", 0.91, (440, 60, 570, 200)),
        ("d20_17", 0.95, (590, 55, 720, 210)),
    ]
    roll = interpret_detections(mock)
    print("="*50)
    for die in roll.dice:
        print(f"  {die}")
    print(f"\n  {roll.summary()}")
    print("="*50)
    import json
    print(json.dumps(roll.to_dict(), indent=2, ensure_ascii=False))
