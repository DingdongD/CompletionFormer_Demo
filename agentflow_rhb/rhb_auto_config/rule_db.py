import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from .schema import DEFAULT_RULE_DB, Rule
except ImportError:  # pragma: no cover - supports direct script-style imports
    from schema import DEFAULT_RULE_DB, Rule


class RuleDB:
    def __init__(self, rules: Iterable[Rule]):
        self.rules = list(rules)

    @classmethod
    def load(cls, path: Path = DEFAULT_RULE_DB) -> "RuleDB":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(Rule.from_dict(item) for item in data["rules"])

    def by_decision(self) -> Dict[str, List[Rule]]:
        groups: Dict[str, List[Rule]] = defaultdict(list)
        for rule in self.rules:
            groups[rule.decision].append(rule)
        return dict(groups)

    def by_kind(self) -> Dict[str, List[Rule]]:
        groups: Dict[str, List[Rule]] = defaultdict(list)
        for rule in self.rules:
            groups[rule.kind].append(rule)
        return dict(groups)

    def summary_lines(self) -> List[str]:
        kind_counts = Counter(rule.kind for rule in self.rules)
        decision_counts = Counter(rule.decision for rule in self.rules)
        lines = [
            f"rules: {len(self.rules)}",
            "by kind:",
        ]
        lines.extend(f"  {kind}: {count}" for kind, count in sorted(kind_counts.items()))
        lines.append("by decision:")
        lines.extend(f"  {decision}: {count}" for decision, count in sorted(decision_counts.items()))
        return lines

    def risk_rules(self) -> List[Rule]:
        return [rule for rule in self.rules if "high" in rule.risk.lower() or rule.decision in {"forbid", "host"}]
