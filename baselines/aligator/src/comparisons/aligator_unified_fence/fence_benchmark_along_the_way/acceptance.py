"""Experiment gate; physical audit residuals are unchanged."""
THRESHOLD = 2e-3

def classify(audit):
    audit["passed_strict_1e3"] = bool(audit.get("finite", False) and max(audit.get("node_max", float("inf")), audit.get("fine_max", float("inf"))) <= 1e-3)
    audit["acceptance_threshold"] = THRESHOLD
    audit["passed"] = bool(audit.get("finite", False) and max(audit.get("node_max", float("inf")), audit.get("fine_max", float("inf"))) <= THRESHOLD)
    return audit

def metric_gate(metric, threshold=THRESHOLD):
    metric["acceptance_threshold"] = threshold
    metric["passed_strict_1e3"] = bool(metric.get("audit_valid", True) and metric["max_violation"] <= 1e-3)
    metric["passed"] = bool(metric.get("audit_valid", True) and metric["max_violation"] <= threshold)
    return metric
