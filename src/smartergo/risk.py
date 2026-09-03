def ergonomic_risk(neck_angle: float, trunk_angle: float, elbow_angle: float) -> dict[str, object]:
    score = 1
    score += 2 if neck_angle > 30 else 1 if neck_angle > 15 else 0
    score += 3 if trunk_angle > 60 else 2 if trunk_angle > 20 else 0
    score += 1 if elbow_angle < 60 or elbow_angle > 120 else 0
    level = "high" if score >= 6 else "medium" if score >= 3 else "low"
    return {"score": score, "level": level}
