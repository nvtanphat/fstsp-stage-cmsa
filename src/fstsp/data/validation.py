from fstsp.domain.instance import FSTSPInstance


def validate_instance(instance: FSTSPInstance) -> list[str]:
    issues: list[str] = []
    if instance.n < 1:
        issues.append("instance must contain at least one customer")
    if instance.drone_endurance <= instance.recovery_time:
        issues.append("drone endurance should exceed recovery time")
    return issues
