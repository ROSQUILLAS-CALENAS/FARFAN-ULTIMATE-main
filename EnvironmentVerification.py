# verify_environment.py
# Runtime check:
# 1) Verify each installed package's Requires-Python matches your interpreter
# 2) Verify inter-package dependency specifiers vs installed versions
#
# # # # No external packages required. Uses importlib.metadata from stdlib.  # Module not found  # Module not found  # Module not found

import sys
import re
# # # from typing import Dict, List, Tuple, Optional  # Module not found  # Module not found  # Module not found
# # # from importlib.metadata import distributions, requires, PackageNotFoundError  # Module not found  # Module not found  # Module not found

def parse_version_tuple(s: str) -> Tuple[int, ...]:
    parts = []
    for token in re.split(r"[.\-+_]", s):
        if token.isdigit():
            parts.append(int(token))
        else:
            break
    return tuple(parts) if parts else (0,)

def compare_versions(a: str, b: str) -> int:
    ta, tb = parse_version_tuple(a), parse_version_tuple(b)
    n = max(len(ta), len(tb))
    ta += (0,) * (n - len(ta))
    tb += (0,) * (n - len(tb))
    if ta < tb: return -1
    if ta > tb: return 1
    return 0

def eval_specifier(spec: str, installed: str) -> bool:
    m = re.match(r"^(==|!=|>=|<=|>|<)\s*([0-9][^,\s;]*)$", spec.strip())
    if not m:
        return True
    op, ver = m.groups()
    cmp = compare_versions(installed, ver)
    return {
        "==": cmp == 0,
        "!=": cmp != 0,
        ">=": cmp >= 0,
        "<=": cmp <= 0,
        ">":  cmp > 0,
        "<":  cmp < 0
    }[op]

def eval_specifier_set(specs: str, installed: str) -> bool:
    if not specs:
        return True
    return all(eval_specifier(s.strip(), installed) for s in specs.split(",") if s.strip())

def parse_requirement(req_str: str) -> Tuple[str, str, Optional[str]]:
    parts = req_str.split(";", 1)
    core = parts[0].strip()
    marker = parts[1].strip() if len(parts) == 2 else None
    core = re.sub(r"\[.*?\]", "", core)             # remove extras
    core = re.sub(r"\s*\(([^)]+)\)", r" \1", core)  # '(>=1.0)' -> ' >=1.0'
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*(.*)$", core)
    if not m:
        return (core, "", marker)
    name, spec_set = m.groups()
    return (name.strip(), spec_set.strip(), marker)

def eval_marker(marker: Optional[str]) -> bool:
    if not marker:
        return True
    m = re.search(r"python_version\s*(==|!=|>=|<=|>|<)\s*\"([0-9.]+)\"", marker)
    if not m:
        return True
    op, ver = m.groups()
    current = ".".join(map(str, sys.version_info[:3]))
    cmp = compare_versions(current, ver)
    return {
        "==": cmp == 0,
        "!=": cmp != 0,
        ">=": cmp >= 0,
        "<=": cmp <= 0,
        ">":  cmp > 0,
        "<":  cmp < 0
    }[op]

def verify_environment():
    print("=" * 80)
    print("PART 2: RUNTIME ENVIRONMENT VERIFICATION")
    print("=" * 80)

    py_ver = ".".join(map(str, sys.version_info[:3]))
    print(f"🐍 Python version: {py_ver}")
    print("-" * 80)

    dists = list(distributions())
    installed: Dict[str, str] = {}
    requires_python_map: Dict[str, str] = {}
    reqs_map: Dict[str, List[str]] = {}

    for dist in dists:
        name = (dist.metadata.get("Name") or dist.metadata.get("name") or "").strip()
        if not name:
            continue
        ver = (dist.version or "").strip()
        installed[name] = ver

        rp = (dist.metadata.get("Requires-Python") or "").strip()
        if rp:
            requires_python_map[name] = rp

        try:
            reqs_map[name] = list(requires(name) or [])
        except (PackageNotFoundError, Exception):
            reqs_map[name] = []

    issues = 0

    print("🔍 Checking Requires-Python constraints...")
    for name, rp in sorted(requires_python_map.items(), key=lambda x: x[0].lower()):
        ok = all(eval_specifier(p.strip(), py_ver) for p in rp.split(",") if p.strip())
        if not ok:
            print(f"  🚨 Python incompatibility: {name}=={installed.get(name, '?')} requires Python '{rp}', current is {py_ver}")
            issues += 1
    if issues == 0:
        print("  ✅ All packages compatible with current Python version.")
    print("-" * 80)

    print("🔍 Checking inter-package dependency constraints...")
    for name, req_list in sorted(reqs_map.items(), key=lambda x: x[0].lower()):
        for req_str in req_list:
            dep_name, spec_set, marker = parse_requirement(req_str)
            if not eval_marker(marker):
                continue

            dep_installed_ver = None
            for k, v in installed.items():
                if k.lower() == dep_name.lower():
                    dep_installed_ver = v
                    break

            if dep_installed_ver is None:
                print(f"  🚨 Missing dependency: {name}=={installed.get(name, '?')} requires '{dep_name}{(' ' + spec_set) if spec_set else ''}', but it is not installed.")
                issues += 1
                continue

            spec_set_norm = spec_set.replace(" ", "")
            if spec_set_norm and not eval_specifier_set(spec_set_norm, dep_installed_ver):
                print(f"  🚨 Version conflict: {name}=={installed.get(name, '?')} requires '{dep_name}{spec_set}', but installed is {dep_name}=={dep_installed_ver}.")
                issues += 1

    if issues == 0:
        print("  ✅ No inter-package version conflicts detected.")
    print("-" * 80)

    print("📋 Summary:")
    print(f"  Total installed packages scanned: {len(installed)}")
    if issues == 0:
        print("  ✅ Environment is consistent and compatible.")
    else:
        print(f"  ❌ Found {issues} compatibility issue(s). Adjust pinned versions in requirements.txt and reinstall.")

if __name__ == "__main__":
    verify_environment()
