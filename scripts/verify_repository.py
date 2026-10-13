"""Check preserved-source hashes and package structure without importing agents."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    errors: list[str] = []
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        relative = Path(item["path"])
        path = (ROOT / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(ROOT):
            errors.append(f"Unsafe manifest path: {item['path']}")
            continue
        if not path.is_file():
            errors.append(f"Missing source: {item['path']}")
            continue
        data = path.read_bytes()
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            errors.append(f"Source bytes changed: {item['path']}")

    counts = {"python": 0, "toml": 0, "json": 0}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
            continue
        try:
            if path.suffix == ".py":
                ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
                counts["python"] += 1
            elif path.suffix == ".toml":
                tomllib.loads(path.read_text(encoding="utf-8-sig"))
                counts["toml"] += 1
            elif path.suffix == ".json" and ".vscode" not in path.parts:
                json.loads(path.read_text(encoding="utf-8-sig"))
                counts["json"] += 1
        except (SyntaxError, ValueError, UnicodeError) as exc:
            errors.append(f"Parse error in {path.relative_to(ROOT)}: {exc}")

    external: set[str] = set()
    for path in ROOT.rglob("algo_conf_*.toml"):
        project = path.parent.parent
        data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        for settings in data.values():
            if not isinstance(settings, dict):
                continue
            for key, dotted in settings.items():
                if not isinstance(dotted, str) or "." not in dotted:
                    continue
                module = dotted.rsplit(".", 1)[0]
                if module.startswith("agent_"):
                    target = project.joinpath(*module.split(".")).with_suffix(".py")
                    if not target.is_file():
                        errors.append(f"Missing {key} module: {target.relative_to(ROOT)}")
                elif module.startswith("tools."):
                    external.add(module)

    print(f"Preserved-source hashes checked: {len(manifest['files'])}")
    print(f"Parsed files: {counts}")
    print("Platform modules referenced: " + ", ".join(sorted(external)))
    print("VS Code launch.json uses JSON with comments and is excluded from strict JSON parsing.")
    if errors:
        for error in errors:
            print("ERROR: " + error, file=sys.stderr)
        return 1
    print("PASS: source integrity, syntax and local algorithm entry modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
