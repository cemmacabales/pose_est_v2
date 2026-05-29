#!/usr/bin/env python3
"""Check that required dependencies are installed and importable."""

import sys
import importlib

# Required dependencies: (import_name, min_version_or_None, pip_name_for_requirements)
DEPS = [
    ("tensorflow", "2.13", "tensorflow"),
    ("tflite_runtime", None, "tflite-runtime"),
    ("tensorflow_hub", None, "tensorflow-hub"),
    ("cv2", None, "opencv-python"),
    ("numpy", None, "numpy"),
    ("PIL", None, "Pillow"),
    ("tkinter", None, "tk"),
]


def get_version(module):
    """Return version string if available, otherwise None."""
    if hasattr(module, "__version__"):
        return module.__version__
    if hasattr(module, "VERSION"):
        return module.VERSION
    if hasattr(module, "version"):
        ver = module.version
        if isinstance(ver, str):
            return ver
    return None


def version_satisfies(installed, required):
    """Simple PEP-440-like version comparison for >= required."""
    # Split into numeric parts
    def to_tuple(v):
        parts = []
        for p in v.split("."):
            num = ""
            for ch in p:
                if ch.isdigit():
                    num += ch
                else:
                    break
            parts.append(int(num) if num else 0)
        return tuple(parts)

    return to_tuple(installed) >= to_tuple(required)


def main():
    failed = []
    req_lines = []

    # Python version check
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info < (3, 9):
        print(f"Python      {py_version}  FAIL (requires >= 3.9)")
        failed.append("python")
    else:
        print(f"Python      {py_version}  PASS")

    for import_name, min_ver, pip_name in DEPS:
        try:
            mod = importlib.import_module(import_name)
            ver = get_version(mod)
            ver_str = ver if ver else "unknown"
            if min_ver and ver:
                ok = version_satisfies(ver, min_ver)
            else:
                ok = True
            status = "PASS" if ok else f"FAIL (requires >= {min_ver})"
            if not ok:
                failed.append(pip_name)
            print(f"{import_name:<15} {ver_str:<15} {status}")
            if ver:
                req_lines.append(f"{pip_name}=={ver}")
            else:
                req_lines.append(pip_name)
        except Exception as e:
            print(f"{import_name:<15} {'N/A':<15} FAIL ({e})")
            failed.append(pip_name)
            req_lines.append(pip_name)

    if failed:
        print("\nMISSING DEPS — fix the above before continuing")
    else:
        print("\nALL DEPS OK — ready to proceed")

    with open("requirements.txt", "w") as f:
        for line in req_lines:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
