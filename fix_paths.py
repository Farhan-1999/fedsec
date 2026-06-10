import pathlib, re
root = pathlib.Path(".")
bad = "."
n = 0
for p in list(root.rglob("*.py")):
    if ".venv" in p.parts or "__pycache__" in p.parts:
        continue
    t = p.read_text(encoding="utf-8")
    if bad in t:
        # replace the hardcoded absolute path with a repo-root-relative resolve
        t2 = t.replace('"."', 'str(pathlib.Path(__file__).resolve().parents[2] if "plots" in __file__ else pathlib.Path(__file__).resolve().parents[1])')
        # simpler: just strip the prefix so paths become relative
        t2 = t.replace("", "").replace(".", ".")
        p.write_text(t2, encoding="utf-8")
        n += 1
        print("fixed", p)
print(f"patched {n} files")
