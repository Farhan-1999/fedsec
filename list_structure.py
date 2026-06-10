import pathlib
root = pathlib.Path('.')
skip = {'.git', '__pycache__', '.venv', 'node_modules', '.pytest_cache', 'data'}
lines = []
for p in sorted(root.rglob('*'), key=lambda x: str(x.relative_to(root)).lower()):
    if any(s in p.parts for s in skip):
        continue
    if p.name.endswith('.egg-info') or '.egg-info' in str(p):
        continue
    depth = len(p.relative_to(root).parts) - 1
    indent = '  ' * depth
    if p.is_dir():
        lines.append(f'{indent}{p.name}/')
    else:
        lines.append(f'{indent}{p.name}  ({p.stat().st_size} bytes)')
out = '\n'.join(lines)
pathlib.Path('codebase_structure.txt').write_text(out, encoding='utf-8')
print('wrote codebase_structure.txt with', len(lines), 'entries')
