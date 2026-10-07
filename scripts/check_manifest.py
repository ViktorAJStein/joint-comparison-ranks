"""Verify the exact allowlist and SHA256 payload; caches/generated runs are ignored."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    records={}
    for line in (ROOT/'manifest.sha256').read_text().splitlines():
        checksum,name=line.split('  ',1)
        if Path(name).is_absolute() or '..' in Path(name).parts or name in records:raise ValueError('Unsafe/duplicate manifest')
        records[name]=checksum
        path=ROOT/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('Manifest mismatch: '+name)
    files=set()
    for path in ROOT.rglob('*'):
        if any(p in ('.git','.venv','__pycache__','.pytest_cache','runs','downloads','reproduced-figures','imgs','dist','build') or p.endswith('.egg-info') for p in path.relative_to(ROOT).parts):continue
        if path.is_file() and path.name!='manifest.sha256':files.add(str(path.relative_to(ROOT)))
    if files!=set(records):raise ValueError('Unlisted/missing files: '+str(files^set(records)))
    print(f'PASS: {len(records)} allowlisted files match SHA256')

if __name__=='__main__':main()
