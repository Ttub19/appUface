"""
Split a trained .keras file into <= 25 MB parts + a .sha256 file, the way
backend/models stores models (GitHub rejects files over 100 MB and warns over 50 MB).
The server joins the parts back on start-up and checks the checksum.

    python scripts/split_model.py path/to/face_shape_female.keras path/to/face_shape_male.keras

Writes into backend/models/ and removes any old parts of the same model first.
"""
import hashlib
import sys
from pathlib import Path

PART_SIZE = 25 * 1024 * 1024
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def split(src: Path, out_dir: Path = MODELS_DIR, part_size: int = PART_SIZE) -> list:
    data = src.read_bytes()
    for old in out_dir.glob(f"{src.name}.part*"):
        old.unlink()
    parts = []
    for i in range(0, len(data), part_size):
        part = out_dir / f"{src.name}.part{i // part_size:02d}"
        part.write_bytes(data[i:i + part_size])
        parts.append(part)
    (out_dir / f"{src.name}.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n")
    whole = out_dir / src.name
    if whole.exists() and whole.resolve() != src.resolve():
        whole.unlink()          # stale joined copy from an older model
    return parts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        parts = split(Path(arg))
        print(f"{arg}: {len(parts)} parts -> {MODELS_DIR}")
