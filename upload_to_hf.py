# -*- coding: utf-8 -*-
"""
upload_to_hf.py
Memoni Audio Pipeline — Step 2: Build & Upload HuggingFace Dataset

Uploads raw MP3 files as an AudioFolder dataset to HuggingFace Hub.

Directory layout expected (produced by process_audio.py):
    <base-dir>/
        metadata_backup.json
        cleaned/
            <uid>_seg0001_cleaned.mp3
            ...

Repo layout on HuggingFace after upload:
    data/train/
        00042_seg0001_cleaned.mp3
        ...
    metadata.jsonl
    README.md

Example usage
─────────────
Single job:
    python upload_to_hf.py \
        --base-dir  /scratch/hamza95/memoni_audio/processed \
        --repo-id   hamzahanif/memoni-audio

Multi-job merge:
    python upload_to_hf.py \
        --base-dir  /scratch/hamza95/memoni_audio/processed \
        --repo-id   hamzahanif/memoni-audio \
        --merge

Token from env:
    export HF_TOKEN=$(cat /home/hamza95/.hf_token)

python /project/ctb-stelzer/hamza95/memoni_audio/upload_to_hf.py \
    --base-dir /scratch/hamza95/memoni_audio/processed \
    --repo-id  hamzahanif/memoni-audio \
    --merge

"""

import os
import json
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = "/project/ctb-stelzer/hamza95/memoni_audio"
LOG_DIR    = f"{SCRIPT_DIR}/logs"


# ─────────────────────────────────────────────────────────────────────────────
#  METADATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata_single(base_dir: Path) -> list[dict]:
    """Load segments from a single metadata_backup.json."""
    meta_path = base_dir / "metadata_backup.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata_backup.json not found in {base_dir}")
    with open(meta_path) as f:
        data = json.load(f)
    segments = data.get("segments", [])
    print(f"  Loaded {len(segments)} segment records from {meta_path}")
    return segments


def load_metadata_merged(base_dir: Path) -> list[dict]:
    """
    Scan base_dir for job_*/ sub-directories and merge all
    metadata_backup.json files. Deduplicates by unique_id.
    """
    all_segments: list[dict] = []
    seen_ids: set[str] = set()

    candidates = sorted(base_dir.glob("job_*/"))
    if not candidates:
        candidates = [base_dir]

    print(f"\nMerging metadata from {len(candidates)} job director(ies)...")

    for job_dir in candidates:
        meta_path = job_dir / "metadata_backup.json"
        if not meta_path.exists():
            print(f"  [WARN] No metadata_backup.json in {job_dir} — skipping.")
            continue
        with open(meta_path) as f:
            data = json.load(f)
        segs = data.get("segments", [])
        new = 0
        for seg in segs:
            uid = seg.get("unique_id", "")
            if uid and uid not in seen_ids:
                seen_ids.add(uid)
                all_segments.append(seg)
                new += 1
        print(f"  {job_dir.name}: {len(segs)} segments ({new} new after dedup)")

    print(f"\n  Total unique segments : {len(all_segments)}")
    return all_segments


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_segments(segments: list[dict]) -> list[dict]:
    """Filter out segments whose audio file no longer exists on disk."""
    valid = []
    missing = 0
    for seg in segments:
        path = seg.get("audio_path", "")
        if path and Path(path).exists():
            valid.append(seg)
        else:
            print(f"  [WARN] Missing: {path} — dropping {seg.get('unique_id')}")
            missing += 1
    if missing:
        print(f"\n  {missing} segment(s) dropped (missing files).")
    print(f"  {len(valid)} valid segment(s) remaining.")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET CARD
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_card(segments: list[dict], repo_id: str) -> str:
    total_duration_min = sum(
        s.get("speech_duration_seconds", 0) for s in segments
    ) / 60.0

    categories = sorted({s.get("category", "")    for s in segments if s.get("category")})
    platforms  = sorted({s.get("platform",  "")   for s in segments if s.get("platform")})
    channels   = sorted({s.get("channel_name", "") for s in segments if s.get("channel_name")})

    return f"""---
language:
- ar
license: other
task_categories:
- automatic-speech-recognition
- audio-classification
pretty_name: Memoni Audio
size_categories:
- 1K<n<10K
---

# Memoni Audio Dataset

Cleaned, voice-activity-detected speech segments collected from online video
sources for Arabic/regional dialect speech research.

## Statistics

| Metric | Value |
|---|---|
| Total segments | {len(segments):,} |
| Total speech duration | {total_duration_min:.1f} min ({total_duration_min/60:.1f} h) |
| Categories | {len(categories)} |
| Platforms | {', '.join(platforms) or 'N/A'} |
| Channels | {len(channels)} |

## Processing Pipeline

1. **Download** — YouTube audio downloaded as MP3.
2. **VAD** — TenVad at 16 kHz mono; original sample rate preserved for all audio ops.
3. **Chunking** — Speech regions < 30 s kept as-is; >= 30 s split into ~30 s sub-chunks.
4. **Denoising** — dpdfnet (`dpdfnet8_48khz_hr`, attn_limit_db=12).
5. **Export** — 192 kbps MP3, peak-normalised to 0 dBFS.

## Dataset Fields

| Field | Type | Description |
|---|---|---|
| `audio` | Audio | MP3 file |
| `unique_id` | string | Segment identifier (e.g. `00042_seg0003`) |
| `source_video_id` | string | Original video unique_id |
| `youtube_link` | string | Source YouTube URL |
| `category` | string | Content category |
| `channel_name` | string | Source channel name |
| `platform` | string | Source platform |
| `segment_start_sec` | float | Start offset in original audio (seconds) |
| `segment_end_sec` | float | End offset in original audio (seconds) |
| `speech_duration_seconds` | float | Clean speech duration of this segment |
| `sample_rate_hz` | int | Native sample rate |
| `channels` | int | Audio channels (1=mono, 2=stereo) |

## Categories

{', '.join(f'`{c}`' for c in categories) or 'N/A'}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
sample = ds[0]
print(sample["unique_id"], sample["speech_duration_seconds"])
```
"""


# ─────────────────────────────────────────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def upload_dataset(
    segments  : list[dict],
    repo_id   : str,
    token     : str,
    split     : str  = "train",
    private   : bool = False,
    card_text : str  = "",
):
    """
    Upload raw MP3 files + metadata.jsonl to HuggingFace as an AudioFolder
    dataset. The Hub stores the actual .mp3 files directly.

    Repo layout on HuggingFace:
        data/train/
            00042_seg0001_cleaned.mp3
            ...
        metadata.jsonl
        README.md
    """
    print(f"\n{'='*65}")
    print(f"Uploading to HuggingFace Hub (raw MP3 / AudioFolder)")
    print(f"  Repo   : {repo_id}")
    print(f"  Split  : {split}")
    print(f"  Private: {private}")
    print(f"  Files  : {len(segments)}")
    print(f"{'='*65}")

    api = HfApi(token=token)

    # ── create repo ───────────────────────────────────────────────────────────
    api.create_repo(
        repo_id   = repo_id,
        repo_type = "dataset",
        private   = private,
        exist_ok  = True,
        token     = token,
    )
    print(f"  Repo ready: https://huggingface.co/datasets/{repo_id}")

    # ── upload README ─────────────────────────────────────────────────────────
    if card_text:
        api.upload_file(
            path_or_fileobj = card_text.encode(),
            path_in_repo    = "README.md",
            repo_id         = repo_id,
            repo_type       = "dataset",
            token           = token,
        )
        print("  README.md uploaded.")

    # ── write metadata.jsonl ──────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        audio_stage = tmp / "audio"
        audio_stage.mkdir()

        jsonl_path = tmp / "metadata.jsonl"
        with open(jsonl_path, "w") as jf:
            for seg in segments:
                src   = Path(seg["audio_path"])
                fname = src.name   # e.g. 00042_seg0001_cleaned.mp3
                shutil.copy2(str(src), str(audio_stage / fname))

                record = {
                    "file_name":               f"data/{split}/{fname}",
                    "unique_id":               seg.get("unique_id",               ""),
                    "source_video_id":         seg.get("source_video_id",         ""),
                    "youtube_link":            seg.get("youtube_link",            ""),
                    "category":                seg.get("category",                ""),
                    "channel_name":            seg.get("channel_name",            ""),
                    "platform":                seg.get("platform",                ""),
                    "segment_start_sec":       seg.get("segment_start_sec",       0.0),
                    "segment_end_sec":         seg.get("segment_end_sec",         0.0),
                    "speech_duration_seconds": seg.get("speech_duration_seconds", 0.0),
                    "sample_rate_hz":          seg.get("sample_rate_hz",          0),
                    "channels":                seg.get("channels",                1),
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  metadata.jsonl written ({len(segments)} lines).")

        # ── upload metadata.jsonl ─────────────────────────────────────────────
        api.upload_file(
            path_or_fileobj = str(jsonl_path),
            path_in_repo    = "metadata.jsonl",
            repo_id         = repo_id,
            repo_type       = "dataset",
            token           = token,
        )
        print("  metadata.jsonl uploaded.")

        # ── upload MP3 folder ─────────────────────────────────────────────────
        print(f"  Uploading {len(segments)} MP3 files...")
        api.upload_folder(
            folder_path    = str(audio_stage),
            path_in_repo   = f"data/{split}",
            repo_id        = repo_id,
            repo_type      = "dataset",
            token          = token,
            commit_message = f"Upload {len(segments)} audio segments ({split} split)",
        )

    print(f"\nUpload complete!")
    print(f"  https://huggingface.co/datasets/{repo_id}")


# ─────────────────────────────────────────────────────────────────────────────
#  SLURM HELPER
# ─────────────────────────────────────────────────────────────────────────────

def generate_slurm_script(args, output_path: str):
    merge_flag   = "--merge"   if args.merge   else ""
    private_flag = "--private" if args.private else ""

    script = f"""#!/bin/bash
#SBATCH --job-name=memoni_upload
#SBATCH --account={args.account}
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output={LOG_DIR}/memoni_upload_%j.out
#SBATCH --error={LOG_DIR}/memoni_upload_%j.err

set -euo pipefail

module load python/3.11
source /scratch/hamza95/df_env/bin/activate

# Load HF token
export HF_TOKEN=$(cat /home/hamza95/.hf_token)

python {SCRIPT_DIR}/upload_to_hf.py \\
    --base-dir {args.base_dir} \\
    --repo-id  {args.repo_id} \\
    --split    {args.split} \\
    {merge_flag} \\
    {private_flag}
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(script)
    print(f"SLURM script written: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Memoni Step 2 — Upload raw MP3s as HuggingFace AudioFolder dataset"
    )
    p.add_argument("--base-dir",  required=True,
                   help="Base directory from process_audio.py")
    p.add_argument("--repo-id",   required=True,
                   help="HuggingFace repo id, e.g. 'hamzahanif/memoni-audio'")
    p.add_argument("--token",     default=None,
                   help="HuggingFace write token (default: reads $HF_TOKEN env var)")
    p.add_argument("--split",     default="train",
                   help="Dataset split name (default: train)")
    p.add_argument("--private",   action="store_true",
                   help="Create a private HuggingFace repo")
    p.add_argument("--merge",     action="store_true",
                   help="Merge metadata from job_*/ subdirectories under --base-dir")
    p.add_argument("--dry-run",   action="store_true",
                   help="Validate and build metadata but skip the upload step")
    p.add_argument("--account",   default="def-mdanning",
                   help="SBATCH --account value (for --generate-slurm)")
    p.add_argument("--generate-slurm", metavar="OUTPUT_SCRIPT", default=None,
                   help="Write a SLURM batch script to OUTPUT_SCRIPT and exit")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.generate_slurm:
        generate_slurm_script(args, args.generate_slurm)
        raise SystemExit(0)

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"ERROR: --base-dir does not exist: {base_dir}")

    # ── token ─────────────────────────────────────────────────────────────────
    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "ERROR: No HuggingFace token provided.\n"
            "  Pass --token hf_... or set the $HF_TOKEN environment variable.\n"
            "  Tip: export HF_TOKEN=$(cat /home/hamza95/.hf_token)"
        )
    login(token=hf_token)
    print("Logged into Hugging Face!")

    # ── load metadata ─────────────────────────────────────────────────────────
    if args.merge:
        segments = load_metadata_merged(base_dir)
    else:
        segments = load_metadata_single(base_dir)

    if not segments:
        raise SystemExit("ERROR: No segments found in metadata. Nothing to upload.")

    # ── validate files exist ──────────────────────────────────────────────────
    segments = validate_segments(segments)
    if not segments:
        raise SystemExit("ERROR: No valid audio files found on disk. Nothing to upload.")

    # ── stats ─────────────────────────────────────────────────────────────────
    total_min = sum(s.get("speech_duration_seconds", 0) for s in segments) / 60.0
    print(f"\n{'='*65}")
    print(f"Dataset summary")
    print(f"  Segments     : {len(segments):,}")
    print(f"  Total speech : {total_min:.1f} min  ({total_min/60:.2f} h)")
    print(f"  Target repo  : https://huggingface.co/datasets/{args.repo_id}")
    print(f"{'='*65}")

    # ── generate card ─────────────────────────────────────────────────────────
    card_text = generate_dataset_card(segments, args.repo_id)

    # ── upload ────────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\nDRY RUN — skipping upload.")
        out_path = base_dir / "dataset_card_preview.md"
        out_path.write_text(card_text)
        print(f"Dataset card preview written to: {out_path}")
    else:
        upload_dataset(
            segments  = segments,
            repo_id   = args.repo_id,
            token     = hf_token,
            split     = args.split,
            private   = args.private,
            card_text = card_text,
        )