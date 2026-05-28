# -*- coding: utf-8 -*-
"""
process_audio.py
Memoni Audio Pipeline — Step 1: Process Pre-Downloaded Files (VAD + Denoise)

This script assumes audio files are ALREADY downloaded and saved as:
    <audio-dir>/<unique_id>.mp3   (e.g. 00000.mp3, 00001.mp3, ...)

Steps performed:
  1. Read metadata CSV (columns: unique_id, youtube_link, category, channel_name, platform)
  2. For each row, locate the corresponding .mp3 in --audio-dir
  3. Run TenVad on a TEMPORARY mono-16kHz copy to find speech timestamps ONLY
     (the original file is never resampled or mixed down for processing)
  4. Use pydub to slice the ORIGINAL audio at those timestamps (sample rate + channels preserved)
  5. If segment < 30s: denoise as-is
     If segment >= 30s: split into ~30s sub-chunks, denoise each
  6. dpdfnet denoising via soundfile (preserves original sr/channels)
  7. Export final cleaned .mp3 per segment/chunk
  8. Write metadata_backup.json for use by upload_to_hf.py

Key guarantee
─────────────
  librosa is used ONLY to produce a throw-away 16 kHz mono array fed to TenVad.
  All slicing, denoising, and saving operates on the original audio so that
  sample rate and channel count are NEVER altered.

Example usage:
    python process_audio.py \
        --csv-path     /scratch/hamza95/memoni_audio/videos.csv \
        --audio-dir    /scratch/hamza95/memoni_audio/audios \
        --base-dir     /scratch/hamza95/memoni_audio/processed

SLURM slice (optional):
    --job-index   0   (0-based slice index)
    --chunk-size  50  (rows per job)

Generate + submit all SLURM jobs at once:
    --submit-all

python /project/ctb-stelzer/hamza95/memoni_audio/process_audio.py \
    --csv-path /scratch/hamza95/memoni_audio/id_link_mapping.csv \
    --audio-dir /scratch/hamza95/memoni_audio/audios \
    --chunk-size 5 --submit-all

python /project/ctb-stelzer/hamza95/memoni_audio/process_audio.py \
    --csv-path  /scratch/hamza95/memoni_audio/id_link_mapping.csv \
    --audio-dir /scratch/hamza95/memoni_audio/audios \
    --base-dir  /scratch/hamza95/memoni_audio/processed/test \
    --test
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, argparse, shutil, tempfile
from pathlib import Path

import numpy as np
import librosa                          # used ONLY for VAD pre-processing
import soundfile as sf
from pydub import AudioSegment
import pandas as pd
from ten_vad import TenVad

# dpdfnet denoising
import dpdfnet

print("All libraries imported successfully.")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = "/project/ctb-stelzer/hamza95/memoni_audio"
LOG_DIR    = f"{SCRIPT_DIR}/logs"

# TenVad requires exactly 256 samples per frame at 16 kHz (= 16 ms per frame)
_TENVAD_SAMPLE_RATE   = 16000
_TENVAD_FRAME_SAMPLES = 256
_TENVAD_FRAME_MS      = _TENVAD_FRAME_SAMPLES / _TENVAD_SAMPLE_RATE  # 0.016 s


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def deepfilter_denoise(input_wav: Path, output_wav: Path):
    """
    Apply dpdfnet noise suppression.

    Reads the WAV with soundfile (preserving the file's own sample rate and
    channel layout), enhances with dpdfnet8_48khz_hr, then writes the result
    back with soundfile at the same sample rate.
    """
    audio, sr = sf.read(str(input_wav))
    enhanced = dpdfnet.enhance(
        audio,
        sample_rate=sr,
        model="dpdfnet8_48khz_hr",
        attn_limit_db=12,
    )
    sf.write(str(output_wav), enhanced, sr)


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AudioProcessor:
    """
    Processes pre-downloaded audio files: VAD → split → denoise → export.

    Design principle
    ────────────────
    TenVad needs 16 kHz mono PCM.  We satisfy that by asking librosa for a
    temporary downmixed/resampled array that is used ONLY to obtain speech
    timestamps.  All actual audio work (slicing, denoising, saving) is done
    on pydub AudioSegment objects loaded directly from the original file, so
    the original sample rate and channel count are preserved end-to-end.
    """

    def __init__(self, audio_dir: str, base_dir: str = "/tmp/memoni_processed"):
        self.audio_dir = Path(audio_dir)
        self.base_dir  = Path(base_dir)
        self.clean_dir = self.base_dir / "cleaned"
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        self.metadata    : list[dict] = []
        self.audio_files : list[dict] = []

        self._load_metadata()

        print(f"\nAudioProcessor initialised.")
        print(f"  Audio source dir : {self.audio_dir}")
        print(f"  Clean output dir : {self.clean_dir}")

    # ── persistence ──────────────────────────────────────────────────────────

    def _metadata_path(self) -> Path:
        return self.base_dir / "metadata_backup.json"

    def _save_metadata(self):
        with open(self._metadata_path(), "w") as f:
            json.dump({"processed": self.metadata, "segments": self.audio_files}, f, indent=2)
        print(f"  Metadata saved → {self._metadata_path()}")
        print(f"  ({len(self.metadata)} source files, {len(self.audio_files)} segments)")

    def _load_metadata(self):
        p = self._metadata_path()
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            self.metadata    = data.get("processed", [])
            self.audio_files = data.get("segments",  [])
            print(f"  Resumed: {len(self.metadata)} previously processed source files, "
                  f"{len(self.audio_files)} segments.")

    def _processed_source_ids(self) -> set:
        return {m["unique_id"] for m in self.metadata}

    # ── VAD  (timestamps only — no permanent resampling) ─────────────────────

    def _find_speech_segments(self, audio_path: Path) -> list[tuple[float, float]]:
        """
        Detect speech regions via TenVad and return a list of
        (start_sec, end_sec) timestamps relative to the original file.

        Implementation note
        ───────────────────
        librosa.load() is called here with sr=16000, mono=True so that
        TenVad gets the exact format it requires.  The resulting array is
        ONLY used for VAD — it is never written to disk or used downstream.
        The timestamps produced are in seconds and are valid for the
        original file regardless of its sample rate, because time is
        sample-rate–independent.

        Post-processing:
          min_speech_frames : 4   (~64 ms) — discard very short hits
          merge_gap_sec     : 0.5          — merge nearby segments
        """
        MIN_SPEECH_FRAMES = 4
        MERGE_GAP_SEC     = 0.5

        # ── temporary mono-16kHz array for VAD only ──────────────────────────
        audio_16k_mono, _ = librosa.load(
            str(audio_path),
            sr   = _TENVAD_SAMPLE_RATE,
            mono = True,          # TenVad requires single channel
        )
        audio_int16 = (audio_16k_mono * 32767).astype(np.int16)
        # audio_16k_mono is intentionally not stored on self — it is garbage-
        # collected after this function returns.

        vad = TenVad(hop_size=_TENVAD_FRAME_SAMPLES, threshold=0.5)

        raw_segments  = []
        current_start = None
        speech_frames = 0

        for i in range(0, len(audio_int16) - _TENVAD_FRAME_SAMPLES, _TENVAD_FRAME_SAMPLES):
            frame            = audio_int16[i : i + _TENVAD_FRAME_SAMPLES]
            _prob, is_speech = vad.process(frame)

            if is_speech:
                speech_frames += 1
                if current_start is None:
                    current_start = i / _TENVAD_SAMPLE_RATE   # seconds
            else:
                if current_start is not None:
                    if speech_frames >= MIN_SPEECH_FRAMES:
                        end_sec = i / _TENVAD_SAMPLE_RATE
                        raw_segments.append((current_start, end_sec))
                    current_start = None
                    speech_frames = 0

        # flush final open segment
        if current_start is not None and speech_frames >= MIN_SPEECH_FRAMES:
            raw_segments.append((current_start, len(audio_16k_mono) / _TENVAD_SAMPLE_RATE))

        # merge segments whose gap is < MERGE_GAP_SEC
        merged = []
        for seg in raw_segments:
            if merged and seg[0] - merged[-1][1] < MERGE_GAP_SEC:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(list(seg))

        print(f"  TenVad: {len(merged)} speech segment(s) detected.")
        return merged

    # ── chunking ──────────────────────────────────────────────────────────────

    def _chunk_speech_segments(
        self,
        full_audio : AudioSegment,      # original audio, any sr / channels
        speech_segs: list[tuple[float, float]],
        chunk_sec  : float = 30.0,
    ) -> list[tuple[AudioSegment, float, float]]:
        """
        Slice the original AudioSegment using VAD timestamps.

        Segments < 30 s  → pass through unchanged.
        Segments >= 30 s → split into ~30 s sub-chunks.

        Slicing is done in milliseconds, which is sample-rate–agnostic,
        so the original sr and channel count of full_audio are preserved.
        """
        chunks = []

        for start, end in speech_segs:
            # millisecond slice — works regardless of sample rate
            seg     = full_audio[int(start * 1000) : int(end * 1000)]
            seg_dur = end - start

            if seg_dur < chunk_sec:
                chunks.append((seg, start, end))
            else:
                offset = 0.0
                while offset < seg_dur:
                    sub_end   = min(offset + chunk_sec, seg_dur)
                    sub_audio = seg[int(offset * 1000) : int(sub_end * 1000)]
                    chunks.append((sub_audio, start + offset, start + sub_end))
                    offset = sub_end

        short = sum(1 for _, s, e in chunks if (e - s) < chunk_sec)
        long  = len(chunks) - short
        print(f"  Chunking: {len(speech_segs)} VAD segment(s) → "
              f"{len(chunks)} chunk(s)  ({short} short as-is, {long} split from long).")
        return chunks

    # ── denoise ───────────────────────────────────────────────────────────────

    def _denoise(self, input_wav: Path, output_wav: Path):
        """Denoise with dpdfnet; fall back to copy on error."""
        try:
            deepfilter_denoise(input_wav, output_wav)
        except Exception as e:
            print(f"  dpdfnet error ({e}) — copying input as fallback.")
            shutil.copy(input_wav, output_wav)

    # ── per-segment saving ────────────────────────────────────────────────────

    def _save_segment(
        self,
        seg        : AudioSegment,   # sliced from original — sr/channels intact
        start_sec  : float,
        end_sec    : float,
        seg_id     : str,
        row        : pd.Series,
        source_uid : str,
    ) -> dict | None:
        """
        Denoise and save one audio chunk as MP3 without altering sample rate
        or channel count.

        Workflow
        ────────
        1. Export the pydub slice to a temporary WAV.
           pydub writes the WAV at the segment's own sample_width, frame_rate
           and channels — no conversion happens.

        2. Pass that WAV to dpdfnet via soundfile read/write.
           dpdfnet operates at the file's native sample rate; sf.write outputs
           the enhanced audio at the same sr that was read in.

        3. Load the denoised WAV back with pydub (which reads whatever sr the
           file declares) and export as 192 k MP3.  No manual resampling.

        Note: amplitude normalisation (peak-normalise to 0 dBFS) is performed
        via pydub's normalize() so we stay in the pydub domain and never touch
        numpy/librosa for the actual audio data.
        """
        if len(seg) < 200:      # skip clips shorter than 200 ms
            return None
        if len(seg) > 100:
            seg = seg[50:-50]   # trim 50 ms edges to avoid click artefacts

        duration_sec = len(seg) / 1000.0

        # ── Step 1: export original-quality slice to temp WAV ────────────────
        # Use a proper temp directory so concurrent jobs don't collide.
        with tempfile.TemporaryDirectory(dir=self.clean_dir) as tmpdir:
            tmp = Path(tmpdir)
            temp_wav     = tmp / f"{seg_id}_raw.wav"
            denoised_wav = tmp / f"{seg_id}_denoised.wav"

            # pydub writes WAV at the segment's own frame_rate + channels
            seg.export(str(temp_wav), format="wav")

            # ── Step 2: dpdfnet denoising ─────────────────────────────────────
            print(f"  [{seg_id}] Denoising {duration_sec:.1f}s chunk …")
            self._denoise(temp_wav, denoised_wav)

            # ── Step 3: reload denoised WAV with pydub ───────────────────────
            # pydub reads the WAV header, so sr / channels come from the file.
            denoised_seg = AudioSegment.from_wav(str(denoised_wav))

        # ── Step 4: peak-normalise (stays in pydub — no numpy resampling) ────
        # normalize() scales to 0 dBFS without changing sr or channels.
        denoised_seg = denoised_seg.normalize()

        # ── Step 5: export final MP3 ─────────────────────────────────────────
        output_mp3 = self.clean_dir / f"{seg_id}_cleaned.mp3"
        denoised_seg.export(str(output_mp3), format="mp3", bitrate="192k")

        size_mb = output_mp3.stat().st_size / (1024 ** 2)
        print(f"  [{seg_id}] Saved {duration_sec:.1f}s  "
              f"({size_mb:.2f} MB)  "
              f"sr={denoised_seg.frame_rate} Hz  "
              f"ch={denoised_seg.channels}  "
              f"→ {output_mp3.name}")

        return {
            "unique_id":               seg_id,
            "source_video_id":         source_uid,
            "youtube_link":            str(row.get("youtube_link",  "")),
            "category":                str(row.get("category",      "")),
            "channel_name":            str(row.get("channel_name",  "")),
            "platform":                str(row.get("platform",      "")),
            "segment_start_sec":       round(start_sec,    3),
            "segment_end_sec":         round(end_sec,      3),
            "speech_duration_seconds": round(duration_sec, 3),
            "sample_rate_hz":          denoised_seg.frame_rate,
            "channels":                denoised_seg.channels,
            "audio_path":              str(output_mp3),
        }

    # ── per-source-file processing ────────────────────────────────────────────

    def process_file(self, row: pd.Series, unique_id: str) -> list[dict]:
        """
        Process one pre-downloaded audio file.

        Pipeline
        ────────
        1. Load the original MP3 with pydub (sr + channels preserved).
        2. Run _find_speech_segments() which internally resamples to 16 kHz
           mono ONLY for VAD; it returns timestamps in seconds.
        3. Slice the ORIGINAL pydub AudioSegment using those timestamps.
        4. Denoise and save each chunk.
        """
        mp3_path = self.audio_dir / f"{unique_id}.mp3"
        if not mp3_path.exists():
            print(f"  [{unique_id}] Audio file not found: {mp3_path}  — skipping.")
            return []

        link = str(row.get("youtube_link", row.get("videolink", "")))
        print(f"\n{'='*65}")
        print(f"Processing : {unique_id}  |  {link[:55]}")
        print(f"  category={row.get('category', '')}   "
              f"channel={row.get('channel_name', '')}")
        print(f"  File: {mp3_path}")
        print(f"{'='*65}")

        # ── VAD: timestamps only ──────────────────────────────────────────────
        speech_segs = self._find_speech_segments(mp3_path)
        if not speech_segs:
            print(f"  No speech detected in {unique_id} — skipping.")
            return []

        # ── load original (preserves sr + channels) ───────────────────────────
        full_audio = AudioSegment.from_file(str(mp3_path))
        print(f"  Original audio: sr={full_audio.frame_rate} Hz  "
              f"ch={full_audio.channels}  "
              f"dur={len(full_audio)/1000:.1f}s")

        # ── chunk using VAD timestamps ────────────────────────────────────────
        chunks = self._chunk_speech_segments(full_audio, speech_segs, chunk_sec=30.0)

        results = []
        for chunk_idx, (chunk_audio, start, end) in enumerate(chunks, 1):
            seg_id = f"{unique_id}_seg{chunk_idx:04d}"
            rec = self._save_segment(chunk_audio, start, end, seg_id, row, unique_id)
            if rec:
                results.append(rec)
                self.audio_files.append(rec)

        self.metadata.append({
            "unique_id":    unique_id,
            "youtube_link": link,
            "category":     str(row.get("category",     "")),
            "channel_name": str(row.get("channel_name", "")),
            "platform":     str(row.get("platform",     "")),
            "segments":     len(results),
        })

        print(f"  {len(results)}/{len(chunks)} chunks saved for {unique_id}.")
        return results

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, csv_path: str, job_index: int = 0, chunk_size: int = 50):
        print(f"\n{'='*65}")
        print(f"Memoni Audio Processor")
        print(f"  CSV       : {csv_path}")
        print(f"  job_index : {job_index}  |  chunk_size : {chunk_size}")
        print(f"{'='*65}")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if "youtube_link" not in df.columns and "videolink" in df.columns:
            df = df.rename(columns={"videolink": "youtube_link"})

        required = {"unique_id", "youtube_link"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        print(f"Loaded {len(df)} rows from CSV.")

        done = self._processed_source_ids()
        candidates = [
            (str(row["unique_id"]).strip().zfill(5), row)
            for _, row in df.iterrows()
            if str(row["unique_id"]).strip().zfill(5) not in done
        ]
        print(f"Already processed : {len(done)}")
        print(f"Remaining         : {len(candidates)}")

        start      = job_index * chunk_size
        end        = start + chunk_size
        to_process = candidates[start:end]

        print(f"This job's slice  : [{start}:{end}]  → {len(to_process)} file(s)")

        if not to_process:
            print("Nothing in this job's slice — exiting.")
            self._save_metadata()
            return

        all_results = []
        succeeded = failed = 0
        for i, (uid, row) in enumerate(to_process, 1):
            print(f"\n[{i}/{len(to_process)}] {uid}")
            try:
                results = self.process_file(row, uid)
                if results:
                    all_results.extend(results)
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  Unexpected error on {uid}: {e}")
                failed += 1
            time.sleep(1)

        self._save_metadata()

        print(f"\n{'='*65}")
        print(f"Processing complete.")
        print(f"  Total in slice         : {len(to_process)}")
        print(f"  Successfully processed : {succeeded}")
        print(f"  Failed / skipped       : {failed}")
        print(f"  Segments created       : {len(all_results)}")
        print(f"  metadata_backup.json   : {self._metadata_path()}")
        print(f"{'='*65}")
        print(f"\nNext step → run upload_to_hf.py --base-dir {self.base_dir} ...")

    # ── summary ───────────────────────────────────────────────────────────────

    def show_summary(self):
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        if not self.metadata:
            print("  No files processed yet.")
            return
        total_sp = sum(f.get("speech_duration_seconds", 0) for f in self.audio_files)
        print(f"  Source files processed : {len(self.metadata)}")
        print(f"  Speech segments created: {len(self.audio_files)}")
        print(f"  Total clean speech     : {total_sp / 60:.1f} min")
        print(f"  Local output dir       : {self.clean_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Memoni Step 1 — VAD + denoise pre-downloaded audio"
    )
    p.add_argument("--csv-path",     required=True,
                   help="CSV with columns: unique_id, youtube_link, category, channel_name, platform")
    p.add_argument("--audio-dir",    default="/scratch/hamza95/memoni_audio/audios",
                   help="Directory containing <unique_id>.mp3 files")
    p.add_argument("--base-dir",     default="/scratch/hamza95/memoni_audio/processed",
                   help="Working directory for cleaned segments and metadata_backup.json")
    p.add_argument("--job-index",    type=int, default=0,
                   help="0-based SLURM job slice index (default: 0)")
    p.add_argument("--chunk-size",   type=int, default=50,
                   help="Rows per SLURM job (default: 50)")
    p.add_argument("--account",      default="def-mdanning",
                   help="SBATCH --account value")
    p.add_argument("--generate-slurm", metavar="OUTPUT_SCRIPT", default=None,
                   help="Write a SLURM batch script to OUTPUT_SCRIPT and exit")
    p.add_argument("--submit-all",   action="store_true",
                   help="Generate and sbatch one script per chunk, then exit")
    p.add_argument("--test",         action="store_true",
                   help="Process only 1 file (overrides --chunk-size to 1)")
    p.add_argument("--resubmit",     action="store_true",
                   help="Scan logs for timed-out/OOM jobs and resubmit with bumped time/memory")
    for action in p._actions:
        if hasattr(action, "option_strings") and "--csv-path" in action.option_strings:
            action.required = False
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  SLURM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_slurm_script(args, output_path: str, job_index: int,
                          chunk_size: int, test: bool = False):
    suffix    = f"memoni_proc_{job_index}" + ("_test" if test else "")
    test_flag = "--test" if test else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={suffix}
#SBATCH --account={args.account}
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output={LOG_DIR}/{suffix}_%j.out
#SBATCH --error={LOG_DIR}/{suffix}_%j.err

set -euo pipefail

module load llvm

export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/llvmcore/21.1.5/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH

module load python/3.11
source /scratch/hamza95/df_env/bin/activate
echo "Job {job_index} / chunk-size {chunk_size}{' (TEST)' if test else ''}"

python {SCRIPT_DIR}/process_audio.py \\
    --csv-path   {args.csv_path} \\
    --audio-dir  {args.audio_dir} \\
    --base-dir   {args.base_dir}/job_{job_index} \\
    --job-index  {job_index} \\
    --chunk-size {chunk_size} \\
    {test_flag}
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(script)
    print(f"SLURM script written: {output_path}")
    return output_path


def submit_all_jobs(args):
    import subprocess

    df         = pd.read_csv(args.csv_path)
    total      = len(df)
    chunk_size = 1 if args.test else args.chunk_size
    num_jobs   = (total + chunk_size - 1) // chunk_size

    print(f"\nTotal rows : {total}")
    print(f"Chunk size : {chunk_size}")
    print(f"Jobs       : {num_jobs}")

    jobs_dir = Path(SCRIPT_DIR) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    submitted = []
    for ji in range(num_jobs):
        script_path = str(jobs_dir / f"proc_job_{ji}.sh")
        generate_slurm_script(args, script_path, ji, chunk_size, test=args.test)
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Submitted proc_job_{ji}.sh  → {result.stdout.strip()}")
            submitted.append(result.stdout.strip())
        else:
            print(f"  ERROR proc_job_{ji}.sh : {result.stderr.strip()}")

    print(f"\nDone. {len(submitted)}/{num_jobs} jobs submitted.")
    print("Monitor with:  squeue -u $USER")


# ─────────────────────────────────────────────────────────────────────────────
#  RESUBMIT FAILED JOBS
# ─────────────────────────────────────────────────────────────────────────────

_RETRY_TIME = ["06:00:00", "12:00:00", "24:00:00"]
_RETRY_MEM  = ["128G",     "192G",     "256G"    ]

def _parse_job_index_from_log(log_path: Path) -> int | None:
    try:
        for line in log_path.read_text(errors="replace").splitlines():
            if line.startswith("Job ") and "/ chunk-size" in line:
                return int(line.split()[1])
    except Exception:
        pass
    return None


def _detect_failure_reason(log_path: Path) -> str | None:
    text = ""
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        pass

    err_path = log_path.with_suffix(".err")
    try:
        text += "\n" + err_path.read_text(errors="replace")
    except Exception:
        pass

    if not text.strip():
        return None

    tl = text.lower()

    if any(k in tl for k in [
        "due to time limit", "job cancelled at", "time limit",
        "duetimelimit", "cancelled due to time",
    ]):
        return "timeout"

    if any(k in tl for k in [
        "out of memory", "oom killer", "killed",
        "memory limit", "exceeded memory", "bus error",
    ]):
        return "oom"

    if "processing complete." not in tl and any(k in tl for k in [
        "error", "traceback", "exception", "slurmstepd",
    ]):
        return "error"

    return None


def _read_sbatch_value(script_path: Path, directive: str) -> str | None:
    prefix = f"#SBATCH --{directive}="
    for line in script_path.read_text().splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return None


def _bump_resources(script_path: Path, new_time: str, new_mem: str) -> Path:
    text = script_path.read_text()

    import re
    text = re.sub(r"(#SBATCH --time=)\S+", rf"\g<1>{new_time}", text)
    text = re.sub(r"(#SBATCH --mem=)\S+",  rf"\g<1>{new_mem}",  text)
    text = re.sub(r"(#SBATCH --job-name=\S+)", rf"\1_retry", text)

    stem    = script_path.stem
    parent  = script_path.parent
    attempt = 1
    while True:
        new_path = parent / f"{stem}_retry{attempt}.sh"
        if not new_path.exists():
            break
        attempt += 1

    new_path.write_text(text)
    return new_path


def resubmit_failed_jobs(args):
    import subprocess, re

    log_dir  = Path(LOG_DIR)
    jobs_dir = Path(SCRIPT_DIR) / "jobs"

    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        return

    out_files = sorted(log_dir.glob("memoni_proc_*.out"))
    if not out_files:
        print(f"No log files found in {log_dir}")
        return

    err_count = sum(1 for f in out_files if f.with_suffix(".err").exists())

    print(f"\n{'='*65}")
    print(f"Scanning {len(out_files)} log file(s) in {log_dir}")
    print(f"  .out files : {len(out_files)}")
    print(f"  .err files : {err_count}  (both are checked for each job)")
    print(f"{'='*65}")

    resubmitted = skipped = already_ok = 0

    for out_file in out_files:
        reason = _detect_failure_reason(out_file)

        if reason is None:
            already_ok += 1
            print(f"  OK       {out_file.name}")
            continue

        if reason not in ("timeout", "oom"):
            skipped += 1
            print(f"  SKIP     {out_file.name}  ({reason} — manual review needed)")
            continue

        m = re.match(r"memoni_proc_(\d+)", out_file.stem)
        if not m:
            print(f"  SKIP     {out_file.name}  (can't parse job index from name)")
            skipped += 1
            continue

        job_index   = int(m.group(1))
        script_path = jobs_dir / f"proc_job_{job_index}.sh"

        if not script_path.exists():
            retries = sorted(jobs_dir.glob(f"proc_job_{job_index}_retry*.sh"))
            if retries:
                script_path = retries[-1]
            else:
                print(f"  SKIP     {out_file.name}  (no script found at {script_path})")
                skipped += 1
                continue

        retry_m = re.search(r"_retry(\d+)\.sh$", str(script_path))
        attempt = int(retry_m.group(1)) if retry_m else 0

        if attempt >= len(_RETRY_TIME):
            print(f"  SKIP     {out_file.name}  (already at max retry {attempt})")
            skipped += 1
            continue

        new_time = _RETRY_TIME[attempt]
        new_mem  = _RETRY_MEM[attempt]

        print(f"  RESUBMIT {out_file.name}")
        print(f"           reason={reason}  job_index={job_index}  "
              f"attempt={attempt+1}  time={new_time}  mem={new_mem}")

        new_script = _bump_resources(script_path, new_time, new_mem)
        print(f"           new script → {new_script.name}")

        result = subprocess.run(["sbatch", str(new_script)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(f"           submitted  → {result.stdout.strip()}")
            resubmitted += 1
        else:
            print(f"           ERROR      : {result.stderr.strip()}")
            skipped += 1

    print(f"\n{'='*65}")
    print(f"Resubmit scan complete.")
    print(f"  OK (no action)  : {already_ok}")
    print(f"  Resubmitted     : {resubmitted}")
    print(f"  Skipped         : {skipped}")
    print(f"{'='*65}")
    if resubmitted:
        print("Monitor with:  squeue -u $USER")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.generate_slurm:
        chunk_size = 1 if args.test else args.chunk_size
        generate_slurm_script(args, args.generate_slurm,
                               job_index=args.job_index,
                               chunk_size=chunk_size, test=args.test)
        raise SystemExit(0)

    if args.submit_all:
        submit_all_jobs(args)
        raise SystemExit(0)

    if args.resubmit:
        resubmit_failed_jobs(args)
        raise SystemExit(0)

    chunk_size = 1 if args.test else args.chunk_size
    if args.test:
        print("TEST MODE: processing 1 file only.")

    processor = AudioProcessor(
        audio_dir = args.audio_dir,
        base_dir  = args.base_dir,
    )
    processor.run(
        csv_path   = args.csv_path,
        job_index  = args.job_index,
        chunk_size = chunk_size,
    )
    processor.show_summary()