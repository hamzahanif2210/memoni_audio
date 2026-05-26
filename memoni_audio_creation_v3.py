# -*- coding: utf-8 -*-
"""
memoni_audio_creation_v3.py
Updated Memoni Audio Dataset Pipeline

Changes from v2:
  - Audio is saved per VAD speech segment (not 1-minute fixed chunks)
  - Unique IDs are zero-padded integers only: 00000, 00001, ...
  - SLURM support via argparse:
      --job-index       : which job slice to process (0-based)
      --chunk-size      : how many videos per job (default 200)
      --account         : SBATCH --account value (e.g. def-mdanning)
      --generate-slurm  : write a ready-to-submit .sh script for this job
                          instead of running the pipeline directly
  - Duplicate videolink detection before processing
  - Auto-fills empty platform column from URL domain
    (youtube.com -> youtube, instagram.com -> instagram, tiktok.com -> tiktok)

python /project/ctb-stelzer/hamza95/memoni_audio/memoni_audio_creation_v3.py \
    --account def-mdanning \
    --hf-username hamzahanif \
    --dataset-name memoni_clean_audio_test \
    --base-dir /scratch/hamza95/memoni \
    --proxy-file /project/ctb-stelzer/hamza95/memoni_audio/proxies.txt \
    --cookies-file /project/ctb-stelzer/hamza95/memoni_audio/cookies.txt \
    --chunk-size 50 \
    --submit-all
"""

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, json, time, re, argparse
from pathlib import Path
from io import StringIO
from urllib.parse import urlparse

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import yt_dlp
import webrtcvad
import pandas as pd
import requests
from datasets import Dataset, Audio, Features, Value, load_dataset
from huggingface_hub import login

# DeepFilterNet
from df.enhance import enhance, init_df, load_audio, save_audio

print("All libraries imported!")

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/hamzahanif2210/memoni_audio"
    "/refs/heads/main/videos_links.csv"
)
MEMON_PATTERN = re.compile(r'#?memoni?\b', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_from_github(url: str) -> pd.DataFrame:
    """Fetch the CSV from GitHub raw URL and return a DataFrame."""
    print(f"Fetching CSV from: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    df.columns = df.columns.str.strip()

    required_cols = {"videolink", "category", "channel_name", "post_text", "platform"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    print(f"Loaded {len(df)} rows.  Columns: {df.columns.tolist()}")
    return df


def should_process_video(row: pd.Series) -> bool:
    """
    Return True if the video should be processed.
    - Empty/NaN post_text: always include
    - Non-empty post_text: include only when it contains memon/memoni
    """
    text = row.get("post_text", "")
    if pd.isna(text) or str(text).strip() == "":
        return True
    return bool(MEMON_PATTERN.search(str(text)))


def make_unique_id(global_index: int) -> str:
    """
    Return a zero-padded 5-digit integer string.
    e.g. 0 -> '00000',  1 -> '00001',  999 -> '00999'
    """
    return f"{global_index:05d}"


# Domain -> platform name mapping (order matters: most specific first)
_PLATFORM_MAP = {
    "youtube.com":   "youtube",
    "youtu.be":      "youtube",
    "instagram.com": "instagram",
    "tiktok.com":    "tiktok",
}

def infer_platform(url: str) -> str:
    """
    Return platform name derived from the URL domain.
    Returns empty string if the domain is not recognised.
    """
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
        for domain, name in _PLATFORM_MAP.items():
            if host == domain or host.endswith("." + domain):
                return name
    except Exception:
        pass
    return ""


def fill_platform(df: pd.DataFrame) -> pd.DataFrame:
    """
    For any row where `platform` is empty/NaN, infer it from `videolink`.
    Logs a summary of how many values were filled.
    """
    mask = df["platform"].isna() | (df["platform"].str.strip() == "")
    if mask.any():
        df = df.copy()
        df.loc[mask, "platform"] = df.loc[mask, "videolink"].apply(infer_platform)
        filled = mask.sum()
        print(f"Platform column: filled {filled} empty value(s) from URL.")
    else:
        print("Platform column: no empty values.")
    return df


def drop_duplicate_links(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with duplicate videolink values (keep first occurrence).
    Prints a report of any duplicates found.
    """
    # Drop rows with no link at all
    before = len(df)
    df = df.dropna(subset=["videolink"]).reset_index(drop=True)
    dropped_nan = before - len(df)
    if dropped_nan:
        print(f"Duplicate check: dropped {dropped_nan} row(s) with empty videolink.")

    dupes = df[df.duplicated(subset="videolink", keep=False)]
    if dupes.empty:
        print("Duplicate check: no duplicate links found.")
        return df

    dup_links = df[df.duplicated(subset="videolink", keep="first")]["videolink"]
    print(f"Duplicate check: found {len(dup_links)} duplicate row(s) — removing:")
    for link in dup_links.values:
        print(f"  DUPLICATE (removed): {str(link)[:80]}")

    return df.drop_duplicates(subset="videolink", keep="first").reset_index(drop=True)


def deepfilter_denoise(input_wav: Path, output_wav: Path):
    """Apply DeepFilterNet noise suppression."""
    model, df_state, _ = init_df()
    audio, _ = load_audio(str(input_wav), sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)
    save_audio(str(output_wav), enhanced, df_state.sr())


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class MemonAudioPipeline:
    """
    Full pipeline:
      1. Read CSV from GitHub
      2. Filter rows by memon/memoni keyword
      3. Download audio with yt-dlp
      4. Run VAD (webrtcvad) to find speech segments
      5. Save EACH speech segment as a separate audio file
      6. Denoise each segment with DeepFilterNet
      7. Upload to Hugging Face with rich metadata
    """

    def __init__(self, hf_username: str, dataset_name: str, base_dir: str = "/tmp/memoni", cookies_file: str = None, proxy: str = None, proxy_file: str = None):
        self.username     = hf_username
        self.dataset_name = dataset_name
        self.repo_name    = f"{hf_username}/{dataset_name}"
        self.cookies_file = cookies_file
        self.proxy        = proxy
        self._proxy_list  = []
        self._proxy_index = 0

        if proxy_file:
            self._load_proxy_file(proxy_file)
        elif proxy:
            self._proxy_list = [proxy]

        self.temp_dir  = Path(base_dir)
        self.raw_dir   = self.temp_dir / "downloaded"
        self.clean_dir = self.temp_dir / "cleaned"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.clean_dir.mkdir(parents=True, exist_ok=True)

        self.metadata    = []
        self.audio_files = []

        self._load_metadata()

        print(f"Pipeline ready!")
        print(f"Raw dir:     {self.raw_dir}")
        print(f"Cleaned dir: {self.clean_dir}")
        print(f"HF repo:     https://huggingface.co/datasets/{self.repo_name}")
        if self._proxy_list:
            print(f"Proxies loaded: {len(self._proxy_list)}")

    def _load_proxy_file(self, proxy_file: str):
        """
        Parse a proxy file in host:port:user:pass format (one per line).
        Converts each line to http://user:pass@host:port
        """
        proxies = []
        with open(proxy_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(":")
                if len(parts) == 4:
                    host, port, user, pwd = parts
                    proxies.append(f"http://{user}:{pwd}@{host}:{port}")
                else:
                    print(f"   Skipping malformed proxy line: {line}")
        self._proxy_list = proxies
        print(f"Loaded {len(proxies)} proxies from {proxy_file}")

    def _next_proxy(self) -> str | None:
        """Return the next proxy in round-robin rotation."""
        if not self._proxy_list:
            return None
        proxy = self._proxy_list[self._proxy_index % len(self._proxy_list)]
        self._proxy_index += 1
        return proxy

    # ── persistence ──────────────────────────────────────────────────────────

    def _metadata_path(self) -> Path:
        return self.temp_dir / "metadata_backup.json"

    def _save_metadata(self):
        with open(self._metadata_path(), "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Metadata saved ({len(self.metadata)} entries)")

    def _load_metadata(self):
        p = self._metadata_path()
        if p.exists():
            with open(p) as f:
                self.metadata = json.load(f)
            print(f"Loaded {len(self.metadata)} previously processed videos")

    def _processed_ids(self) -> set:
        return {m["unique_id"] for m in self.metadata}

    # ── download ─────────────────────────────────────────────────────────────

    def _download_audio(self, url: str, unique_id: str) -> tuple[Path | None, float]:
        """Download audio from YouTube as MP3."""
        output_path = self.raw_dir / f"{unique_id}.mp3"
        ydl_opts = {
            "format": "bestaudio/best",
            "format_sort": ["abr", "asr"],
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": str(self.raw_dir / unique_id),
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": False,
            "sleep_interval":       3,   # wait 3s between requests
            "max_sleep_interval":   6,   # random up to 6s
            "sleep_interval_requests": 1, # apply to every request
        }
        if self.cookies_file:
            ydl_opts["cookiefile"] = self.cookies_file
        proxy = self._next_proxy()
        if proxy:
            ydl_opts["proxy"] = proxy
            print(f"   Using proxy: {proxy.split('@')[-1]}")  # only show host:port
        try:
            print(f"   Downloading {url[:60]}...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            duration = info.get("duration", 0) or 0
            print(f"   Downloaded ({duration:.0f}s)")
            return output_path, duration
        except Exception as e:
            err = str(e)
            if "Requested format is not available" in err:
                print(f"   Format unavailable, retrying with any format...")
                try:
                    ydl_opts["format"] = "best"
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                    duration = info.get("duration", 0) or 0
                    print(f"   Downloaded with fallback format ({duration:.0f}s)")
                    return output_path, duration
                except Exception as e2:
                    print(f"   Download failed after fallback: {e2}")
                    return None, 0
            print(f"   Download failed: {e}")
            return None, 0

    # ── VAD ───────────────────────────────────────────────────────────────────

    def _find_speech_segments(self, audio_path: Path) -> list[tuple[float, float]]:
        """
        Use webrtcvad to locate speech regions in an audio file.
        Returns list of (start_sec, end_sec) tuples.
        """
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_int16 = (audio * 32767).astype(np.int16)

        vad        = webrtcvad.Vad(2)
        frame_ms   = 30
        frame_size = int(sr * frame_ms / 1000)

        segments, current_start = [], None
        for i in range(0, len(audio_int16) - frame_size, frame_size):
            frame     = audio_int16[i : i + frame_size].tobytes()
            is_speech = vad.is_speech(frame, sr)
            if is_speech:
                if current_start is None:
                    current_start = i / sr
            else:
                if current_start is not None:
                    segments.append((current_start, i / sr))
                    current_start = None

        if current_start is not None:
            segments.append((current_start, len(audio) / sr))

        # Merge gaps < 0.5 s
        merged = []
        for seg in segments:
            if merged and seg[0] - merged[-1][1] < 0.5:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(list(seg))

        print(f"   Found {len(merged)} speech segments")
        return merged

    # ── denoise ───────────────────────────────────────────────────────────────

    def _denoise_with_deepfilternet(self, input_wav: Path, output_wav: Path):
        """Apply DeepFilterNet noise suppression."""
        print(f"   DeepFilterNet denoising...")
        try:
            deepfilter_denoise(input_wav, output_wav)
            print(f"   Denoised -> {output_wav.name}")
        except Exception as e:
            print(f"   DeepFilterNet failed ({e}), copying input as fallback")
            import shutil
            shutil.copy(input_wav, output_wav)

    # ── per-segment saving ────────────────────────────────────────────────────

    def _save_speech_segment(
        self,
        full_audio:  AudioSegment,
        start_sec:   float,
        end_sec:     float,
        seg_id:      str,
        csv_row:     pd.Series,
        source_uid:  str,
    ) -> dict | None:
        """
        Extract a single speech segment, denoise it, and save as MP3.
        One output file per speech segment (replaces 1-minute chunking).
        Returns a metadata record dict, or None on failure.
        """
        seg = full_audio[int(start_sec * 1000) : int(end_sec * 1000)]
        if len(seg) < 200:          # skip segments shorter than 200 ms
            return None
        if len(seg) > 100:
            seg = seg[50:-50]       # trim 50 ms edges to avoid clicks

        duration_sec = len(seg) / 1000.0

        # Export to temp WAV
        temp_wav = self.clean_dir / f"{seg_id}_raw.wav"
        seg.export(temp_wav, format="wav")

        # Denoise
        denoised_wav = self.clean_dir / f"{seg_id}_denoised.wav"
        self._denoise_with_deepfilternet(temp_wav, denoised_wav)
        temp_wav.unlink(missing_ok=True)

        # Normalize & save final MP3
        y, sr = librosa.load(denoised_wav, sr=16000)
        peak  = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        denoised_wav.unlink(missing_ok=True)

        output_mp3 = self.clean_dir / f"{seg_id}_cleaned.mp3"
        temp_norm  = self.clean_dir / f"{seg_id}_norm.wav"
        sf.write(temp_norm, y, sr)
        AudioSegment.from_wav(temp_norm).export(output_mp3, format="mp3", bitrate="192k")
        temp_norm.unlink(missing_ok=True)

        file_size_mb = output_mp3.stat().st_size / (1024 ** 2)
        print(f"   {seg_id}: {duration_sec:.1f}s,  {file_size_mb:.2f} MB")

        return {
            "unique_id":               seg_id,
            "source_video_id":         source_uid,
            "videolink":               str(csv_row.get("videolink",    "")),
            "category":                str(csv_row.get("category",     "")),
            "channel_name":            str(csv_row.get("channel_name", "")),
            "post_text":               str(csv_row.get("post_text",    "")),
            "platform":                str(csv_row.get("platform",     "")),
            "segment_start_sec":       start_sec,
            "segment_end_sec":         end_sec,
            "speech_duration_seconds": duration_sec,
            "audio_path":              str(output_mp3),
        }

    # ── per-video processing ──────────────────────────────────────────────────

    def process_video(self, row: pd.Series, unique_id: str) -> list[dict]:
        """
        Full pipeline for one CSV row:
          download -> VAD -> save each speech segment -> collect records
        Returns list of output records (one per speech segment).
        """
        url = str(row["videolink"]).strip()
        print(f"\n{'='*65}")
        print(f"Video: {unique_id}  |  {url[:55]}...")
        print(f"  category={row.get('category','')}  channel={row.get('channel_name','')}")
        print(f"{'='*65}")

        raw_path, duration = self._download_audio(url, unique_id)
        if raw_path is None:
            return []

        self.metadata.append({
            "unique_id":                 unique_id,
            "videolink":                 url,
            "category":                  str(row.get("category",    "")),
            "channel_name":              str(row.get("channel_name","")),
            "post_text":                 str(row.get("post_text",   "")),
            "platform":                  str(row.get("platform",    "")),
            "original_duration_seconds": duration,
        })

        # VAD over full audio — each segment becomes its own output file
        speech_segs = self._find_speech_segments(raw_path)
        if not speech_segs:
            print(f"   No speech found in {unique_id}")
            raw_path.unlink(missing_ok=True)
            return []

        full_audio = AudioSegment.from_file(raw_path)

        results = []
        for seg_idx, (start, end) in enumerate(speech_segs, 1):
            seg_id = f"{unique_id}_seg{seg_idx:04d}"
            rec = self._save_speech_segment(
                full_audio, start, end, seg_id, row, unique_id
            )
            if rec:
                results.append(rec)
                self.audio_files.append(rec)

        raw_path.unlink(missing_ok=True)

        print(f"   {len(results)}/{len(speech_segs)} segments saved for {unique_id}")
        return results

    # ── main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        csv_url:    str = CSV_URL,
        csv_path:   str | None = None,
        job_index:  int = 0,
        chunk_size: int = 50,
    ):
        """
        Full pipeline with SLURM slice support.

        csv_path   : path to pre-cleaned local CSV (skips fetch/dedup/platform fill)
        csv_url    : fallback GitHub URL when csv_path is not provided
        job_index  : 0-based index of this job's slice (from --job-index)
        chunk_size : videos per SLURM job         (from --chunk-size)
        """
        print(f"\n{'='*65}")
        print(f"Starting Memoni Audio Pipeline")
        print(f"  job_index={job_index}  chunk_size={chunk_size}")
        print(f"{'='*65}")

        # 1. Load CSV — from local cleaned file or fetch from GitHub
        if csv_path:
            print(f"Reading pre-cleaned CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            total_rows = len(df)
            print(f"Loaded {total_rows} rows (already cleaned, no dedup/platform fill needed).")
        else:
            df = load_csv_from_github(csv_url)
            total_rows = len(df)
            df = drop_duplicate_links(df)
            df = fill_platform(df)

        df_filtered = df[df.apply(should_process_video, axis=1)].reset_index(drop=True)
        skipped     = total_rows - len(df_filtered)
        print(f"\nKeyword filter: {len(df_filtered)}/{total_rows} rows pass "
              f"(skipped {skipped} without memon/memoni in post_text)")

        # 2. Assign zero-padded numeric unique IDs (global sequential)
        processed_ids = self._processed_ids()
        all_candidates = []
        for global_idx, (_, row) in enumerate(df_filtered.iterrows()):
            uid = make_unique_id(global_idx)
            if uid not in processed_ids:
                all_candidates.append((uid, row))

        # 3. Slice for this SLURM job
        start      = job_index * chunk_size
        end        = start + chunk_size
        to_process = all_candidates[start:end]

        print(f"\nTotal unprocessed : {len(all_candidates)}")
        print(f"This job's slice  : [{start}:{end}]  -> {len(to_process)} videos")

        if not to_process:
            print("Nothing in this job's slice!")
            self._save_metadata()
            return

        # 4. Process videos
        all_results = []
        succeeded   = 0
        failed      = 0
        for i, (uid, row) in enumerate(to_process, 1):
            print(f"\n[{i}/{len(to_process)}] Processing {uid}")
            try:
                results = self.process_video(row, uid)
                if results:
                    all_results.extend(results)
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"   Unexpected error on {uid}: {e}")
                failed += 1
            time.sleep(5)

        self._save_metadata()

        print(f"\n{'='*65}")
        print(f"Processing complete.")
        print(f"  Total in slice          : {len(to_process)}")
        print(f"  Successfully processed  : {succeeded}")
        print(f"  Failed / skipped        : {failed}")
        print(f"  Audio segments created  : {len(all_results)}")
        print(f"{'='*65}")

        if all_results:
            self.upload_to_huggingface()
        else:
            print("No audio files to upload.")

    # ── Hugging Face upload ───────────────────────────────────────────────────

    def upload_to_huggingface(self):
        """Merge existing HF dataset with new local files and re-upload."""
        print(f"\n{'='*65}")
        print(f"Syncing with Hugging Face: {self.repo_name}")
        print(f"{'='*65}")

        META_FIELDS = [
            "unique_id", "source_video_id",
            "videolink", "category", "channel_name", "platform",
            "segment_start_sec", "segment_end_sec",
            "speech_duration_seconds",
        ]

        existing: dict[str, list] = {f: [] for f in ["audio"] + META_FIELDS}
        existing_ids: set[str]    = set()

        try:
            print("Fetching existing dataset from HF...")
            ex_ds = load_dataset(self.repo_name, split="train")
            for item in ex_ds:
                existing["audio"].append(item["audio"]["path"])
                for field in META_FIELDS:
                    existing[field].append(item.get(field, ""))
            existing_ids = set(existing["unique_id"])
            print(f"   Found {len(existing_ids)} existing entries on HF")
        except Exception as e:
            print(f"No existing dataset (or error): {e}")
            print("   Will create a new dataset.")

        new_records = [
            rec for rec in self.audio_files
            if rec["unique_id"] not in existing_ids
               and Path(rec["audio_path"]).exists()
        ]

        if not new_records and existing_ids:
            print(f"No new files. HF already has {len(existing_ids)} entries.")
            return

        print(f"   New files to add: {len(new_records)}")

        merged: dict[str, list] = {
            "audio": existing["audio"] + [r["audio_path"] for r in new_records],
        }
        for field in META_FIELDS:
            merged[field] = existing[field] + [r.get(field, "") for r in new_records]

        print(f"   Total after merge: {len(merged['audio'])} entries")

        df_out   = pd.DataFrame(merged)
        csv_path = str(self.temp_dir / "merged_dataset.csv")
        df_out.to_csv(csv_path, index=False)

        feature_dict = {
            "audio":                   Audio(),
            "unique_id":               Value("string"),
            "source_video_id":         Value("string"),
            "videolink":               Value("string"),
            "category":                Value("string"),
            "channel_name":            Value("string"),
            "platform":                Value("string"),
            "segment_start_sec":       Value("float32"),
            "segment_end_sec":         Value("float32"),
            "speech_duration_seconds": Value("float32"),
        }

        hf_ds = load_dataset(
            "csv",
            data_files=csv_path,
            features=Features(feature_dict),
            split="train",
        )

        print(f"\nUploading {len(merged['audio'])} files to HF...")
        hf_ds.push_to_hub(
            self.repo_name,
            private=False,
            commit_message=(
                f"Added {len(new_records)} new files. "
                f"Total: {len(merged['audio'])}"
            ),
        )

        print(f"\nUpload complete!")
        print(f"Dataset: https://huggingface.co/datasets/{self.repo_name}")
        return hf_ds

    # ── summary ───────────────────────────────────────────────────────────────

    def show_summary(self):
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        if not self.metadata:
            print("   No videos processed yet.")
            return

        total_src = sum(m.get("original_duration_seconds", 0) for m in self.metadata)
        total_sp  = sum(f.get("speech_duration_seconds", 0)   for f in self.audio_files)

        print(f"   Source videos processed : {len(self.metadata)}")
        print(f"   Speech segments created : {len(self.audio_files)}")
        print(f"   Original duration       : {total_src/60:.1f} min")
        print(f"   Clean speech duration   : {total_sp/60:.1f} min")
        print(f"   Removed silence/noise   : {(total_src - total_sp)/60:.1f} min")
        if total_src > 0:
            pct = (1 - total_sp / total_src) * 100
            print(f"   Reduction               : {pct:.1f}%")
        print(f"\n   Local files : {self.clean_dir}")
        print(f"   HF dataset  : https://huggingface.co/datasets/{self.repo_name}")


# ─────────────────────────────────────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Memoni Audio Dataset Pipeline (SLURM-ready)"
    )
    parser.add_argument(
        "--job-index", type=int, default=0,
        help="0-based index of this SLURM job's slice (default: 0)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=200,
        help="Number of videos per SLURM job (default: 200)"
    )
    parser.add_argument(
        "--account", type=str, default="def-mdanning",
        help="SBATCH --account value used by the submit script (default: def-mdanning)"
    )
    parser.add_argument(
        "--hf-username", type=str, default="Aqiba",
        help="Hugging Face username (default: Aqiba)"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="memoni_clean_audio",
        help="Hugging Face dataset name (default: memoni_clean_audio)"
    )
    parser.add_argument(
        "--base-dir", type=str, default="/tmp/memoni",
        help="Base directory for temp files (default: /tmp/memoni)"
    )
    parser.add_argument(
        "--csv-path", type=str, default=None,
        help=(
            "Path to a pre-cleaned local CSV file to use instead of fetching "
            "from GitHub. Set automatically by --submit-all."
        )
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--cookies-file", type=str, default=None,
        help="Path to a Netscape-format cookies file for yt-dlp (bypasses YouTube bot detection)"
    )
    parser.add_argument(
        "--proxy", type=str, default=None,
        help="Single proxy URL e.g. http://user:pass@host:port"
    )
    parser.add_argument(
        "--proxy-file", type=str, default=None,
        help="Path to proxy file in host:port:user:pass format (one per line). Rotated per video."
    )
    parser.add_argument(
        "--generate-slurm", type=str, default=None,
        metavar="OUTPUT_SCRIPT",
        help=(
            "Instead of running the pipeline, write a ready-to-submit SLURM "
            "batch script to OUTPUT_SCRIPT (e.g. run_job0.sh) and exit."
        )
    )
    parser.add_argument(
        "--test", action="store_true",
        help=(
            "Test mode: process only 1 video (overrides --chunk-size to 1). "
            "When combined with --generate-slurm, the generated script also "
            "runs in test mode."
        )
    )
    parser.add_argument(
        "--submit-all", action="store_true",
        help=(
            "Fetch the CSV, count filtered videos, generate one SLURM script "
            "per chunk, and sbatch all of them automatically."
        )
    )
    parser.add_argument(
        "--test-slurm", action="store_true",
        help=(
            "Like --submit-all but submits only the first chunk (job-index 0) "
            "with chunk-size 1. Quick end-to-end sanity check on the cluster."
        )
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  SLURM SCRIPT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = "/project/ctb-stelzer/hamza95/memoni_audio"
LOG_DIR    = f"{SCRIPT_DIR}/logs"

def generate_slurm_script(args, output_path: str):
    """
    Write a self-contained SBATCH script for this job index to output_path.
    The script can then be submitted with:  sbatch <output_path>
    """
    token_line = (
        f'export HF_TOKEN="{args.hf_token}"'
        if args.hf_token
        else 'export HF_TOKEN=$(cat "${HOME}/.hf_token")'
    )

    test_flag   = "--test" if args.test else ""
    job_suffix  = f"memoni_audio_{args.job_index}" + ("_test" if args.test else "")
    chunk_size  = 1 if args.test else args.chunk_size

    csv_path_line     = f"    --csv-path      {args.csv_path} \\" if args.csv_path else ""
    cookies_file_line = f"    --cookies-file  {args.cookies_file} \\" if getattr(args, "cookies_file", None) else ""
    proxy_line        = f"    --proxy         {args.proxy} \\" if getattr(args, "proxy", None) else ""
    proxy_file_line   = f"    --proxy-file    {args.proxy_file} \\" if getattr(args, "proxy_file", None) else ""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_suffix}
#SBATCH --account={args.account}
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output={LOG_DIR}/{job_suffix}_%j.out
#SBATCH --error={LOG_DIR}/{job_suffix}_%j.err

set -euo pipefail

module load python/3.11
source /scratch/hamza95/df_env/bin/activate

# Hugging Face token
{token_line}

echo "Starting job index {args.job_index} / chunk-size {chunk_size}{' (TEST MODE)' if args.test else ''}"

python {SCRIPT_DIR}/memoni_audio_creation_v3.py \\
    --job-index    {args.job_index} \\
    --chunk-size   {chunk_size} \\
    --account      {args.account} \\
    --hf-username  {args.hf_username} \\
    --dataset-name {args.dataset_name} \\
    --base-dir     {args.base_dir}/job_{args.job_index} \\
    --hf-token     "${{HF_TOKEN}}" \\
{csv_path_line}
{cookies_file_line}
{proxy_line}
{proxy_file_line}
    {test_flag}
"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(script)
    print(f"SLURM script written to: {output_path}")
    print(f"Submit with:  sbatch {output_path}")


def submit_all_jobs(args, test_slurm: bool = False):
    """
    Fetch the CSV, count filtered videos after dedup, generate one SLURM
    script per chunk into SCRIPT_DIR/jobs/, and sbatch each one.

    test_slurm=True  →  submit only job-index 0 with chunk-size 1.
    """
    import subprocess

    print(f"\n{'='*65}")
    print(f"SUBMIT ALL — fetching CSV to count videos...")
    print(f"{'='*65}")

    # Load & prepare CSV once — save cleaned version for all jobs to reuse
    resp = requests.get(CSV_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = df.columns.str.strip()
    df = drop_duplicate_links(df)
    df = fill_platform(df)
    df_filtered = df[df.apply(should_process_video, axis=1)].reset_index(drop=True)
    total_videos = len(df_filtered)

    # Save the cleaned+filtered CSV so jobs don't re-fetch or re-clean
    cleaned_csv_path = str(Path(SCRIPT_DIR) / "cleaned_videos.csv")
    df_filtered.to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned CSV saved: {cleaned_csv_path}  ({total_videos} videos)")

    if test_slurm:
        num_jobs   = 1
        chunk_size = 1
        print(f"TEST SLURM MODE: 1 job, chunk-size 1  (first video only)")
    else:
        chunk_size = args.chunk_size
        num_jobs   = (total_videos + chunk_size - 1) // chunk_size
        print(f"Filtered videos : {total_videos}")
        print(f"Chunk size      : {chunk_size}")
        print(f"Jobs to submit  : {num_jobs}")

    jobs_dir = Path(SCRIPT_DIR) / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    submitted = []
    for job_idx in range(num_jobs):
        # Temporarily override job_index on args so generate_slurm_script picks it up
        args.job_index  = job_idx
        args.chunk_size = chunk_size
        args.test       = test_slurm

        suffix      = f"job_{job_idx}" + ("_test" if test_slurm else "")
        script_path = str(jobs_dir / f"{suffix}.sh")

        args.csv_path = cleaned_csv_path
        generate_slurm_script(args, script_path)

        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip()
            print(f"  Submitted {suffix}.sh  ->  {job_id}")
            submitted.append(job_id)
        else:
            print(f"  ERROR submitting {suffix}.sh: {result.stderr.strip()}")

    print(f"\n{'='*65}")
    print(f"Done. {len(submitted)}/{num_jobs} jobs submitted.")
    print(f"Monitor with:  squeue -u $USER")
    print(f"{'='*65}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # ── Submit all jobs automatically then exit ───────────────────────────────
    if args.submit_all or args.test_slurm:
        submit_all_jobs(args, test_slurm=args.test_slurm)
        raise SystemExit(0)

    # ── Generate a single SLURM script and exit ───────────────────────────────
    if args.generate_slurm:
        generate_slurm_script(args, args.generate_slurm)
        raise SystemExit(0)

    # ── Test mode: override chunk_size to 1 ──────────────────────────────────
    chunk_size = 1 if args.test else args.chunk_size
    if args.test:
        print("TEST MODE: processing 1 video only (chunk-size overridden to 1)")

    # HF login
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged into Hugging Face!")
    else:
        print("HF_TOKEN not set — upload will fail unless already logged in.")

    pipeline = MemonAudioPipeline(
        hf_username  = args.hf_username,
        dataset_name = args.dataset_name,
        base_dir     = args.base_dir,
        cookies_file = args.cookies_file,
        proxy        = args.proxy,
        proxy_file   = args.proxy_file,
    )

    pipeline.run(
        csv_url    = CSV_URL,
        csv_path   = args.csv_path,
        job_index  = args.job_index,
        chunk_size = chunk_size,
    )

    pipeline.show_summary()