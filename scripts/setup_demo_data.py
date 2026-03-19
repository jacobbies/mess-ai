#!/usr/bin/env python3
"""Create a tiny synthetic dataset for first-run onboarding."""

from __future__ import annotations

import argparse
import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoTrack:
    track_id: str
    frequency_hz: float
    composer: str
    title: str


DEMO_TRACKS: tuple[DemoTrack, ...] = (
    DemoTrack(
        track_id="Beethoven_Op027No1-01",
        frequency_hz=261.63,
        composer="Ludwig van Beethoven",
        title="Synthetic demo tone in C4",
    ),
    DemoTrack(
        track_id="Chopin_Op028-15",
        frequency_hz=329.63,
        composer="Frederic Chopin",
        title="Synthetic demo tone in E4",
    ),
    DemoTrack(
        track_id="Bach_BWV849-01",
        frequency_hz=392.00,
        composer="Johann Sebastian Bach",
        title="Synthetic demo tone in G4",
    ),
)


def _sine_pcm16(frequency_hz: float, sample_rate: int, duration_seconds: float) -> bytes:
    total_samples = max(1, int(sample_rate * duration_seconds))
    amplitude = 0.25 * 32767.0
    fade_seconds = min(0.02, max(duration_seconds / 10.0, 0.001))
    fade_samples = max(1, int(sample_rate * fade_seconds))

    frames = bytearray(total_samples * 2)
    two_pi_f = 2.0 * math.pi * frequency_hz

    for index in range(total_samples):
        t = index / sample_rate
        sample = math.sin(two_pi_f * t)

        if index < fade_samples:
            envelope = index / fade_samples
        elif index >= total_samples - fade_samples:
            envelope = (total_samples - index - 1) / fade_samples
        else:
            envelope = 1.0

        scaled = int(max(-32767, min(32767, sample * envelope * amplitude)))
        frames[index * 2:index * 2 + 2] = scaled.to_bytes(2, byteorder="little", signed=True)

    return bytes(frames)


def _write_wav(path: Path, pcm16_mono: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), mode="wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16_mono)


def _write_metadata(path: Path, tracks: list[DemoTrack]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track_id",
        "dataset",
        "composer",
        "title",
        "source",
        "license",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for track in tracks:
            writer.writerow(
                {
                    "track_id": track.track_id,
                    "dataset": "smd",
                    "composer": track.composer,
                    "title": track.title,
                    "source": "generated_by_scripts/setup_demo_data.py",
                    "license": "CC0-1.0",
                }
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate small synthetic WAV files for onboarding demos"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Data root directory (default: ./data)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=8.0,
        help="Duration per track in seconds (default: 8.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate for generated WAV files (default: 24000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0")

    audio_dir = args.data_root / "audio" / "smd" / "wav-44"
    metadata_path = args.data_root / "metadata" / "smd_metadata.csv"

    selected_tracks = list(DEMO_TRACKS)
    created_files: list[Path] = []
    skipped_files: list[Path] = []

    for track in selected_tracks:
        target_path = audio_dir / f"{track.track_id}.wav"
        if target_path.exists() and not args.force:
            skipped_files.append(target_path)
            continue

        pcm16 = _sine_pcm16(
            frequency_hz=track.frequency_hz,
            sample_rate=args.sample_rate,
            duration_seconds=args.duration_seconds,
        )
        _write_wav(target_path, pcm16, sample_rate=args.sample_rate)
        created_files.append(target_path)

    if created_files or args.force or not metadata_path.exists():
        _write_metadata(metadata_path, selected_tracks)

    print(f"Data root: {args.data_root}")
    print(f"Audio dir: {audio_dir}")
    print(f"Metadata : {metadata_path}")
    print(f"Created  : {len(created_files)} file(s)")
    print(f"Skipped  : {len(skipped_files)} file(s)")

    if created_files:
        print("\nGenerated tracks:")
        for path in created_files:
            print(f"- {path.name}")

    if skipped_files:
        print("\nSkipped existing tracks (use --force to overwrite):")
        for path in skipped_files:
            print(f"- {path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
