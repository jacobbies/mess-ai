#!/usr/bin/env python3
"""Build/publish production FAISS artifacts for track or clip retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path

from mess.config import mess_config
from mess.search.faiss_index import (
    build_clip_artifact,
    build_track_artifact,
    find_latest_artifact_dir,
    save_artifact,
    upload_artifact_to_s3,
)


def _default_features_dir(dataset: str, kind: str) -> Path:
    base = mess_config.data_root / "embeddings" / f"{dataset}-emb"
    return base / ("segments" if kind == "clip" else "aggregated")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and optionally publish FAISS artifacts")
    parser.add_argument("--dataset", default="smd", help="Dataset name (default: smd)")
    parser.add_argument(
        "--kind",
        choices=["track", "clip"],
        default="clip",
        help="Artifact type to build (default: clip)",
    )
    parser.add_argument(
        "--artifact-name",
        default=None,
        help="Artifact namespace name (default: clip_index or track_index)",
    )
    parser.add_argument(
        "--features-dir",
        default=None,
        help="Features source dir; defaults to data/embeddings/<dataset>-emb/{aggregated|segments}",
    )
    parser.add_argument("--layer", type=int, default=None, help="Optional layer 0-12")
    parser.add_argument(
        "--index-type",
        choices=["flatip", "ivfflat"],
        default="flatip",
        help="FAISS index type (default: flatip)",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=None,
        help="IVF coarse centroid count (only used for ivfflat)",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(mess_config.data_root / "indices"),
        help="Root output directory for artifacts",
    )
    parser.add_argument("--model-name", default=mess_config.model_name, help="Model identifier")
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=mess_config.segment_duration,
        help="Clip segmentation duration in seconds",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=mess_config.overlap_ratio,
        help="Clip segmentation overlap ratio",
    )
    parser.add_argument("--s3-bucket", default=None, help="S3 bucket for upload")
    parser.add_argument(
        "--s3-prefix",
        default="mess/faiss",
        help="S3 prefix root for uploaded artifacts",
    )
    parser.add_argument(
        "--upload-latest",
        action="store_true",
        help="Upload latest artifact under --artifact-name to S3",
    )

    args = parser.parse_args()

    artifact_name = args.artifact_name or ("clip_index" if args.kind == "clip" else "track_index")
    features_dir = Path(args.features_dir) if args.features_dir else _default_features_dir(
        args.dataset, args.kind
    )

    if not features_dir.exists():
        print(f"Error: features directory not found: {features_dir}")
        return 1

    if args.kind == "clip":
        artifact = build_clip_artifact(
            dataset=args.dataset,
            features_dir=features_dir,
            artifact_name=artifact_name,
            layer=args.layer,
            segment_duration=args.segment_duration,
            overlap_ratio=args.overlap_ratio,
            index_type=args.index_type,
            model_name=args.model_name,
            nlist=args.nlist or 1024,
        )
    else:
        artifact = build_track_artifact(
            dataset=args.dataset,
            features_dir=features_dir,
            artifact_name=artifact_name,
            layer=args.layer,
            index_type=args.index_type,
            model_name=args.model_name,
            nlist=args.nlist or 256,
        )

    saved_dir = save_artifact(artifact, artifact_root=args.artifact_root)
    print(f"Saved artifact: {saved_dir}")

    if args.s3_bucket:
        if args.upload_latest:
            to_upload = find_latest_artifact_dir(args.artifact_root, artifact_name=artifact_name)
        else:
            to_upload = saved_dir

        uploaded = upload_artifact_to_s3(
            to_upload,
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            upload_latest_pointer=True,
        )
        print(f"Uploaded {len(uploaded)} object(s) to s3://{args.s3_bucket}/{args.s3_prefix}")
        print(f"Artifact uploaded: {to_upload}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
