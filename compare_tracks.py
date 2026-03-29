"""
compare_tracks.py
-----------------
CLI entry point. Run:
    python compare_tracks.py song_a.mp3 song_b.mp3
    python compare_tracks.py song_a.mp3 song_b.mp3 --clap
"""

import argparse
import json
from feature_extractor import FeatureExtractor
from similarity_engine import compare_tracks


def main():
    parser = argparse.ArgumentParser(
        description="Compare two audio tracks for AI attribution."
    )
    parser.add_argument("track_a", help="Path to the first track (original)")
    parser.add_argument("track_b", help="Path to the second track (suspected AI)")
    parser.add_argument(
        "--clap", action="store_true",
        help="Use CLAP neural embeddings (slower, more accurate)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output in JSON format"
    )
    args = parser.parse_args()

    # Feature extraction
    extractor = FeatureExtractor(use_clap=args.clap)

    print(f"\nTrack A: {args.track_a}")
    features_a = extractor.extract(args.track_a)

    print(f"\nTrack B: {args.track_b}")
    features_b = extractor.extract(args.track_b)

    # Comparison
    print("\nCalculating similarity...")
    result = compare_tracks(features_a, features_b, verbose=not args.json)

    if args.json:
        # Remove chunk_scores from JSON output for clarity
        output = {
            "track_a": args.track_a,
            "track_b": args.track_b,
            "attribution_score": round(result["attribution_score"], 4),
            "interpretation": result["interpretation"],
            "feature_breakdown": {
                k: round(v, 4) if v is not None else None
                for k, v in result["feature_breakdown"].items()
            }
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()