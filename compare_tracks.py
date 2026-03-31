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
from similarity_engine import compare_tracks, ai_detection_score


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

    # AI detection (per track)
    ai_a = ai_detection_score(features_a)
    ai_b = ai_detection_score(features_b)

    # Attribution (pair-level)
    print("\nCalculating similarity...")
    result = compare_tracks(features_a, features_b, verbose=not args.json)

    if args.json:
        output = {
            "track_a": args.track_a,
            "track_b": args.track_b,
            "attribution_score": round(result["attribution_score"], 4),
            "interpretation": result["interpretation"],
            "feature_breakdown": {
                k: round(v, 4) if v is not None else None
                for k, v in result["feature_breakdown"].items()
            },
            "diagnostics": {
                k: round(v, 4) if v is not None else None
                for k, v in result.get("diagnostics", {}).items()
            },
            "ai_detection": {
                "track_a": ai_a,
                "track_b": ai_b,
            },
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*50}")
        print(f"AI Detection - Track A: {ai_a['ai_score']:.3f} - {ai_a['interpretation']}")
        print(f"AI Detection - Track B: {ai_b['ai_score']:.3f} - {ai_b['interpretation']}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()