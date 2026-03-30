"""
evaluate_mippia.py
------------------
Runs the attribution system on all complete pairs in the MIPPIA SMP dataset.
"""

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from feature_extractor import FeatureExtractor
from similarity_engine import compare_tracks

DATASET_DIR  = Path("smp_dataset/final_dataset")
METADATA_CSV = Path("smp_dataset/Final_dataset_pairs.csv")
RESULTS_FILE = "mippia_results.json"


def get_complete_pairs(dataset_dir: Path):
    """Return only pairs that contain exactly 2 audio files."""
    pairs = []
    for pair_dir in sorted(dataset_dir.iterdir(), key=lambda p: int(p.name)):
        if not pair_dir.is_dir():
            continue
        files = sorted(pair_dir.glob("*.wav"), key=lambda path: path.name.casefold())
        if len(files) == 2:
            pairs.append({
                "pair_id": pair_dir.name,
                "track_a": str(files[0]),
                "track_b": str(files[1]),
            })
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run attribution evaluation only on complete pairs from the MIPPIA SMP dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to the smp_dataset/final_dataset directory",
    )
    parser.add_argument(
        "--results-file",
        default=RESULTS_FILE,
        help="JSON file for storing results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on complete pairs for a quick dry run",
    )
    return parser.parse_args()


def load_metadata_lookup(csv_path: Path) -> dict:
    """Return a dict: pair_id -> {relation, ori_title, comp_title}."""
    if not csv_path.exists():
        return {}
    import csv
    lookup = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            pid = str(row.get("pair_number", "")).strip()
            if pid:
                lookup[pid] = {
                    "relation":   row.get("relation", "").strip(),
                    "ori_title":  row.get("ori_title", "").strip(),
                    "comp_title": row.get("comp_title", "").strip(),
                }
    return lookup


def load_existing_results(results_file: str) -> tuple[list, set]:
    """Load existing results and return (results, done_pair_ids)."""
    path = Path(results_file)
    if not path.exists():
        return [], set()
    try:
        with open(path, encoding="utf-8") as f:
            results = json.load(f)
        done = {str(r["pair_id"]) for r in results if "pair_id" in r}
        print(f"Loaded {len(results)} existing results. Resuming from pair {max(done, key=int) if done else '?'}...\n")
        return results, done
    except Exception as e:
        print(f"WARNING: Could not load existing results: {e}\n")
        return [], set()


def save_results(results: list, results_file: str) -> None:
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")

    pairs = get_complete_pairs(args.dataset_dir)
    if args.limit is not None:
        pairs = pairs[:args.limit]

    print(f"Found {len(pairs)} complete pairs.\n")

    meta = load_metadata_lookup(METADATA_CSV)

    results, done_ids = load_existing_results(args.results_file)
    pairs_to_run = [p for p in pairs if p["pair_id"] not in done_ids]
    if done_ids:
        print(f"Skipping {len(done_ids)} already processed pairs. Remaining: {len(pairs_to_run)}\n")

    extractor = FeatureExtractor(use_clap=False)

    for i, pair in enumerate(pairs_to_run):
        print(f"[{i+1}/{len(pairs_to_run)}] Pair {pair['pair_id']}")
        print(f"  A: {Path(pair['track_a']).name}")
        print(f"  B: {Path(pair['track_b']).name}")

        try:
            feats_a = extractor.extract(pair["track_a"])
            feats_b = extractor.extract(pair["track_b"])
            result = compare_tracks(feats_a, feats_b, verbose=False)

            score = round(result["attribution_score"], 4)
            interp = result["interpretation"]
            print(f"  Score: {score:.4f} — {interp}\n")

            pair_meta = meta.get(pair["pair_id"], {})
            results.append({
                "pair_id":           pair["pair_id"],
                "relation":          pair_meta.get("relation"),
                "ori_title":         pair_meta.get("ori_title"),
                "comp_title":        pair_meta.get("comp_title"),
                "track_a":           Path(pair["track_a"]).name,
                "track_b":           Path(pair["track_b"]).name,
                "attribution_score": score,
                "interpretation":    interp,
                "feature_breakdown": {
                    k: round(v, 4) if v is not None else None
                    for k, v in result["feature_breakdown"].items()
                },
            })
            save_results(results, args.results_file)

        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "pair_id": pair["pair_id"],
                "track_a": Path(pair["track_a"]).name,
                "track_b": Path(pair["track_b"]).name,
                "error":   str(e),
            })
            save_results(results, args.results_file)

    save_results(results, args.results_file)
    print(f"\nResults saved to: {args.results_file}")

    # Summary
    scored = [r for r in results if "attribution_score" in r]
    if scored:
        scores = [r["attribution_score"] for r in scored]
        print(f"\n{'='*50}")
        print(f"Total evaluated pairs: {len(scored)}")
        print(f"Mean Attribution Score: {sum(scores)/len(scores):.4f}")
        print(f"Max: {max(scores):.4f} | Min: {min(scores):.4f}")
        print(f"\nInterpretation distribution:")
        from collections import Counter
        for interp, count in Counter(r["interpretation"] for r in scored).most_common():
            print(f"  {count:2d}x  {interp}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
