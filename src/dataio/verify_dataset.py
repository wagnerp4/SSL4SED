import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Verify that annotation TSV files reference existing audio files.")
    parser.add_argument(
        "--conf",
        default="src/training/confs/stage2.yaml",
        type=str,
        help="Path to the training configuration YAML used to resolve annotation and audio folders.",
    )
    parser.add_argument(
        "--root",
        default=".",
        type=str,
        help="Project root used to resolve relative paths found in the configuration.",
    )
    parser.add_argument(
        "--tsv",
        action="append",
        default=None,
        help="Optional overrides in the form 'annotations.tsv:audio_folder'. Can be repeated. When provided, the configuration file is ignored.",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        default=False,
        help="Exit with status code 1 if any referenced files are missing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of missing file paths to display per dataset. Use 0 to print all.",
    )
    parser.add_argument(
        "--path-width",
        type=int,
        default=80,
        help="Maximum characters to display per missing path entry.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_path(root: Path, reference: str) -> Path:
    candidate = Path(reference)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def iter_config_pairs(config: Dict, root: Path) -> Iterable[Tuple[Path, Path, str]]:
    data_cfg = config.get("data", {})
    for key, value in data_cfg.items():
        if not key.endswith("_tsv"):
            continue
        folder = infer_audio_folder(key[:-4], data_cfg)
        if folder is None:
            continue
        tsv_path = resolve_path(root, value)
        audio_folder = resolve_path(root, folder)
        yield tsv_path, audio_folder, key


def infer_audio_folder(prefix: str, data_cfg: Dict) -> Optional[str]:
    direct_key = f"{prefix}folder"
    if direct_key in data_cfg:
        return data_cfg[direct_key]
    underscored_key = f"{prefix}_folder"
    if underscored_key in data_cfg:
        return data_cfg[underscored_key]
    parts = prefix.split("_")
    for end in range(len(parts) - 1, 0, -1):
        candidate = "_".join(parts[:end]) + "_folder"
        if candidate in data_cfg:
            return data_cfg[candidate]
    return None


def parse_manual_pairs(pairs: List[str], root: Path) -> Iterable[Tuple[Path, Path, str]]:
    for entry in pairs:
        if ":" not in entry:
            raise ValueError(f"Invalid --tsv entry '{entry}'. Expected 'annotations.tsv:audio_folder'.")
        tsv_ref, audio_ref = entry.split(":", 1)
        tsv_path = resolve_path(root, tsv_ref)
        audio_folder = resolve_path(root, audio_ref)
        label = Path(tsv_ref).stem
        yield tsv_path, audio_folder, label


def collect_missing_files(tsv_path: Path, audio_folder: Path) -> Dict[str, int]:
    df = pd.read_csv(tsv_path, sep="\t")
    missing: Dict[str, int] = {}
    for _, row in df.iterrows():
        filename = row.get("filename")
        if not isinstance(filename, str):
            continue
        resolved = resolve_audio_path(audio_folder, filename)
        if not resolved.is_file():
            key = str(resolved)
            missing[key] = missing.get(key, 0) + 1
    return missing


def resolve_audio_path(audio_folder: Path, filename: str) -> Path:
    candidate = audio_folder / filename
    if candidate.is_file():
        return candidate
    if "strong_real_16k" in str(audio_folder) and filename.endswith(".wav") and not filename.endswith("_16k.wav"):
        adjusted = audio_folder / f"{filename[:-4]}_16k.wav"
        return adjusted
    return candidate


def shorten_path(path: str, max_width: int) -> str:
    if len(path) <= max_width or max_width <= 0:
        return path
    if max_width <= 4:
        return path[-max_width:]
    segment = max_width - 3
    prefix_length = segment // 2
    suffix_length = segment - prefix_length
    return f"{path[:prefix_length]}...{path[-suffix_length:]}"


def format_entries(entries: List[Tuple[str, int]], limit: int, max_width: int, root: Path) -> List[Tuple[str, int]]:
    if limit <= 0 or limit >= len(entries):
        selected = entries
    else:
        head_count = limit // 2
        tail_count = limit - head_count
        head = entries[:head_count]
        tail = entries[-tail_count:]
        selected = head + tail
    formatted: List[Tuple[str, int]] = []
    for path, count in selected:
        display_path = path
        try:
            resolved = Path(path)
            display_path = str(resolved.relative_to(root))
        except Exception:
            display_path = path
        display_path = shorten_path(display_path, max_width)
        formatted.append((display_path, count))
    return formatted


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if args.tsv:
        pairs = list(parse_manual_pairs(args.tsv, root))
    else:
        config_path = resolve_path(root, args.conf)
        config = load_yaml_config(config_path)
        pairs = list(iter_config_pairs(config, root))
    if not pairs:
        print("No TSV to audio folder mappings could be determined.")
        sys.exit(1)
    any_missing = False
    for tsv_path, audio_folder, label in pairs:
        if not tsv_path.is_file():
            print(f"[{label}] Missing annotation file: {tsv_path}")
            any_missing = True
            continue
        if not audio_folder.is_dir():
            print(f"[{label}] Missing audio folder: {audio_folder}")
            any_missing = True
            continue
        missing_files = collect_missing_files(tsv_path, audio_folder)
        if missing_files:
            any_missing = True
            unique_count = len(missing_files)
            total_refs = sum(missing_files.values())
            print(f"[{label}] Missing {unique_count} files ({total_refs} references) in {audio_folder}.")
            entries = sorted(missing_files.items(), key=lambda item: item[0])
            to_show = format_entries(entries, args.limit, args.path_width, root)
            if to_show:
                max_len = max(len(p) for p, _ in to_show)
                column_width = min(max_len, args.path_width if args.path_width > 0 else max_len)
                for path, count in to_show:
                    count_label = f"{count}x" if count > 1 else "1x"
                    print(f"    {path.ljust(column_width)}  {count_label:>4}")
            if args.limit > 0 and args.limit < unique_count:
                remaining = unique_count - len(to_show)
                print(f"    ... {remaining} more not shown. Increase --limit to view all.")
        else:
            print(f"[{label}] All files referenced by {tsv_path} are present in {audio_folder}.")
        print("")
    
    if any_missing and args.fail_on_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()

