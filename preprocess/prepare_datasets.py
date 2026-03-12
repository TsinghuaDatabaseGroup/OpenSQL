#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

SUPPORTED_DATASETS = ("BIRD_dev", "BIRD_train", "Spider")
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


def unzip_file(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(output_dir)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def rename_if_needed(src: Path, dst: Path, dataset_tag: str) -> None:
    if src.exists():
        if dst.exists():
            print(f"[{dataset_tag}] Rename skipped: both exist, keep target {dst.name}.")
            return
        src.rename(dst)
        print(f"[{dataset_tag}] Renamed {src.name} -> {dst.name}")
        return

    if dst.exists():
        print(f"[{dataset_tag}] Rename skipped: {dst.name} already exists.")
        return

    print(f"[{dataset_tag}] Rename skipped: neither {src.name} nor {dst.name} exists.")


def prepare_bird_dev(dataset_dir: Path) -> None:
    dev_zip = dataset_dir / "dev.zip"
    extracted_dir = dataset_dir / "dev_20240627"
    target_dir = dataset_dir / "BIRD_dev"
    dev_databases_zip = target_dir / "dev_databases.zip"

    print("[BIRD_dev] Step1: unzip dev.zip")
    if extracted_dir.exists() and target_dir.exists():
        raise RuntimeError(f"Both {extracted_dir} and {target_dir} exist. Please keep only one of them before running again.")
    if not extracted_dir.exists() and not target_dir.exists():
        unzip_file(dev_zip, dataset_dir)
    else:
        print("[BIRD_dev] Step1 skipped: archive already extracted.")

    print("[BIRD_dev] Step2: rename dev_20240627 to BIRD_dev")
    if extracted_dir.exists() and not target_dir.exists():
        extracted_dir.rename(target_dir)
    elif target_dir.exists() and not extracted_dir.exists():
        print("[BIRD_dev] Step2 skipped: target directory already named BIRD_dev.")
    else:
        raise FileNotFoundError(f"Cannot continue: neither {extracted_dir} nor {target_dir} exists.")

    print("[BIRD_dev] Step3: unzip BIRD_dev/dev_databases.zip")
    if dev_databases_zip.exists():
        unzip_file(dev_databases_zip, target_dir)
    else:
        print("[BIRD_dev] Step3 skipped: dev_databases.zip is missing (possibly already processed).")

    print("[BIRD_dev] Step4: remove BIRD_dev/__MACOSX")
    remove_path(target_dir / "__MACOSX")

    print("[BIRD_dev] Step5: remove BIRD_dev/dev_databases.zip")
    remove_path(dev_databases_zip)

    print(f"[BIRD_dev] Done. Output directory: {target_dir}")


def prepare_bird_train(dataset_dir: Path) -> None:
    train_zip = dataset_dir / "train.zip"
    extracted_dir = dataset_dir / "train"
    target_dir = dataset_dir / "BIRD_train"
    train_databases_zip = target_dir / "train_databases.zip"

    print("[BIRD_train] Step1: unzip train.zip")
    if extracted_dir.exists() and target_dir.exists():
        raise RuntimeError(f"Both {extracted_dir} and {target_dir} exist. Please keep only one of them before running again.")
    if not extracted_dir.exists() and not target_dir.exists():
        unzip_file(train_zip, dataset_dir)
    else:
        print("[BIRD_train] Step1 skipped: archive already extracted.")

    print("[BIRD_train] Step2: rename train to BIRD_train")
    if extracted_dir.exists() and not target_dir.exists():
        extracted_dir.rename(target_dir)
    elif target_dir.exists() and not extracted_dir.exists():
        print("[BIRD_train] Step2 skipped: target directory already named BIRD_train.")
    else:
        raise FileNotFoundError(f"Cannot continue: neither {extracted_dir} nor {target_dir} exists.")

    print("[BIRD_train] Step3: unzip BIRD_train/train_databases.zip")
    if train_databases_zip.exists():
        unzip_file(train_databases_zip, target_dir)
    else:
        print("[BIRD_train] Step3 skipped: train_databases.zip is missing (possibly already processed).")

    print("[BIRD_train] Step4: remove BIRD_train/__MACOSX")
    remove_path(target_dir / "__MACOSX")

    print("[BIRD_train] Step5: remove BIRD_train/train_databases.zip")
    remove_path(train_databases_zip)

    print("[BIRD_train] Step6: remove dataset/__MACOSX")
    remove_path(dataset_dir / "__MACOSX")

    print("[BIRD_train] Step7: rename app_store description files for schema alignment")
    app_store_desc_dir = target_dir / "train_databases" / "app_store" / "database_description"
    rename_if_needed(
        app_store_desc_dir / "googleplaystore.csv",
        app_store_desc_dir / "playstore.csv",
        dataset_tag="BIRD_train",
    )
    rename_if_needed(
        app_store_desc_dir / "googleplaystore_user_reviews.csv",
        app_store_desc_dir / "user_reviews.csv",
        dataset_tag="BIRD_train",
    )
    coinmarketcap_desc_dir = target_dir / "train_databases" / "coinmarketcap" / "database_description"
    rename_if_needed(
        coinmarketcap_desc_dir / "Coins.csv",
        coinmarketcap_desc_dir / "coins.csv",
        dataset_tag="BIRD_train",
    )
    rename_if_needed(
        coinmarketcap_desc_dir / "Historical.csv",
        coinmarketcap_desc_dir / "historical.csv",
        dataset_tag="BIRD_train",
    )
    craftbeer_desc_dir = target_dir / "train_databases" / "craftbeer" / "database_description"
    rename_if_needed(
        craftbeer_desc_dir / "Breweries.csv",
        craftbeer_desc_dir / "breweries.csv",
        dataset_tag="BIRD_train",
    )
    rename_if_needed(
        craftbeer_desc_dir / "Beers.csv",
        craftbeer_desc_dir / "beers.csv",
        dataset_tag="BIRD_train",
    )
    mondial_geo_desc_dir = target_dir / "train_databases" / "mondial_geo" / "database_description"
    rename_if_needed(
        mondial_geo_desc_dir / "mergeswith.csv",
        mondial_geo_desc_dir / "mergesWith.csv",
        dataset_tag="BIRD_train",
    )
    rename_if_needed(
        mondial_geo_desc_dir / "mountainonisland.csv",
        mondial_geo_desc_dir / "mountainOnIsland.csv",
        dataset_tag="BIRD_train",
    )
    shooting_desc_dir = target_dir / "train_databases" / "shooting" / "database_description"
    rename_if_needed(
        shooting_desc_dir / "Incidents.csv",
        shooting_desc_dir / "incidents.csv",
        dataset_tag="BIRD_train",
    )
    student_loan_desc_dir = target_dir / "train_databases" / "student_loan" / "database_description"
    rename_if_needed(
        student_loan_desc_dir / "filed_for_bankruptcy.csv",
        student_loan_desc_dir / "filed_for_bankrupcy.csv",
        dataset_tag="BIRD_train",
    )
    wdi_desc_dir = target_dir / "train_databases" / "world_development_indicators" / "database_description"
    rename_if_needed(
        wdi_desc_dir / "FootNotes.csv",
        wdi_desc_dir / "Footnotes.csv",
        dataset_tag="BIRD_train",
    )

    print(f"[BIRD_train] Done. Output directory: {target_dir}")


def load_json_list(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {path}, got {type(data).__name__}.")
    return data


def prepare_spider(dataset_dir: Path) -> None:
    spider_zip = dataset_dir / "spider_data.zip"
    spider_dir = dataset_dir / "spider_data"
    train_spider_path = spider_dir / "train_spider.json"
    train_others_path = spider_dir / "train_others.json"
    train_path = spider_dir / "train.json"

    print("[Spider] Step1: unzip spider_data.zip")
    if not spider_dir.exists():
        unzip_file(spider_zip, dataset_dir)
    else:
        print("[Spider] Step1 skipped: archive already extracted.")
    if not spider_dir.exists():
        raise FileNotFoundError(f"Cannot continue: {spider_dir} was not found after unzip.")

    print("[Spider] Step2: merge train_spider.json and train_others.json to train.json")
    if not train_spider_path.exists():
        raise FileNotFoundError(f"Missing required file: {train_spider_path}")
    if not train_others_path.exists():
        raise FileNotFoundError(f"Missing required file: {train_others_path}")
    merged_data = load_json_list(train_spider_path)
    merged_data.extend(load_json_list(train_others_path))
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print("[Spider] Step3: remove dataset/__MACOSX")
    remove_path(dataset_dir / "__MACOSX")

    print(f"[Spider] Done. Output directory: {spider_dir}")


def prepare_single_dataset(dataset_name: str, dataset_dir: Path) -> None:
    if dataset_name == "BIRD_dev":
        prepare_bird_dev(dataset_dir)
        return
    if dataset_name == "BIRD_train":
        prepare_bird_train(dataset_dir)
        return
    if dataset_name == "Spider":
        prepare_spider(dataset_dir)
        return
    raise NotImplementedError(f"{dataset_name} is not implemented yet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official SQL benchmark datasets into project layout.")
    select_group = parser.add_mutually_exclusive_group(required=True)
    select_group.add_argument("--dataset", choices=SUPPORTED_DATASETS, help="Prepare only one dataset.")
    select_group.add_argument("--all", action="store_true", help="Prepare all supported datasets.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help=f"Directory that stores official zip files (default: {DEFAULT_DATASET_DIR}).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    dataset_names = SUPPORTED_DATASETS if args.all else (args.dataset,)
    not_implemented: list[str] = []

    for dataset_name in dataset_names:
        try:
            prepare_single_dataset(dataset_name, dataset_dir)
        except NotImplementedError as exc:
            print(f"[{dataset_name}] {exc}", file=sys.stderr)
            not_implemented.append(dataset_name)

    if not_implemented:
        print("Pending datasets not implemented yet: " + ", ".join(not_implemented), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
