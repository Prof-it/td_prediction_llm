"""End-to-end CLI: `python scripts/run_pipeline.py --stage all`.

Stages: mine, label, train, xai, analysis, all.
Each stage is idempotent when its cache exists. Used by `make reproduce`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def stage_mine(args) -> None:
    from td_prediction import mining
    mining.mine_all(force=args.force)


def stage_label(args) -> None:
    from td_prediction import config, mining
    from td_prediction.labeling import llm_judge
    parsed = llm_judge.parse_results(variant_filter=args.variant)
    # Persist raw parsed outputs (label, confidence, rationale per commit) for audit trail.
    parsed_out = config.PATHS.results / f"parsed_labels_{args.variant}.csv"
    parsed.to_csv(parsed_out, index=False)
    df = mining.load_all_features()
    df = llm_judge.attach_llm_labels(df, variant=args.variant, parsed=parsed)
    out = config.PATHS.data / "features_with_llm_labels.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} (positive rate: {df['label_llm'].mean():.3f})")


def _refresh_labels(split, labels_df: "pd.DataFrame"):
    """Replace label columns in a cached split with fresh labels from features file.

    Splits are cached with whatever label_llm was current at generation time.
    After re-labeling (e.g. v1→v2) the fresh labels must be merged in so
    training uses up-to-date labels, not stale ones baked into the split CSVs.
    """
    label_cols = ["commit_uid", "label_llm", "label_satd", "label_consolidated"]
    cols = [c for c in label_cols if c in labels_df.columns]
    fresh = labels_df[cols]

    from td_prediction.data.splits import Split
    def _merge(df):
        df = df.drop(columns=[c for c in cols if c != "commit_uid" and c in df.columns])
        return df.merge(fresh, on="commit_uid", how="left")

    result = Split(
        train=_merge(split.train),
        val=_merge(split.val),
        test=_merge(split.test),
        name=split.name,
    )
    for part_name, part in [("train", result.train), ("val", result.val), ("test", result.test)]:
        n_missing = part["label_llm"].isna().sum() if "label_llm" in part.columns else len(part)
        if n_missing > 0:
            raise ValueError(
                f"{n_missing} commits in {split.name}/{part_name} have no matching labels "
                f"in features_with_llm_labels.csv. Re-run --stage label first."
            )
    return result


def stage_train(args) -> None:
    import pandas as pd
    from td_prediction import config
    from td_prediction.data.splits import FeatureSet, load_split
    from td_prediction.models import trainer

    labels_df = pd.read_csv(config.PATHS.data / "features_with_llm_labels.csv")

    label_col = args.label_col
    print(f"Training on label column: {label_col}")
    split = _refresh_labels(load_split(config.PATHS.splits, "time"), labels_df)
    bundles = trainer.sweep(
        split=split, label_col=label_col,
        model_kinds=("rf", "lgbm", "xgb"),
        imbalance_strategies=("none", "class_weight", "smote"),  # smoteenn excluded: too slow for sweep
        feature_sets=(FeatureSet.ALL, FeatureSet.CHANGE_ONLY),
    )
    trainer.bundles_to_df(bundles).to_csv(config.PATHS.results / "metrics_time.csv", index=False)
    for b in bundles:
        if b.feature_set == "all" and b.imbalance == "class_weight":
            trainer.save_bundle(b, config.PATHS.models)

    if args.lopo:
        lopo_bundles = []
        for repo in ["fastapi", "flask", "keras", "requests", "scrapy"]:
            s = _refresh_labels(load_split(config.PATHS.splits, f"lopo_{repo}"), labels_df)
            lopo_bundles.extend(trainer.sweep(
                split=s, label_col=label_col,
                model_kinds=("rf", "lgbm", "xgb"),
                imbalance_strategies=("none", "class_weight"),
                feature_sets=(FeatureSet.ALL, FeatureSet.CHANGE_ONLY),
            ))
        trainer.bundles_to_df(lopo_bundles).to_csv(config.PATHS.results / "metrics_lopo.csv", index=False)
    print("training done.")


def stage_xai(args) -> None:
    import joblib
    import pandas as pd
    from td_prediction import config
    from td_prediction.data.splits import FeatureSet, load_split, prepare_xy
    from td_prediction.xai import shap_explainer, lime_explainer, permutation
    bundle_path = next(config.PATHS.models.glob(f"time__{args.model}__class_weight__all.joblib"), None)
    if bundle_path is None:
        raise FileNotFoundError("No trained bundle found. Run --stage train first.")
    b = joblib.load(bundle_path)
    model, cols = b["model"], b["feature_cols"]
    split = load_split(config.PATHS.splits, "time")
    X_tr, _, _, fill_vals = prepare_xy(split.train, label_col="label_llm", feature_set=FeatureSet.ALL, feature_cols=cols)
    X_te, y_te, _, _ = prepare_xy(split.test, label_col="label_llm", feature_set=FeatureSet.ALL, feature_cols=cols, fill_values=fill_vals)
    shap_explainer.summary_plot(model, X_te, title=f"SHAP — {args.model}",
                                out_path=config.PATHS.figures / f"shap_summary_{args.model}.png")
    shap_r = shap_explainer.feature_importance(model, X_te)
    lime_r = lime_explainer.global_importance_sampled(model, X_tr, X_te, n_sample=60)
    perm_r = permutation.permutation_importance_df(model, X_te, y_te, n_repeats=5)
    shap_r.to_csv(config.PATHS.results / f"shap_ranking_{args.model}.csv", index=False)
    lime_r.to_csv(config.PATHS.results / f"lime_ranking_{args.model}.csv", index=False)
    perm_r.to_csv(config.PATHS.results / f"perm_ranking_{args.model}.csv", index=False)
    permutation.compare_rankings({"shap": shap_r, "lime": lime_r, "perm": perm_r}, top_k=15).to_csv(
        config.PATHS.results / f"xai_topk_{args.model}.csv", index=False)
    print("xai done.")


def stage_analysis(args) -> None:
    import pandas as pd
    from td_prediction import config
    from td_prediction.labeling import human_review
    df = pd.read_csv(config.PATHS.data / "features_with_llm_labels.csv")
    sample = human_review.sample_for_review(df, n_per_cell=25)
    human_review.write_review_sheet(sample, config.PATHS.results / "human_review_sheet.csv")
    print("analysis done.")


STAGES = {
    "mine": stage_mine,
    "label": stage_label,
    "train": stage_train,
    "xai": stage_xai,
    "analysis": stage_analysis,
}


def main() -> None:
    _ensure_src_on_path()
    import numpy as np
    from td_prediction import config
    np.random.seed(config.SEED)
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    p.add_argument("--force", action="store_true", help="re-mine even if cached")
    p.add_argument("--variant", default="v1_satd_filtered",
                   help="LLM prompt variant to use as label_llm")
    p.add_argument("--model", default="lgbm", choices=["rf", "lgbm", "xgb"])
    p.add_argument("--lopo", action="store_true", help="also run leave-one-project-out")
    p.add_argument("--label-col", default="label_consolidated",
                   choices=["label_consolidated", "label_llm", "label_satd"],
                   help="Label column to train against. Default: consolidated GT (human consensus where available, LLM otherwise).")
    args = p.parse_args()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for s in stages:
        print(f"\n=== STAGE: {s} ===")
        STAGES[s](args)


if __name__ == "__main__":
    main()
