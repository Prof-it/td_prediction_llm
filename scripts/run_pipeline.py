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
    df = mining.load_all_features()
    df = llm_judge.attach_llm_labels(df, variant=args.variant, parsed=parsed)
    out = config.PATHS.data / "features_with_llm_labels.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} (positive rate: {df['label_llm'].mean():.3f})")


def stage_train(args) -> None:
    import pandas as pd
    from pathlib import Path
    from td_prediction import config
    from td_prediction.data.splits import FeatureSet, load_split
    from td_prediction.models import trainer

    split = load_split(config.PATHS.splits, "time")
    bundles = trainer.sweep(
        split=split, label_col="label_llm",
        model_kinds=("rf", "lgbm", "xgb"),
        imbalance_strategies=("none", "class_weight", "smote"),
        feature_sets=(FeatureSet.ALL, FeatureSet.NO_MATURITY, FeatureSet.CHANGE_ONLY),
    )
    trainer.bundles_to_df(bundles).to_csv(config.PATHS.results / "metrics_time.csv", index=False)
    for b in bundles:
        if b.feature_set == "all" and b.imbalance == "class_weight":
            trainer.save_bundle(b, config.PATHS.models)

    if args.lopo:
        lopo_bundles = []
        for repo in ["fastapi", "flask", "keras", "requests", "scrapy"]:
            s = load_split(config.PATHS.splits, f"lopo_{repo}")
            lopo_bundles.extend(trainer.sweep(
                split=s, label_col="label_llm",
                model_kinds=("rf", "lgbm", "xgb"),
                imbalance_strategies=("none", "class_weight"),
                feature_sets=(FeatureSet.ALL, FeatureSet.NO_MATURITY),
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
    X_tr, _, _ = prepare_xy(split.train, label_col="label_llm", feature_set=FeatureSet.ALL, feature_cols=cols)
    X_te, y_te, _ = prepare_xy(split.test, label_col="label_llm", feature_set=FeatureSet.ALL, feature_cols=cols)
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
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    p.add_argument("--force", action="store_true", help="re-mine even if cached")
    p.add_argument("--variant", default="v1_satd_filtered",
                   help="LLM prompt variant to use as label_llm")
    p.add_argument("--model", default="lgbm", choices=["rf", "lgbm", "xgb"])
    p.add_argument("--lopo", action="store_true", help="also run leave-one-project-out")
    args = p.parse_args()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for s in stages:
        print(f"\n=== STAGE: {s} ===")
        STAGES[s](args)


if __name__ == "__main__":
    main()
