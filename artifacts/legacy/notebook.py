import marimo
# This notebook can be opened and run using Marimo (https://github.com/marimo-team/marimo) or as a regular Python (.py) file

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bachelor's Thesis – Prediction of Technical Debt Using Machine Learning Based on Code Metrics and Commit Histories

    (For full reproduction: install the imported modules, clone the repos and configure their paths if necessary, and set the OPENAI_API_KEY environment variable)
    """)
    return


@app.cell
def _():
    import os, re, json, math, asyncio, datetime as dt, collections, hashlib
    from pathlib import Path
    import pandas as pd, numpy as np
    from pydriller import Repository
    from pydriller.domain.commit import ModificationType as MT
    from tqdm import tqdm

    BASE_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
    REPOS = {
        'requests':  {'path': 'repos/requests',        'branch': 'main'},
        'fastapi':  {'path': 'repos/fastapi',        'branch': 'master'},
        'scrapy':   {'path': 'repos/scrapy',         'branch': 'master'},
        'flask':    {'path': 'repos/flask',          'branch': 'main'},
        'keras':    {'path': 'repos/keras',          'branch': 'master'},
    }

    CUTOFF=dt.datetime(2025,6,15,23,59,59, tzinfo=dt.timezone.utc)
    SATD=re.compile(r'\b(TODO|FIXME|BUG|HACK|XXX|WORKAROUND|TEMP|KLUDGE|UGLY|DIRTY|BROKEN|FIX)\b',re.I)

    LLM_FRAC=.25
    LLM_MODEL='gpt-4.1-mini'

    OUT=Path('data'); OUT.mkdir(exist_ok=True)
    SPLIT=Path('splits'); SPLIT.mkdir(exist_ok=True)
    LLM_DIR=Path('llm_batch'); LLM_DIR.mkdir(exist_ok=True)
    return (
        BASE_DIR,
        CUTOFF,
        LLM_DIR,
        LLM_MODEL,
        OUT,
        Path,
        REPOS,
        Repository,
        SATD,
        SPLIT,
        collections,
        dt,
        json,
        np,
        os,
        pd,
        re,
        tqdm,
    )


@app.cell
def _(SATD):
    def is_comment_or_docstring(line):
        _line = _line.strip()
        return _line.startswith('#') or _line.startswith('"""') or _line.startswith("'''") or _line.endswith('"""') or _line.endswith("'''")

    def satd_delta(mod):
        add = rem = 0
        for _, _line in mod.diff_parsed['added']:
            if is_comment_or_docstring(_line) and SATD.search(_line):
                add = add + 1
        for _, _line in mod.diff_parsed['deleted']:
            if is_comment_or_docstring(_line) and SATD.search(_line):
                rem = rem + 1
        return add - rem

    def is_py(m):
        fp = m.new_path or m.old_path or ''
        return fp.endswith('.py')

    def quick_hunks_count(mod):
        diff = mod.diff.splitlines()
        state = False
        h = 0
        for _line in diff:
            if _line.startswith(('+', '-')):
                if not state:
                    state = True
                    h = h + 1
            else:
                state = False
        return h

    def diff_snippet(txt, max_lines=3000):
        return '\n'.join(txt.split('\n')[:max_lines])

    def summary(r):
        return f"adds {r['lines_added']} LOC ({r['lines_deleted']} del) across {r['files_changed']} py‑files; ΔCCmax {r['cc_delta_max']}; methods {r['n_methods_changed']}; commits90d {r['n_commits_file_past90d']}; authors total {r['n_authors_till_now']}"

    return diff_snippet, is_py, quick_hunks_count, satd_delta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mining & Feature‑Engineering
    """)
    return


@app.cell
def _():
    import csv
    from pydriller.metrics.process.commits_count import CommitsCount
    from pydriller.metrics.process.contributors_experience import ContributorsExperience
    from pydriller.metrics.process.history_complexity import HistoryComplexity
    GENERATE_LLM_BATCH = True
    MAX_REQ = 50000
    MAX_MB = 180
    # LLM Batch Setup
    # Constants
    COLS = ['repo_id', 'commit_hash', 'commit_uid', 'commit_date', 'lines_added', 'lines_deleted', 'files_changed', 'hunks', 'n_methods_changed', 'cc_delta_sum', 'cc_delta_max', 'complexity_current_sum', 'churn_delta', 'churn_cum', 'contributors_count', 'contributors_cum', 'n_authors_till_now', 'n_commits_file_past90d', 'commits_count_file', 'contributors_experience', 'history_complexity', 'dmm_unit_complexity', 'dmm_unit_size', 'dmm_unit_interfacing', 'satd_delta', 'label_td_satd']
    return (
        COLS,
        CommitsCount,
        ContributorsExperience,
        GENERATE_LLM_BATCH,
        HistoryComplexity,
        MAX_MB,
        csv,
    )


@app.cell
def _(LLM_MODEL, SATD, re):
    # Helper functions for LLM prompt
    def summary_llm(r):
        fields = [
            f"Lines Added: {r['lines_added']}",
            f"Lines Deleted: {r['lines_deleted']}",
            f"Files Changed: {r['files_changed']}",
            f"Hunks: {r['hunks']}",
            f"Methods Changed: {r['n_methods_changed']}",
            f"Complexity Δ (Sum/Max): {r['cc_delta_sum']}/{r['cc_delta_max']}",
            f"Churn Δ: {r['churn_delta']}",
            f"Churn Cumulative: {r['churn_cum']}",
            f"Contributors (this commit): {r['contributors_count']}",
            f"Commits (past 90d): {r['n_commits_file_past90d']}",
            f"Contributors (cumulative): {r['contributors_cum']}",
            f"DMM Complexity: {r['dmm_unit_complexity']}"
        ]
        return " | ".join(fields)

    def prepare_llm_prompt_original(row, diff_text):
        prompt = (
            "You are a senior reviewer.\n\n"
            "Commit Summary:\n"
            f"{summary_llm(row)}\n\n"
            "DIFF:\n"
            f"{diff_text}\n\n"
            "Question: Does this commit introduce technical debt? Answer yes or no."
        )
        return {
            'custom_id': row['commit_uid'] + "-original-prompt",
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': LLM_MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1,
                'temperature': 0
            }
        }

    def prepare_llm_prompt_satd_filtered(row, diff_text):
        filtered_diff = re.sub(SATD, "", diff_text)
        prompt = (
            "You are a senior code reviewer. Based on the code change and metrics summary below, "
            "assess if this change might lead to long-term maintainability issues. Answer with 'yes' or 'no'.\n\n"
            f"Commit Summary:\n{summary_llm(row)}\n\n"
            f"DIFF:\n{filtered_diff}"
        )
        return {
            'custom_id': row['commit_uid'] + "-satd-filtered",
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': LLM_MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1,
                'temperature': 0
            }
        }

    def prepare_llm_prompt_diff_removed(row):
        prompt = (
            "You are a senior code reviewer. Based on the code change and metrics summary below, "
            "assess if this change might lead to long-term maintainability issues. Answer with 'yes' or 'no'.\n\n"
            f"Commit Summary:\n{summary_llm(row)}"
        )
        return {
            'custom_id': row['commit_uid'] + "-diff-removed",
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': {
                'model': LLM_MODEL,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1,
                'temperature': 0
            }
        }

    return (
        prepare_llm_prompt_diff_removed,
        prepare_llm_prompt_original,
        prepare_llm_prompt_satd_filtered,
    )


@app.cell
def _(
    BASE_DIR,
    COLS,
    CUTOFF,
    CommitsCount,
    ContributorsExperience,
    GENERATE_LLM_BATCH,
    HistoryComplexity,
    LLM_DIR,
    MAX_MB,
    OUT,
    REPOS,
    Repository,
    collections,
    csv,
    diff_snippet,
    dt,
    is_py,
    json,
    prepare_llm_prompt_diff_removed,
    prepare_llm_prompt_original,
    prepare_llm_prompt_satd_filtered,
    quick_hunks_count,
    satd_delta,
    tqdm,
):
    for _repo, cfg in tqdm(REPOS.items(), desc='Repos'):
        _path = (BASE_DIR / cfg['path']).expanduser().resolve()
        if not _path.is_dir():
            raise FileNotFoundError(f'{_path} does not exist – check the REPOS entry!')
        file_prev_cc = collections.defaultdict(int)
        file_current_cc = collections.defaultdict(int)
        file_churn_cum = collections.defaultdict(int)
        file_authors = collections.defaultdict(set)
        file_times = collections.defaultdict(list)
        file_contributors = collections.defaultdict(set)
        START_DATE = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        print(f'[{_repo}] Computing process metrics...')
        commits_count_dict = CommitsCount(str(_path), since=START_DATE, to=CUTOFF).count()
        contrib_exp_dict = ContributorsExperience(str(_path), since=START_DATE, to=CUTOFF).count()
        hist_complexity_dict = HistoryComplexity(str(_path), since=START_DATE, to=CUTOFF).count()
        print(f'[{_repo}] Process metrics computed.')
        csv_file = OUT / f'features_{_repo}.csv'
        writer = csv.DictWriter(csv_file.open('w', newline=''), fieldnames=COLS)
        writer.writeheader()
        part = 1
        handles = {'orig': None, 'nosatd': None, 'nodiff': None}
        cur_sizes = {'orig': 0, 'nosatd': 0, 'nodiff': 0}
        commits = Repository(path_to_repo=str(_path), only_in_branch=cfg['branch'], to=CUTOFF, only_modifications_with_file_types=['.py'], num_workers=64, skip_whitespaces=True, histogram_diff=True).traverse_commits()
        for c in tqdm(commits, desc=_repo, leave=False):
            py = [m for m in c.modified_files if is_py(m)]
            if not py:
                continue
            la = sum((m.added_lines for m in py))
            ld = sum((m.deleted_lines for m in py))
            files_changed = len(py)
            hunks = sum((quick_hunks_count(m) for m in py))
            n_methods = sum((len(m.changed_methods) for m in py))
            cc_delta_sum = 0
            cc_delta_max = 0
            churn_delta = 0
            complexity_current_sum = 0
            churn_cum_sum = 0
            contributors_in_commit = set()
            for m in py:
                fp = m.new_path or m.old_path
                delta_cc = (m.complexity or 0) - file_prev_cc[fp]
                cc_delta_sum = cc_delta_sum + delta_cc
                cc_delta_max = max(cc_delta_max, delta_cc)
                file_prev_cc[fp] = m.complexity or 0
                file_current_cc[fp] = m.complexity or 0
                complexity_current_sum = complexity_current_sum + file_current_cc[fp]
                churn_this = m.added_lines + m.deleted_lines
                churn_delta = churn_delta + churn_this
                file_churn_cum[fp] = file_churn_cum[fp] + churn_this
                churn_cum_sum = churn_cum_sum + file_churn_cum[fp]
                file_contributors[fp].add(c.author.email)
                contributors_in_commit.update(file_contributors[fp])
                file_authors[fp].add(c.author.email)
                file_times[fp].append(c.author_date)
            cutoff90 = c.author_date - dt.timedelta(days=90)
            n_commits90 = sum((len([t for t in ts if t >= cutoff90]) for fp, ts in file_times.items() if fp in [m.new_path or m.old_path for m in py]))
            commits_count_file = sum((commits_count_dict.get(m.new_path or m.old_path, 0) for m in py))
            contributors_experience = sum((contrib_exp_dict.get(m.new_path or m.old_path, 0) for m in py))
            history_complexity = sum((hist_complexity_dict.get(m.new_path or m.old_path, 0) for m in py))
            satd = sum((satd_delta(m) for m in py))
            label_td_satd = 1 if satd > 0 else 0
            row_dict = {'repo_id': _repo, 'commit_hash': c.hash, 'commit_uid': f'{_repo}#{c.hash}', 'commit_date': c.author_date.isoformat(), 'lines_added': la, 'lines_deleted': ld, 'files_changed': files_changed, 'hunks': hunks, 'n_methods_changed': n_methods, 'cc_delta_sum': cc_delta_sum, 'cc_delta_max': cc_delta_max, 'complexity_current_sum': complexity_current_sum, 'churn_delta': churn_delta, 'churn_cum': churn_cum_sum, 'contributors_count': len(contributors_in_commit), 'contributors_cum': sum((len(file_contributors[fp]) for m in py for fp in [m.new_path or m.old_path])), 'n_authors_till_now': len({a for s in file_authors.values() for a in s}), 'n_commits_file_past90d': n_commits90, 'commits_count_file': commits_count_file, 'contributors_experience': contributors_experience, 'history_complexity': history_complexity, 'dmm_unit_complexity': c.dmm_unit_complexity, 'dmm_unit_size': c.dmm_unit_size, 'dmm_unit_interfacing': c.dmm_unit_interfacing, 'satd_delta': satd, 'label_td_satd': label_td_satd}
            writer.writerow(row_dict)
            if GENERATE_LLM_BATCH:
                diff_text = '\n'.join([diff_snippet(m.diff) for m in py if m.diff])
                outfiles = {'orig': LLM_DIR / f'{_repo}_part{part}_original_prompt.jsonl', 'nosatd': LLM_DIR / f'{_repo}_part{part}_satd_filtered.jsonl', 'nodiff': LLM_DIR / f'{_repo}_part{part}_diff_removed.jsonl'}
                for key in handles:
                    if handles[key] is None:
                        handles[key] = outfiles[key].open('w', encoding='utf-8')
                jsonl_lines = {'orig': json.dumps(prepare_llm_prompt_original(row_dict, diff_text), ensure_ascii=False) + '\n', 'nosatd': json.dumps(prepare_llm_prompt_satd_filtered(row_dict, diff_text), ensure_ascii=False) + '\n', 'nodiff': json.dumps(prepare_llm_prompt_diff_removed(row_dict), ensure_ascii=False) + '\n'}
                for key in handles:
                    _line = jsonl_lines[key]
                    if cur_sizes[key] + len(_line.encode('utf-8')) > MAX_MB * 1000000:
                        handles[key].close()
                        part = part + 1
                        handles[key] = (LLM_DIR / f'{_repo}_part{part}_{key}.jsonl').open('w', encoding='utf-8')
                        cur_sizes[key] = 0
                    handles[key].write(_line)
                    cur_sizes[key] = cur_sizes[key] + len(_line.encode('utf-8'))
        for h in handles.values():
            if h:
                h.close()
        print(f'[{_repo}] completed.')
    print('All repos processed.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splits
    """)
    return


@app.cell
def _(OUT, SPLIT, pd):
    _csv_files = list(OUT.glob('features_*.csv'))
    _df = pd.concat([pd.read_csv(_f) for _f in _csv_files], ignore_index=True)
    _df['commit_dt'] = pd.to_datetime(_df.commit_date, utc=True)
    _train_idx = []
    _test_idx = []
    for _repo, _g in _df.groupby('repo_id'):
        _g = _g.sort_values('commit_dt')
        _n = int(0.7 * len(_g))
        _train_idx = _train_idx + list(_g.index[:_n])
        _test_idx = _test_idx + list(_g.index[_n:])
    SPLIT.joinpath('time_train.csv').write_text(_df.loc[_train_idx].to_csv(index=False))
    SPLIT.joinpath('time_test.csv').write_text(_df.loc[_test_idx].to_csv(index=False))
    for _repo in _df.repo_id.unique():
        SPLIT.joinpath(f'lopo_train_excl_{_repo}.csv').write_text(_df[_df.repo_id != _repo].to_csv(index=False))
        SPLIT.joinpath(f'lopo_test_{_repo}.csv').write_text(_df[_df.repo_id == _repo].to_csv(index=False))
    print('Global splits created.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iteration 1
    """)
    return


@app.cell
def _(Path, pd):
    import lightgbm as lgb
    import xgboost as xgb
    import shap
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
    SPLIT_1 = Path('splits')
    _train_df = pd.read_csv(SPLIT_1 / 'time_train.csv')
    _test_df = pd.read_csv(SPLIT_1 / 'time_test.csv')

    def _prepare_xy(df):
        X = _df.drop(columns=['repo_id', 'commit_hash', 'commit_uid', 'commit_date', 'commit_dt', 'satd_delta', 'label_td_satd'])
        y = _df['label_td_satd']
        return (X, y)

    def _evaluate_model(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f'\n{_name} – Classification Report:')
        print(classification_report(_y_test, y_pred, digits=3))
        print(f'Confusion Matrix ({_name}):')
        print(confusion_matrix(_y_test, y_pred))
        print(f'ROC AUC ({_name}): {roc_auc_score(_y_test, y_proba):.4f}')
        return (y_pred, y_proba)
    _X_train, _y_train = _prepare_xy(_train_df)
    X_test, _y_test = _prepare_xy(_test_df)
    MODELS_DIR = Path('trained_models')
    MODELS_DIR.mkdir(exist_ok=True)
    rf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    rf.fit(_X_train, _y_train)
    joblib.dump(rf, MODELS_DIR / 'rf_model.joblib')
    _evaluate_model('Random Forest', rf, X_test, _y_test)
    lgbm = lgb.LGBMClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    lgbm.fit(_X_train, _y_train)
    joblib.dump(lgbm, MODELS_DIR / 'lgbm_model.joblib')
    _evaluate_model('LightGBM', lgbm, X_test, _y_test)
    xgbm = xgb.XGBClassifier(n_estimators=250, random_state=42, n_jobs=-1, use_label_encoder=False)
    xgbm.fit(_X_train, _y_train)
    joblib.dump(xgbm, MODELS_DIR / 'xgb_model.joblib')
    _evaluate_model('XGBoost', xgbm, X_test, _y_test)
    print('All models trained and saved.')
    return (
        RandomForestClassifier,
        classification_report,
        confusion_matrix,
        joblib,
        lgb,
        precision_recall_curve,
        roc_auc_score,
        shap,
        xgb,
    )


@app.cell
def _(
    Path,
    RandomForestClassifier,
    classification_report,
    confusion_matrix,
    lgb,
    pd,
    roc_auc_score,
    xgb,
):
    SPLIT_2 = Path('splits')
    _repos = sorted([p.name.replace('lopo_test_', '').replace('.csv', '') for p in SPLIT_2.glob('lopo_test_*.csv')])

    def _prepare_xy(df):
        X = _df.drop(columns=['repo_id', 'commit_hash', 'commit_uid', 'commit_date', 'commit_dt', 'satd_delta', 'label_td_satd'])
        y = _df['label_td_satd']
        return (X, y)

    def _evaluate_model(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f'\n{_name} – LOPO Results:')
        print(classification_report(_y_test, y_pred, digits=3))
        print(f'Confusion Matrix ({_name}):')
        print(confusion_matrix(_y_test, y_pred))
        print(f'ROC AUC ({_name}): {roc_auc_score(_y_test, y_proba):.4f}')
    for _repo in _repos:
        print(f'\n=== LOPO: {_repo} excluded ===')
        _train_df = pd.read_csv(SPLIT_2 / f'lopo_train_excl_{_repo}.csv')
        _test_df = pd.read_csv(SPLIT_2 / f'lopo_test_{_repo}.csv')
        _X_train, _y_train = _prepare_xy(_train_df)
        X_test_1, _y_test = _prepare_xy(_test_df)
        rf_1 = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
        rf_1.fit(_X_train, _y_train)
        _evaluate_model(f'Random Forest (LOPO: {_repo})', rf_1, X_test_1, _y_test)
        lgbm_1 = lgb.LGBMClassifier(n_estimators=250, random_state=42, n_jobs=-1)
        lgbm_1.fit(_X_train, _y_train)
        _evaluate_model(f'LightGBM (LOPO: {_repo})', lgbm_1, X_test_1, _y_test)
        xgbm_1 = xgb.XGBClassifier(n_estimators=250, random_state=42, n_jobs=-1, use_label_encoder=False)
        xgbm_1.fit(_X_train, _y_train)
        _evaluate_model(f'XGBoost (LOPO: {_repo})', xgbm_1, X_test_1, _y_test)
    return X_test_1, lgbm_1, rf_1, xgbm_1


@app.cell
def _(X_test_1, lgbm_1, rf_1, shap, xgbm_1):
    import matplotlib.pyplot as plt
    explainer_rf = shap.TreeExplainer(rf_1)
    explainer_lgbm = shap.TreeExplainer(lgbm_1)
    # Prepare TreeExplainer (for tree models: RF, LightGBM, XGBoost)
    explainer_xgbm = shap.TreeExplainer(xgbm_1)
    shap_values_rf = explainer_rf.shap_values(X_test_1)
    shap_values_lgbm = explainer_lgbm.shap_values(X_test_1)
    shap_values_xgbm = explainer_xgbm.shap_values(X_test_1)
    # Compute SHAP values for the test set
    shap.summary_plot(shap_values_rf, X_test_1, show=False)
    plt.title('SHAP Summary – Random Forest')
    plt.savefig('shap_rf_summary.png', bbox_inches='tight')
    plt.close()
    # SHAP Summary Plot for Random Forest
    shap.summary_plot(shap_values_lgbm, X_test_1, show=False)
    plt.title('SHAP Summary – LightGBM')
    plt.savefig('shap_lgbm_summary.png', bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values_xgbm, X_test_1, show=False)
    # SHAP Summary Plot for LightGBM
    plt.title('SHAP Summary – XGBoost')
    plt.savefig('shap_xgbm_summary.png', bbox_inches='tight')
    plt.close()
    # SHAP Summary Plot for XGBoost
    print('SHAP analysis complete. Plots saved as PNG.')
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM-as-Judge OpenAI GPT-API Batching (Upload & Batch Start)
    """)
    return


@app.cell
def _(Path, json, os, tqdm):
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    assert _client.api_key, 'OPENAI_API_KEY environment variable is not set.'
    LLM_DIR_1 = Path('llm_batch')
    _BATCH_INFO_FILE = LLM_DIR_1 / 'batch_metadata.json'
    _batch_metadata = {}
    _jsonl_files = sorted(LLM_DIR_1.glob('*.jsonl'))
    for jsonl_file in tqdm(_jsonl_files, desc='Batch uploads + creations'):
        with open(jsonl_file, 'rb') as _f:
            batch_input_file = _client.files.create(file=_f, purpose='batch')
        file_id = batch_input_file.id
        print(f'File {jsonl_file.name} uploaded: {file_id}')
        _batch = _client.batches.create(input_file_id=file_id, endpoint='/v1/chat/completions', completion_window='24h', metadata={'description': f'TD Detection Batch for {jsonl_file.name}'})
        _batch_id = _batch.id
        print(f'Batch started for {jsonl_file.name}: {_batch_id}')
        _batch_metadata[jsonl_file.name] = {'file_id': file_id, 'batch_id': _batch_id, 'status': 'submitted'}
    with open(_BATCH_INFO_FILE, 'w', encoding='utf-8') as _f:
        json.dump(_batch_metadata, _f, indent=2)
    print(f'All batches started. Metadata saved to {_BATCH_INFO_FILE}')
    return (OpenAI,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM-as-Judge OpenAI GPT-API Batching (Status & Optional Download)
    """)
    return


@app.cell
def _(OpenAI, Path, json, os, tqdm):
    _client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    LLM_DIR_2 = Path('llm_batch')
    _BATCH_INFO_FILE = LLM_DIR_2 / 'batch_metadata.json'
    BATCH_STATUS_FILE = LLM_DIR_2 / 'batch_status.json'
    with open(_BATCH_INFO_FILE, 'r', encoding='utf-8') as _f:
        _batch_metadata = json.load(_f)
    batch_status_results = {}
    for _name, meta in tqdm(_batch_metadata.items(), desc='Batch Status Check'):
        _batch_id = meta['batch_id']
        _batch = _client.batches.retrieve(_batch_id)
        status_info = {'status': _batch.status, 'input_file_id': _batch.input_file_id, 'output_file_id': _batch.output_file_id, 'error_file_id': _batch.error_file_id, 'request_counts': _batch.request_counts}
        if _batch.output_file_id:
            output_path = LLM_DIR_2 / f'{_name}_output.jsonl'
            with open(output_path, 'wb') as out_f:
                _content = _client.files.content(_batch.output_file_id)
                out_f.write(_content.read())
        if _batch.error_file_id:
            error_path = LLM_DIR_2 / f'{_name}_errors.jsonl'
            with open(error_path, 'wb') as err_f:
                _content = _client.files.content(_batch.error_file_id)
                err_f.write(_content.read())
        batch_status_results[_name] = status_info

    def safe_json(obj):
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)
    with open(BATCH_STATUS_FILE, 'w', encoding='utf-8') as _f:
        json.dump({k: {kk: safe_json(vv) for kk, vv in v.items()} for k, v in batch_status_results.items()}, _f, indent=2)
    print(f'Batch status and results saved to {BATCH_STATUS_FILE}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iteration 2
    """)
    return


@app.cell
def _(
    Path,
    RandomForestClassifier,
    classification_report,
    confusion_matrix,
    joblib,
    json,
    lgb,
    pd,
    plt,
    roc_auc_score,
    shap,
    tqdm,
    xgb,
):
    LLM_DIR_3 = Path('llm_batch')
    OUT_1 = Path('data')
    SPLIT_3 = Path('splits')
    SPLIT_3.mkdir(exist_ok=True)
    _llm_labels = {}
    _jsonl_files = sorted(LLM_DIR_3.glob('*_output.jsonl'))
    for _file in tqdm(_jsonl_files, desc='Reading LLM judgements'):
        with open(_file, 'r', encoding='utf-8') as _f:
            for _line in _f:
                _data = json.loads(_line)
                _commit_uid = _data.get('custom_id')
                _choice = _data.get('response', {}).get('body', {}).get('choices', [{}])[0]
                _content = _choice.get('message', {}).get('content', '').strip().lower()
                if _content in ['yes', 'no']:
                    _llm_labels[_commit_uid] = 1 if _content == 'yes' else 0
    print(f'LLM labels loaded: {len(_llm_labels)}')
    _csv_files = list(OUT_1.glob('features_*.csv'))
    _dfs = []
    for _file in tqdm(_csv_files, desc='Processing CSV files'):
        _df = pd.read_csv(_file)
        _df['label_llm'] = _df['commit_uid'].map(_llm_labels).fillna(0).astype(int)
        _df['label_td_combined'] = ((_df['label_td_satd'] == 1) | (_df['label_llm'] == 1)).astype(int)
        _dfs.append(_df)
    _df_all = pd.concat(_dfs, ignore_index=True)
    _df_all['commit_dt'] = pd.to_datetime(_df_all.commit_date, utc=True)
    _train_idx, _test_idx = ([], [])
    for _repo, _g in _df_all.groupby('repo_id'):
        _g = _g.sort_values('commit_dt')
        _n = int(0.7 * len(_g))
        _train_idx = _train_idx + list(_g.index[:_n])
        _test_idx = _test_idx + list(_g.index[_n:])
    SPLIT_3.joinpath('time_train_iteration-2.csv').write_text(_df_all.loc[_train_idx].to_csv(index=False))
    SPLIT_3.joinpath('time_test_iteration-2.csv').write_text(_df_all.loc[_test_idx].to_csv(index=False))
    for _repo in _df_all.repo_id.unique():
        SPLIT_3.joinpath(f'lopo_train_excl_{_repo}_iteration-2.csv').write_text(_df_all[_df_all.repo_id != _repo].to_csv(index=False))
        SPLIT_3.joinpath(f'lopo_test_{_repo}_iteration-2.csv').write_text(_df_all[_df_all.repo_id == _repo].to_csv(index=False))
    print('New splits created (Iteration 2)')

    def _prepare_xy(df):
        X = _df.drop(columns=['repo_id', 'commit_hash', 'commit_uid', 'commit_date', 'commit_dt', 'satd_delta', 'label_td_satd', 'label_llm'])
        y = _df['label_td_combined']
        return (X, y)

    def _evaluate_model(name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        print(f'\n{_name} – Evaluation:')
        print(classification_report(_y_test, y_pred, digits=3))
        print(f'Confusion Matrix ({_name}):\n{confusion_matrix(_y_test, y_pred)}')
        print(f'ROC AUC ({_name}): {roc_auc_score(_y_test, y_proba):.4f}')
    _train_df = pd.read_csv(SPLIT_3 / 'time_train_iteration-2.csv')
    _test_df = pd.read_csv(SPLIT_3 / 'time_test_iteration-2.csv')
    _X_train, _y_train = _prepare_xy(_train_df)
    X_test_2, _y_test = _prepare_xy(_test_df)
    rf_2 = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    rf_2.fit(_X_train, _y_train)
    _evaluate_model('Random Forest (Iteration 2)', rf_2, X_test_2, _y_test)
    lgbm_2 = lgb.LGBMClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    lgbm_2.fit(_X_train, _y_train)
    _evaluate_model('LightGBM (Iteration 2)', lgbm_2, X_test_2, _y_test)
    xgbm_2 = xgb.XGBClassifier(n_estimators=250, random_state=42, n_jobs=-1, use_label_encoder=False)
    xgbm_2.fit(_X_train, _y_train)
    _evaluate_model('XGBoost (Iteration 2)', xgbm_2, X_test_2, _y_test)
    joblib.dump(rf_2, 'trained_rf_iteration2.joblib')
    joblib.dump(lgbm_2, 'trained_lgbm_iteration2.joblib')
    joblib.dump(xgbm_2, 'trained_xgbm_iteration2.joblib')
    _repos = _df_all.repo_id.unique()
    for _repo in _repos:
        print(f'\n=== LOPO: {_repo} excluded ===')
        _train_df = pd.read_csv(SPLIT_3 / f'lopo_train_excl_{_repo}_iteration-2.csv')
        _test_df = pd.read_csv(SPLIT_3 / f'lopo_test_{_repo}_iteration-2.csv')
        _X_train, _y_train = _prepare_xy(_train_df)
        X_test_2, _y_test = _prepare_xy(_test_df)
        rf_2.fit(_X_train, _y_train)
        _evaluate_model(f'Random Forest (LOPO {_repo})', rf_2, X_test_2, _y_test)
        lgbm_2.fit(_X_train, _y_train)
        _evaluate_model(f'LightGBM (LOPO {_repo})', lgbm_2, X_test_2, _y_test)
        xgbm_2.fit(_X_train, _y_train)
        _evaluate_model(f'XGBoost (LOPO {_repo})', xgbm_2, X_test_2, _y_test)
    _explainer = shap.TreeExplainer(lgbm_2)
    _shap_values = _explainer.shap_values(X_test_2)
    shap.summary_plot(_shap_values, X_test_2, show=False)
    plt.title('SHAP Summary – LightGBM – Iteration 2')
    plt.savefig('shap_lgbm_iteration2.png', bbox_inches='tight')
    plt.close()
    print('Iteration 2 completed.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ablation Test: Prompt Sensitivity
    """)
    return


@app.cell
def _(Path, json):
    import random
    LLM_DIR_4 = Path('llm_batch')
    input_files = ['requests_part1_original_prompt.jsonl', 'fastapi_part1_original_prompt.jsonl', 'flask_part1_original_prompt.jsonl', 'scrapy_part1_original_prompt.jsonl', 'keras_part1_original_prompt.jsonl']
    _output_file = LLM_DIR_4 / 'prompt_sensitivity_ablation_test.jsonl'
    all_lines = []
    for filename in input_files:
        _path = LLM_DIR_4 / filename
        with open(_path, 'r', encoding='utf-8') as _f:
            all_lines.extend(_f.readlines())
    print(f'Total available lines: {len(all_lines)}')
    sampled_lines = random.sample(all_lines, 10000)
    final_lines = []
    enhanced_prompt = "You are a senior code reviewer. Based on the code change and metrics summary below, assess if this change might lead to long-term maintainability issues. Answer with 'yes' or 'no'."
    for _line in sampled_lines:
        obj = json.loads(_line)
        for msg in obj['body']['messages']:
            if msg['role'] == 'user':
                _content = msg['content']
                _content = _content.replace('You are a senior reviewer.', enhanced_prompt)
                _content = _content.replace('Question: Does this commit introduce technical debt? Answer yes or no.', '')
                msg['content'] = _content
        obj['custom_id'] = obj['custom_id'].replace('-original-prompt', '-original-diff')
        final_lines.append(json.dumps(obj, ensure_ascii=False) + '\n')
        no_diff_obj = json.loads(json.dumps(obj))
        for msg in no_diff_obj['body']['messages']:
            if msg['role'] == 'user':
                _content = msg['content']
                if 'DIFF:' in _content:
                    _content = _content.split('DIFF:')[0].rstrip()
                msg['content'] = _content
        no_diff_obj['custom_id'] = no_diff_obj['custom_id'].replace('-original-diff', '-no-diff')
        final_lines.append(json.dumps(no_diff_obj, ensure_ascii=False) + '\n')
    with open(_output_file, 'w', encoding='utf-8') as _f:
        _f.writelines(final_lines)
    print(f'Created {len(final_lines)} lines in {_output_file} (comparing {len(final_lines) / 2} unique lines)')
    return (LLM_DIR_4,)


@app.cell
def _(LLM_DIR_4, json):
    from collections import defaultdict
    _output_file = LLM_DIR_4 / 'prompt_sensitivity_ablation_test.jsonl_result.jsonl'
    _counts = defaultdict(lambda: {'yes': 0, 'total': 0})
    with _output_file.open('r', encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line:
                continue
            _data = json.loads(_line)
            custom_id = _data['custom_id']
            if custom_id.endswith('-original-diff'):
                variant = 'original-diff'
            elif custom_id.endswith('-no-diff'):
                variant = 'no-diff'
            else:
                continue
            if not _data.get('response') or not _data['response'].get('body'):
                continue
            try:
                _content = _data['response']['body']['choices'][0]['message']['content'].strip().lower()
            except (KeyError, IndexError, AttributeError):
                continue
            if _content.startswith('yes'):
                _counts[variant]['yes'] = _counts[variant]['yes'] + 1
            _counts[variant]['total'] = _counts[variant]['total'] + 1
    for variant, stats in _counts.items():
        yes_ratio = stats['yes'] / stats['total'] * 100 if stats['total'] else 0
        print(f"{variant} → Yes: {stats['yes']}/{stats['total']} = {yes_ratio:.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iteration 3
    """)
    return


@app.cell
def _(
    Path,
    RandomForestClassifier,
    classification_report,
    confusion_matrix,
    display,
    f1_score,
    json,
    lgb,
    matthews_corrcoef,
    np,
    pd,
    plt,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    shap,
    tqdm,
    xgb,
):
    from sklearn.metrics import precision_score
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.feature_selection import VarianceThreshold
    import warnings
    from sklearn.metrics import roc_curve, auc
    from sklearn.metrics import balanced_accuracy_score, average_precision_score
    LLM_DIR_5 = Path('llm_batch')
    OUT_2 = Path('data')
    SPLIT_4 = Path('splits')
    SPLIT_4.mkdir(exist_ok=True)
    _llm_labels = {}
    _jsonl_files = sorted(LLM_DIR_5.glob('*_satd_filtered.jsonl_output.jsonl'))
    for _file in tqdm(_jsonl_files, desc='Reading LLM Judgements'):
        with open(_file, 'r', encoding='utf-8') as _f:
            for _line in _f:
                _data = json.loads(_line)
                _commit_uid = _data.get('custom_id')
                if '-' in _commit_uid:
                    _commit_uid = _commit_uid.split('-', 1)[0]
                _choice = _data.get('response', {}).get('body', {}).get('choices', [{}])[0]
                _content = _choice.get('message', {}).get('content', '').strip().lower()
                if _content in ['yes', 'no']:
                    _llm_labels[_commit_uid] = 1 if _content == 'yes' else 0
    print(f'LLM labels read: {len(_llm_labels)}')
    _csv_files = list(OUT_2.glob('features_*.csv'))
    _dfs = []
    for _file in tqdm(_csv_files, desc='Processing CSV-Files'):
        _df = pd.read_csv(_file)
        _df['label_llm'] = _df['commit_uid'].map(_llm_labels).fillna(0).astype(int)
        _df.to_csv(_file, index=False)
        _dfs.append(_df)
    _df_all = pd.concat(_dfs, ignore_index=True)
    _df_all['commit_dt'] = pd.to_datetime(_df_all.commit_date, utc=True)
    _train_idx, _test_idx = ([], [])
    for _repo, _g in _df_all.groupby('repo_id'):
        _g = _g.sort_values('commit_dt')
        _n = int(0.7 * len(_g))
        _train_idx = _train_idx + list(_g.index[:_n])
        _test_idx = _test_idx + list(_g.index[_n:])
    SPLIT_4.joinpath('time_train_iteration-3.csv').write_text(_df_all.loc[_train_idx].to_csv(index=False))
    SPLIT_4.joinpath('time_test_iteration-3.csv').write_text(_df_all.loc[_test_idx].to_csv(index=False))
    for _repo in _df_all.repo_id.unique():
        SPLIT_4.joinpath(f'lopo_train_excl_{_repo}_iteration-3.csv').write_text(_df_all[_df_all.repo_id != _repo].to_csv(index=False))
        SPLIT_4.joinpath(f'lopo_test_{_repo}_iteration-3.csv').write_text(_df_all[_df_all.repo_id == _repo].to_csv(index=False))
    print('New Splits created (Iteration 3)')

    def _prepare_xy(df):
        X = _df.drop(columns=['repo_id', 'commit_hash', 'commit_uid', 'commit_date', 'commit_dt', 'satd_delta', 'label_td_satd', 'label_llm'])
        y = _df['label_llm']
        return (X, y)

    def _evaluate_model(name, model, X_test, y_test, threshold=0.5, eval_type='Main Split'):
        """
        Evaluate model at a given threshold.  # save updated features file with label_llm
        - Debt = positive class (1)
        Returns per-class metrics (Debt), macro/weighted/micro F1, MCC, balanced acc,
        ROC-AUC, PR-AUC (Debt), and confusion matrix counts.
        """
        if hasattr(model, 'predict_proba') and len(getattr(model, 'classes_', [0, 1])) > 1:
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
            roc_auc = roc_auc_score(_y_test, y_proba)
            pr_auc_debt = average_precision_score(_y_test, y_proba)
        else:
            y_pred = model.predict(X_test)
            y_proba = None
            roc_auc = float('nan')
            pr_auc_debt = float('nan')
        p_debt = precision_score(_y_test, y_pred, pos_label=1, zero_division=0)
        r_debt = recall_score(_y_test, y_pred, pos_label=1, zero_division=0)
        f1_debt = f1_score(_y_test, y_pred, pos_label=1, zero_division=0)
        f1_macro = f1_score(_y_test, y_pred, average='macro', zero_division=0)
        f1_weight = f1_score(_y_test, y_pred, average='weighted', zero_division=0)
        f1_micro = f1_score(_y_test, y_pred, average='micro', zero_division=0)
        mcc = matthews_corrcoef(_y_test, y_pred)
        bal_acc = balanced_accuracy_score(_y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(_y_test, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        print(f'\n{_name} – {eval_type} (threshold={threshold:.3f}):')
        print(classification_report(_y_test, y_pred, digits=3))
        print(f'Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]')
        print(f'Debt P/R/F1: {p_debt:.3f}/{r_debt:.3f}/{f1_debt:.3f} | Macro-F1: {f1_macro:.3f} | Weighted-F1: {f1_weight:.3f} | Micro-F1: {f1_micro:.3f}')
        print(f'ROC-AUC: {roc_auc:.3f} | PR-AUC (Debt): {pr_auc_debt:.3f} | MCC: {mcc:.3f} | Balanced Acc: {bal_acc:.3f} | Specificity (No-Debt): {specificity:.3f}')
        return {'Model': _name, 'Evaluation Type': eval_type, 'Threshold': threshold, 'P_Debt': p_debt, 'R_Debt': r_debt, 'F1_Debt': f1_debt, 'F1_Macro': f1_macro, 'F1_Weighted': f1_weight, 'F1_Micro': f1_micro, 'MCC': mcc, 'Balanced_Acc': bal_acc, 'ROC_AUC': roc_auc, 'PR_AUC_Debt': pr_auc_debt, 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    _train_df = pd.read_csv(SPLIT_4 / 'time_train_iteration-3.csv')
    _test_df = pd.read_csv(SPLIT_4 / 'time_test_iteration-3.csv')
    _X_train, _y_train = _prepare_xy(_train_df)
    X_test_3, _y_test = _prepare_xy(_test_df)
    rf_3 = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1).fit(_X_train, _y_train)
    lgbm_3 = lgb.LGBMClassifier(n_estimators=250, random_state=42, n_jobs=-1).fit(_X_train, _y_train)
    xgbm_3 = xgb.XGBClassifier(n_estimators=250, random_state=42, n_jobs=-1).fit(_X_train, _y_train)
    threshold_results, best_thresholds = ([], {})
    for _name, model in [('LightGBM', lgbm_3), ('Random Forest', rf_3), ('XGBoost', xgbm_3)]:
        y_proba = model.predict_proba(X_test_3)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(_y_test, y_proba)
        f1_scores, mcc_scores = ([], [])
        for t in thresholds:
            preds = (y_proba >= t).astype(int)
            f1_scores.append(f1_score(_y_test, preds))
            mcc_scores.append(matthews_corrcoef(_y_test, preds))
        best_idx = int(np.argmax(f1_scores))
        best_threshold = thresholds[best_idx]
        best_thresholds[_name] = best_threshold
        threshold_results.append({'Model': _name, 'Best Threshold': best_threshold, 'Best F1': f1_scores[best_idx], 'Best MCC': mcc_scores[best_idx]})
        plt.figure()
        plt.plot(recalls, precisions, label='PR curve')
        plt.scatter(recalls[best_idx], precisions[best_idx], color='red', label=f'Best F1={f1_scores[best_idx]:.3f} @ {best_threshold:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve – {_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"precision_recall_curve_{_name.lower().replace(' ', '_')}_iteration-3.png", bbox_inches='tight')
        plt.close()
        fpr, tpr, _ = roc_curve(_y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve – {_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"roc_curve_{_name.lower().replace(' ', '_')}_iteration-3.png", bbox_inches='tight')
        plt.close()
    pd.DataFrame(threshold_results).to_csv('threshold_tuning_results_iteration-3.csv', index=False)
    metrics_results = []
    metrics_results.append(_evaluate_model('Random Forest', rf_3, X_test_3, _y_test, threshold=best_thresholds['Random Forest']))
    metrics_results.append(_evaluate_model('LightGBM', lgbm_3, X_test_3, _y_test, threshold=best_thresholds['LightGBM']))
    metrics_results.append(_evaluate_model('XGBoost', xgbm_3, X_test_3, _y_test, threshold=best_thresholds['XGBoost']))
    _repos = _df_all.repo_id.unique()
    for _repo in _repos:
        print(f'\n=== LOPO: {_repo} excluded ===')
        _train_df = pd.read_csv(SPLIT_4 / f'lopo_train_excl_{_repo}_iteration-3.csv')
        _test_df = pd.read_csv(SPLIT_4 / f'lopo_test_{_repo}_iteration-3.csv')
        _X_train, _y_train = _prepare_xy(_train_df)
        X_test_3, _y_test = _prepare_xy(_test_df)
        rf_3.fit(_X_train, _y_train)
        metrics_results.append(_evaluate_model('Random Forest', rf_3, X_test_3, _y_test, threshold=best_thresholds['Random Forest'], eval_type=f'LOPO {_repo}'))
        lgbm_3.fit(_X_train, _y_train)
        metrics_results.append(_evaluate_model('LightGBM', lgbm_3, X_test_3, _y_test, threshold=best_thresholds['LightGBM'], eval_type=f'LOPO {_repo}'))
        xgbm_3.fit(_X_train, _y_train)
        metrics_results.append(_evaluate_model('XGBoost', xgbm_3, X_test_3, _y_test, threshold=best_thresholds['XGBoost'], eval_type=f'LOPO {_repo}'))
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df.to_csv('model_metrics_iteration-3.csv', index=False)
    print('\nModel metrics saved to model_metrics_iteration-3.csv')
    display(metrics_df)
    warnings.filterwarnings('ignore', message='.*LightGBM binary classifier.*')
    models = {'Random Forest': rf_3, 'LightGBM': lgbm_3, 'XGBoost': xgbm_3}
    for _name, model in models.items():
        print(f'\nGenerating SHAP summary for {_name}...')
        _explainer = shap.TreeExplainer(model)
        _shap_values = _explainer.shap_values(X_test_3)
        if isinstance(_shap_values, list) and len(_shap_values) == 2:
            _shap_values = _shap_values[1]
        shap.summary_plot(_shap_values, X_test_3, show=False)
        plt.title(f'SHAP Summary – {_name} – Iteration 3')
        plt.savefig(f"shap_{_name.lower().replace(' ', '_')}_iteration3.png", bbox_inches='tight')
        plt.close()
    print('SHAP plots saved.')
    print('\nGenerating LIME explanations for a few instances...')
    selector = VarianceThreshold(threshold=0.0)
    X_train_lime = pd.DataFrame(selector.fit_transform(_X_train), columns=_X_train.columns[selector.get_support()])
    X_test_lime = pd.DataFrame(selector.transform(X_test_3), columns=_X_train.columns[selector.get_support()])
    X_train_lime = X_train_lime.fillna(X_train_lime.mean())
    X_test_lime = X_test_lime.fillna(X_train_lime.mean())
    _explainer = LimeTabularExplainer(training_data=X_train_lime.values, feature_names=X_train_lime.columns, class_names=['No Debt', 'Debt'], mode='classification', discretize_continuous=False)
    for i in range(3):
        exp = _explainer.explain_instance(data_row=X_test_lime.iloc[i].values, predict_fn=lambda x: lgbm_3.predict_proba(pd.DataFrame(x, columns=_X_train.columns)))
        exp.save_to_file(f'lime_explanation_instance_{i}.html')
        print(f'LIME explanation saved: lime_explanation_instance_{i}.html')
    print('LIME and SHAP done.')
    print('Iteration 3 completed.')
    return (SPLIT_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample Review

    - for manually checking if the LLM-Judge detects TD beyond simple SATD keywords
    - extracts n=50 commits with:
      - label_td_satd == 0
      - label_llm == 1
    """)
    return


@app.cell
def _(Path, pd):
    from datetime import datetime
    OUT_3 = Path('data')
    _csv_files = list(OUT_3.glob('features_*.csv'))
    _dfs = [pd.read_csv(_file) for _file in _csv_files]
    _df_all = pd.concat(_dfs, ignore_index=True)
    condition_str = 'label_llm == 1 && label_td_satd == 0'
    condition = (_df_all['label_llm'] == 1) & (_df_all['label_td_satd'] == 0)
    sample_df = _df_all[condition]
    sample_commits = sample_df.sample(n=50, random_state=42)
    repo_url_map = {'keras': 'https://github.com/keras-team/keras', 'scrapy': 'https://github.com/scrapy/scrapy', 'fastapi': 'https://github.com/fastapi/fastapi', 'flask': 'https://github.com/pallets/flask', 'requests': 'https://github.com/psf/requests'}
    links = []
    for _, row in sample_commits.iterrows():
        _repo = row['repo_id']
        commit_hash = row['commit_hash']
        if _repo in repo_url_map:
            links.append(f'{repo_url_map[_repo]}/commit/{commit_hash}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _output_file = Path(f'sample_review_{timestamp}.txt')
    with open(_output_file, 'w', encoding='utf-8') as _f:
        _f.write(f'{condition_str}\n')
        for link in links:
            _f.write(link + '\n')
    print(f'Sample review file saved: {_output_file}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### To be investigated:

    Picked example from sample review: https://github.com/scrapy/scrapy/commit/cd6aa72d7f13a7169a4ce0204559f194fdf229f3

    The gpt-4.1-mini batching judges the scrapy commit with hash cd6aa72d7f13a7169a4ce0204559f194fdf229f3 to potentially lead to maintainability issues (by answering "Yes" to the prompt given). But this answer is neither logical (looking at the simple one-line commit) nor deterministic / reproducible (although the batching temperature is set to 0).
    When asking the ChatGPT web UI with the same model (gpt-4.1-mini), the answer was "No" in a manual test. The prompt was exactly the same as during the batching (copied from the batch input file for this commit).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LOPO Table

    Code for creating a table summarizing the LOPO (leave-one-project-out) splits
    """)
    return


@app.cell
def _(SPLIT_4, display, pd):
    split_dir = SPLIT_4
    rows = []
    for test_file in sorted(split_dir.glob('lopo_test_*_iteration-3.csv')):
        test_project = test_file.stem.replace('lopo_test_', '').replace('_iteration-3', '')
        _test_df = pd.read_csv(test_file)
        num_test_commits = len(_test_df)
        train_file = split_dir / f'lopo_train_excl_{test_project}_iteration-3.csv'
        _train_df = pd.read_csv(train_file)
        num_train_commits = len(_train_df)
        train_projects = sorted(_train_df['repo_id'].unique())
        rows.append({'Fold #': len(rows) + 1, 'Test Project': test_project, '#Commits (Test)': num_test_commits, 'Train Projects': ', '.join(train_projects), '#Commits (Train)': num_train_commits})
    lopo_summary_df = pd.DataFrame(rows)
    lopo_summary_df = lopo_summary_df[['Fold #', 'Test Project', '#Commits (Test)', 'Train Projects', '#Commits (Train)']]
    print('\nLOPO Splits Summary:')
    display(lopo_summary_df)
    summary_file = split_dir / 'lopo_splits_summary.csv'
    lopo_summary_df.to_csv(summary_file, index=False)
    print(f'\nLOPO split summary saved to: {summary_file}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Debt / No Debt Distribution
    """)
    return


@app.cell
def _(pd):
    _df_all = pd.concat([pd.read_csv('data/features_fastapi.csv'), pd.read_csv('data/features_flask.csv'), pd.read_csv('data/features_keras.csv'), pd.read_csv('data/features_requests.csv'), pd.read_csv('data/features_scrapy.csv')], ignore_index=True)
    label_counts = _df_all['label_llm'].value_counts()
    total = label_counts.sum()
    label_percentages = (label_counts / total * 100).round(2)
    print('Technical Debt Distribution (label_llm) - Full Dataset:')
    for label, count in label_counts.items():
        label_name = 'Debt' if label == 1 else 'No Debt'
        print(f'{label_name}: {count} ({label_percentages[label]}%)')
    return


@app.cell
def _(pd):
    _train_df = pd.read_csv('splits/time_train_iteration-3.csv')
    _counts = _train_df['label_llm'].value_counts().rename(index={0: 'No Debt', 1: 'Debt'})
    ratios = (_counts / _counts.sum() * 100).round(2)
    class_distribution = pd.DataFrame({'Count': _counts, 'Ratio (%)': ratios})
    print('Technical Debt Distribution (label_llm) - Only Chronological Training Partition:')
    print(class_distribution)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Class Imbalance, SMOTE, Undersampling Tests
    """)
    return


@app.cell
def _(os, pd, plt):
    os.environ['SKLEARN_ARRAY_API_DISPATCH'] = '0'
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    _train_df = pd.read_csv('splits/time_train_iteration-3.csv')

    def _prepare_xy(df):
        X = _df.drop(columns=['repo_id', 'commit_hash', 'commit_uid', 'commit_date', 'commit_dt', 'satd_delta', 'label_td_satd', 'label_llm'])
        y = _df['label_llm']
        return (X, y)
    _X_train, _y_train = _prepare_xy(_train_df)
    non_numeric = _X_train.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        print('Non-numeric features detected (SMOTE may fail):', non_numeric)
    else:
        print('All features are numeric — SMOTE-friendly.')
    nan_count = _X_train.isna().sum().sum()
    if nan_count > 0:
        print(f'Found {nan_count} missing values — filling with column means.')
        _X_train = _X_train.fillna(_X_train.mean())
    else:
        print('No missing values detected.')
    counts_before = _y_train.value_counts().rename(index={0: 'No Debt', 1: 'Debt'})
    ratios_before = (counts_before / counts_before.sum() * 100).round(2)
    print('\nClass distribution before balancing:')
    print(pd.DataFrame({'Count': counts_before, 'Ratio (%)': ratios_before}))
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(_X_train, _y_train)
    under = RandomUnderSampler(random_state=42)
    X_under, y_under = under.fit_resample(_X_train, _y_train)
    smoteenn = SMOTEENN(random_state=42)
    X_smoteenn, y_smoteenn = smoteenn.fit_resample(_X_train, _y_train)
    counts_smote = y_smote.value_counts().rename(index={0: 'No Debt', 1: 'Debt'})
    counts_under = y_under.value_counts().rename(index={0: 'No Debt', 1: 'Debt'})
    counts_smoteenn = y_smoteenn.value_counts().rename(index={0: 'No Debt', 1: 'Debt'})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    hatches = ['///', '\\\\\\', 'xxx', '---']
    axes[0].bar(counts_before.index, counts_before.values, color='lightgray', hatch=hatches[0], edgecolor='black')
    axes[0].set_title('Unbalanced')
    axes[0].set_ylabel('Number of commits')
    axes[1].bar(counts_smote.index, counts_smote.values, color='lightgray', hatch=hatches[1], edgecolor='black')
    axes[1].set_title('After SMOTE')
    axes[2].bar(counts_under.index, counts_under.values, color='lightgray', hatch=hatches[2], edgecolor='black')
    axes[2].set_title('After undersampling')
    axes[3].bar(counts_smoteenn.index, counts_smoteenn.values, color='lightgray', hatch=hatches[3], edgecolor='black')
    axes[3].set_title('After SMOTEENN hybrid')
    for ax in axes:
        ax.set_xticks(range(len(counts_before.index)))
        ax.set_xticklabels(['No Debt', 'Debt'])
    plt.tight_layout()
    plt.savefig('class_balance_greyscale.pdf', bbox_inches='tight')
    plt.savefig('class_balance_greyscale.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved 'class_balance_greyscale.pdf' and 'class_balance_greyscale.png'.")
    return


@app.cell
def _(json):
    import glob
    files = glob.glob('llm_batch/*_satd_filtered.jsonl_output.jsonl')
    total_prompt_tokens = 0
    total_output_tokens = 0
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as _f:
            for _line in _f:
                if not _line.strip():
                    continue
                try:
                    _data = json.loads(_line)
                    usage = _data.get('response', {}).get('body', {}).get('usage', {})
                    total_prompt_tokens = total_prompt_tokens + usage.get('prompt_tokens', 0)
                    total_output_tokens = total_output_tokens + usage.get('completion_tokens', 0)
                except json.JSONDecodeError as e:
                    print(f'Skipping invalid JSON in {file_path}: {e}')
    print(f'Total prompt tokens: {total_prompt_tokens}')
    print(f'Total output tokens: {total_output_tokens}')
    return


if __name__ == "__main__":
    app.run()
