"""
models.py — تدريب النماذج الأربعة وتقييمها وتفسير توقعاتها.

لكل فئة (أفراد / شركات) نفعل ما يلي بصورة مستقلة:
1. نفصل البيانات إلى تدريب واختبار (الاختبار لا يُستعمل أبدًا في التدريب).
2. نعالج المتغيرات: الرقمية تُوحَّد مقاييسها (للنماذج التي تحتاج ذلك)، والفئوية تُحوَّل إلى أعمدة 0/1.
3. ندرّب: الانحدار اللوجستي، شجرة القرار، الغابة العشوائية، الشبكة العصبية (MLP).
4. نقيس الأداء على مجموعة الاختبار فقط.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data import TARGET

RANDOM_STATE = 42

# المتغيرات المستعملة في التنبؤ. لاحظ غياب المعرّف وغياب default_12m.
GROUPS = {
    "individuals": {
        "id": "person_id",
        "num": ["age", "monthly_income", "job_stability_years", "total_debt", "monthly_obligations",
                "debt_to_income", "loan_amount", "loan_term_months", "late_payments_24m"],
        "cat": ["job_type", "payment_history"],
        "test_size": 0.25,
        "min_leaf": 15,
    },
    "companies": {
        "id": "company_id",
        # نستعمل النِّسَب بدل المبالغ الخام لأن حجم الشركات يختلف كثيرًا؛ النسبة تقارن شركة صغيرة بكبيرة بعدل.
        "num": ["company_age", "annual_revenue", "net_profit_margin", "cash_flow_margin", "current_ratio",
                "debt_ratio", "debt_service_coverage", "loan_to_revenue"],
        "cat": ["sector", "payment_history"],
        "test_size": 0.30,
        "min_leaf": 8,
    },
}

MODEL_NAMES = {
    "logistic": "الانحدار اللوجستي",
    "tree": "شجرة القرار",
    "forest": "الغابة العشوائية",
    "mlp": "الشبكة العصبية",
}
MODEL_EN = {"logistic": "Logistic Regression", "tree": "Decision Tree",
            "forest": "Random Forest", "mlp": "Neural Network (MLP)"}


def _preprocessor(num, cat, scale):
    """النماذج الخطية والشبكة العصبية تحتاج توحيد المقاييس؛ الأشجار لا تحتاجه، ونتركها بالقيم الأصلية لتبقى شروطها مفهومة."""
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])


def _make_models(min_leaf):
    return {
        "logistic": (LogisticRegression(max_iter=3000), True),
        "tree": (DecisionTreeClassifier(max_depth=4, min_samples_leaf=min_leaf, random_state=RANDOM_STATE), False),
        "forest": (RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                          random_state=RANDOM_STATE), False),
        "mlp": (MLPClassifier(hidden_layer_sizes=(8,), alpha=3.0, max_iter=3000,
                              random_state=RANDOM_STATE), True),
    }


def train_group(df, group):
    """يدرّب النماذج الأربعة لفئة واحدة ويعيد كل ما تحتاجه الواجهة."""
    cfg = GROUPS[group]
    features = cfg["num"] + cfg["cat"]
    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=RANDOM_STATE)

    pipes, proba_test, auc_train = {}, {}, {}
    for key, (clf, scale) in _make_models(cfg["min_leaf"]).items():
        pipe = Pipeline([("prep", _preprocessor(cfg["num"], cfg["cat"], scale)), ("model", clf)])
        pipe.fit(X_train, y_train)
        pipes[key] = pipe
        proba_test[key] = pipe.predict_proba(X_test)[:, 1]
        auc_train[key] = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])

    return {
        "group": group, "cfg": cfg, "features": features, "pipes": pipes,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "train_ids": set(df.loc[X_train.index, cfg["id"]]),
        "proba_test": proba_test, "auc_train": auc_train,
        "base_rate": float(y_train.mean()),
    }


def train_all(individuals, companies):
    return {"individuals": train_group(individuals, "individuals"),
            "companies": train_group(companies, "companies")}


# ---------------------------------------------------------------- تقييم الأداء
def metrics_table(res, threshold=0.5):
    rows = []
    for key, p in res["proba_test"].items():
        pred = (p >= threshold).astype(int)
        rows.append({
            "key": key,
            "model": MODEL_NAMES[key],
            "auc": roc_auc_score(res["y_test"], p),
            "auc_train": res["auc_train"][key],
            "precision": precision_score(res["y_test"], pred, zero_division=0),
            "recall": recall_score(res["y_test"], pred, zero_division=0),
            "f1": f1_score(res["y_test"], pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def confusion(res, key, threshold=0.5):
    pred = (res["proba_test"][key] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(res["y_test"], pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def roc_points(res):
    out = []
    for key, p in res["proba_test"].items():
        fpr, tpr, _ = roc_curve(res["y_test"], p)
        out.append(pd.DataFrame({"fpr": fpr, "tpr": tpr, "model": MODEL_NAMES[key]}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- التوقع
def row_frame(res, record):
    """يحوّل سجلًا (Series) إلى DataFrame بسطر واحد يحتوي متغيرات التنبؤ فقط."""
    return pd.DataFrame([record[res["features"]].to_dict()])[res["features"]]


def predict_all(res, record):
    X = row_frame(res, record)
    return {key: float(pipe.predict_proba(X)[0, 1]) for key, pipe in res["pipes"].items()}


def risk_level(p, low_t, high_t):
    if p < low_t:
        return "low"
    if p < high_t:
        return "mid"
    return "high"


# ---------------------------------------------------------------- التفسير
def _transformed_names(res, pipe):
    """قائمة (اسم المتغير الأصلي، الفئة أو None) لكل عمود بعد المعالجة."""
    prep = pipe.named_steps["prep"]
    enc = prep.named_transformers_["cat"]
    names = [(c, None) for c in res["cfg"]["num"]]
    for col, cats in zip(res["cfg"]["cat"], enc.categories_):
        names += [(col, v) for v in cats]
    return names


def what_if(res, key, record):
    """
    تفسير محلي يصلح لكل النماذج:
    نستبدل كل متغير بقيمته "النموذجية" في بيانات التدريب (الوسيط أو الفئة الأكثر شيوعًا)
    ونقيس كم يتغيّر احتمال التعثر. الفرق الموجب = هذا المتغير يرفع المخاطر عند هذا المقترض.
    """
    pipe = res["pipes"][key]
    X = row_frame(res, record)
    base = float(pipe.predict_proba(X)[0, 1])
    rows = []
    for col in res["features"]:
        if col in res["cfg"]["num"]:
            ref = float(res["X_train"][col].median())
            same = np.isclose(float(X[col].iloc[0]), ref)
        else:
            ref = res["X_train"][col].mode().iloc[0]
            same = X[col].iloc[0] == ref
        X2 = X.copy()
        X2[col] = ref
        new = base if same else float(pipe.predict_proba(X2)[0, 1])
        rows.append({"feature": col, "value": X[col].iloc[0], "reference": ref,
                     "proba_if_typical": new, "effect": base - new})
    out = pd.DataFrame(rows)
    out["abs"] = out["effect"].abs()
    return base, out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def logistic_contributions(res, record):
    """مساهمة كل متغير في "درجة" الانحدار اللوجستي لهذا السجل = المعامل × القيمة الموحَّدة."""
    pipe = res["pipes"]["logistic"]
    Xt = pipe.named_steps["prep"].transform(row_frame(res, record))[0]
    coef = pipe.named_steps["model"].coef_[0]
    names = _transformed_names(res, pipe)
    df = pd.DataFrame({"feature": [n[0] for n in names], "contribution": Xt * coef})
    df = df.groupby("feature", as_index=False)["contribution"].sum()
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def logistic_coefficients(res):
    pipe = res["pipes"]["logistic"]
    names = _transformed_names(res, pipe)
    coef = pipe.named_steps["model"].coef_[0]
    df = pd.DataFrame({"feature": [n[0] for n in names], "category": [n[1] for n in names], "coef": coef})
    df["abs"] = df["coef"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def tree_path(res, record):
    """المسار الذي سلكه السجل داخل الشجرة المدرَّبة، شرطًا بشرط."""
    pipe = res["pipes"]["tree"]
    clf = pipe.named_steps["model"]
    Xt = pipe.named_steps["prep"].transform(row_frame(res, record))
    names = _transformed_names(res, pipe)
    t = clf.tree_
    node_ids = clf.decision_path(Xt).indices
    leaf = int(clf.apply(Xt)[0])
    steps = []
    for node in node_ids:
        if node == leaf:
            continue
        f = t.feature[node]
        col, cat = names[f]
        thr = float(t.threshold[node])
        val = float(Xt[0, f])
        went_left = val <= thr
        steps.append({"feature": col, "category": cat, "threshold": thr, "value": val,
                      "left": went_left, "samples": int(t.n_node_samples[node])})
    leaf_samples = int(t.n_node_samples[leaf])
    leaf_rate = float(clf.predict_proba(Xt)[0, 1])
    return steps, leaf_samples, leaf_rate


def forest_votes(res, record):
    """كم شجرة من أشجار الغابة توقعت التعثر لهذا السجل."""
    pipe = res["pipes"]["forest"]
    clf = pipe.named_steps["model"]
    Xt = pipe.named_steps["prep"].transform(row_frame(res, record))
    probs = np.array([est.predict_proba(Xt)[0, 1] for est in clf.estimators_])
    return int((probs >= 0.5).sum()), len(probs)


def forest_importance(res):
    pipe = res["pipes"]["forest"]
    names = _transformed_names(res, pipe)
    imp = pipe.named_steps["model"].feature_importances_
    df = pd.DataFrame({"feature": [n[0] for n in names], "importance": imp})
    return df.groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)


def mlp_info(res):
    clf = res["pipes"]["mlp"].named_steps["model"]
    n_inputs = clf.coefs_[0].shape[0]
    n_weights = sum(w.size for w in clf.coefs_) + sum(b.size for b in clf.intercepts_)
    return {"inputs": n_inputs, "hidden": list(clf.hidden_layer_sizes), "weights": int(n_weights),
            "iterations": int(clf.n_iter_), "loss": float(clf.loss_), "loss_curve": list(clf.loss_curve_)}
