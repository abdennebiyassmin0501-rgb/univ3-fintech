"""
app.py — واجهة Univ 3 Fintech (Streamlit).

التشغيل:   streamlit run app.py
"""
import hashlib
import hmac
import json
import random
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import data
import models
from texts import (DEFAULT_DEFINITION, GLOSSARY, HOME_QA, LABELS, MODEL_TEXT, MONEY_COLS, PERCENT_COLS,
                   PIPELINE, RISK_NAMES, SYNTHETIC_NOTE)

AUTH_PATH = Path(__file__).parent / "auth.json"
PRIMARY, ACCENT = "#14213D", "#A8741F"
RISK_COLORS = {"low": "#2C7A57", "mid": "#B9830F", "high": "#B0372C"}
GROUP_AR = {"individuals": "الأفراد", "companies": "الشركات"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select, label {
  font-family: 'IBM Plex Sans Arabic', 'Segoe UI', Tahoma, sans-serif; }
.stApp, section[data-testid="stSidebar"] { direction: rtl; }
.stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp .stMarkdown { text-align: right; }
[data-testid="stDataFrame"], [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"],
[data-testid="stImage"], [data-testid="stPyplot"] { direction: ltr; }
[data-testid="stMetricValue"] { direction: ltr; text-align: right; }
.pipeline { display:flex; flex-wrap:wrap; gap:6px; align-items:stretch; margin: 6px 0 4px; }
.pipeline .step { flex:1 1 150px; background:#F1F4F8; border:1px solid #D6DCE5; border-radius:8px; padding:10px 12px; }
.pipeline .step b { display:block; color:#14213D; margin-bottom:4px; }
.pipeline .step span { font-size:.88rem; color:#4A5468; }
.pipeline .arrow { align-self:center; color:#A8741F; font-size:1.4rem; font-weight:700; }
.kv-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(230px,1fr)); gap:6px 18px; }
.kv-grid div { display:flex; justify-content:space-between; gap:10px; border-bottom:1px dashed #D6DCE5; padding:4px 0; font-size:.92rem; }
.kv-grid div span:first-child { color:#5A6479; }
.kv-grid div span:last-child { font-weight:600; direction:ltr; unicode-bidi:isolate; }
.pill { display:inline-block; padding:3px 14px; border-radius:99px; font-weight:700; }
.pill-low { background:#DDEFE5; color:#1E6B49; } .pill-mid { background:#F6EBCF; color:#8A6208; } .pill-high { background:#F5DDD9; color:#8E2A21; }
.big-pd { font-size:2.6rem; font-weight:700; color:#14213D; direction:ltr; display:inline-block; }
.cm { border-collapse:collapse; width:100%; font-size:.85rem; }
.cm td, .cm th { border:1px solid #D6DCE5; padding:5px 6px; text-align:center; }
.cm th { background:#F1F4F8; font-weight:600; }
.login-head { text-align:center !important; }
div[data-testid="stExpander"] details summary p { font-weight:600; }
</style>
"""


# ======================================================================= أدوات عامة
def fmt_value(col, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if col in PERCENT_COLS:
        return f"{float(v) * 100:.1f}%"
    if col in MONEY_COLS:
        return f"{float(v):,.0f}".replace(",", " ") + " دج"
    if col == "default_12m":
        return "نعم" if int(v) == 1 else "لا"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):g}"
    return str(v)


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def rate(mask, df):
    """معدل التعثر داخل مجموعة، مع عددها."""
    sub = df.loc[mask, "default_12m"]
    return (float(sub.mean()) if len(sub) else 0.0), int(len(sub))


def label(col):
    return LABELS.get(col, col)


def risk_pill(level):
    return f'<span class="pill pill-{level}">{RISK_NAMES[level]}</span>'


def display_df(df):
    out = df.copy()
    for c in PERCENT_COLS & set(out.columns):
        out[c] = (out[c] * 100).round(1)
    if "default_12m" in out:
        out["default_12m"] = out["default_12m"].map({1: "نعم", 0: "لا"})
    names = {c: label(c) + (" %" if c in PERCENT_COLS else "") for c in out.columns}
    return out.rename(columns=names)


def details_card(record, cols):
    cells = "".join(f"<div><span>{label(c)}</span><span>{fmt_value(c, record[c])}</span></div>" for c in cols)
    st.markdown(f'<div class="kv-grid">{cells}</div>', unsafe_allow_html=True)


def bar_chart(df, x, y, x_title, y_title, color=PRIMARY, y_format=None, height=260):
    axis = alt.Axis(format=y_format) if y_format else alt.Axis()
    chart = alt.Chart(df).mark_bar(color=color, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X(f"{x}:N", sort=None, title=x_title, axis=alt.Axis(labelAngle=0, labelLimit=120)),
        y=alt.Y(f"{y}:Q", title=y_title, axis=axis),
        tooltip=[alt.Tooltip(f"{x}:N", title=x_title), alt.Tooltip(f"{y}:Q", title=y_title, format=y_format or ",")],
    ).properties(height=height)
    st.altair_chart(chart)


def binned_counts(series, bins, labels_):
    cut = pd.cut(series, bins=bins, labels=labels_, include_lowest=True)
    return cut.value_counts().reindex(labels_).fillna(0).astype(int).rename_axis("الفئة").reset_index(name="العدد")


def default_rate_by(df, col_or_series, order=None):
    g = df.groupby(col_or_series, observed=True)["default_12m"].agg(["mean", "size"]).reset_index()
    g.columns = ["الفئة", "معدل التعثر", "العدد"]
    if order is not None:
        g["الفئة"] = pd.Categorical(g["الفئة"], categories=order, ordered=True)
        g = g.sort_values("الفئة")
        g["الفئة"] = g["الفئة"].astype(str)
    return g


def why_box(element, function, need, problem, how, link, limits):
    with st.expander(f"لماذا استخدمنا هذا؟ — {element}"):
        st.markdown(
            f"- **وظيفته:** {function}\n"
            f"- **لماذا نحتاجه:** {need}\n"
            f"- **المشكلة التي يحلّها:** {problem}\n"
            f"- **كيف يعمل:** {how}\n"
            f"- **علاقته بتقييم المخاطر:** {link}\n"
            f"- **حدوده:** {limits}")


def pick_record(df, id_col, key, prefix, default_id=None):
    """بحث برقم التعريف + قائمة اختيار. يعيد السجل المختار (Series)."""
    q = st.text_input("ابحث برقم التعريف الاصطناعي", placeholder=f"مثال: {prefix}-0042", key=f"{key}_q")
    ids = df[id_col]
    q_norm = q.strip().upper()
    matches = ids[ids.str.contains(q_norm, regex=False)] if q_norm else ids
    if q_norm and matches.empty:
        st.warning(f"لا يوجد سجل يحتوي «{q.strip()}». تأكّد من الصيغة ({prefix}-0001 إلى {ids.iloc[-1]}).")
        matches = ids
    options = matches.tolist()
    index = options.index(default_id) if default_id in options else 0
    sel = st.selectbox(f"اختر السجل ({len(options)} نتيجة)", options, index=index, key=f"{key}_sel_{q_norm}")
    return df[df[id_col] == sel].iloc[0]


# ======================================================================= البيانات والنماذج (مع التخزين المؤقت)
@st.cache_data(show_spinner="جارٍ تحميل البيانات من SQLite...")
def get_data(version):
    return data.load_data()


@st.cache_resource(show_spinner="جارٍ تدريب النماذج الأربعة للأفراد والشركات...")
def get_models(version):
    ind, comp = get_data(version)
    return models.train_all(ind, comp)


# ======================================================================= تسجيل الدخول
def load_auth():
    """على الاستضافة نقرأ الحساب من «أسرار» Streamlit؛ على الجهاز من ملف auth.json."""
    try:
        if "auth" in st.secrets:
            return dict(st.secrets["auth"])
    except Exception:
        pass
    if AUTH_PATH.exists():
        return json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    return None


def verify_password(username, password):
    """نقارن بصمة كلمة المرور (PBKDF2-SHA256 مع ملح) ولا نخزن كلمة المرور نفسها."""
    cfg = load_auth()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(cfg["salt"]), int(cfg["iterations"]))
    user_ok = hmac.compare_digest(username.strip().encode("utf-8"), cfg["username"].encode("utf-8"))
    pass_ok = hmac.compare_digest(digest.hex(), cfg["hash"])
    return user_ok and pass_ok


def login_page():
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        st.markdown("<h1 class='login-head'>Univ 3 Fintech</h1>", unsafe_allow_html=True)
        st.markdown("<p class='login-head'>منصة تعليمية لتقييم المخاطر الائتمانية بالذكاء الاصطناعي</p>",
                    unsafe_allow_html=True)
        if load_auth() is None:
            st.error("لا يوجد حساب. على الجهاز: شغّل python set_password.py. على الاستضافة: أضف قسم [auth] في الأسرار (Secrets).")
            return
        wait = st.session_state.get("lock_until", 0) - time.time()
        if wait > 0:
            st.error(f"محاولات خاطئة كثيرة. أعد المحاولة بعد {int(wait) + 1} ثانية.")
            return
        with st.form("login_form"):
            username = st.text_input("اسم المستخدم")
            password = st.text_input("كلمة المرور", type="password")
            submitted = st.form_submit_button("تسجيل الدخول", type="primary")
        if submitted:
            if verify_password(username, password):
                st.session_state.update(auth=True, user=username.strip(), fails=0)
                st.rerun()
            else:
                fails = st.session_state.get("fails", 0) + 1
                st.session_state["fails"] = fails
                if fails >= 5:
                    st.session_state.update(lock_until=time.time() + 60, fails=0)
                    st.error("5 محاولات خاطئة. تم إيقاف الدخول لمدة دقيقة.")
                else:
                    st.error(f"اسم المستخدم أو كلمة المرور غير صحيحة (المحاولة {fails} من 5).")


# ======================================================================= 1. الرئيسية
def page_home(ind, comp, res, meta):
    st.title("Univ 3 Fintech")
    st.markdown("#### منصة تعليمية لفهم تقييم المخاطر الائتمانية للأفراد والشركات بالذكاء الاصطناعي")
    st.info(SYNTHETIC_NOTE)

    c = st.columns(4)
    c[0].metric("الأفراد في القاعدة", f"{len(ind)}")
    c[1].metric("الشركات في القاعدة", f"{len(comp)}")
    c[2].metric("معدل تعثر الأفراد", pct(ind.default_12m.mean()))
    c[3].metric("معدل تعثر الشركات", pct(comp.default_12m.mean()))

    st.subheader("مفاهيم أساسية")
    cols = st.columns(2)
    for i, (q, a) in enumerate(HOME_QA):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{q}**\n\n{a}")

    st.subheader("تعريف التعثر في المنصة")
    st.markdown(DEFAULT_DEFINITION)

    st.subheader("مراحل العمل")
    steps = '<span class="arrow">←</span>'.join(
        f'<div class="step"><b>{i + 1}. {t}</b><span>{d}</span></div>' for i, (t, d) in enumerate(PIPELINE))
    st.markdown(f'<div class="pipeline">{steps}</div>', unsafe_allow_html=True)

    ri, rc = res["individuals"], res["companies"]
    mi = models.metrics_table(ri)
    best = mi.sort_values("auc", ascending=False).iloc[0]
    why_box(
        "مسار العمل من البيانات إلى التفسير",
        "يرتّب خطوات المشروع بحيث تعتمد كل خطوة على السابقة.",
        "بدون فصل واضح بين التدريب والاختبار قد نخدع أنفسنا بنتائج تبدو ممتازة.",
        f"يمنع أن نقيّم النموذج على بيانات حفظها: تدرّبت نماذج الأفراد على {len(ri['X_train'])} سجل "
        f"واختُبرت على {len(ri['X_test'])} سجل لم تَرَها؛ والشركات على {len(rc['X_train'])} واختُبرت على {len(rc['X_test'])}.",
        "كل مرحلة تأخذ مخرجات السابقة: بيانات ← أعمدة رقمية ← نماذج مدرّبة ← مقاييس ← احتمالات ← تفسيرات.",
        f"النتيجة النهائية احتمال تعثر لكل مقترض؛ مثلًا أفضل نموذج للأفراد حاليًا هو «{best['model']}» "
        f"بـ ROC-AUC = {best['auc']:.2f} على مجموعة الاختبار.",
        "البيانات اصطناعية، فالنتائج توضّح المنهج ولا تصلح لاتخاذ قرارات إقراض حقيقية.")

    st.subheader("إدارة البيانات")
    st.caption(f"أُنشئت القاعدة الحالية في {meta['created_at']} (بذرة التوليد: {meta['seed']}). "
               "تبقى البيانات نفسها في كل تشغيل إلى أن تطلب إعادة إنشائها.")
    with st.expander("إعادة إنشاء البيانات الاصطناعية"):
        st.write("سيُولَّد 800 فرد و200 شركة جدد، وتُعاد تدريب النماذج. كل النتائج ستتغير.")
        sure = st.checkbox("أفهم أن البيانات الحالية ستُستبدل", key="regen_ok")
        if st.button("إعادة إنشاء البيانات", disabled=not sure, key="regen_btn"):
            data.create_database(seed=random.randint(1, 999_999))
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("تم إنشاء بيانات جديدة وإعادة تدريب النماذج.")
            st.rerun()


# ======================================================================= 2. الأفراد
IND_SHOW = ["person_id", "age", "monthly_income", "job_type", "job_stability_years", "total_debt",
            "monthly_obligations", "debt_to_income", "loan_amount", "loan_term_months", "late_payments_24m",
            "payment_history", "default_12m"]


def page_individuals(ind, res):
    st.title("بيانات الأفراد")
    st.caption("800 فرد اصطناعي. " + "البيانات للتعلم فقط.")
    c = st.columns(4)
    c[0].metric("عدد الأفراد", len(ind))
    c[1].metric("وسيط الدخل الشهري", fmt_value("monthly_income", ind.monthly_income.median()))
    c[2].metric("وسيط نسبة الديون إلى الدخل", pct(ind.debt_to_income.median()))
    c[3].metric("معدل التعثر خلال 12 شهرًا", pct(ind.default_12m.mean()))

    st.subheader("البحث عن فرد")
    left, right = st.columns([1, 2])
    with left:
        rec = pick_record(ind, "person_id", "ind", "P")
    with right:
        st.markdown(f"**تفاصيل {rec['person_id']}**")
        details_card(rec, IND_SHOW[1:])
        st.caption("«تعثّر خلال 12 شهرًا» هو المتغير المستهدف: نعرفه في البيانات التاريخية فقط، ولا يدخل في متغيرات التنبؤ.")

    st.subheader("جدول البيانات")
    st.dataframe(display_df(ind[IND_SHOW]), hide_index=True, height=320)

    st.subheader("رسوم بيانية")
    t1, t2, t3, t4 = st.tabs(["توزيع الدخل", "توزيع نسبة الديون", "التعثر حسب سجل السداد", "التعثر حسب نسبة الديون"])
    with t1:
        bins = [0, 40000, 60000, 80000, 100000, 150000, 1e9]
        labs = ["< 40 ألف", "40–60 ألف", "60–80 ألف", "80–100 ألف", "100–150 ألف", "> 150 ألف"]
        bar_chart(binned_counts(ind.monthly_income, bins, labs), "الفئة", "العدد", "الدخل الشهري (دج)", "عدد الأفراد")
    with t2:
        bins = [-0.01, 0, 0.1, 0.2, 0.3, 0.4, 1]
        labs = ["بلا ديون", "0–10%", "10–20%", "20–30%", "30–40%", "> 40%"]
        bar_chart(binned_counts(ind.debt_to_income, bins, labs), "الفئة", "العدد", "نسبة الديون إلى الدخل", "عدد الأفراد")
    with t3:
        bar_chart(default_rate_by(ind, "payment_history", data.HISTORY_LEVELS), "الفئة", "معدل التعثر",
                  "سجل السداد السابق", "معدل التعثر", color=RISK_COLORS["high"], y_format=".0%")
    with t4:
        buckets = pd.cut(ind.debt_to_income, [-0.01, 0, 0.1, 0.2, 0.3, 1],
                         labels=["بلا ديون", "0–10%", "10–20%", "20–30%", "> 30%"])
        bar_chart(default_rate_by(ind, buckets), "الفئة", "معدل التعثر", "نسبة الديون إلى الدخل", "معدل التعثر",
                  color=RISK_COLORS["high"], y_format=".0%")

    st.subheader("لماذا نستعمل هذه المؤشرات؟")
    q1, q3 = ind.monthly_income.quantile([0.25, 0.75])
    r_low_inc, n1 = rate(ind.monthly_income <= q1, ind)
    r_high_inc, n2 = rate(ind.monthly_income >= q3, ind)
    r_dti_hi, n3 = rate(ind.debt_to_income > 0.30, ind)
    r_dti_lo, n4 = rate(ind.debt_to_income <= 0.30, ind)
    r_st_lo, n5 = rate(ind.job_stability_years < 2, ind)
    r_st_hi, n6 = rate(ind.job_stability_years >= 5, ind)
    hist = ind.groupby("payment_history")["default_12m"].mean()
    lti = ind.loan_amount / (ind.monthly_income * 12)
    r_lti_hi, n7 = rate(lti > 1.5, ind)
    r_lti_lo, n8 = rate(lti <= 0.75, ind)
    r_term_long, n9 = rate(ind.loan_term_months >= 60, ind)
    r_term_short, n10 = rate(ind.loan_term_months <= 24, ind)
    dti = rec["debt_to_income"]

    items = [
        ("الدخل الشهري", "يقيس قدرة الفرد على تغطية الأقساط. الدخل الأعلى يترك هامشًا أكبر للطوارئ.",
         f"في بياناتنا: الربع الأقل دخلًا (≤ {fmt_value('monthly_income', q1)}) تعثّر منه {pct(r_low_inc)} "
         f"({n1} فرد)، مقابل {pct(r_high_inc)} في الربع الأعلى دخلًا ({n2} فرد)."),
        ("نسبة الديون إلى الدخل", "الجزء من الدخل الذي يذهب مسبقًا إلى أقساط ديون قائمة. كلما ارتفع قلّ ما يبقى للقرض الجديد.",
         f"عند {rec['person_id']} النسبة = {pct(dti)}: من كل 100 دج من دخله يذهب {dti * 100:.0f} دج لأقساط قائمة. "
         f"في البيانات: من نسبتهم فوق 30% تعثّر {pct(r_dti_hi)} ({n3} فرد)، مقابل {pct(r_dti_lo)} للباقين."),
        ("الاستقرار الوظيفي", "عدد سنوات العمل في الوضعية الحالية؛ الدخل المستقر أكثر قابلية للتوقع.",
         f"من لديهم أقل من سنتين تعثّر منهم {pct(r_st_lo)} ({n5} فرد)، ومن لديهم 5 سنوات فأكثر {pct(r_st_hi)} ({n6} فرد)."),
        ("سجل السداد", "السلوك الماضي من أقوى المؤشرات على السلوك المستقبلي.",
         "معدل التعثر حسب السجل: " + "، ".join(f"{h}: {pct(hist.get(h, 0))}" for h in data.HISTORY_LEVELS) + "."),
        ("مبلغ القرض ومدته", "قرض كبير مقارنة بالدخل يعني قسطًا أثقل؛ والمدة الطويلة تخفف القسط لكنها تطيل فترة التعرض للمخاطر.",
         f"من طلبوا قرضًا يتجاوز 1.5 مرة دخلهم السنوي تعثّر منهم {pct(r_lti_hi)} ({n7} فرد) مقابل {pct(r_lti_lo)} "
         f"لمن طلبوا أقل من 0.75 مرة ({n8}). المدد ≥ 60 شهرًا: {pct(r_term_long)}، والمدد ≤ 24 شهرًا: {pct(r_term_short)}."),
    ]
    for title, why, fact in items:
        with st.container(border=True):
            st.markdown(f"**{title}** — {why}\n\n{fact}")

    n_hi_ok = int(((ind.debt_to_income > 0.30) & (ind.default_12m == 0)).sum())
    n_lo_def = int(((ind.debt_to_income <= 0.10) & (ind.default_12m == 1)).sum())
    st.warning(f"**مؤشر واحد لا يكفي للقرار:** في بياناتنا {n_hi_ok} فردًا نسبة ديونهم فوق 30% ولم يتعثروا، "
               f"و{n_lo_def} فردًا نسبتهم 10% أو أقل وتعثّروا. لذلك تجمع النماذج كل المؤشرات معًا.")

    why_box("جدول الأفراد والبحث والرسوم",
            "يعرض البيانات الخام التي تتعلّم منها النماذج، ويسمح بفحص أي فرد.",
            "قبل بناء أي نموذج يجب أن نفهم البيانات: قيمها، توزيعها، والقيم الشاذة فيها.",
            "يكشف الأخطاء والعلاقات قبل التدريب، بدل أن نثق بالنموذج دون فهم.",
            f"الرسوم تجمع الأفراد في شرائح وتحسب معدل التعثر داخل كل شريحة. مثلًا {rec['person_id']} "
            f"دخله {fmt_value('monthly_income', rec['monthly_income'])} ونسبة ديونه {pct(dti)}.",
            "تُظهر الرسوم أيّ المؤشرات ترتبط بالتعثر، وهي نفس المؤشرات التي ستستعملها النماذج.",
            "الرسم يُظهر علاقة بين متغيرين فقط في كل مرة، ولا يثبت السببية.")


# ======================================================================= 3. الشركات
COMP_SHOW = ["company_id", "sector", "company_age", "annual_revenue", "net_profit", "operating_cash_flow",
             "total_assets", "total_liabilities", "total_debt", "current_ratio", "debt_ratio", "net_profit_margin",
             "debt_service_coverage", "loan_amount", "payment_history", "default_12m"]


def page_companies(comp, ind, res):
    st.title("بيانات الشركات")
    st.caption("200 شركة اصطناعية. البيانات للتعلم فقط.")
    c = st.columns(4)
    c[0].metric("عدد الشركات", len(comp))
    c[1].metric("وسيط رقم الأعمال", f"{comp.annual_revenue.median() / 1e6:,.1f} مليون دج")
    c[2].metric("شركات خاسرة", pct((comp.net_profit < 0).mean()))
    c[3].metric("معدل التعثر خلال 12 شهرًا", pct(comp.default_12m.mean()))

    st.subheader("البحث عن شركة")
    left, right = st.columns([1, 2])
    with left:
        rec = pick_record(comp, "company_id", "comp", "C")
    with right:
        st.markdown(f"**تفاصيل {rec['company_id']}**")
        details_card(rec, COMP_SHOW[1:])
        st.caption("«تعثّر خلال 12 شهرًا» هو المتغير المستهدف، ولا يدخل في متغيرات التنبؤ.")

    st.subheader("جدول البيانات")
    st.dataframe(display_df(comp[COMP_SHOW]), hide_index=True, height=320)

    st.subheader("رسوم بيانية")
    t1, t2, t3, t4 = st.tabs(["رقم الأعمال", "الربحية", "المديونية والتعثر", "التعثر حسب القطاع"])
    with t1:
        bins = [0, 20e6, 50e6, 100e6, 250e6, 500e6, 1e13]
        labs = ["< 20", "20–50", "50–100", "100–250", "250–500", "> 500"]
        bar_chart(binned_counts(comp.annual_revenue, bins, labs), "الفئة", "العدد", "رقم الأعمال (مليون دج)", "عدد الشركات")
    with t2:
        bins = [-1, -0.05, 0, 0.05, 0.10, 0.15, 1]
        labs = ["< −5%", "−5% إلى 0", "0–5%", "5–10%", "10–15%", "> 15%"]
        bar_chart(binned_counts(comp.net_profit_margin, bins, labs), "الفئة", "العدد", "هامش الربح الصافي", "عدد الشركات")
    with t3:
        buckets = pd.cut(comp.debt_ratio, [0, 0.2, 0.3, 0.4, 0.5, 1], labels=["< 20%", "20–30%", "30–40%", "40–50%", "> 50%"])
        bar_chart(default_rate_by(comp, buckets), "الفئة", "معدل التعثر", "نسبة المديونية (الديون ÷ الأصول)",
                  "معدل التعثر", color=RISK_COLORS["high"], y_format=".0%")
    with t4:
        g = default_rate_by(comp, "sector").sort_values("معدل التعثر", ascending=False)
        bar_chart(g, "الفئة", "معدل التعثر", "القطاع", "معدل التعثر", color=RISK_COLORS["high"], y_format=".0%")

    st.subheader("لماذا نستعمل هذه المؤشرات؟")
    q1, q3 = comp.annual_revenue.quantile([0.25, 0.75])
    facts = {
        "rev": (rate(comp.annual_revenue <= q1, comp), rate(comp.annual_revenue >= q3, comp)),
        "margin": (rate(comp.net_profit_margin < 0, comp), rate(comp.net_profit_margin > 0.10, comp)),
        "cf": (rate(comp.cash_flow_margin < 0, comp), rate(comp.cash_flow_margin >= 0, comp)),
        "liq": (rate(comp.current_ratio < 1, comp), rate(comp.current_ratio >= 1.5, comp)),
        "debt": (rate(comp.debt_ratio > 0.5, comp), rate(comp.debt_ratio <= 0.3, comp)),
        "dscr": (rate(comp.debt_service_coverage < 1, comp), rate(comp.debt_service_coverage >= 1.5, comp)),
    }
    hist = comp.groupby("payment_history")["default_12m"].mean()

    def f(k):
        (a, na), (b, nb) = facts[k]
        return a, na, b, nb

    a, na, b, nb = f("rev")
    items = [("رقم الأعمال", "حجم النشاط؛ الشركات الأكبر عادة أكثر تنوعًا وقدرة على امتصاص الصدمات.",
              f"أصغر ربع (≤ {q1 / 1e6:,.0f} مليون دج) تعثّر منه {pct(a)} ({na} شركة)، وأكبر ربع {pct(b)} ({nb} شركة).")]
    a, na, b, nb = f("margin")
    items.append(("الربحية", "الشركة الرابحة تولّد موارد لسداد ديونها؛ الخسارة المتكررة تستنزفها.",
                  f"الشركات الخاسرة تعثّر منها {pct(a)} ({na})، مقابل {pct(b)} لمن هامشها فوق 10% ({nb}). "
                  f"هامش {rec['company_id']}: {pct(rec['net_profit_margin'])}."))
    a, na, b, nb = f("cf")
    items.append(("التدفقات النقدية", "الربح المحاسبي قد لا يكون نقدًا متوفرًا؛ الديون تُسدَّد بالنقد.",
                  f"تدفق نقدي سالب: تعثّر {pct(a)} ({na})، تدفق موجب: {pct(b)} ({nb})."))
    a, na, b, nb = f("liq")
    items.append(("السيولة (النسبة الجارية)", "الأصول القصيرة الأجل ÷ الالتزامات القصيرة الأجل؛ أقل من 1 يعني صعوبة دفع ما يستحق قريبًا.",
                  f"نسبة < 1: تعثّر {pct(a)} ({na})، نسبة ≥ 1.5: {pct(b)} ({nb}). عند {rec['company_id']}: {rec['current_ratio']:g}."))
    a, na, b, nb = f("debt")
    items.append(("نسبة المديونية", "الديون ÷ الأصول؛ كلما ارتفعت زادت هشاشة الشركة أمام أي تراجع.",
                  f"فوق 50%: تعثّر {pct(a)} ({na})، و30% أو أقل: {pct(b)} ({nb}). عند {rec['company_id']}: {pct(rec['debt_ratio'])}."))
    a, na, b, nb = f("dscr")
    items.append(("القدرة على سداد الالتزامات", "التدفق النقدي ÷ خدمة الدين السنوية التقديرية (مع القرض الجديد). أقل من 1 = النقد لا يغطي الأقساط.",
                  f"أقل من 1: تعثّر {pct(a)} ({na})، و1.5 فأكثر: {pct(b)} ({nb}). عند {rec['company_id']}: {rec['debt_service_coverage']:g}."))
    items.append(("سجل السداد", "تأخرات الشركة مع البنوك والموردين في الماضي.",
                  "معدل التعثر حسب السجل: " + "، ".join(f"{h}: {pct(hist.get(h, 0))}" for h in data.HISTORY_LEVELS if h in hist) + "."))
    for title, why, fact in items:
        with st.container(border=True):
            st.markdown(f"**{title}** — {why}\n\n{fact}")

    ratio = comp.annual_revenue.median() / (ind.monthly_income.median() * 12)
    st.info(
        "**لماذا نحلّل الشركات بطريقة مختلفة؟** الفرد له دخل شهري واحد تقريبًا ونقيس عبئه عليه. الشركة لها قوائم مالية كاملة: "
        "إيرادات ومصاريف وأصول والتزامات، وقد تربح محاسبيًا وتعاني نقصًا في النقد. كذلك تختلف الأحجام كثيرًا: وسيط رقم أعمال "
        f"شركاتنا يعادل نحو {ratio:,.0f} مرة الدخل السنوي الوسيط للأفراد، لذلك نعتمد على **النِّسَب** (الهامش، المديونية، السيولة) "
        "بدل المبالغ الخام. الهدف واحد: احتمال التعثر، لكن المؤشرات والنماذج مستقلة.")

    why_box("مؤشرات الشركات",
            "تلخّص القوائم المالية لكل شركة في أرقام قابلة للمقارنة.",
            "لا يمكن مقارنة شركة رقم أعمالها 20 مليون بأخرى 1 مليار بالمبالغ الخام.",
            "النسب تعطي صورة عادلة عن الصحة المالية مهما كان حجم الشركة.",
            "كل نسبة = مبلغ ÷ مبلغ آخر من القوائم المالية، تُحسب قبل منح القرض.",
            "هذه النسب هي المتغيرات التي تدخل نماذج الشركات.",
            f"عدد الشركات صغير ({len(comp)})، فمعدلات التعثر داخل الشرائح الصغيرة قد تتغير كثيرًا بتغيير بضع شركات.")


# ======================================================================= 4. النماذج
def cond_text(step):
    lab = label(step["feature"])
    if step["category"] is not None:
        return f"{lab} ≠ «{step['category']}»" if step["left"] else f"{lab} = «{step['category']}»"
    op = "≤" if step["left"] else ">"
    return f"{lab} {op} {fmt_value(step['feature'], step['threshold'])} (القيمة: {fmt_value(step['feature'], step['value'])})"


def show_tree_path(R, rec):
    steps, leaf_n, leaf_rate = models.tree_path(R, rec)
    lines = [f"{i + 1}. {cond_text(s)} — كان في هذه العقدة {s['samples']} سجل تدريب" for i, s in enumerate(steps)]
    st.markdown("\n".join(lines))
    st.markdown(f"**الورقة النهائية:** {leaf_n} سجل تدريب، نسبة المتعثرين بينهم {pct(leaf_rate)} ← "
                f"وهذا هو احتمال التعثر الذي تعطيه الشجرة.")


def model_section(key, R, rec, probs, id_col):
    t = MODEL_TEXT[key]
    a, b = st.columns(2)
    with a:
        st.markdown(f"**ما هو هذا النموذج؟** {t['what']}\n\n**لماذا اخترناه؟** {t['why']}\n\n**كيف يعمل؟** {t['how']}")
    with b:
        st.markdown(f"**نوع البيانات:** {t['data']}\n\n**المميزات:** {t['pros']}\n\n**العيوب والحدود:** {t['cons']}")

    n_tr = len(R["X_train"])
    used = {
        "logistic": f"LogisticRegression من scikit-learn، دُرِّب على {n_tr} سجل بعد توحيد المقاييس وتحويل الفئات.",
        "tree": f"DecisionTreeClassifier بعمق أقصى 4 وحد أدنى {R['cfg']['min_leaf']} سجل في كل ورقة، على {n_tr} سجل.",
        "forest": f"RandomForestClassifier بـ 200 شجرة عمقها الأقصى 6، على {n_tr} سجل.",
        "mlp": f"MLPClassifier بطبقة خفية من 8 خلايا وعقوبة alpha=3 لمنع الحفظ، على {n_tr} سجل.",
    }[key]
    st.markdown(f"**كيف استخدمناه في التطبيق؟** {used}")

    st.markdown(f"##### النتيجة عند تقييم {rec[id_col]}")
    st.metric("احتمال التعثر حسب هذا النموذج", pct(probs[key]))
    if key == "logistic":
        contrib = models.logistic_contributions(R, rec).head(5)
        lines = [f"- {label(r.feature)}: {'يرفع' if r.contribution > 0 else 'يخفّض'} الدرجة ({r.contribution:+.2f})"
                 for r in contrib.itertuples()]
        st.markdown("أكبر المساهمات في درجة هذا السجل (المعامل × القيمة الموحَّدة):\n" + "\n".join(lines))
        coefs = models.logistic_coefficients(R).head(8)
        coefs["المتغير"] = [label(f) + (f" = {c}" if c else "") for f, c in zip(coefs.feature, coefs.category)]
        coefs["الاتجاه"] = np.where(coefs.coef > 0, "يرفع المخاطر", "يخفّض المخاطر")
        with st.expander("أكبر معاملات النموذج (عامة لكل المقترضين)"):
            st.dataframe(coefs[["المتغير", "coef", "الاتجاه"]].rename(columns={"coef": "المعامل"}).round(3), hide_index=True)
    elif key == "tree":
        st.markdown("المسار الفعلي لهذا السجل داخل الشجرة المدرَّبة:")
        show_tree_path(R, rec)
        with st.expander("رسم الشجرة كاملة"):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.tree import plot_tree
            pipe = R["pipes"]["tree"]
            names = pipe.named_steps["prep"].get_feature_names_out()
            fig, ax = plt.subplots(figsize=(18, 7))
            plot_tree(pipe.named_steps["model"], feature_names=list(names), class_names=["no default", "default"],
                      filled=True, impurity=False, proportion=True, fontsize=7, ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("أسماء المتغيرات في الرسم بالإنجليزية (أسماء الأعمدة في قاعدة البيانات) لأن مكتبة الرسم لا تدعم الحروف العربية جيدًا.")
    elif key == "forest":
        yes, total = models.forest_votes(R, rec)
        st.markdown(f"**{yes} شجرة من {total}** توقعت تعثّر هذا السجل (احتمالها ≥ 50%). "
                    "الاحتمال النهائي هو متوسط احتمالات الأشجار كلها.")
        imp = models.forest_importance(R).head(8)
        imp["المتغير"] = imp.feature.map(label)
        bar_chart(imp, "المتغير", "importance", "المتغير", "الأهمية في الغابة", color=ACCENT, y_format=".2f")
    else:
        info = models.mlp_info(R)
        c = st.columns(3)
        c[0].metric("مدخلات الشبكة", info["inputs"])
        c[1].metric("عدد الأوزان المتعلَّمة", info["weights"])
        c[2].metric("دورات التدريب", info["iterations"])
        st.markdown("منحنى الخطأ أثناء التدريب: ينخفض كلما عدّلت الشبكة أوزانها.")
        st.line_chart(pd.DataFrame({"الخطأ (loss)": info["loss_curve"]}), height=200)
        _, w = models.what_if(R, "mlp", rec)
        top = w.head(3)
        st.markdown("بما أن الشبكة صندوق أسود، نفسّرها بالتجربة: أكثر 3 مؤشرات غيّرت احتمالها لهذا السجل:\n" + "\n".join(
            f"- {label(r.feature)}: {'+' if r.effect > 0 else '−'}{abs(r.effect) * 100:.1f} نقطة مئوية" for r in top.itertuples()))


def page_models(ind, comp, res):
    st.title("نماذج الذكاء الاصطناعي")
    choice = st.radio("الفئة", ["الأفراد", "الشركات"], horizontal=True, key="m_group")
    g = "individuals" if choice == "الأفراد" else "companies"
    R, df = res[g], (ind if g == "individuals" else comp)
    id_col = R["cfg"]["id"]
    st.markdown(f"ندرّب **أربعة نماذج** على نفس متغيرات {choice} ({len(R['features'])} متغيرًا)، باستعمال "
                f"{len(R['X_train'])} سجل للتدريب، ونختبرها على {len(R['X_test'])} سجل منفصل. "
                "نماذج الأفراد ونماذج الشركات مستقلة تمامًا.")
    first_test = df.loc[R["X_test"].index[0], id_col]
    rec = pick_record(df, id_col, f"m_{g}", "P" if g == "individuals" else "C", default_id=first_test)
    in_train = rec[id_col] in R["train_ids"]
    st.caption(("هذا السجل من **مجموعة التدريب**: النموذج رآه أثناء التعلم." if in_train else
                "هذا السجل من **مجموعة الاختبار**: النموذج لم يره أثناء التدريب."))
    probs = models.predict_all(R, rec)
    tabs = st.tabs([models.MODEL_NAMES[k] for k in models.MODEL_NAMES])
    for key, tab in zip(models.MODEL_NAMES, tabs):
        with tab:
            model_section(key, R, rec, probs, id_col)

    why_box("أربعة نماذج بدل نموذج واحد",
            "نقارن طرقًا مختلفة لحل المشكلة نفسها على البيانات نفسها.",
            "لا يوجد نموذج أفضل دائمًا؛ الأفضل يتحدد بالتجربة على بيانات لم يرها النموذج.",
            "يكشف إن كان نموذج بسيط ومفهوم (الانحدار اللوجستي) يكفي، أو إن كان التعقيد يضيف دقة حقيقية.",
            "كل نموذج يتعلم من مجموعة التدريب، ثم يعطي احتمالًا لكل سجل جديد.",
            f"لهذا السجل تتراوح احتمالات النماذج بين {pct(min(probs.values()))} و{pct(max(probs.values()))}.",
            "كل النماذج تتعلم من الماضي؛ إذا تغيّرت الظروف الاقتصادية قد تخطئ كلها معًا.")


# ======================================================================= 5. تقييم المخاطر
IND_EVAL = ["age", "monthly_income", "job_type", "job_stability_years", "monthly_obligations", "debt_to_income",
            "loan_amount", "loan_term_months", "late_payments_24m", "payment_history"]
COMP_EVAL = ["sector", "company_age", "annual_revenue", "net_profit_margin", "cash_flow_margin", "current_ratio",
             "debt_ratio", "debt_service_coverage", "loan_amount", "loan_to_revenue", "payment_history"]


def page_evaluate(ind, comp, res):
    st.title("تقييم المخاطر الائتمانية")
    kind = st.radio("نوع التقييم", ["تقييم فرد", "تقييم شركة"], horizontal=True, key="e_kind")
    g = "individuals" if kind == "تقييم فرد" else "companies"
    R, df = res[g], (ind if g == "individuals" else comp)
    id_col = R["cfg"]["id"]

    with st.expander("حدود تصنيف المخاطر (قابلة للتعديل)"):
        low_t, high_t = st.slider("احتمال التعثر: ما دون الحد الأول منخفض، وما فوق الحد الثاني مرتفع",
                                  0.0, 1.0, (0.10, 0.30), step=0.01, key="e_thr")
        if low_t >= high_t:
            st.warning("الحد الأول يجب أن يكون أقل من الحد الثاني.")
        st.caption(f"منخفض: أقل من {pct(low_t, 0)} · متوسط: من {pct(low_t, 0)} إلى أقل من {pct(high_t, 0)} · "
                   f"مرتفع: {pct(high_t, 0)} فأكثر. معدل التعثر العام في بيانات التدريب: {pct(R['base_rate'])}.")

    rec = pick_record(df, id_col, f"e_{g}", "P" if g == "individuals" else "C")
    st.markdown(f"**بيانات {rec[id_col]}**")
    details_card(rec, IND_EVAL if g == "individuals" else COMP_EVAL)

    probs = models.predict_all(R, rec)
    avg = float(np.mean(list(probs.values())))
    level = models.risk_level(avg, low_t, high_t)

    st.subheader("النتيجة")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("احتمال التعثر المتوقع خلال الـ 12 شهرًا القادمة (متوسط النماذج الأربعة)")
        st.markdown(f'<span class="big-pd">{pct(avg)}</span>', unsafe_allow_html=True)
    with c2:
        st.markdown("مستوى المخاطر")
        st.markdown(risk_pill(level), unsafe_allow_html=True)
    st.caption("احتمال التعثر **تقدير إحصائي** مبني على مقترضين يشبهون هذا الملف، وليس تأكيدًا بأنه سيتعثر أو لن يتعثر. "
               f"احتمال {pct(avg, 0)} يعني: من بين 100 ملف مشابه نتوقع أن يتعثر نحو {avg * 100:.0f}.")

    st.markdown("##### نتائج النماذج الأربعة")
    table = pd.DataFrame({"key": list(probs), "النموذج": [models.MODEL_NAMES[k] for k in probs],
                          "احتمال التعثر": list(probs.values())})
    table["المستوى"] = [RISK_NAMES[models.risk_level(p, low_t, high_t)] for p in table["احتمال التعثر"]]
    bars = alt.Chart(table).mark_bar(color=PRIMARY, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("النموذج:N", sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("احتمال التعثر:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        tooltip=["النموذج", alt.Tooltip("احتمال التعثر:Q", format=".1%"), "المستوى"])
    rules = alt.Chart(pd.DataFrame({"y": [low_t, high_t], "c": [RISK_COLORS["mid"], RISK_COLORS["high"]]})).mark_rule(
        strokeDash=[5, 4], strokeWidth=2).encode(y="y:Q", color=alt.Color("c:N", scale=None))
    left, right = st.columns([1.3, 1])
    with left:
        st.altair_chart((bars + rules).properties(height=260))
        st.caption("الخطان المتقطعان: حدّا المستوى المتوسط والمرتفع.")
    with right:
        show = table[["النموذج", "احتمال التعثر", "المستوى"]].copy()
        show["احتمال التعثر"] = show["احتمال التعثر"].map(pct)
        st.dataframe(show, hide_index=True)

    levels = set(table["المستوى"])
    spread = max(probs.values()) - min(probs.values())
    hi_k, lo_k = max(probs, key=probs.get), min(probs, key=probs.get)
    if len(levels) > 1 or spread > 0.15:
        st.warning(f"**النماذج لا تتفق تمامًا:** أعلى تقدير من «{models.MODEL_NAMES[hi_k]}» ({pct(probs[hi_k])}) وأدناه من "
                   f"«{models.MODEL_NAMES[lo_k]}» ({pct(probs[lo_k])})، أي فرق {spread * 100:.0f} نقطة مئوية. "
                   "يحدث هذا لأن كل نموذج يتعلم العلاقات بطريقة مختلفة (خطية، بشروط، بمتوسط أشجار، بأوزان). "
                   "في الواقع، ملف كهذا يُحال عادة إلى محلل بشري لدراسة أعمق.")
    else:
        st.success(f"النماذج الأربعة متقاربة (الفرق {spread * 100:.0f} نقطة مئوية) وتضع الملف في المستوى نفسه، "
                   "وهذا يزيد الثقة في التقدير.")

    with st.expander("ماذا حدث فعلًا لهذا المقترض في البيانات التاريخية؟"):
        actual = int(rec["default_12m"])
        where = "مجموعة التدريب (النموذج رأى نتيجته)" if rec[id_col] in R["train_ids"] else "مجموعة الاختبار (النموذج لم يرَ نتيجته)"
        st.markdown(f"النتيجة الفعلية: **{'تعثّر' if actual else 'لم يتعثر'}** خلال 12 شهرًا. السجل من {where}. "
                    "تذكّر: احتمال 20% لا يعني أن الملف «خطأ» إذا تعثّر؛ يعني أن 1 من كل 5 ملفات مشابهة يتعثر.")

    st.subheader("لماذا حصل هذا المقترض على هذه النتيجة؟")
    exp_key = st.selectbox("النموذج المستعمل في التفسير", list(models.MODEL_NAMES), index=2,
                           format_func=lambda k: models.MODEL_NAMES[k], key=f"e_exp_{g}")
    base, w = models.what_if(R, exp_key, rec)
    st.caption("الطريقة: نستبدل كل مؤشر بقيمته النموذجية في بيانات التدريب (الوسيط أو الفئة الأكثر شيوعًا) "
               "ونقيس كم يتغيّر احتمال التعثر. هذا التفسير يُحسب من جديد لكل سجل ولكل نموذج.")
    strong = w[w.effect.abs() >= 0.005].head(6)
    if strong.empty:
        st.markdown("لا يوجد مؤشر يغيّر الاحتمال بمفرده بشكل ملحوظ: الملف قريب من الملف النموذجي في كل المؤشرات.")
    else:
        for r in strong.itertuples():
            arrow = "▲ رفع" if r.effect > 0 else "▼ خفّض"
            st.markdown(f"- **{label(r.feature)} = {fmt_value(r.feature, r.value)}** (القيمة النموذجية: "
                        f"{fmt_value(r.feature, r.reference)}): {arrow} الاحتمال بـ {abs(r.effect) * 100:.1f} نقطة مئوية. "
                        f"لو كان نموذجيًا لأصبح الاحتمال {pct(r.proba_if_typical)} بدل {pct(base)}.")
        chart_df = strong.assign(المؤشر=strong.feature.map(label), الأثر=strong.effect * 100)
        chart = alt.Chart(chart_df).mark_bar().encode(
            y=alt.Y("المؤشر:N", sort=None, title=None),
            x=alt.X("الأثر:Q", title="الأثر على الاحتمال (نقطة مئوية)"),
            color=alt.condition(alt.datum["الأثر"] > 0, alt.value(RISK_COLORS["high"]), alt.value(RISK_COLORS["low"])),
            tooltip=["المؤشر", alt.Tooltip("الأثر:Q", format="+.1f")]).properties(height=34 * len(chart_df) + 30)
        st.altair_chart(chart)
        up = [label(f) for f, e in zip(strong.feature, strong.effect) if e > 0]
        down = [label(f) for f, e in zip(strong.feature, strong.effect) if e < 0]
        summary = f"حسب «{models.MODEL_NAMES[exp_key]}» ({pct(base)}): "
        summary += ("أهم ما رفع المخاطر: " + "، ".join(up[:3]) + ". ") if up else "لا يوجد مؤشر يرفع المخاطر بوضوح. "
        summary += ("وأهم ما خفّضها: " + "، ".join(down[:3]) + ".") if down else ""
        st.info(summary)
    if exp_key == "tree":
        st.markdown("**مسار القرار داخل الشجرة:**")
        show_tree_path(R, rec)

    why_box("صفحة التقييم",
            "تحوّل بيانات مقترض واحد إلى احتمال تعثر ومستوى مخاطر وتفسير.",
            "هذا ما يحتاجه موظف القرض فعلًا: رقم يمكن مقارنته وسبب يمكن شرحه للعميل.",
            "يستبدل الحكم الحدسي بتقدير مبني على مئات الملفات السابقة.",
            f"كل نموذج يعطي احتمالًا، نأخذ متوسطها ({pct(avg)}) ونقارنه بالحدّين {pct(low_t, 0)} و{pct(high_t, 0)}.",
            "المستوى (منخفض/متوسط/مرتفع) يترجم الاحتمال إلى قرار: قبول، قبول بضمانات، أو رفض/دراسة أعمق.",
            "الحدود اختيار إداري يعكس مدى تحمّل البنك للمخاطر، وليست حقيقة علمية؛ والتفسير يقيس أثر كل مؤشر منفردًا.")


# ======================================================================= 6. مقارنة النماذج
def confusion_html(cm):
    return (f'<table class="cm"><tr><th></th><th>توقع: متعثر</th><th>توقع: سليم</th></tr>'
            f'<tr><th>فعلًا متعثر</th><td>{cm["tp"]}</td><td>{cm["fn"]}</td></tr>'
            f'<tr><th>فعلًا سليم</th><td>{cm["fp"]}</td><td>{cm["tn"]}</td></tr></table>')


def compare_group(R, thr, name):
    mt = models.metrics_table(R, thr)
    n_test, n_def = len(R["y_test"]), int(R["y_test"].sum())
    st.caption(f"مجموعة الاختبار لـ{name}: {n_test} سجل، منها {n_def} حالة تعثر. لم تُستعمل في التدريب.")
    show = mt.drop(columns="key").rename(columns={
        "model": "النموذج", "auc": "ROC-AUC (اختبار)", "auc_train": "ROC-AUC (تدريب)",
        "precision": "Precision", "recall": "Recall", "f1": "F1-score"}).round(3)
    st.dataframe(show, hide_index=True)

    best_auc = mt.loc[mt.auc.idxmax()]
    best_rec = mt.loc[mt.recall.idxmax()]
    best_pre = mt.loc[mt.precision.idxmax()]
    notes = [f"أفضل تمييز بين المتعثرين وغيرهم: **{best_auc.model}** (ROC-AUC = {best_auc.auc:.2f}).",
             f"أكثر نموذج يكتشف المتعثرين عند العتبة {pct(thr, 0)}: **{best_rec.model}** (Recall = {best_rec.recall:.2f}).",
             f"أقل إنذارات خاطئة: **{best_pre.model}** (Precision = {best_pre.precision:.2f})."]
    over = mt[mt.auc_train - mt.auc > 0.07]
    if not over.empty:
        notes.append("فرق واضح بين أداء التدريب والاختبار (علامة إفراط في التعلّم) عند: " + "، ".join(
            f"{r.model} ({r.auc_train:.2f} ← {r.auc:.2f})" for r in over.itertuples()) + ".")
    if n_def < 20:
        notes.append(f"عدد حالات التعثر في الاختبار قليل ({n_def})، لذلك قد تتغير المقاييس كثيرًا مع بيانات أخرى.")
    st.markdown("\n".join(f"- {n}" for n in notes))

    roc = models.roc_points(R)
    lines = alt.Chart(roc).mark_line(strokeWidth=2).encode(
        x=alt.X("fpr:Q", title="نسبة الإنذارات الخاطئة (FPR)"), y=alt.Y("tpr:Q", title="نسبة المتعثرين المكتشفين (TPR)"),
        color=alt.Color("model:N", title="النموذج"))
    diag = alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})).mark_line(strokeDash=[4, 4], color="#999").encode(x="fpr", y="tpr")
    st.markdown("**منحنى ROC** — كلما اقترب المنحنى من الزاوية العليا اليسرى كان النموذج أفضل؛ الخط المتقطع = تخمين عشوائي.")
    st.altair_chart((diag + lines).properties(height=300))

    st.markdown(f"**مصفوفات الالتباس** (العتبة {pct(thr, 0)})")
    cols = st.columns(4)
    for col, key in zip(cols, models.MODEL_NAMES):
        with col:
            st.markdown(f"**{models.MODEL_NAMES[key]}**")
            st.markdown(confusion_html(models.confusion(R, key, thr)), unsafe_allow_html=True)


def page_compare(res):
    st.title("مقارنة النماذج")
    thr = st.slider("عتبة التصنيف: نعتبر المقترض «متوقَّع التعثر» إذا تجاوز احتماله هذه القيمة",
                    0.05, 0.95, 0.30, step=0.05, key="c_thr")
    st.caption("لماذا 30% وليس 50%؟ لأن المتعثرين أقلية (نحو 20%)، فعتبة 50% تجعل النماذج تفوّت كثيرًا منهم. "
               "غيّر العتبة ولاحظ كيف يرتفع Recall وينخفض Precision أو العكس. ROC-AUC لا يتأثر بالعتبة.")
    t1, t2 = st.tabs(["الأفراد", "الشركات"])
    with t1:
        compare_group(res["individuals"], thr, "الأفراد")
    with t2:
        compare_group(res["companies"], thr, "الشركات")

    st.subheader("معنى المقاييس")
    st.markdown(
        "- **ROC-AUC:** مدى قدرة النموذج على التمييز بين حالات التعثر وعدم التعثر. 0.5 = تخمين عشوائي، 1 = تمييز مثالي.\n"
        "- **Precision:** من بين من توقّع النموذج تعثرهم، كم تعثّر فعلًا؟ (الدقة في الإنذار)\n"
        "- **Recall:** من بين المتعثرين فعلًا، كم اكتشف النموذج؟ (القدرة على الاكتشاف)\n"
        "- **F1-score:** مقياس يجمع Precision وRecall في رقم واحد.\n"
        "- **مصفوفة الالتباس:** تعدّ الأخطاء بنوعيها: متعثر لم يُكتشف (خسارة مالية)، وسليم رُفض خطأً (خسارة عميل).")
    ri = models.metrics_table(res["individuals"], thr)
    rec_best = ri.loc[ri.recall.idxmax()]
    pre_best = ri.loc[ri.precision.idxmax()]
    st.subheader("أي نموذج لأي هدف؟")
    st.markdown(
        f"- إذا كان الهدف **عدم تفويت أي متعثر** (حماية أموال البنك)، نفضّل Recall أعلى: عند الأفراد «{rec_best.model}» "
        f"يكتشف {pct(rec_best.recall, 0)} من المتعثرين.\n"
        f"- إذا كان الهدف **عدم رفض عملاء جيدين** (نمو المحفظة)، نفضّل Precision أعلى: «{pre_best.model}» "
        f"صحيح في {pct(pre_best.precision, 0)} من إنذاراته.\n"
        "- إذا كان الهدف **الشرح للعميل وللمراقب البنكي**، قد نختار الانحدار اللوجستي أو الشجرة حتى لو كانت أقل دقة قليلًا.\n"
        "- النموذج الأفضل **ليس** الذي يعطي أقل احتمالات تعثر، بل الذي يميّز جيدًا على بيانات لم يتدرب عليها.")

    why_box("مقاييس الأداء ومجموعة الاختبار",
            "تقيس جودة كل نموذج برقم موضوعي.",
            "التوقعات وحدها لا تخبرنا إن كان النموذج جيدًا؛ نموذج يقول «لا أحد سيتعثر» يبدو مطمئنًا لكنه عديم الفائدة.",
            f"الدقة البسيطة مضلّلة: نموذج يتوقع «لا تعثر» للجميع يكون صحيحًا في {pct(1 - res['individuals']['y_test'].mean(), 0)} "
            "من حالات الأفراد دون أن يكتشف متعثرًا واحدًا. لذلك نستعمل ROC-AUC وPrecision وRecall.",
            "نقارن توقعات كل نموذج على مجموعة الاختبار بالنتائج الفعلية ونعدّ الإصابات والأخطاء.",
            "تحدد أي نموذج نثق به لتقييم المقترضين الجدد، وأي عتبة نستعمل.",
            "مجموعة الاختبار صغيرة (خاصة للشركات)، فالفروق الصغيرة بين النماذج قد تكون صدفة.")


# ======================================================================= 7. المصطلحات
def page_glossary(ind, comp, res):
    st.title("قاموس المصطلحات")
    ri, rc = res["individuals"], res["companies"]
    mi = models.metrics_table(ri)
    examples = {
        "التعثر الائتماني": f"في بياناتنا تعثّر {pct(ind.default_12m.mean())} من الأفراد و{pct(comp.default_12m.mean())} من الشركات.",
        "البيانات الاصطناعية": f"{len(ind)} فرد و{len(comp)} شركة مولَّدون آليًا ومحفوظون في SQLite.",
        "المتغير المستهدف": "العمود «تعثّر خلال 12 شهرًا» (default_12m).",
        "المتغيرات التفسيرية": f"{len(ri['features'])} متغيرًا للأفراد و{len(rc['features'])} للشركات.",
        "تدريب النموذج": f"{len(ri['X_train'])} فرد و{len(rc['X_train'])} شركة للتدريب.",
        "مجموعة الاختبار": f"{len(ri['X_test'])} فرد و{len(rc['X_test'])} شركة لم تستعملها النماذج في التعلّم.",
        "ROC-AUC": "أفضل قيمة حاليًا للأفراد: " + f"{mi.auc.max():.2f}.",
        "الإفراط في التعلّم (Overfitting)": "قارن عمودي ROC-AUC (تدريب) و(اختبار) في صفحة مقارنة النماذج.",
    }
    q = st.text_input("ابحث عن مصطلح", key="g_q")
    for term, definition in GLOSSARY:
        if q and q.strip() not in term and q.strip() not in definition:
            continue
        ex = examples.get(term)
        with st.container(border=True):
            st.markdown(f"**{term}** — {definition}" + (f"\n\n*في منصتنا:* {ex}" if ex else ""))
    why_box("القاموس",
            "يجمع المصطلحات الأساسية في مكان واحد.",
            "مصطلحات مثل Recall والعتبة تتكرر في كل الصفحات وقد تُنسى.",
            "يمنع الخلط بين مفاهيم متقاربة مثل الدقة وPrecision.",
            "تعريف قصير لكل مصطلح، مع مثال من أرقام المنصة الفعلية عند الإمكان.",
            "فهم المصطلحات شرط لقراءة نتائج التقييم ومقارنة النماذج بشكل صحيح.",
            "التعريفات مبسطة عمدًا؛ للدراسة المعمّقة راجع مراجع الإحصاء وتعلم الآلة.")


# ======================================================================= التشغيل
PAGES = ["الرئيسية", "بيانات الأفراد", "بيانات الشركات", "نماذج الذكاء الاصطناعي",
         "تقييم المخاطر", "مقارنة النماذج", "المصطلحات"]


def main():
    st.set_page_config(page_title="Univ 3 Fintech", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if not st.session_state.get("auth"):
        login_page()
        return

    meta = data.load_meta()
    version = f"{meta['created_at']}-{meta['seed']}"
    ind, comp = get_data(version)
    res = get_models(version)

    with st.sidebar:
        st.markdown("## Univ 3 Fintech")
        st.caption("منصة تعليمية لتقييم المخاطر الائتمانية")
        page = st.radio("التنقل", PAGES, key="page")
        st.divider()
        st.caption(f"مستخدم: {st.session_state.get('user', '')}")
        if st.button("تسجيل الخروج", key="logout"):
            st.session_state.clear()
            st.rerun()
        st.caption("جميع البيانات اصطناعية ومخصصة للتعلم فقط.")

    if page == "الرئيسية":
        page_home(ind, comp, res, meta)
    elif page == "بيانات الأفراد":
        page_individuals(ind, res)
    elif page == "بيانات الشركات":
        page_companies(comp, ind, res)
    elif page == "نماذج الذكاء الاصطناعي":
        page_models(ind, comp, res)
    elif page == "تقييم المخاطر":
        page_evaluate(ind, comp, res)
    elif page == "مقارنة النماذج":
        page_compare(res)
    else:
        page_glossary(ind, comp, res)


if __name__ == "__main__":
    main()
