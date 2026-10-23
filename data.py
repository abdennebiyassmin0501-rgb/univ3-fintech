"""
data.py — توليد البيانات الاصطناعية وحفظها في SQLite.

الفكرة:
- عند أول تشغيل، نولّد 800 فرد و200 شركة ونحفظهم في ملف univ3_fintech.db.
- في كل تشغيل لاحق نقرأ البيانات من الملف، فلا تتغيّر إلا إذا طلب المستخدم إعادة إنشائها.

تعريف التعثر (ثابت في كل المشروع):
    default_12m = 1  إذا تأخر المقترض عن السداد 90 يومًا أو أكثر
                     خلال الاثني عشر شهرًا التي تلي منح القرض.
كل المتغيرات الأخرى معروفة "يوم تقديم الطلب"، أي قبل وقوع أي تعثر،
لذلك لا يوجد تسرّب لمعلومة التعثر داخل متغيرات التنبؤ.
"""
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "univ3_fintech.db"
DEFAULT_SEED = 2026
N_INDIVIDUALS = 800
N_COMPANIES = 200
TARGET = "default_12m"

JOB_TYPES = ["موظف قطاع عام", "موظف قطاع خاص", "عمل حر", "تاجر", "عامل مؤقت", "متقاعد"]
SECTORS = ["تجارة", "صناعة", "بناء وأشغال عمومية", "خدمات", "فلاحة", "سياحة", "تكنولوجيا"]
HISTORY_LEVELS = ["ممتاز", "جيد", "متوسط", "ضعيف"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _calibrate(z, target_rate):
    """نبحث عن ثابت نضيفه إلى z حتى يصبح متوسط احتمال التعثر قريبًا من النسبة المطلوبة."""
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return z + (lo + hi) / 2


def _history_from_late(late, rng):
    """سجل السداد السابق: تقييم نوعي يعتمد على التأخرات السابقة مع شيء من الضوضاء."""
    h = late + rng.normal(0, 0.8, len(late))
    return np.select([h < 0.5, h < 1.5, h < 3.0], HISTORY_LEVELS[:3], default=HISTORY_LEVELS[3])


def generate_individuals(rng, n=N_INDIVIDUALS):
    age = rng.integers(21, 64, n)
    job = rng.choice(JOB_TYPES, n, p=[0.30, 0.28, 0.14, 0.12, 0.12, 0.04])
    # المتقاعد لا يكون شابًا
    young_retired = (job == "متقاعد") & (age < 55)
    age = np.where(young_retired, rng.integers(55, 64, n), age)

    income_mult = pd.Series(job).map({
        "موظف قطاع عام": 1.0, "موظف قطاع خاص": 1.1, "عمل حر": 1.15,
        "تاجر": 1.3, "عامل مؤقت": 0.65, "متقاعد": 0.8}).to_numpy()
    income = 60000 * income_mult * np.exp(rng.normal(0, 0.45, n)) * (1 + (age - 21) * 0.01)
    income = np.clip(np.round(income / 500) * 500, 22000, 600000)

    stab_scale = pd.Series(job).map({
        "موظف قطاع عام": 5.0, "موظف قطاع خاص": 3.0, "عمل حر": 2.5,
        "تاجر": 4.0, "عامل مؤقت": 0.7, "متقاعد": 2.0}).to_numpy()
    stability = np.round(np.minimum(rng.gamma(2.0, stab_scale), age - 20), 1)

    has_debt = rng.random(n) < 0.6
    obligations = np.where(has_debt, income * rng.beta(2, 6, n) * 0.9, 0)
    obligations = np.round(obligations / 100) * 100
    total_debt = np.round(obligations * rng.integers(6, 60, n) / 1000) * 1000
    dti = np.round(obligations / income, 3)

    loan_amount = np.clip(np.round(income * rng.uniform(3, 30, n) / 10000) * 10000, 50000, 8_000_000)
    ratio = loan_amount / income
    term = np.where(ratio < 8, rng.choice([12, 24], n),
                    np.where(ratio < 16, rng.choice([24, 36, 48], n), rng.choice([48, 60, 84], n)))

    lam = 0.25 + 2.2 * dti + 0.4 * (job == "عامل مؤقت") + 0.4 * (income < 35000)
    late = np.clip(rng.poisson(lam), 0, 12)
    history = _history_from_late(late, rng)

    # آلية التعثر "الخفية" (لا يراها النموذج، بل يحاول اكتشافها من البيانات)
    new_payment = loan_amount * (1 + 0.065 * term / 12) / term
    burden = (obligations + new_payment) / income
    hist_eff = pd.Series(history).map({"ممتاز": -0.5, "جيد": 0.0, "متوسط": 0.4, "ضعيف": 0.9}).to_numpy()
    job_eff = pd.Series(job).map({"موظف قطاع عام": -0.4, "موظف قطاع خاص": 0.0, "عمل حر": 0.25,
                                  "تاجر": 0.15, "عامل مؤقت": 0.55, "متقاعد": -0.2}).to_numpy()
    z = 1.6 * (3.0 * burden + 0.45 * late + 1.4 * hist_eff + 1.3 * job_eff - 0.09 * np.minimum(stability, 12)
               - 0.7 * np.log(income / 60000) + 0.35 * (age < 26) + 0.006 * (term - 36)) \
        + rng.normal(0, 0.6, n)
    p = _sigmoid(_calibrate(z, 0.18))
    default = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "person_id": [f"P-{i:04d}" for i in range(1, n + 1)],
        "age": age.astype(int),
        "monthly_income": income.astype(int),
        "job_type": job,
        "job_stability_years": stability,
        "total_debt": total_debt.astype(int),
        "monthly_obligations": obligations.astype(int),
        "debt_to_income": dti,
        "loan_amount": loan_amount.astype(int),
        "loan_term_months": term.astype(int),
        "late_payments_24m": late.astype(int),
        "payment_history": history,
        TARGET: default,
    })


def generate_companies(rng, n=N_COMPANIES):
    sector = rng.choice(SECTORS, n, p=[0.22, 0.15, 0.15, 0.18, 0.10, 0.08, 0.12])
    company_age = np.clip(1 + rng.gamma(2.0, 5.0, n), 1, 50).astype(int)
    revenue = 80e6 * np.exp(rng.normal(0, 1.0, n)) * (1 + np.minimum(company_age, 20) * 0.03)
    revenue = np.clip(np.round(revenue / 1e5) * 1e5, 3e6, 5e9)

    sector_margin = pd.Series(sector).map({"تجارة": 0.0, "صناعة": 0.01, "بناء وأشغال عمومية": -0.02,
                                           "خدمات": 0.02, "فلاحة": -0.01, "سياحة": -0.01,
                                           "تكنولوجيا": 0.03}).to_numpy()
    margin = np.clip(rng.normal(0.05 + sector_margin, 0.07, n), -0.30, 0.30)
    net_profit = np.round(revenue * margin / 1e4) * 1e4
    cash_flow = np.round((net_profit + revenue * rng.normal(0.03, 0.04, n)) / 1e4) * 1e4

    assets = np.round(revenue * rng.uniform(0.5, 1.8, n) / 1e5) * 1e5
    liab_ratio = np.clip(rng.beta(4, 3, n), 0.10, 0.98)
    liabilities = np.round(assets * liab_ratio / 1e5) * 1e5
    total_debt = np.round(liabilities * rng.uniform(0.45, 0.85, n) / 1e5) * 1e5
    current_ratio = np.round(np.clip(rng.normal(1.5 - 0.6 * (liab_ratio - 0.55), 0.45, n), 0.3, 4.0), 2)
    debt_ratio = np.round(total_debt / assets, 3)
    loan_amount = np.round(revenue * rng.uniform(0.05, 0.35, n) / 1e5) * 1e5
    # قدرة السداد: التدفق النقدي مقسومًا على خدمة الدين السنوية التقديرية (الديون القائمة + القرض الجديد)
    debt_service = total_debt * 0.18 + loan_amount * 0.25
    dscr = np.round(cash_flow / debt_service, 2)

    late = np.clip(rng.poisson(0.3 + 2.0 * np.maximum(0, debt_ratio - 0.45) + 0.8 * (margin < 0)), 0, 10)
    history = _history_from_late(late, rng)

    hist_eff = pd.Series(history).map({"ممتاز": -0.5, "جيد": 0.0, "متوسط": 0.45, "ضعيف": 1.0}).to_numpy()
    sector_eff = pd.Series(sector).map({"تجارة": 0.0, "صناعة": 0.1, "بناء وأشغال عمومية": 0.45,
                                        "خدمات": -0.1, "فلاحة": 0.3, "سياحة": 0.35,
                                        "تكنولوجيا": -0.15}).to_numpy()
    z = 1.8 * (5.0 * (debt_ratio - 0.40) - 9.0 * margin - 0.55 * np.clip(dscr, -2, 4)
               - 0.9 * (current_ratio - 1.4) + 1.3 * hist_eff + 1.2 * sector_eff
               - 0.04 * np.minimum(company_age, 20) + 0.6 * np.log(loan_amount / revenue / 0.2)) \
        + rng.normal(0, 0.6, n)
    p = _sigmoid(_calibrate(z, 0.20))
    default = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "company_id": [f"C-{i:04d}" for i in range(1, n + 1)],
        "sector": sector,
        "company_age": company_age,
        "annual_revenue": revenue.astype(np.int64),
        "net_profit": net_profit.astype(np.int64),
        "operating_cash_flow": cash_flow.astype(np.int64),
        "total_assets": assets.astype(np.int64),
        "total_liabilities": liabilities.astype(np.int64),
        "total_debt": total_debt.astype(np.int64),
        "current_ratio": current_ratio,
        "debt_ratio": debt_ratio,
        "net_profit_margin": np.round(net_profit / revenue, 3),
        "cash_flow_margin": np.round(cash_flow / revenue, 3),
        "debt_service_coverage": dscr,
        "loan_amount": loan_amount.astype(np.int64),
        "loan_to_revenue": np.round(loan_amount / revenue, 3),
        "payment_history": history,
        TARGET: default,
    })


def create_database(seed=DEFAULT_SEED, db_path=DB_PATH):
    """يولّد البيانات ويحفظها (يستبدل البيانات القديمة إن وجدت)."""
    rng = np.random.default_rng(seed)
    individuals = generate_individuals(rng)
    companies = generate_companies(rng)
    meta = pd.DataFrame([{
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(seed),
        "n_individuals": len(individuals),
        "n_companies": len(companies),
    }])
    with closing(sqlite3.connect(db_path)) as con, con:
        individuals.to_sql("individuals", con, if_exists="replace", index=False)
        companies.to_sql("companies", con, if_exists="replace", index=False)
        meta.to_sql("meta", con, if_exists="replace", index=False)


def database_ready(db_path=DB_PATH):
    if not Path(db_path).exists():
        return False
    try:
        with closing(sqlite3.connect(db_path)) as con, con:
            n_i = con.execute("SELECT COUNT(*) FROM individuals").fetchone()[0]
            n_c = con.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        return n_i > 0 and n_c > 0
    except sqlite3.Error:
        return False


def ensure_database(db_path=DB_PATH):
    """يُنشئ قاعدة البيانات عند أول تشغيل فقط."""
    if not database_ready(db_path):
        create_database(db_path=db_path)


def load_meta(db_path=DB_PATH):
    ensure_database(db_path)
    with closing(sqlite3.connect(db_path)) as con, con:
        return pd.read_sql("SELECT * FROM meta", con).iloc[0].to_dict()


def load_data(db_path=DB_PATH):
    ensure_database(db_path)
    with closing(sqlite3.connect(db_path)) as con, con:
        individuals = pd.read_sql("SELECT * FROM individuals", con)
        companies = pd.read_sql("SELECT * FROM companies", con)
    return individuals, companies
