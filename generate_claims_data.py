"""
Synthetic TPA claims data generator.

Produces three CSV files that mirror the Yuzu claims schema:
  members.csv       — one row per member (no PII)
  claims.csv        — one row per claim with adjudication totals
  member_summary.csv — one row per member, aggregated features + target label

Target variable: is_high_cost (plan paid > $25,000 in a plan year)

The cohort is controlled by STUDENT_SEED and N_MEMBERS in config.py, so each
student can generate their own dataset: same code, different members, different
prevalence, different fairness story.

Usage:
    python generate_claims_data.py
    STUDENT_SEED=7 python generate_claims_data.py     # your own cohort
"""

import csv
import random
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    HIGH_COST_THRESHOLD_CENTS,
    HIGH_COST_THRESHOLD_USD,
    N_MEMBERS,
    STUDENT_SEED,
)

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

PLAN_YEAR_START = date(2024, 1, 1)
PLAN_YEAR_END = date(2024, 12, 31)

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
SEX = ["M", "F"]

# ICD-10 codes with rough prevalence weights and chronic-condition flag
DIAGNOSIS_CODES = [
    ("Z00.00", "Encounter for general adult medical exam", 0.15, False),
    ("M54.5",  "Low back pain",                           0.12, False),
    ("I10",    "Essential hypertension",                  0.10, True),
    ("E11.9",  "Type 2 diabetes mellitus",                0.08, True),
    ("E78.5",  "Hyperlipidemia",                          0.07, True),
    ("F32.9",  "Major depressive disorder",               0.06, True),
    ("J06.9",  "Acute upper respiratory infection",       0.06, False),
    ("J45.909","Unspecified asthma",                      0.05, True),
    ("J44.1",  "COPD with acute exacerbation",            0.04, True),
    ("I25.10", "Coronary artery disease",                 0.04, True),
    ("N18.3",  "Chronic kidney disease stage 3",          0.03, True),
    ("C50.919","Malignant neoplasm of breast",            0.02, True),
    ("M17.11", "Unilateral primary osteoarthritis, knee", 0.03, True),
    ("K21.0",  "GERD with esophagitis",                   0.04, False),
    ("F41.1",  "Generalized anxiety disorder",            0.05, True),
    ("Z12.31", "Encounter for colon cancer screening",    0.06, False),
]
DX_CODES  = [d[0] for d in DIAGNOSIS_CODES]
DX_WEIGHTS = np.array([d[2] for d in DIAGNOSIS_CODES], dtype=float)
DX_WEIGHTS /= DX_WEIGHTS.sum()
DX_CHRONIC = {d[0]: d[3] for d in DIAGNOSIS_CODES}

# CPT codes with typical billed / allowed ranges (in cents)
PROCEDURE_CODES = {
    "99213": ("Office visit – low complexity",      15000,  30000,  10000, 20000),
    "99214": ("Office visit – moderate complexity", 25000,  50000,  15000, 30000),
    "99215": ("Office visit – high complexity",     40000,  80000,  25000, 50000),
    "99232": ("Subsequent hospital care",           30000,  60000,  20000, 40000),
    "99283": ("ED visit – moderate severity",       60000, 150000,  40000, 90000),
    "27447": ("Total knee replacement",            800000,1800000, 500000,900000),
    "70553": ("MRI brain with contrast",            90000, 200000,  50000,100000),
    "93000": ("ECG",                                 5000,  15000,   3000,  8000),
    "80053": ("Comprehensive metabolic panel",       4000,  10000,   2500,  6000),
    "71046": ("Chest X-ray 2 views",                10000,  25000,   6000, 14000),
    "90837": ("Psychotherapy 60 min",               25000,  50000,  15000, 30000),
    "43239": ("Upper GI endoscopy w/ biopsy",      200000, 450000, 120000,250000),
    "47562": ("Laparoscopic cholecystectomy",       500000,1200000, 300000,700000),
    "36415": ("Routine venipuncture",                2000,   5000,   1000,  3000),
    "99395": ("Preventive visit 18-39 years",       25000,  60000,  15000, 35000),
}

# Pharmacy drug profiles: (name, therapeutic_class, specialty, billed_range_cents, plan_paid_range_cents)
PHARMACY_DRUGS = [
    ("Lisinopril 10mg",   "ACE Inhibitor",          False, (1000, 3000),   (500,  2000)),
    ("Metformin 500mg",   "Biguanide",               False, (800,  2500),   (400,  1500)),
    ("Atorvastatin 20mg", "Statin",                  False, (1500, 4000),   (800,  2500)),
    ("Sertraline 50mg",   "SSRI",                    False, (1200, 3500),   (600,  2000)),
    ("Albuterol inhaler", "Bronchodilator",           False, (3000, 7000),  (2000, 5000)),
    ("Omeprazole 20mg",   "PPI",                     False, (1000, 2500),   (500,  1500)),
    ("Metoprolol 25mg",   "Beta Blocker",             False, (900,  2200),   (400,  1200)),
    ("Humira 40mg",       "TNF Inhibitor",            True, (450000,600000),(400000,550000)),
    ("Keytruda 100mg",    "PD-1 Inhibitor",           True, (1800000,2200000),(1600000,2000000)),
    ("Ozempic 0.5mg",     "GLP-1 Agonist",            True, (80000,120000), (70000,100000)),
    ("Eliquis 5mg",       "Factor Xa Inhibitor",     False, (35000, 60000), (25000, 45000)),
    ("Dupixent 300mg",    "IL-4/IL-13 Inhibitor",    True, (550000,750000),(490000,680000)),
]

AGE_BUCKETS = ["18-29", "30-39", "40-49", "50-59", "60-64"]
AGE_WEIGHTS = [0.15, 0.22, 0.28, 0.25, 0.10]


def seeded_uuid() -> str:
    """UUID4 drawn from the seeded `random` module.

    uuid.uuid4() reads os.urandom(), which ignores STUDENT_SEED — so using it
    here would make member_ids differ on every run even at a fixed seed. Drawing
    the bits from `random` instead keeps the whole dataset reproducible.
    """
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def make_member(member_id: str) -> dict:
    age_bucket = random.choices(AGE_BUCKETS, weights=AGE_WEIGHTS)[0]
    # Chronic burden increases with age
    age_idx = AGE_BUCKETS.index(age_bucket)
    chronic_prob = 0.15 + age_idx * 0.12
    has_chronic = random.random() < chronic_prob
    return {
        "member_id": member_id,
        "age_bucket": age_bucket,
        "sex": random.choice(SEX),
        "region": random.choice(REGIONS),
        "plan_year_start": PLAN_YEAR_START.isoformat(),
        "plan_year_end": PLAN_YEAR_END.isoformat(),
        "_has_chronic": has_chronic,
    }


def claim_count_for_member(member: dict) -> int:
    base = 3 if not member["_has_chronic"] else 8
    return max(1, int(np.random.negative_binomial(n=2, p=0.25) + base))


def make_medical_claim(claim_id: str, member_id: str, member: dict) -> dict:
    # High-cost members skew toward institutional / inpatient procedure codes
    if member["_has_chronic"] and random.random() < 0.15:
        cpt = random.choice(["27447", "47562", "43239", "70553", "99283"])
    else:
        cpt = random.choices(list(PROCEDURE_CODES.keys()),
                             weights=[1]*len(PROCEDURE_CODES))[0]

    _, billed_lo, billed_hi, allowed_lo, allowed_hi = PROCEDURE_CODES[cpt]
    billed  = random.randint(billed_lo, billed_hi)
    allowed = random.randint(allowed_lo, min(allowed_hi, billed))

    dx = np.random.choice(DX_CODES, p=DX_WEIGHTS)

    deductible   = int(allowed * random.uniform(0.0, 0.20))
    copay        = random.choice([0, 2000, 3000, 4000, 5000])
    coinsurance  = int(max(0, allowed - deductible - copay) * random.uniform(0.0, 0.20))
    plan_paid    = max(0, allowed - deductible - copay - coinsurance)
    out_of_pocket = deductible + copay + coinsurance

    svc_date = random_date(PLAN_YEAR_START, PLAN_YEAR_END)

    return {
        "claim_id":          claim_id,
        "member_id":         member_id,
        "claim_type":        random.choices(["Professional", "Institutional"], weights=[0.7, 0.3])[0],
        "service_date":      svc_date.isoformat(),
        "adjudication_status": "approved" if random.random() > 0.08 else "denied",
        "diagnosis_code":    dx,
        "procedure_code":    cpt,
        "billed_cents":      billed,
        "allowed_cents":     allowed,
        "plan_paid_cents":   plan_paid,
        "deductible_cents":  deductible,
        "copay_cents":       copay,
        "coinsurance_cents": coinsurance,
        "out_of_pocket_cents": out_of_pocket,
        "is_pharmacy":       False,
        "drug_name":         "",
        "therapeutic_class": "",
        "is_specialty_drug": False,
    }


def make_pharmacy_claim(claim_id: str, member_id: str, member: dict) -> dict:
    # Chronic members more likely to be on specialty drugs
    specialty_weight = 0.25 if member["_has_chronic"] else 0.04
    drug_weights = [
        specialty_weight if d[2] else (1 - specialty_weight)
        for d in PHARMACY_DRUGS
    ]
    total = sum(drug_weights)
    drug_weights = [w / total for w in drug_weights]
    drug = random.choices(PHARMACY_DRUGS, weights=drug_weights)[0]

    billed_lo, billed_hi = drug[3]
    paid_lo,   paid_hi   = drug[4]
    billed    = random.randint(billed_lo, billed_hi)
    plan_paid = random.randint(paid_lo, min(paid_hi, billed))
    allowed   = plan_paid + random.randint(0, int(billed * 0.05))

    svc_date = random_date(PLAN_YEAR_START, PLAN_YEAR_END)

    return {
        "claim_id":          claim_id,
        "member_id":         member_id,
        "claim_type":        "Pharmacy",
        "service_date":      svc_date.isoformat(),
        "adjudication_status": "approved",
        "diagnosis_code":    "",
        "procedure_code":    "",
        "billed_cents":      billed,
        "allowed_cents":     allowed,
        "plan_paid_cents":   plan_paid,
        "deductible_cents":  0,
        "copay_cents":       random.choice([500, 1000, 2000, 3000]),
        "coinsurance_cents": 0,
        "out_of_pocket_cents": random.choice([500, 1000, 2000, 3000]),
        "is_pharmacy":       True,
        "drug_name":         drug[0],
        "therapeutic_class": drug[1],
        "is_specialty_drug": drug[2],
    }


def generate():
    # Seed inside generate() (not at import time) so the cohort is reproducible
    # even when this function is called more than once in a process, e.g. by tests.
    random.seed(STUDENT_SEED)
    np.random.seed(STUDENT_SEED)

    members_rows = []
    claims_rows  = []

    for _ in range(N_MEMBERS):
        member_id = seeded_uuid()
        member = make_member(member_id)
        members_rows.append(member)

        n_claims = claim_count_for_member(member)
        # ~30% pharmacy mix for chronic members, ~15% otherwise
        pharmacy_rate = 0.30 if member["_has_chronic"] else 0.15

        for _ in range(n_claims):
            claim_id = seeded_uuid()
            if random.random() < pharmacy_rate:
                claims_rows.append(make_pharmacy_claim(claim_id, member_id, member))
            else:
                claims_rows.append(make_medical_claim(claim_id, member_id, member))

    # Strip internal helper field before writing members CSV
    members_out = [{k: v for k, v in m.items() if not k.startswith("_")} for m in members_rows]

    with open(OUT_DIR / "members.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(members_out[0].keys()))
        writer.writeheader()
        writer.writerows(members_out)

    with open(OUT_DIR / "claims.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(claims_rows[0].keys()))
        writer.writeheader()
        writer.writerows(claims_rows)

    # Build member-level summary for ML
    members_df = pd.DataFrame(members_rows).drop(columns=["_has_chronic"])
    claims_df  = pd.DataFrame(claims_rows)

    approved = claims_df[claims_df["adjudication_status"] == "approved"]

    agg = approved.groupby("member_id").agg(
        total_claims        =("claim_id",          "count"),
        medical_claims      =("is_pharmacy",        lambda x: (~x).sum()),
        pharmacy_claims     =("is_pharmacy",        "sum"),
        specialty_drug_fills=("is_specialty_drug",  "sum"),
        total_billed_cents  =("billed_cents",        "sum"),
        total_allowed_cents =("allowed_cents",       "sum"),
        total_plan_paid_cents=("plan_paid_cents",    "sum"),
        total_oop_cents     =("out_of_pocket_cents", "sum"),
        unique_diagnosis_codes=("diagnosis_code",   lambda x: x[x != ""].nunique()),
        inpatient_claims    =("claim_type",         lambda x: (x == "Institutional").sum()),
        has_specialty_drug  =("is_specialty_drug",  lambda x: x.any()),
    ).reset_index()

    agg["is_high_cost"] = (agg["total_plan_paid_cents"] > HIGH_COST_THRESHOLD_CENTS).astype(int)

    summary = members_df.merge(agg, on="member_id", how="left").fillna(0)

    summary.to_csv(OUT_DIR / "member_summary.csv", index=False)

    total     = len(summary)
    high_cost = summary["is_high_cost"].sum()
    print(f"Generated {total} members and {len(claims_df)} claims (STUDENT_SEED={STUDENT_SEED}).")
    print(f"High-cost threshold: ${HIGH_COST_THRESHOLD_USD:,} in plan-paid claims")
    print(f"High-cost members: {high_cost} ({100 * high_cost / total:.1f}%)")
    print(f"Output written to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    generate()
