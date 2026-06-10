from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENT = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
YES_NO = ["Yes", "No"]

# веса логит-модели оттока
_W = {
    "intercept": -1.15,
    "tenure": -0.045,
    "contract": {"Month-to-month": 1.05, "One year": -0.45, "Two year": -1.30},
    "monthly_charges": 0.020,
    "internet": {"Fiber optic": 0.75, "DSL": 0.0, "No": -0.65},
    "tech_support": {"Yes": -0.55, "No": 0.30, "No internet service": 0.0},
    "online_security": {"Yes": -0.45, "No": 0.25, "No internet service": 0.0},
    "payment": {
        "Electronic check": 0.70,
        "Mailed check": 0.05,
        "Bank transfer": -0.20,
        "Credit card": -0.25,
    },
    "paperless": {"Yes": 0.25, "No": -0.10},
    "senior": 0.35,
    "partner": {"Yes": -0.22, "No": 0.10},
    "dependents": {"Yes": -0.28, "No": 0.08},
    "support_calls": 0.23,
    "streaming_tv": {"Yes": 0.12, "No": 0.0, "No internet service": 0.0},
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(
    n_customers: int = 7000,
    random_state: int = 42,
    target_churn_rate: float = 0.26,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n = n_customers

    senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = rng.choice(YES_NO, size=n, p=[0.48, 0.52])
    dependents = rng.choice(YES_NO, size=n, p=[0.30, 0.70])

    tenure = np.clip(
        np.where(
            rng.random(n) < 0.45,
            rng.exponential(scale=12, size=n),
            rng.normal(loc=48, scale=18, size=n),
        ),
        0, 72,
    ).round().astype(int)

    contract = np.where(
        tenure < 12,
        rng.choice(CONTRACTS, size=n, p=[0.75, 0.18, 0.07]),
        rng.choice(CONTRACTS, size=n, p=[0.35, 0.33, 0.32]),
    )

    internet = rng.choice(INTERNET, size=n, p=[0.34, 0.44, 0.22])
    has_internet = internet != "No"

    def _service_col(prob_yes: float) -> np.ndarray:
        col = np.where(
            rng.random(n) < prob_yes, "Yes", "No"
        )
        return np.where(has_internet, col, "No internet service")

    online_security = _service_col(0.38)
    tech_support = _service_col(0.37)
    streaming_tv = _service_col(0.50)

    base_charge = np.where(
        internet == "Fiber optic", 75.0,
        np.where(internet == "DSL", 50.0, 20.0),
    )
    monthly_charges = np.clip(
        base_charge
        + (streaming_tv == "Yes") * 12.0
        + (online_security == "Yes") * 6.0
        + (tech_support == "Yes") * 6.0
        + rng.normal(0, 8, size=n),
        18.0, 120.0,
    ).round(2)

    total_charges = (
        tenure * monthly_charges * rng.normal(1.0, 0.05, size=n)
    ).round(2)
    total_charges = np.clip(total_charges, 0, None)

    payment_method = rng.choice(PAYMENT, size=n, p=[0.34, 0.23, 0.22, 0.21])
    paperless_billing = rng.choice(YES_NO, size=n, p=[0.59, 0.41])

    num_support_calls = rng.poisson(lam=0.9, size=n)
    num_support_calls = np.clip(num_support_calls, 0, 9)

    monthly_centered = monthly_charges - 65.0
    logit = (
        _W["intercept"]
        + _W["tenure"] * tenure
        + np.vectorize(_W["contract"].get)(contract)
        + _W["monthly_charges"] * monthly_centered
        + np.vectorize(_W["internet"].get)(internet)
        + np.vectorize(_W["tech_support"].get)(tech_support)
        + np.vectorize(_W["online_security"].get)(online_security)
        + np.vectorize(_W["payment"].get)(payment_method)
        + np.vectorize(_W["paperless"].get)(paperless_billing)
        + _W["senior"] * senior
        + np.vectorize(_W["partner"].get)(partner)
        + np.vectorize(_W["dependents"].get)(dependents)
        + _W["support_calls"] * num_support_calls
        + np.vectorize(_W["streaming_tv"].get)(streaming_tv)
    )

    is_month = contract == "Month-to-month"
    is_fiber = internet == "Fiber optic"
    logit += 0.00075 * (monthly_charges - 58.0) ** 2
    logit += (
        (tenure < 3) * 0.95
        + (num_support_calls >= 3) * 1.35
        + (is_month & (monthly_centered > 20)) * 1.10
        + (is_fiber & (tech_support == "No")) * 0.95
        + ((senior == 1) & (payment_method == "Electronic check")) * 0.75
        + (is_fiber & (tenure < 12)) * 0.65
        - ((tenure > 36) & (contract == "Two year")) * 0.70
    )

    logit = logit + rng.normal(0, 0.55, size=n)

    # подгоняем свободный член под нужную долю оттока
    for _ in range(40):
        rate = _sigmoid(logit).mean()
        logit += (np.log(target_churn_rate) - np.log(rate)) * 0.5
        if abs(rate - target_churn_rate) < 0.003:
            break

    churn_prob = _sigmoid(logit)
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "contract": contract,
            "internet_service": internet,
            "online_security": online_security,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "num_support_calls": num_support_calls,
            "churn": churn,
        }
    )

    # у новых клиентов ещё нет total_charges
    df.loc[df["tenure"] == 0, "total_charges"] = np.nan
    miss_idx = rng.choice(n, size=max(1, int(0.005 * n)), replace=False)
    df.loc[miss_idx, "total_charges"] = np.nan

    return df


if __name__ == "__main__":
    sample = generate_dataset(2000)
    print(sample.head())
    print(f"\nРазмер: {sample.shape}")
    print(f"Доля оттока: {sample['churn'].mean():.3f}")
    print(f"Пропусков в total_charges: {sample['total_charges'].isna().sum()}")
