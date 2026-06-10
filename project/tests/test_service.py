from __future__ import annotations

from src.service.app import create_app
from src.service.schemas import EXAMPLE_REQUEST, validate_customer
from src.utils.config import get_settings


def _client():
    return create_app().test_client()


def _model_exists() -> bool:
    return get_settings().model_path.exists()


def test_index_lists_endpoints() -> None:
    resp = _client().get("/")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "endpoints" in body and "predict_example_request" in body


def test_health_endpoint_responds() -> None:
    resp = _client().get("/health")
    assert resp.status_code in (200, 503)
    assert "model_loaded" in resp.get_json()


def test_metrics_endpoint_responds() -> None:
    resp = _client().get("/metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "total_requests" in body and "predictions" in body


def test_unknown_endpoint_returns_404() -> None:
    resp = _client().get("/no-such-endpoint")
    assert resp.status_code == 404


def test_validate_customer_accepts_valid_payload() -> None:
    cleaned = validate_customer(EXAMPLE_REQUEST)
    assert cleaned["contract"] == "Month-to-month"


def test_validate_customer_rejects_bad_category() -> None:
    bad = dict(EXAMPLE_REQUEST, contract="Lifetime")
    try:
        validate_customer(bad)
        assert False, "Ожидалась ValidationError"
    except Exception as exc:
        assert "contract" in str(exc)


def test_predict_returns_probability_when_model_ready() -> None:
    if not _model_exists():
        return
    resp = _client().post("/predict", json=EXAMPLE_REQUEST)
    assert resp.status_code == 200
    body = resp.get_json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_category"] in ("low", "medium", "high")
    assert len(body["key_factors"]) > 0


def test_predict_rejects_invalid_payload() -> None:
    if not _model_exists():
        return
    resp = _client().post("/predict", json={"tenure": 5})
    assert resp.status_code == 422
    assert "details" in resp.get_json()


def test_predict_high_vs_low_risk_ordering() -> None:
    if not _model_exists():
        return
    client = _client()
    risky = client.post("/predict", json=EXAMPLE_REQUEST).get_json()
    loyal_payload = dict(
        EXAMPLE_REQUEST,
        tenure=65, contract="Two year", monthly_charges=42.0,
        total_charges=2730.0, num_support_calls=0,
        internet_service="DSL", tech_support="Yes",
        payment_method="Credit card",
    )
    loyal = client.post("/predict", json=loyal_payload).get_json()
    assert risky["churn_probability"] > loyal["churn_probability"]
