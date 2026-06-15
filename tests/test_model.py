"""
Tests for model loading and inference — verifies the pipeline works end-to-end.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODEL_PATH, METADATA_PATH


@pytest.fixture(scope="module")
def pipeline():
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metadata():
    import json
    with open(METADATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sample_input():
    return pd.DataFrame([{
        "loan_amnt":            10_000,
        "term":                 " 36 months",
        "int_rate":             12.5,
        "installment":          332.14,
        "grade":                "B",
        "sub_grade":            "B3",
        "emp_length":           "5 years",
        "home_ownership":       "RENT",
        "annual_inc":           65_000,
        "verification_status":  "Verified",
        "purpose":              "debt_consolidation",
        "addr_state":           "CA",
        "dti":                  18.5,
        "delinq_2yrs":          0,
        "inq_last_6mths":       1,
        "open_acc":             8,
        "pub_rec":              0,
        "revol_bal":            12_000,
        "revol_util":           45.0,
        "total_acc":            20,
        "mort_acc":             1,
        "pub_rec_bankruptcies": 0,
    }])


class TestModelLoads:
    def test_model_file_exists(self):
        assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"

    def test_metadata_file_exists(self):
        assert METADATA_PATH.exists(), f"Metadata not found at {METADATA_PATH}"

    def test_pipeline_loads(self, pipeline):
        assert pipeline is not None

    def test_pipeline_has_preprocessor(self, pipeline):
        assert "preprocessor" in pipeline.named_steps

    def test_pipeline_has_model(self, pipeline):
        assert "model" in pipeline.named_steps


class TestInference:
    def test_predict_proba_returns_valid_probability(self, pipeline, sample_input):
        prob = pipeline.predict_proba(sample_input)[0][1]
        assert 0.0 <= prob <= 1.0

    def test_predict_returns_binary(self, pipeline, sample_input):
        pred = pipeline.predict(sample_input)[0]
        assert pred in [0, 1]

    def test_high_risk_input_lower_probability(self, pipeline):
        risky = pd.DataFrame([{
            "loan_amnt":            35_000,
            "term":                 " 60 months",
            "int_rate":             28.0,
            "installment":          900.0,
            "grade":                "G",
            "sub_grade":            "G5",
            "emp_length":           "< 1 year",
            "home_ownership":       "RENT",
            "annual_inc":           20_000,
            "verification_status":  "Not Verified",
            "purpose":              "small_business",
            "addr_state":           "NV",
            "dti":                  45.0,
            "delinq_2yrs":          5,
            "inq_last_6mths":       6,
            "open_acc":             15,
            "pub_rec":              2,
            "revol_bal":            25_000,
            "revol_util":           95.0,
            "total_acc":            10,
            "mort_acc":             0,
            "pub_rec_bankruptcies": 1,
        }])
        safe = pd.DataFrame([{
            "loan_amnt":            5_000,
            "term":                 " 36 months",
            "int_rate":             6.5,
            "installment":          152.0,
            "grade":                "A",
            "sub_grade":            "A1",
            "emp_length":           "10+ years",
            "home_ownership":       "MORTGAGE",
            "annual_inc":           150_000,
            "verification_status":  "Source Verified",
            "purpose":              "car",
            "addr_state":           "CA",
            "dti":                  5.0,
            "delinq_2yrs":          0,
            "inq_last_6mths":       0,
            "open_acc":             5,
            "pub_rec":              0,
            "revol_bal":            2_000,
            "revol_util":           5.0,
            "total_acc":            25,
            "mort_acc":             2,
            "pub_rec_bankruptcies": 0,
        }])
        risky_prob = pipeline.predict_proba(risky)[0][1]
        safe_prob  = pipeline.predict_proba(safe)[0][1]
        assert safe_prob > risky_prob, (
            f"Safe loan ({safe_prob:.3f}) should score higher than risky ({risky_prob:.3f})"
        )

    def test_unknown_state_does_not_crash(self, pipeline, sample_input):
        df = sample_input.copy()
        df["addr_state"] = "ZZ"   # unknown state not in training data
        prob = pipeline.predict_proba(df)[0][1]
        assert 0.0 <= prob <= 1.0


class TestMetadata:
    def test_threshold_in_valid_range(self, metadata):
        assert 0.0 < metadata["threshold"] < 1.0

    def test_roc_auc_reasonable(self, metadata):
        assert metadata["roc_auc"] > 0.60, "AUC below 0.60 suggests model issues"

    def test_features_list_not_empty(self, metadata):
        assert len(metadata["features"]) > 0

    def test_numerical_and_categorical_cover_all_features(self, metadata):
        all_f = set(metadata["numerical"]) | set(metadata["categorical"])
        assert all_f == set(metadata["features"])
