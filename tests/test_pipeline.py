"""
Tests for pipeline integrity — schema, config, and data contract.
"""
import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ROOT, MODEL_PATH, METADATA_PATH,
    GRADE_OPTIONS, SUBGRADE_OPTIONS, TERM_OPTIONS,
    EMP_LENGTH_OPTIONS, HOME_OWNERSHIP_OPTIONS,
    VERIFICATION_OPTIONS, PURPOSE_OPTIONS, STATE_OPTIONS,
)


class TestConfig:
    def test_root_is_directory(self):
        assert ROOT.is_dir()

    def test_model_path_is_pkl(self):
        assert MODEL_PATH.suffix == ".pkl"

    def test_metadata_path_is_json(self):
        assert METADATA_PATH.suffix == ".json"

    def test_grade_options_correct(self):
        assert GRADE_OPTIONS == ["A", "B", "C", "D", "E", "F", "G"]

    def test_subgrade_options_count(self):
        # 7 grades × 5 subgrades = 35
        assert len(SUBGRADE_OPTIONS) == 35

    def test_subgrade_starts_with_grade(self):
        for sg in SUBGRADE_OPTIONS:
            assert sg[0] in GRADE_OPTIONS

    def test_term_options(self):
        assert len(TERM_OPTIONS) == 2
        assert all("months" in t for t in TERM_OPTIONS)

    def test_state_options_count(self):
        assert len(STATE_OPTIONS) == 50


class TestMetadataSchema:
    @pytest.fixture(scope="class")
    def metadata(self):
        with open(METADATA_PATH) as f:
            return json.load(f)

    def test_required_keys_present(self, metadata):
        required = {"best_model", "roc_auc", "threshold", "features",
                    "numerical", "categorical", "metrics"}
        assert required.issubset(metadata.keys())

    def test_metrics_keys_present(self, metadata):
        required = {"accuracy", "precision", "recall", "f1", "roc_auc",
                    "charged_off_recall", "charged_off_precision"}
        assert required.issubset(metadata["metrics"].keys())

    def test_all_metrics_between_0_and_1(self, metadata):
        for key, val in metadata["metrics"].items():
            assert 0.0 <= val <= 1.0, f"Metric '{key}' = {val} is out of range"

    def test_no_feature_in_both_lists(self, metadata):
        overlap = set(metadata["numerical"]) & set(metadata["categorical"])
        assert overlap == set(), f"Features in both lists: {overlap}"

    def test_best_model_is_string(self, metadata):
        assert isinstance(metadata["best_model"], str)
        assert len(metadata["best_model"]) > 0


class TestRequirements:
    def test_requirements_file_exists(self):
        req = ROOT / "requirements.txt"
        assert req.exists()

    def test_all_packages_pinned(self):
        req = ROOT / "requirements.txt"
        lines = [l.strip() for l in req.read_text().splitlines()
                 if l.strip() and not l.startswith("#")]
        unpinned = [l for l in lines if "==" not in l]
        assert unpinned == [], f"Unpinned packages: {unpinned}"

    def test_xgboost_in_requirements(self):
        req = (ROOT / "requirements.txt").read_text()
        assert "xgboost" in req

    def test_scikit_learn_in_requirements(self):
        req = (ROOT / "requirements.txt").read_text()
        assert "scikit-learn" in req
