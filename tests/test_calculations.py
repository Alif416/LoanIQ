"""
Tests for pure calculation functions — no model or data loading required.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app import emi, risk_label, validate_inputs


class TestEmi:
    def test_standard_loan(self):
        # $10,000 at 12% for 36 months — known answer $332.14
        result = emi(10_000, 12.0, 36)
        assert abs(result - 332.14) < 0.10

    def test_zero_interest(self):
        # Zero interest → simple division
        result = emi(12_000, 0.0, 12)
        assert result == pytest.approx(1000.0)

    def test_zero_principal(self):
        assert emi(0, 12.0, 36) == 0.0

    def test_zero_months(self):
        assert emi(10_000, 12.0, 0) == 0.0

    def test_total_repayment_exceeds_principal(self):
        # Total paid must always be >= principal
        monthly = emi(5_000, 10.0, 24)
        assert monthly * 24 >= 5_000

    def test_longer_term_lower_payment(self):
        short = emi(10_000, 12.0, 36)
        long_ = emi(10_000, 12.0, 60)
        assert long_ < short


class TestRiskLabel:
    def test_low_risk(self):
        label, color, badge = risk_label(0.80)
        assert label == "Low Risk"
        assert badge == "badge-green"

    def test_medium_risk(self):
        label, color, badge = risk_label(0.60)
        assert label == "Medium Risk"
        assert badge == "badge-yellow"

    def test_high_risk(self):
        label, color, badge = risk_label(0.30)
        assert label == "High Risk"
        assert badge == "badge-red"

    def test_boundary_low(self):
        label, _, _ = risk_label(0.75)
        assert label == "Low Risk"

    def test_boundary_medium(self):
        label, _, _ = risk_label(0.50)
        assert label == "Medium Risk"


class TestValidateInputs:
    def test_valid_inputs_no_errors(self):
        errors, warnings = validate_inputs(10_000, 65_000, 18.0, 45.0, 0)
        assert errors == []

    def test_zero_income_raises_error(self):
        errors, _ = validate_inputs(10_000, 0, 18.0, 45.0, 0)
        assert len(errors) == 1
        assert "income" in errors[0].lower()

    def test_zero_loan_raises_error(self):
        errors, _ = validate_inputs(0, 65_000, 18.0, 45.0, 0)
        assert len(errors) == 1
        assert "loan" in errors[0].lower()

    def test_loan_exceeds_income_warns(self):
        _, warnings = validate_inputs(80_000, 50_000, 18.0, 45.0, 0)
        assert any("income" in w.lower() for w in warnings)

    def test_high_dti_warns(self):
        _, warnings = validate_inputs(10_000, 65_000, 55.0, 45.0, 0)
        assert any("dti" in w.lower() for w in warnings)

    def test_high_utilization_warns(self):
        _, warnings = validate_inputs(10_000, 65_000, 18.0, 95.0, 0)
        assert any("utilization" in w.lower() for w in warnings)

    def test_bankruptcy_warns(self):
        _, warnings = validate_inputs(10_000, 65_000, 18.0, 45.0, 1)
        assert any("bankrupt" in w.lower() for w in warnings)
