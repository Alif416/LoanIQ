from pathlib import Path

ROOT          = Path(__file__).parent
MODEL_PATH    = ROOT / "model" / "loan_model_pipeline.pkl"
METADATA_PATH = ROOT / "model" / "model_metadata.json"
DATA_PATH        = ROOT / "data" / "loan.csv"
SAMPLE_DATA_PATH = ROOT / "data" / "loan_sample.csv"

ANALYTICS_COLS = [
    "loan_status", "issue_d", "grade", "loan_amnt", "int_rate",
    "purpose", "home_ownership", "dti", "annual_inc", "addr_state",
    "term", "emp_length",
]


GRADE_OPTIONS    = ["A", "B", "C", "D", "E", "F", "G"]
SUBGRADE_OPTIONS = [f"{g}{n}" for g in GRADE_OPTIONS for n in range(1, 6)]

TERM_OPTIONS             = [" 36 months", " 60 months"]
EMP_LENGTH_OPTIONS       = ["< 1 year", "1 year", "2 years", "3 years", "4 years",
                             "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"]
HOME_OWNERSHIP_OPTIONS   = ["RENT", "MORTGAGE", "OWN", "OTHER"]
VERIFICATION_OPTIONS     = ["Not Verified", "Verified", "Source Verified"]
PURPOSE_OPTIONS          = ["debt_consolidation", "credit_card", "home_improvement",
                             "other", "major_purchase", "small_business", "car",
                             "medical", "moving", "vacation", "house",
                             "wedding", "renewable_energy", "educational"]
STATE_OPTIONS            = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
    "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
    "TX","UT","VT","VA","WA","WV","WI","WY"
]
