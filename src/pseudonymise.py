
"""Raw data exports -> pseudonymised working copies.
Reads  data_raw/*.xlsx      (real, gitignored)
Writes data_working/*.csv   (no direct identifiers, STILL gitignored)
Needs EXPRESS_SALT in .env  (gitignored, never committed).

Run from project root:  python -m src.pseudonymise
"""

import hashlib
import hmac
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw"
WORKING = ROOT / "data_working"

# id_col: the patient reference in THIS table (normalised to patient_key)
# drop:   direct identifiers removed entirely

# --- per-table config -------------------------------------------------
# id_col: the patient reference in THIS table (normalised to patient_key)
# drop:   direct identifiers removed entirely
TABLES = {
    "PatientExport.xlsx": {
        "id_col": "Pat_RefNo",
        "drop": ["Title", "FirstName", "MiddleName", "LastName", "Address1", "Address2"],
        "dob_col": "DOB",
        "postcode_col": "PostCode",
    },
    "AppointmentExport.xlsx": {
        "id_col": "PatntRefceNo",          # note: different spelling from the rest
        "drop": ["FirstName", "MiddleName", "LastName"],
        "dob_col": None,
        "postcode_col": None,
    },
    "Order Report.xlsx": {
        "id_col": "Pat_RefNo",
        "drop": ["DOB"],                    # DOB kept only in PatientExport (master)
        "dob_col": None,                    # drop, don't derive (redundant here)
        "postcode_col": None,
    },
    "PatientRX.xlsx": {
        "id_col": "Pat_RefNo",
        "drop": ["FirstName", "LastName"],
        "dob_col": None,
        "postcode_col": None,
    },
}

# ----------------------------------------------------------------------

def _salt() -> bytes:
    """Get the salt from the environment, or raise an error if not set."""
    load_dotenv()
    salt = os.getenv("EXPRESS_SALT")
    if not salt:
        raise ValueError("EXPRESS_SALT not set in .env")
    return salt.encode("utf-8")


def hash_id(value, salt: bytes) -> str:
    """Hash a value with the given salt using HMAC-SHA256."""
    return hmac.new(salt, str(value).strip().encode("utf-8"), hashlib.sha256).hexdigest()


def outward_code(pc):
    if pd.isna(pc):                        # missing postcode?
        return None
    return str(pc).strip().upper().split(" ")[0]

def pseudonymise(df: pd.DataFrame, cfg: dict, salt: bytes) -> pd.DataFrame:
    out = df.copy()

    # normalise whatever the id is called here -> single canonical key
    out["patient_key"] = out[cfg["id_col"]].map(lambda v: hash_id(v, salt))

    if cfg.get("dob_col") and cfg["dob_col"] in out:
        out["birth_year"] = pd.to_datetime(out[cfg["dob_col"]], errors="coerce").dt.year
    if cfg.get("postcode_col") and cfg["postcode_col"] in out:
        out["outward_code"] = out[cfg["postcode_col"]].map(outward_code)

    drop = set(cfg["drop"]) | {cfg["id_col"]}
    if cfg.get("dob_col"):
        drop.add(cfg["dob_col"])
    if cfg.get("postcode_col"):
        drop.add(cfg["postcode_col"])
    out = out.drop(columns=[c for c in drop if c in out])

    # put patient_key first for readability
    cols = ["patient_key"] + [c for c in out.columns if c != "patient_key"]
    return out[cols]


def main():
    salt = _salt()
    WORKING.mkdir(parents=True, exist_ok=True)

    for fname, cfg in TABLES.items(): #loop over the tables
        src = RAW / fname
        if not src.exists():
            print(f"SKIP  {fname} (not found)")
            continue
        df = pd.read_excel(src)
        clean = pseudonymise(df, cfg, salt) # deidentify it
        out_name = fname.replace(".xlsx", ".csv").replace(" ", "_")
        clean.to_csv(WORKING / out_name, index=False)
        print(f"{fname}: {len(df):>7} rows -> data_working/{out_name}  "
              f"({len(clean.columns)} cols)")


if __name__ == "__main__":
    main()




