"""Generate a fake optical practice matching the VisionPlus export shape,
so the pipeline can be run end-to-end without real patient data.

Deliberately simple distributions: synthetic data needs realistic STRUCTURE
and a genuine feature/outcome relationship, not realistic statistics.

"""


from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data_working" / "synthetic"
TODAY = pd.Timestamp("2026-07-25")
N_PATIENTS = 2000


def make_patients(n, rng):
    """Patient master table: keys, birth year, sex."""
    return pd.DataFrame({
        "patient_key": [f"SYN{i:06d}" for i in range(n)],
        "birth_year": rng.choice(range(1940, 2006), size=n),
        "Sex": rng.choice(["M", "F"], size=n),
    })


def make_traits(patients, rng):
    """Hidden per-patient traits.

    `loyal` is the latent cause that makes visit history and spend predict
    returning. Without a shared cause, features carry no signal and the
    model's AUC would sit at 0.5.
    """
    n = len(patients)
    loyal = rng.random(n) < 0.5
    return pd.DataFrame({
        "loyal": loyal,
        "n_visits": np.where(loyal, rng.choice([3, 4, 5, 6], size=n),
                                    rng.choice([1, 2, 3], size=n)),
        "came_back": rng.random(n) < np.where(loyal, 0.75, 0.30),
        "buy_chance": np.where(loyal, 0.6, 0.25),
    }, index=patients["patient_key"])


def make_prescriptions(patients, rng):
    """Recall intervals; older patients get shorter ones, as in the real data."""
    age = 2026 - patients["birth_year"]
    n = len(patients)
    rea = np.where(age > 60, rng.choice([6, 12], size=n),
                             rng.choice([12, 24], size=n))
    return pd.DataFrame({
        "patient_key": patients["patient_key"],
        "ReDate": TODAY - pd.Timedelta(days=400),
        "ReaMonths": rea,
    })


def make_appointments(patients, traits, prescriptions, rng):
    """Visit history, walked backwards from each patient's last visit.

    Returners' last visit sits inside their recall window; non-returners' is
    past interval + 3 months grace, which is what makes the churn label fire.
    """
    rx = prescriptions.set_index("patient_key")["ReaMonths"]
    rows = []

    for key in patients["patient_key"]:
        interval = rx[key]
        n_visits = traits.loc[key, "n_visits"]

        if traits.loc[key, "came_back"]:
            months_ago = rng.uniform(0, interval)                    # inside window
        else:
            months_ago = rng.uniform(interval + 4, interval + 20)    # lapsed

        date = TODAY - pd.Timedelta(days=months_ago * 30)
        for _ in range(n_visits):
            rows.append({"patient_key": key, "AppointDate": date,
                         "Status": "Completed"})
            date -= pd.Timedelta(days=interval * rng.uniform(0.8, 1.3) * 30)

        # occasional no-show, so the loader's Status filter is exercised
        if rng.random() < 0.25:
            rows.append({"patient_key": key,
                         "AppointDate": TODAY - pd.Timedelta(days=rng.uniform(0, 1500)),
                         "Status": "DNA"})

    return pd.DataFrame(rows)


def make_orders(appointments, traits, rng):
    """Dispenses attached to completed visits.

    Patients who never buy are left with no rows at all — the zero-dispense
    case value.py's shrinkage exists to handle.
    """
    completed = appointments[appointments["Status"] == "Completed"]
    rows = []
    for _, appt in completed.iterrows():
        if rng.random() < traits.loc[appt["patient_key"], "buy_chance"]:
            rows.append({"patient_key": appt["patient_key"],
                         "OrderDate": appt["AppointDate"],
                         "Amount": float(rng.choice([80, 120, 150, 180, 250, 400]))})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(42)      # seeded → identical fake practice every run

    patients = make_patients(N_PATIENTS, rng)
    traits = make_traits(patients, rng)
    prescriptions = make_prescriptions(patients, rng)
    appointments = make_appointments(patients, traits, prescriptions, rng)
    orders = make_orders(appointments, traits, rng)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patients.to_csv(OUT_DIR / "PatientExport.csv", index=False)
    appointments.to_csv(OUT_DIR / "AppointmentExport.csv", index=False)
    prescriptions.to_csv(OUT_DIR / "PatientRX.csv", index=False)
    orders.to_csv(OUT_DIR / "Order_Report.csv", index=False)

    print(f"{len(patients)} patients, {len(appointments)} appointments, "
          f"{len(orders)} orders → {OUT_DIR}")


if __name__ == "__main__":
    main()