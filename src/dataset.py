from pathlib import Path
import pandas as pd




def load_tables(working_dir: Path) -> tuple:
    """read the csv's, cast dates to datetime and 
    returns the four clean tables (pat, app, rx, orders)"""
    pat = pd.read_csv(working_dir / "PatientExport.csv")
    app = pd.read_csv(working_dir / "AppointmentExport.csv")
    rx  = pd.read_csv(working_dir / "PatientRX.csv" , low_memory=False)
    orders = pd.read_csv(working_dir / "Order_Report.csv")


    app["AppointDate"]  = pd.to_datetime(app["AppointDate"], errors="coerce")
    orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], errors="coerce")
    rx["ReDate"]        = pd.to_datetime(rx["ReDate"], errors="coerce")


    return pat, app, rx, orders


def build_model_inputs(pat, app, rx, orders) -> tuple:
    #completed
    completed = app[app["Status"] == "Completed"].copy()
    
    #interval
    latest_rx = rx.sort_values("ReDate").drop_duplicates("patient_key", keep="last")
    interval = latest_rx.set_index("patient_key")["ReaMonths"]

    #data_end
    data_end = completed["AppointDate"].max()

    return completed, interval, data_end