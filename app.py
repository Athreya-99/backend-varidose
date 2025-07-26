from flask import Flask, request, send_file, abort, jsonify
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import json
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import io
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def sum_dosages(dosages):
    total = 0
    for i in dosages:
        try:
            total += float(i)
        except Exception:
            pass
    return total

@app.route('/api/medicines')
def medicines():
    patient_id = request.args.get('id')
    if not patient_id:
        abort(400, description="Missing patient id")
    history_path = os.path.join(BASE_DIR, 'history_new.json')
    if not os.path.exists(history_path):
        abort(500, description="history_new.json not found")
    with open(history_path) as f:
        history_data = json.load(f)
    history_df = pd.DataFrame(history_data)
    # Debug output
    print("Medicines Endpoint - Looking for patient_id:", repr(patient_id))
    print("Available patient IDs:", set(history_df['patient_id'].unique()))
    # Convert IDs to string type for safe comparison
    history_df['patient_id'] = history_df['patient_id'].astype(str)
    history_df = history_df[history_df["patient_id"] == str(patient_id)]
    print("Rows for patient:", len(history_df))
    if history_df.empty:
        return jsonify({"medicines": []})
    medicine_list = sorted(history_df["medicine_name"].unique())
    return jsonify({"medicines": medicine_list})

@app.route('/api/heatmap')
def heatmap():
    patient_id = request.args.get('id')
    if not patient_id:
        abort(400, description="Missing patient id")
    patients_path = os.path.join(BASE_DIR, 'patients_new.json')
    if not os.path.exists(patients_path):
        abort(500, description="patients_new.json not found")
    with open(patients_path) as f:
        patient_data = json.load(f)
    patient_df = pd.DataFrame(patient_data)
    # Debug printout
    print("Heatmap Endpoint - patient_id:", repr(patient_id))
    print("Available _id:", set(patient_df['_id'].astype(str).unique()))
    # IDs to string for comparison
    patient_df['_id'] = patient_df['_id'].astype(str)
    patient_row = patient_df[patient_df["_id"] == str(patient_id)]
    if patient_row.empty:
        abort(404, description="Patient not found")
    patient_name = patient_row["name"].iloc[0]

    history_path = os.path.join(BASE_DIR, 'history_new.json')
    if not os.path.exists(history_path):
        abort(500, description="history_new.json not found")
    with open(history_path) as f:
        history_data = json.load(f)
    history_df = pd.DataFrame(history_data)
    history_df['patient_id'] = history_df['patient_id'].astype(str)
    history_df = history_df[history_df["patient_id"] == str(patient_id)]
    print("History records for heatmap:", len(history_df))
    if history_df.empty:
        abort(404, description="No history for patient")

    alarm_path = os.path.join(BASE_DIR, 'alarm_new.json')
    if not os.path.exists(alarm_path):
        abort(500, description="alarm_new.json not found")
    with open(alarm_path) as f:
        alarm_data = json.load(f)
    alarm_df = pd.DataFrame(alarm_data)
    alarm_df['patient_id'] = alarm_df['patient_id'].astype(str)
    alarm_df = alarm_df[alarm_df["patient_id"] == str(patient_id)]

    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"]).dt.tz_localize(None)
    history_df["date"] = history_df["timestamp"].dt.date

    frequency_df = history_df.groupby(['date', 'medicine_name']).size().reset_index(name='frequency')
    pivot = frequency_df.pivot(index="date", columns="medicine_name", values="frequency")
    pivot = pivot.fillna(0)
    if not pivot.empty:
        pivot.index = pd.to_datetime(pivot.index).strftime("%d-%m-%Y")
    else:
        return abort(404, description="No frequency data found for patient")

    if not alarm_df.empty and "dosage" in alarm_df.columns and "schedule_name" in alarm_df.columns:
        alarm_df["total_scheduled_dosage"] = alarm_df["dosage"].apply(sum_dosages)
        scheduled_dosage_dict = dict(zip(alarm_df["schedule_name"], alarm_df["total_scheduled_dosage"]))
    else:
        scheduled_dosage_dict = {}
    color_matrix = pivot.copy()
    for col in color_matrix.columns:
        sched = scheduled_dosage_dict.get(col, np.inf)
        color_matrix[col] = np.where((pivot[col] == 0) | (pivot[col] > sched), 0, 1)
    color_matrix.index = pivot.index
    custom_cmap = ListedColormap(["#dc4c4c", "#62d666"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        color_matrix,
        annot=pivot,
        fmt=".0f",
        cmap=custom_cmap,
        cbar=False,
        linewidths=1,
        linecolor='black',
        annot_kws={"size": 16}
    )
    plt.title(f"Doses(daily) of {patient_name}", fontsize=16)
    plt.xlabel("medicines taken", fontsize=12)
    plt.ylabel("date", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Correct Dosage'),
        Patch(facecolor='red', label='Missed Dosage/ Overdosage')
    ]
    plt.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.02, 0.5),   # right beside plot, vertically centered
    loc='center left',
    borderaxespad=0.
    prop={'size': 14}
    )
    plt.subplots_adjust(right=0.80) 
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')









@app.route('/api/medicine-heatmap')
def get_medicine_heatmap():
    patient_id = request.args.get('patient')
    medicine_name = request.args.get('medicine')
    if not patient_id or not medicine_name:
        return abort(400, description="Missing patient or medicine info")
    return single_medicine_heatmap(str(patient_id), medicine_name)

def single_medicine_heatmap(patient_id, medicine_name, history_filename='history_new.json', alarm_filename='alarm_new.json'):
    TIME_TOLERANCE = timedelta(minutes=30)
    EXPECTED_INTERVAL = timedelta(hours=6)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    history_path = os.path.join(BASE_DIR, history_filename)
    alarm_path = os.path.join(BASE_DIR, alarm_filename)
    if not os.path.exists(history_path):
        abort(500, description=f"{history_filename} not found")
    with open(history_path) as f:
        history_data = json.load(f)
    history_df = pd.DataFrame(history_data)
    history_df['patient_id'] = history_df['patient_id'].astype(str)
    history_df = history_df[(history_df["patient_id"] == str(patient_id)) &
                            (history_df["medicine_name"] == medicine_name)]
    if not os.path.exists(alarm_path):
        abort(500, description=f"{alarm_filename} not found")
    with open(alarm_path) as f:
        alarm_data = json.load(f)
    alarm_df = pd.DataFrame(alarm_data)
    alarm_df['patient_id'] = alarm_df['patient_id'].astype(str)
    alarm_df = alarm_df[(alarm_df["patient_id"] == str(patient_id)) &
                        (alarm_df["schedule_name"] == medicine_name)]
    if alarm_df.empty:
        return abort(404, description=f"No schedule found for medicine: {medicine_name}")

    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"]).dt.tz_localize(None)
    alarm_df["createdAt"] = pd.to_datetime(alarm_df["createdAt"]).dt.tz_localize(None)

    start_date = alarm_df["createdAt"].min().date()
    end_date = start_date + timedelta(days=6)
    date_range = pd.date_range(start=start_date, end=end_date)

    schedule_entries = []
    for _, row in alarm_df.iterrows():
        medicine_id = row["medicines"][0] if row.get("medicines") and len(row["medicines"]) > 0 else None
        for single_date in date_range:
            weekday = str(single_date.weekday())
            if not row.get("days_of_week") or weekday in row["days_of_week"]:
                for t_str in row['schedule_time']:
                    sched_time = datetime.combine(
                        single_date,
                        datetime.strptime(t_str, "%H:%M").time()
                    )
                    schedule_entries.append({
                        "date": single_date.date(),
                        "time": t_str,
                        "scheduled_datetime": sched_time,
                        "medicine_id": medicine_id,
                        "medicine": medicine_name
                    })

    schedule_df = pd.DataFrame(schedule_entries)
    if schedule_df.empty:
        return abort(404, description="No scheduled doses found for this medicine and week.")

    def classify_dose(row):
        taken = history_df[
            (history_df["medicine_name"] == row["medicine"]) &
            (abs(history_df["timestamp"] - row["scheduled_datetime"]) <= EXPECTED_INTERVAL)
        ]
        if taken.empty:
            return "Missed"
        taken = taken.copy()
        taken["delta"] = abs(taken["timestamp"] - row["scheduled_datetime"])
        closest = taken.sort_values("delta").iloc[0]
        delta = closest["delta"]
        return "On Time" if delta <= TIME_TOLERANCE else "Late"

    schedule_df["status"] = schedule_df.apply(classify_dose, axis=1)

    all_dates = pd.date_range(start=start_date, end=end_date)
    all_times = sorted(alarm_df.iloc[0]["schedule_time"])
    full_grid = []
    for date in all_dates:
        for time in all_times:
            entry = schedule_df[
                (schedule_df["date"] == date.date()) &
                (schedule_df["time"] == time)
            ]
            status = entry["status"].iloc[0] if not entry.empty else "Not Scheduled"
            full_grid.append({
                "date": date.date(),
                "time": time,
                "status": status
            })
    full_df = pd.DataFrame(full_grid)

    pivot = full_df.pivot(index="date", columns="time", values="status")
    pivot.index = pd.to_datetime(pivot.index).strftime("%d-%m-%Y")

    
    status_map = {
        "Missed": 0,
        "Late": 1,
        "On Time": 2
    }
    # NaN (blank cells) for those not scheduled
    filtered_pivot = pivot.replace("Not Scheduled", np.nan)
    heatmap_data = filtered_pivot.replace(status_map).infer_objects(copy=False)
    cmap = ListedColormap(['red', 'orange', 'green'])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        annot=filtered_pivot,  # Shows actual status text in each box
        fmt="",
        cmap=cmap,
        cbar=False,
        linewidths=1,
        linecolor='black'
    )
    plt.title(f"Medicine: {medicine_name}")
    plt.xlabel("Scheduled Time")
    plt.ylabel("Date")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='On Time'),
        Patch(facecolor='orange', label='Late'),
        Patch(facecolor='red', label='Missed')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
