"""Interactive terminal prediction using the saved model."""
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/best_student_model.joblib")
if not MODEL_PATH.exists():
    print("Model not found. First run: python train_model.py")
    raise SystemExit(1)

model = joblib.load(MODEL_PATH)

def ask_number(prompt, low, high, default=None):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"Please enter a number between {low} and {high}.")

def ask_yes_no(prompt):
    while True:
        value = input(prompt + " (yes/no): ").strip().lower()
        if value in ("yes", "no"):
            return value
        print("Please type yes or no.")

print("\n=== STUDENT PERFORMANCE PREDICTOR ===")
print("Note: This demo uses the features available in the public UCI dataset.")
print("Study time is a weekly category (1 to 4), not daily hours.\n")

student = {
    "age": ask_number("Student age (15-22): ", 15, 22),
    "studytime": ask_number("Weekly study-time category (1-4): ", 1, 4),
    "failures": ask_number("Past class failures (0-4): ", 0, 4),
    "absences": ask_number("Number of absences (0-93): ", 0, 93),
    "G1": ask_number("Previous period marks (0-20): ", 0, 20),
    "G2": ask_number("Second period marks (0-20): ", 0, 20),
    "schoolsup": ask_yes_no("Does the student receive school support?"),
    "famsup": ask_yes_no("Does the student receive family support?"),
    "paid": ask_yes_no("Does the student attend paid extra classes?"),
    "activities": ask_yes_no("Does the student take part in activities?"),
    "internet": ask_yes_no("Does the student have internet access at home?"),
    "health": ask_number("Health rating (1-5): ", 1, 5),
    "goout": ask_number("Going out with friends rating (1-5): ", 1, 5),
    "freetime": ask_number("Free-time rating (1-5): ", 1, 5)
}

input_df = pd.DataFrame([student])
prediction = model.predict(input_df)[0]

print("\nPredicted academic performance:", prediction)
if prediction == "Good Performance":
    print("Observation: The entered profile resembles students in the good-performance group.")
    print("Suggestion: Continue the current learning habits.")
elif prediction == "Average Performance":
    print("Observation: There may be room to strengthen study habits and revision.")
    print("Suggestion: Set a regular revision plan and ask for help when needed.")
else:
    print("Observation: The entered profile resembles the needs-improvement group.")
    print("Suggestion: Make a manageable study timetable and seek academic support.")

print("\nThis is a model-based estimate, not a guaranteed result.")
