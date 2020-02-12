#!/usr/bin/env python

import pandas as pd

db = ""
df = pd.read_csv("/tf/data/heart.csv")

# change column names
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
              'cholesterol', 'fasting_blood_sugar', 'rest_ecg',
              'max_heart_rate_achieved', 'exercise_induced_angina',
              'st_depression', 'st_slope', 'num_major_vessels',
              'thalassemia', 'target']

columns_whitelist = ['age', 'chest_pain_type', 'resting_blood_pressure',
                     'cholesterol', 'exercise_induced_angina', 'st_depression',
                     'st_slope', 'num_major_vessels', 'thalassemia']
                    
for i, patient in df.iterrows():
    for (k, v) in patient.items():
        if k in columns_whitelist:
            db += "symptom(p{}, {}, {}).\n".format(i, k, v)
    db += "\n"

f = open("/tf/results/db.pl", "w")
f.write(db)
f.close()
