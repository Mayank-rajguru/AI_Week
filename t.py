import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "previous_sgpa": np.round(np.random.uniform(5.0, 10.0, 200), 1),
    "avg_programming_score": np.round(np.random.uniform(50, 100, 200), 1),
    "avg_practical_score": np.round(np.random.uniform(50, 100, 200), 1),
    "avg_conceptual_score": np.round(np.random.uniform(50, 100, 200), 1),
    "attendance": np.round(np.random.uniform(60, 100, 200), 1),
    "job_hours": np.random.randint(0, 6, 200)
}

df = pd.DataFrame(data)
df.to_csv("student_data_200.csv", index=False)
