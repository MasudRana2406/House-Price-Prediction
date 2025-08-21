import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sample data with multiple features:
# Columns: [Size (sq ft), Bedrooms, Age (years)]
X = np.array([
    [1400, 3, 20],
    [1600, 3, 15],
    [1700, 4, 18],
    [1875, 4, 12],
    [1100, 2, 30],
    [1550, 3, 14],
    [2350, 4, 8],
    [2450, 5, 5],
    [1425, 3, 20],
    [1700, 3, 10]
])

# Target (House Prices)
y = np.array([245000, 312000, 279000, 308000, 199000, 
              219000, 405000, 324000, 319000, 255000])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open('house_price_model_multi.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Multi-feature model trained and saved successfully!")
