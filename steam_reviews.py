import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# 1. LOAD & INITIAL PROCESSING 

df = pd.read_csv("steam.csv")
print("Initial shape:", df.shape)

# Keep game title for error analysis (NOT used in X matrix)
df["game_title"] = df["name"]

# Create total reviews and filter unstable labels (< 20 reviews)
df["total_reviews"] = df["positive_ratings"] + df["negative_ratings"]
df = df[df["total_reviews"] >= 20]

# Compute supervised label y
df["user_rating"] = df["positive_ratings"] / df["total_reviews"] * 100

# Remove any NaNs in the target
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["user_rating"])

# 2. FEATURE ENGINEERING

# Clean genre and tag lists
df["genres"] = df["genres"].fillna("").apply(lambda x: x.split(";"))
df["steamspy_tags"] = df["steamspy_tags"].fillna("").apply(lambda x: x.split(";"))

# Multi-hot encode genres
mlb_genres = MultiLabelBinarizer()
genres_encoded = mlb_genres.fit_transform(df["genres"])
genres_df = pd.DataFrame(genres_encoded, columns=[f"genre_{g}" for g in mlb_genres.classes_])

# Multi-hot encode tags
mlb_tags = MultiLabelBinarizer()
tags_encoded = mlb_tags.fit_transform(df["steamspy_tags"])
tags_df = pd.DataFrame(tags_encoded, columns=[f"tag_{t}" for t in mlb_tags.classes_])

df = pd.concat([df.reset_index(drop=True), genres_df, tags_df], axis=1)

# Convert owners range ("100000-200000") → numeric mean
def owners_to_numeric(s):
    try:
        a, b = s.split("-")
        return (int(a) + int(b)) / 2
    except:
        return np.nan

df["owners_num"] = df["owners"].apply(owners_to_numeric)

# 3. HANDLE MISSING NUMERIC VALUES

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 4. BUILD X AND y MATRICES 

drop_cols = [
    "appid", "name", "genres", "steamspy_tags", "categories",
    "developer", "publisher", "platforms", "release_date", "owners",
    "positive_ratings", "negative_ratings"
]

df = df.drop(columns=[c for c in drop_cols if c in df.columns])

y = df["user_rating"]
X = df.drop(columns=["user_rating", "game_title"])

# Final sanity checks
assert X.isna().sum().sum() == 0, "NaNs detected in X"
assert y.isna().sum() == 0, "NaNs detected in y"

# 5. TRAIN/TEST SPLIT 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# 6. TRAIN MODELS

linreg = LinearRegression()
linreg.fit(X_train, y_train)

rf = RandomForestRegressor(
    n_estimators=350,
    random_state=42,
    max_depth=None,
    min_samples_split=2
)
rf.fit(X_train, y_train)

# 7. EVALUATION 

def evaluate(model, name):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n=== {name} ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}") 
    
    return y_pred

y_pred_lin = evaluate(linreg, "Linear Regression (Baseline)")
y_pred_rf  = evaluate(rf, "Random Forest Regressor")

# 8. IDENTIFY HARD SAMPLES (Slides: "Interesting/Hard Samples")

results = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred_rf
}, index=y_test.index)

# Add titles back
results["title"] = df.loc[results.index, "game_title"]

# Compute absolute errors
results["error"] = abs(results["actual"] - results["predicted"])

worst5 = results.sort_values("error", ascending=False).head(5)

print("\n=== 5 WORST PREDICTIONS ===")
print(worst5[["title", "actual", "predicted", "error"]])

# ================================================================
# 9. VISUALIZATION (Slides: Evaluating Model Fit)
# ================================================================

# Scatter: predicted vs actual (Slides: continuous prediction analysis)
plt.figure(figsize=(8,6))
plt.scatter(results["actual"], results["predicted"], alpha=0.5)
plt.xlabel("Actual Rating (%)")
plt.ylabel("Predicted Rating (%)")
plt.title("Predicted vs Actual Steam User Ratings")
plt.grid(True)
plt.show()

# Feature importance: top 5 (Slides: interpretability)
importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(8,6))
importances.sort_values(ascending=False).head(5).plot(kind='barh')
plt.title("Top 5 Most Important Features Predicting User Rating")
plt.gca().invert_yaxis()
plt.show()
