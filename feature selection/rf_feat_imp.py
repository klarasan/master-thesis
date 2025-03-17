from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

plt.rcParams['font.family'] = 'Times New Roman'

vars = ['PDSI', 'srad', 'ppt', 'tmax', 'def', 'vpd', 'vap', 'pet', 'aet', 'q', 'soil', 'tmin', 'ws', 'swe']

num_years = 5
df = pd.read_csv('data/avg_std_12_years_bilinear_interp_w_outliers.csv', on_bad_lines='skip')

cols = []
for var in vars:
    for year in range(1-num_years, 1):
        cols.append(f'{var}_year{year}_mean')
X = df[cols]
y = df['label']

X_train, y_train = shuffle(X, y, random_state=42)
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

variable_importance = {var: 0 for var in vars}

for var in vars:
    variable_importance[var] = feature_importance_df[
        feature_importance_df['Feature'].str.startswith(var)
    ]['Importance'].sum()

variable_importance_df = pd.DataFrame(
    list(variable_importance.items()), columns=['Variable', 'Total Importance']
).sort_values(by='Total Importance', ascending=False)

print(variable_importance_df)

colors = ["orange", "coral", "thistle", "plum"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

norm = np.linspace(0, 1, len(variable_importance_df))
bar_colors = custom_cmap(norm)

plt.figure(figsize=(8, 3.5))
plt.bar(variable_importance_df['Variable'], variable_importance_df['Total Importance'], color=bar_colors, zorder=2)
plt.grid(True, linestyle='-', alpha=0.5, zorder=1)
plt.xlabel('Input channels')
plt.ylabel('Relevance Estimation (arb. unit)')
plt.tight_layout()
plt.show()
