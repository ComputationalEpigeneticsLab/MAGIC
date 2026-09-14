import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# id map
id_file = pd.read_excel('/id_map.xlsx', dtype={'wsi_id': str})
id_file.set_index('wsi_id', inplace=True)
id_file.loc[:, 'response'] = (id_file['group'] == 'RR').astype(int)

all_proteins = []

for i in range(5):
    protein_data1 = pd.read_csv(
        f"/train_test_split/5_cv_split/fold_{i}/train_protein.csv",
        index_col=0
    )  # row=feature; col=sample
    protein_data = protein_data1.T  # row=sample; col=feature

    X_train_outer = protein_data
    y_train_outer = id_file.loc[X_train_outer.index, 'response'].values

    # inner cv
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    for inner_fold, (train_inner_idx, _) in enumerate(
            inner_cv.split(X_train_outer, y_train_outer)
    ):
        X_train_inner = X_train_outer.iloc[train_inner_idx]
        y_train_inner = y_train_outer[train_inner_idx]

        # train XGBoost
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200
        )
        xgb_model.fit(X_train_inner, y_train_inner)

        # top 180 features
        importances = xgb_model.feature_importances_
        top_idx = np.argsort(-importances, kind='stable')[:180]
        selected_proteins = X_train_inner.columns[top_idx].tolist()

        all_proteins.extend(selected_proteins)

# Frequency Statistics
protein_counts = Counter(all_proteins)
consensus_proteins = sorted(
    protein_counts.items(),
    key=lambda x: (-x[1], x[0])
)[:180]
consensus_proteins = [p for p, cnt in consensus_proteins]

# save
selected_proteins_df = pd.DataFrame(consensus_proteins, columns=["selected_proteins"])
selected_proteins_df.to_csv("/xgboost/selected_proteins.csv", index=False)