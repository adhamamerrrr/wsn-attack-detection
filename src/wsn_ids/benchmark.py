"""Train three classifiers on one untouched holdout; never resample test data."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE


def load_data(path, target='Attack type', drop=('id',), max_rows=None, seed=42):
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if target not in frame or frame[target].isna().any():
        raise ValueError('Target missing or contains null labels')
    frame[target] = frame[target].astype(str).str.strip()
    if frame[target].eq('').any():
        raise ValueError('Empty labels are invalid')
    frame = frame.drop(columns=[c for c in drop if c in frame])
    # Identical observations must not land on both sides of the holdout.
    before = len(frame)
    frame = frame.drop_duplicates()
    removed = before - len(frame)
    if max_rows and len(frame) > max_rows:
        frame, _ = train_test_split(frame, train_size=max_rows, stratify=frame[target], random_state=seed)
    x = frame.drop(columns=target)
    if x.empty or any(not pd.api.types.is_numeric_dtype(x[c]) for c in x):
        raise ValueError('All feature columns must be numeric')
    values = x.to_numpy(dtype=float)
    if np.isinf(values).any() or x.isna().all().any():
        raise ValueError('Infinite values or wholly missing features')
    if frame[target].value_counts().min() < 4 or frame[target].nunique() < 2:
        raise ValueError('Need two classes with at least four rows per class')
    return values, frame[target].to_numpy(), list(x.columns), removed


class Network(nn.Module):
    def __init__(self, features, classes):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, classes))

    def forward(self, x):
        return self.layers(x)


def benchmark(x, y, seed=42, smote=False, epochs=30):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    encoder = LabelEncoder().fit(y)
    labels = encoder.classes_.tolist()
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=.25, stratify=y, random_state=seed)
    train_y, test_y = encoder.transform(y[train_idx]), encoder.transform(y[test_idx])
    imputer, scaler = SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()
    train_x = scaler.fit_transform(imputer.fit_transform(x[train_idx]))
    test_x = scaler.transform(imputer.transform(x[test_idx]))
    before = len(train_y)
    if smote:
        minimum = np.bincount(train_y).min()
        if minimum < 2:
            raise ValueError('SMOTE requires two training rows per class')
        train_x, train_y = SMOTE(random_state=seed, k_neighbors=min(5, minimum-1)).fit_resample(train_x, train_y)
    result = {'seed': seed, 'split': 'stratified random 75/25 holdout', 'train_rows_before_smote': before,
              'train_rows_after_smote': len(train_y), 'test_rows': len(test_y), 'smote': smote, 'epochs': epochs,
              'labels': labels, 'test_index_sha256': hashlib.sha256(test_idx.tobytes()).hexdigest(), 'models': {}}
    predictions = {}
    for name, model in {'decision_tree': DecisionTreeClassifier(max_depth=12, random_state=seed),
                        'logistic_regression': LogisticRegression(max_iter=2000, random_state=seed)}.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(train_x, train_y)
        predictions[name] = (model.predict(test_x), [str(w.message) for w in caught])
    model = Network(train_x.shape[1], len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    features = torch.tensor(train_x, dtype=torch.float32)
    targets = torch.tensor(train_y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, targets), batch_size=256, shuffle=True, generator=generator)
    losses = []
    model.train()
    for _ in range(epochs):
        total = 0.
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(batch_y)
        losses.append(total/len(train_y))
    model.eval()
    with torch.no_grad():
        predicted = model(torch.tensor(test_x, dtype=torch.float32)).argmax(1).numpy()
    predictions['pytorch_mlp'] = (predicted, [])
    for name, (predicted, notices) in predictions.items():
        result['models'][name] = {'report': classification_report(test_y, predicted, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0),
                                 'confusion_matrix': confusion_matrix(test_y, predicted, labels=list(range(len(labels)))).tolist(), 'warnings': notices}
    result['pytorch_training_loss'] = losses
    result['versions'] = {'torch': torch.__version__, 'numpy': np.__version__}
    return result


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', type=Path)
    source.add_argument('--demo', action='store_true')
    p.add_argument('--target', default='Attack type')
    p.add_argument('--drop', nargs='*', default=['id'])
    p.add_argument('--max-rows', type=int)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smote', action='store_true')
    p.add_argument('--output', type=Path, default=Path('results/benchmark.json'))
    a = p.parse_args()
    if a.max_rows is not None and a.max_rows < 8:
        p.error('--max-rows must be at least 8')
    if a.demo:
        x, y = make_classification(n_samples=600, n_features=10, n_informative=6, weights=[.8,.2], random_state=a.seed)
        y = y.astype(str)
        features, removed = [f'synthetic_{i}' for i in range(10)], 0
    else:
        x, y, features, removed = load_data(a.csv, a.target, a.drop, a.max_rows, a.seed)
    result = benchmark(x, y, a.seed, a.smote, a.epochs)
    result.update(data_source='synthetic_smoke_test' if a.demo else a.csv.name, features=features, duplicate_rows_removed=removed,
                  input_sha256=None if a.demo else hashlib.sha256(a.csv.read_bytes()).hexdigest(), original_source_recovered=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({name: round(v['report']['macro avg']['f1-score'],4) for name,v in result['models'].items()}))

if __name__ == '__main__':
    main()
