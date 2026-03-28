import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def load_and_scale_data(batch_size=64):
    train_df = pd.read_csv('data/labelled_train.csv')
    test_df = pd.read_csv('data/labelled_test.csv')
    val_df = pd.read_csv('data/labelled_validation.csv')

    X_train = train_df.drop('sus_label', axis=1).values
    y_train = train_df['sus_label'].values
    X_test = test_df.drop('sus_label', axis=1).values
    y_test = test_df['sus_label'].values
    X_val = val_df.drop('sus_label', axis=1).values
    y_val = val_df['sus_label'].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, (X_val, y_val), (X_test, y_test)