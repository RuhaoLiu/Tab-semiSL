import os
import numpy as np
import torch
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from utils import convert_matrix_to_vector, convert_vector_to_matrix


# Define model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function")

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def xgb_model(x_train, y_train, x_test):
    # Convert labels into proper format
    if len(y_train.shape) > 1:
        y_train = convert_matrix_to_vector(y_train)

    # Define and fit model on training dataset
    model = xgb.XGBClassifier()
    model.fit(x_train, y_train)

    # Predict on x_test
    y_test_hat = model.predict_proba(x_test)

    return y_test_hat

def lightgbm_model(x_train, y_train, x_test):
    # Convert labels into proper format
    if len(y_train.shape) > 1:
        y_train = convert_matrix_to_vector(y_train)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=30)

    # Define and fit model on training dataset
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

    params = {
      'objective': 'multiclass',
      'num_class': 10,
      'metric': 'multi_logloss',
      'boosting_type': 'gbdt',
      'learning_rate': 0.1,
      'num_leaves': 31,
      'feature_fraction': 0.8,
      'bagging_fraction': 0.8,
      'verbose': -1
    }
    model = lgb.train(
      params,
      train_data,
      num_boost_round=200,
      valid_sets=[val_data],
      callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    # Predict on x_test
    # calculate auc
    # y_test_hat = model.predict(x_test)
    # y_test_hat = np.column_stack([1 - y_test_hat, y_test_hat])
    # calculate accuracy
    y_test_hat = model.predict(x_test)

    return y_test_hat

def catboost_model(x_train, y_train, x_test):
    # Convert labels into proper format
    if len(y_train.shape) > 1:
        y_train = convert_matrix_to_vector(y_train)
    # x_train = x_train.numpy()
    # x_test = x_test.numpy()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1,random_state=10)

    # model
    model = CatBoostClassifier(
      iterations=1000,
      learning_rate=0.01,
      depth=8,
      l2_leaf_reg=5,
      early_stopping_rounds=50,
      eval_metric='AUC',
      loss_function='Logloss',
      verbose=False
    )

    model.fit(x_train, y_train, eval_set=(x_val, y_val))
    y_test_hat = model.predict_proba(x_test)
    return y_test_hat

def mlp(x_train, y_train, x_test, parameters):
    # Convert labels into proper format
    if len(y_train.shape) == 1:
        y_train = convert_vector_to_matrix(y_train)

    # Split into train and validation (9:1)
    x_train, x_valid, y_train, y_valid = train_test_split(
      x_train, y_train, test_size=0.1
    )

    # Convert to PyTorch tensors
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    x_valid = torch.FloatTensor(x_valid)
    y_valid = torch.FloatTensor(y_valid)
    x_test = torch.FloatTensor(x_test)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    batch_size = parameters['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # Define basic parameters
    input_dim = len(x_train[0, :])
    output_dim = len(y_train[0, :])

    # Build model
    model = MLP(
      input_dim=input_dim,
      hidden_dim=parameters['hidden_dim'],
      output_dim=output_dim,
      activation=parameters['activation']
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001,
                           # weight_decay=1e-4
                           )

    # Early stopping
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    save_path = './save_model/self_mlp/best_mlp.pth'
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    epochs = parameters['epochs']
    for epoch in range(epochs):
        # print(f'mlp {epoch} epoch')
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)

        # Early stopping check
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(save_path))

    # Predict on test set
    model.eval()
    with torch.no_grad():
        y_test_hat = model(x_test).numpy()

    return y_test_hat