import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_path = (
    r"C:\Users\nebolsinvasili\Documents\rpr\rpr\сalculate\data\test_100_25_50_10_170_1000.csv"
)

df = pd.read_csv(data_path, sep=",", parse_dates=True)

# Выделяем входные и выходные данные
X = df[['x', 'y', 'fi']].values  # Входные признаки
y = df["Ld_3"].values  # Целевые значения

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Нормализуем входные данные
y = y.reshape(-1, 1)  # Нормализуем целевые данные

# Разделение на тренировочные, проверочные и тестовые наборы
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.70, shuffle=True, random_state=42
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, shuffle=True, random_state=42
)

print('Training set shapes:')
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)

print('\nValidation set shapes:')
print('X_valid:', X_valid.shape)
print('y_valid:', y_valid.shape)

print('\nTest set shapes:')
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

# Преобразуем данные в тензоры
X_train, y_train = (
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
X_valid, y_valid = (
    torch.tensor(X_valid, dtype=torch.float32),
    torch.tensor(y_valid, dtype=torch.float32),
)
X_test, y_test = (
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32),
)

if __name__ =="__main__":
    pass