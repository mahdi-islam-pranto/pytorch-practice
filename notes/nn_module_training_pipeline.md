# Simple Neural Network Training Pipeline with `nn.Module`

This note shows the usual steps for building and training a small neural network in PyTorch.
The example is binary classification: the model predicts one of two classes.

## The steps

1. Prepare the data and separate the inputs (`X`) from the answers (`y`).
2. Split the data into training data and test data.
3. Scale the input values and convert the data to PyTorch tensors.
4. Define the network by making a class that inherits from `nn.Module`.
5. Choose a loss function and an optimizer.
6. Train the model: predict, calculate loss, calculate gradients, and update weights.
7. Test the trained model on data it did not train on.

## 1. Prepare and split the data

`X` contains the information the model uses to make a prediction. `y` contains the correct answer.
For this notebook, the target is the diagnosis and the remaining columns are the input features.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Split first so the test data stays separate during training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# Fit the scaler using training inputs only, then use it on both sets.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Turn the two text labels into numbers (0 and 1).
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
```

Scaling puts input values on similar ranges, which usually helps a neural network train.

## 2. Convert the data to tensors

PyTorch models use tensors. The model and the labels must have compatible shapes and data types.

```python
import numpy as np
import torch

X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32))
```

In this example, each input row has 30 features. The label tensor starts with one value per row;
the training loop reshapes it to match the model's one-value output.

## 3. Define the model with `nn.Module`

`nn.Module` is the base class for a PyTorch model. `nn.Sequential` sends the input through
each listed layer in order. This model has one hidden layer with 16 units.

```python
import torch.nn as nn

class MySimpleNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.layers(features)
```

- `Linear` learns weights that combine the values.
- `ReLU` lets the hidden layer learn non-linear patterns.
- The final `Linear` produces one score for each row.
- `Sigmoid` turns that score into a value between 0 and 1.
- `forward` describes how data moves through the model.

The number `16` is the number of hidden units. You can change it or add another hidden layer,
but a bigger network does not automatically mean better test performance.

## 4. Choose how to learn

The **loss function** measures how far the predictions are from the correct answers.
The **optimizer** uses gradients to adjust the model's learned weights.

```python
learning_rate = 0.1
epochs = 25

model = MySimpleNN(num_features=X_train_tensor.shape[1])
loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

`BCELoss` is used here because this is binary classification and the model returns sigmoid
probabilities. `epochs` is how many times the model goes through the training data.

## 5. Train the model

Each training step does four main things:

1. Make predictions (forward pass).
2. Measure the prediction error (loss).
3. Calculate gradients with `backward()`.
4. Update the weights with `optimizer.step()`.

```python
for epoch in range(epochs):
    y_pred = model(X_train_tensor)
    loss = loss_function(y_pred, y_train_tensor.view(-1, 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
```

The loss should generally go down as training progresses, but a lower training loss alone does
not prove that the model will do well on new data.

## 6. Evaluate on the test data

The test data was held aside during training. `torch.no_grad()` avoids calculating gradients
because the model is only making predictions now.

```python
with torch.no_grad():
    probabilities = model(X_test_tensor)
    predictions = (probabilities > 0.5).float()
    accuracy = (predictions == y_test_tensor.view(-1, 1)).float().mean()
    print(f"Accuracy: {accuracy.item()}")
```

Accuracy is the fraction of test predictions that are correct. Check test performance when
comparing models; adding layers may help, make no difference, or hurt performance.

## Quick reminder

**Prepare data -> split -> scale -> make tensors -> define model -> choose loss and optimizer ->
train -> evaluate.**

Use `model(inputs)` to run the forward pass. PyTorch calls the model's `forward` method for you.
