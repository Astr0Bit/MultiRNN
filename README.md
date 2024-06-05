# MultiRNN

The `MultiRNN` class provides tools for building, training, and evaluating a multi-layer Recurrent Neural Network (RNN) using Long Short-Term Memory (LSTM) layers for time series forecasting.

## Features

- Customizable LSTM layers
- Train-test split for time series data
- Data scaling using `MinMaxScaler`
- Model persistence using `joblib`
- Visualization of training performance

## Installation

To use the `MultiRNN` class, ensure you have the following dependencies installed:

```bash
pip install numpy pandas termcolor joblib matplotlib scikit-learn keras
```

## Usage

### Importing the MultiRNN Class

```python
from multirnn import MultiRNN
```

### Creating and Training the Model

```python
import pandas as pd

# Load your time series data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Initialize the model
model = MultiRNN(
    train=train_data, 
    test=test_data, 
    length=10, 
    LSTM_units=[50, 30], 
    activation=['tanh', 'relu']
)

# Train the model
history = model.train(epochs=100, batch_size=32)
```

### Making Predictions

```python
predictions = model.predict()
```

### Plotting the Results

```python
model.plot()
```

### Saving the Model

```python
model.save('model_path')
```

### Loading a Saved Model

```python
model = MultiRNN.load('model_path')
```

## Methods

### `__init__(self, train, test, length, LSTM_units, activation, ... )`
Initializes the `MultiRNN` class with the provided training and testing datasets, sequence length, LSTM units, and activations.

### `train(self, epochs, batch_size)`
Trains the RNN model for a specified number of epochs and batch size. Returns the training history.

### `predict(self)`
Generates predictions for the test dataset.

### `plot(self)`
Plots the training and testing performance over time.

### `save(self, path)`
Saves the trained model to the specified path.

### `load(cls, path)`
Loads a saved model from the specified path.

## Example

Below is a complete example of how to use the `MultiRNN` class.

```python
import pandas as pd
from multirnn import MultiRNN

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Initialize model
model = MultiRNN(
    train=train_data, 
    test=test_data, 
    length=10, 
    LSTM_units=[50, 30], 
    activation=['tanh', 'relu']
)

# Train model
history = model.train(epochs=100, batch_size=32)

# Make predictions
predictions = model.predict()

# Plot results
model.plot()

# Save model
model.save('model_path')

# Load model
loaded_model = MultiRNN.load('model_path')
```

## Contributing

Feel free to open issues or submit pull requests if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.