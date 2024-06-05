# MultiRNN

This class is designed to help design RNN / LSTM models for time series forecasting.
It works by taking in a train and a test dataset with many or just one feature except 
for the index. It then check these datasets for missing values and that they are good to go.

After that, it's up to the user to define the model / models architecture that will be used.
You can either pass in a single int or string per variable, or pass a list with the same amount
of features in the original dataset, and the class will create a unique model for each feature.

Feel free to play around with the already existing demo file under "demo" folder, or why not
add your own dataset to the "data" folder and create your own model / models. 

IMPORTANT: to decrease the execution time, the class will only create 2 models. This can be
changed by commenting out lines 166 and 167 in the "multirnn.py" file. Also, it is highly
recommended using a GPU to train the models, as it will take a long time to train them on a CPU.

## Setup

Before you try to execute any code,
make sure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- tensorflow==2.15
- sklearn
- keras

If unsure, you can install them by running the following command:

```bash
pip install -r requirements.txt
```

## Example Usage

To use the class, you can either create a new instance of it and pass in the train and test datasets, which you need to manuall create, or run the demo file under the "demo" folder.

```python
from multirnn import MultiRNN

# Create train and test datasets
import pandas as pd

df = pd.read_csv("data/path/here.csv")
train = df.iloc[:int(len(df)*0.8)]
test = df.iloc[int(len(df)*0.8):]

# Create instance of MultiRNN --> this also gets the loss before training
multi = MultiRNN(train=train, test=test,
                 length=10, LSTM_units=32,
                 activation="tanh", optimizer="adam",
                 batch_size=10, epochs=5)

# Save the created sub-datasets (optional)
multi.save_datasets(dir_path="path/to/save")

# Save scalers for each feature (optional)
multi.save_scalers(dir_path="path/to/save")

# Save scaled datasets (optional)
multi.save_scaled_datasets(dir_path="path/to/save")

# Train / fit the models
multi.fit_models()

# Save the models (optional)
multi.save_models(dir_path="path/to/save")

# Make prediction on test dataset
multi.predict()

# Plot the predictions for given feature
multi.plot_predictions(column="feature_name",
                       figure_height=5,
                       figure_width=10,
                       save_plot=True,
                       save_plot_name="path/to/save")

# Plot the loss for given feature
multi.plot_loss(column="feature_name",
                figure_height=5,
                figure_width=10,
                save_plot=True,
                save_plot_name="path/to/save")
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```