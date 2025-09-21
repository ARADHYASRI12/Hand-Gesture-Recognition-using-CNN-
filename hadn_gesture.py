import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, plot_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================================================
# Part 1: Hand Gesture Recognition using CNN
# ==============================================================

# Load and preprocess dataset
train_data = pd.read_csv('hand sign/sign_mnist_train.csv')
X_train = train_data.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_data.iloc[:, 0].values

# One-hot encoding
y_train = to_categorical(y_train, num_classes=26)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Build CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.28))
cnn_model.add(Dense(26, activation='softmax'))

# Compile
cnn_model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train
history = cnn_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# Save model
cnn_model.save('HandSignRecog.h5')

# Plot accuracy & loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Over Epochs')
plt.show()

# Save architecture diagram
plot_model(cnn_model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Preprocess function
def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    img_array = img_to_array(img) / 255.0
    return np.reshape(img_array, (1, 28, 28, 1))

# Prediction function
def predict_image(model, image_path, class_labels):
    img_array = preprocess_image(image_path)
    plt.imshow(img_array[0].reshape(28, 28), cmap='gray')
    plt.title("Input Image"); plt.axis('off'); plt.show()
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    print(f"Predicted Class: {class_labels[predicted_class]} (Label: {predicted_class})")

# Class labels (A-Z except J, Z)
class_labels = [chr(i) for i in range(65, 91)]
class_labels.remove('J'); class_labels.remove('Z')

# Test CNN on custom image
image_path = '1127_C.jpg'
predict_image(cnn_model, image_path, class_labels)


# ==============================================================
# Part 2: Time Series Forecasting using LSTM
# ==============================================================

# Load dataset
data = pd.read_csv('individual+household+electric+power+consumption/household_power_consumption.txt',
                   sep=';', low_memory=False, na_values='?')

# DateTime handling
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time
data['Datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))

# Fill missing values
data.ffill(inplace=True)

# Resample daily
data.set_index('Datetime', inplace=True)
daily_data = data['Global_active_power'].resample('D').sum()

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_data.values.reshape(-1, 1))

# Sequence creation
def create_sequences(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 30
X, y = create_sequences(scaled_data, time_steps)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM
history_lstm = lstm_model.fit(
    X_train, y_train, epochs=20, batch_size=32,
    validation_data=(X_test, y_test), verbose=1
)

# Predictions
train_predictions = lstm_model.predict(X_train)
test_predictions = lstm_model.predict(X_test)

# Inverse transform
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluation
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(f"Train MAE: {train_mae}, Train RMSE: {train_rmse}")
print(f"Test MAE: {test_mae}, Test RMSE: {test_rmse}")

# Plot Forecasting
plt.figure(figsize=(10, 6))
plt.plot(daily_data.index[-len(y_test):], y_test, color='red', label='Actual')
plt.plot(daily_data.index[-len(test_predictions):], test_predictions, color='green', label='Predicted')
plt.title('Daily Energy Consumption Forecasting')
plt.xlabel('Date'); plt.ylabel('Energy Consumption (kW)')
plt.legend(); plt.show()
