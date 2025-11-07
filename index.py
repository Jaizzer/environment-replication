import pandas as pd
import numpy as np
import math
import time

# --- TensorFlow/Keras/Prophet Imports ---
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, RNN, LSTM, Dropout, Bidirectional, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
# Assuming 'rnn' is a custom file in the workspace
from rnn import FastGRNNCellKeras, FastRNNCellKeras 
# Assuming 'keras_self_attention' is available
from keras_self_attention import SeqSelfAttention 
from fbprophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from keras_self_attention import SeqSelfAttention


# --- Configuration ---
# All models will share these configuration parameters
LOOKBACK_WINDOW = 10
TRAINING_PERCENTAGE_CUTOFF = 50 # Last 50 points are test data (len(dataset) - 50)
TOTAL_EPOCHS = 40
BATCH_SIZE = 10
LEARNING_RATE = 0.001

# --- Global Results Storage ---
RESULTS = []
PREDICTIONS = {}

# --- Data Loading and Preparation ---

# Load Data
data_1 = pd.read_csv('Apple_2.csv')
print("Initial Data Length: {}".format(len(data_1)))

# Preprocessing (applied once)
df = data_1.filter(['Close']).reset_index(drop=True)
df = df[df['Close'].notna()]
print("Data Length after NaN removal: {}".format(len(df)))

# Convert Time_b for FBProphet later (assuming original 'Time_b' column exists or needs reconstruction)
# NOTE: If 'Time_b' exists in data_1, use data_1['Time_b'] directly
df_fb = data_1.copy()
df_fb['ds'] = pd.to_datetime(df_fb['Time_b'])
df_fb.rename(columns={'Close': 'y'}, inplace=True)
df_fb.index = df_fb['ds']
df_fb = df_fb.filter(['ds', 'y'])
df_fb = df_fb[df_fb['y'].notna()]


# --- Data Scaling and Splitting ---

dataset = df.values
training_data_len = math.ceil(len(dataset) - TRAINING_PERCENTAGE_CUTOFF)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Function to create sequence data (x_train, y_train, x_test, y_test)
def create_sequences(data, training_len, lookback):
    # Training Data
    train_data_scaled = data[0:training_len, :]
    x_train, y_train = [], []
    for i in range(lookback, len(train_data_scaled)):
        x_train.append(train_data_scaled[i-lookback:i, 0])
        y_train.append(train_data_scaled[i, 0])
    
    # Test Data (Includes lookback overlap)
    test_data_scaled = data[training_len - lookback:, :]
    x_test = []
    for i in range(lookback, len(test_data_scaled)):
        x_test.append(test_data_scaled[i-lookback:i, 0])
    
    # True test values (unscaled)
    y_test_unscaled = dataset[training_len:, :]
    
    # Convert to NumPy arrays and reshape for RNN/LSTM [samples, timesteps, features]
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test = np.array(x_test)
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # Ensure y_test_unscaled matches x_test length
    y_test_unscaled = y_test_unscaled[:x_test.shape[0]]

    return x_train, y_train, x_test, y_test_unscaled

x_train, y_train, x_test, y_test = create_sequences(scaled_data, training_data_len, LOOKBACK_WINDOW)

# Validation targets are the scaled data slice
y_val_scaled = scaled_data[training_data_len:training_data_len + len(y_test), :]


# --- Model Configuration Parameters ---
# Parameters for FastRNN/FastGRNN
inputDims = LOOKBACK_WINDOW 
hiddenDims = 50 
update_non_linearity = "relu"
FastCell = FastRNNCellKeras(hiddenDims, update_non_linearity=update_non_linearity)
ADAM_OPTIMIZER = Adam(learning_rate=LEARNING_RATE) 
dataDimension = LOOKBACK_WINDOW


# --- Utility Functions ---

def evaluate_model(model_name, model, x_test_data, y_true_unscaled, start_time, stop_time, output_csv_name):
    predictions = model.predict(x_test_data)
    predictions_unscaled = scaler.inverse_transform(predictions)
    
    # Calculate Metrics
    mse = mean_squared_error(y_true_unscaled, predictions_unscaled)
    rmse = math.sqrt(mse)
    cod = r2_score(y_true_unscaled, predictions_unscaled)
    time_taken = stop_time - start_time
    
    # Store results
    RESULTS.append({
        'Variabel_name': model_name,
        'RMSE': rmse,
        'Time(in sec)': time_taken,
        'Description': "Trained with {} epochs.".format(TOTAL_EPOCHS),
        'R-Square value': cod
    })
    
    # Store predictions
    PREDICTIONS[model_name] = np.squeeze(predictions_unscaled)

    print("\n{} Results:".format(model_name))
    print("Time: {:.3f}s, RMSE: {:.3f}, R2: {:.3f}".format(time_taken, rmse, cod))
    return predictions_unscaled


# --- RNN/LSTM Helper: Reshape for Final Model Layer ---
# This is required for some models that expect [samples, features] instead of [samples, timesteps, features]
def reshape_for_rnn_layer(data):
    # x_train is currently [samples, timesteps (10), 1]
    # Final layer reshape is to [samples, 1, input_dims (10)]
    return np.reshape(data, [data.shape[0], 1, LOOKBACK_WINDOW])


# #################################################
## FAST R N N
# #################################################
def run_fastrnn():
    print("\n--- Running FastRNN ---")
    
    # Model Definition (FastRNN)
    x = inputs = Input(shape=[int(dataDimension / inputDims), inputDims], name='input')
    x_rnn = RNN(FastCell, return_sequences=False, name='rnn')(x)
    out = Dense(1, activation='relu', name='dense')(x_rnn)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=ADAM_OPTIMIZER, loss='mean_squared_error', metrics=['accuracy'])
    
    # Reshape for this specific model's input layers
    x_train_rnn = reshape_for_rnn_layer(x_train)
    x_test_rnn = reshape_for_rnn_layer(x_test)
    
    csv_logger = CSVLogger("model_history_log_fastrnn.csv", append=False)
    
    start = time.time()
    model.fit(x_train_rnn, y_train, BATCH_SIZE, epochs=TOTAL_EPOCHS, 
              validation_data=(x_test_rnn, y_val_scaled), callbacks=[csv_logger], verbose=0)
    stop = time.time()
    
    return evaluate_model("FastRNN", model, x_test_rnn, y_test, start, stop, "model_history_log_fastrnn.csv")

# #################################################
## F B P R O P H E T
# #################################################
def run_fb_prophet():
    print("\n--- Running FBProphet ---")
    # Prophet data must use original unscaled values
    train_fb = df_fb[:training_data_len].filter(['ds', 'y'])
    valid_fb = df_fb[training_data_len:].filter(['ds', 'y'])

    # Fit the model
    model_fb = Prophet(daily_seasonality=True)
    start_fb = time.time()
    model_fb.fit(train_fb)
    stop_fb = time.time()

    # Predictions
    close_prices_fb = model_fb.make_future_dataframe(periods=len(valid_fb), freq='1min')
    forecast_fb = model_fb.predict(close_prices_fb)
    
    forecast_valid_fb = forecast_fb['yhat'][training_data_len:]
    
    # Calculate RMSE and R2 (against original unscaled 'y')
    rmse_fb = np.sqrt(np.mean(np.power((np.array(valid_fb['y'])-np.array(forecast_valid_fb)),2)))
    cod_fb = r2_score(valid_fb['y'], forecast_valid_fb)
    time_fb = stop_fb - start_fb

    # Store results
    RESULTS.append({
        'Variabel_name': 'FB_Prophet',
        'RMSE': rmse_fb,
        'Time(in sec)': time_fb,
        'Description': 'Daily Seasonality=True',
        'R-Square value': cod_fb
    })
    PREDICTIONS['FB_Prophet'] = np.squeeze(np.array(forecast_valid_fb).reshape(-1,1))
    
    print("FB_Prophet Results: Time: {:.3f}s, RMSE: {:.3f}, R2: {:.3f}".format(time_fb, rmse_fb, cod_fb))


# #################################################
## A R I M A
# #################################################
def run_arima():
    print("\n--- Running ARIMA ---")
    # ARIMA uses unscaled data
    X = df['Close'].dropna().values
    train_arima, test_arima = X[0:training_data_len], X[training_data_len:len(X)]
    
    start_arima = time.time()
    # Using fixed order (2,1,1) as per original code
    try:
        model_arima = ARIMA(train_arima, order=(2, 1, 1))
        model_fit = model_arima.fit()
        predictions = model_fit.forecast(len(test_arima))[0]
    except Exception as e:
        print("ARIMA failed to fit: {}".format(e))
        predictions = np.zeros_like(test_arima) # Placeholder for failed model
        error = np.nan
        cod = np.nan
        
    stop_arima = time.time()

    if not np.isnan(predictions).all():
        error = math.sqrt(mean_squared_error(test_arima, predictions))
        cod = r2_score(test_arima, predictions)
    
    time_arima = stop_arima - start_arima

    # Store results
    RESULTS.append({
        'Variabel_name': 'Arima',
        'RMSE': error,
        'Time(in sec)': time_arima,
        'Description': 'Order (2,1,1)',
        'R-Square value': cod
    })
    PREDICTIONS['Arima'] = np.squeeze(predictions)
    
    print("ARIMA Results: Time: {:.3f}s, RMSE: {:.3f}, R2: {:.3f}".format(time_arima, error, cod))

# #################################################
## L S T M / C N N - L S T M M O D E L S
# #################################################
def create_compile_model(model_type, units, learning_rate=0.001, cnn_filters=0, dropout=0.0):
    model = Sequential()
    
    if model_type == 'LSTM_50':
        model.add(LSTM(units=units, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    elif model_type == 'LSTM_1':
        model.add(LSTM(units=units, return_sequences=False, input_shape=(x_train.shape[1], 1)))
    elif model_type == 'CNN_BiLSTM':
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='tanh', input_shape=(x_train.shape[1], 1)))
        model.add(Bidirectional(LSTM(units=units, return_sequences=False)))
        model.add(Dropout(dropout))
    elif model_type == 'LSTM_Attention':
        model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(SeqSelfAttention(attention_activation='relu'))
        model.add(Flatten())
    elif model_type == 'LSTM_CNN':
        model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
    elif model_type == 'FastRNN_Hybrid':
        # This requires functional API due to FastCell RNN layer
        return build_fastrnn_hybrid_model(units)
    elif model_type == 'CNN_2_LSTM_2':
        model.add(Dense(units=128, input_shape=(x_train.shape[1], 1)))
        model.add(Conv1D(filters=112, kernel_size=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dense(units=100))
    else:
        raise ValueError("Invalid model type")

    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def build_fastrnn_hybrid_model(units):
    y = inputs = Input(shape=[int(dataDimension / inputDims), inputDims], name='input')
    y = RNN(FastCell, return_sequences=True, name='rnn')(y)
    y = Conv1D(filters=300, kernel_size=3 , padding='same', activation='relu', name='Conv1D')(y)
    y = MaxPooling1D(pool_size=1,name='MaxPooling1D')(y)
    y = Bidirectional(LSTM(units=units, return_sequences=False))(y)
    out = Dense(units=1)(y)
    
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=ADAM_OPTIMIZER, loss='mean_squared_error')
    
    return model

def run_keras_model(model_type, name, units, cnn_filters=0, learning_rate=LEARNING_RATE, dropout=0.0):
    print("\n--- Running {} ---".format(name))
    
    if model_type == 'FastRNN_Hybrid':
        model = build_fastrnn_hybrid_model(units)
        # Reshape data for FastRNN layers
        x_train_input = reshape_for_rnn_layer(x_train)
        x_test_input = reshape_for_rnn_layer(x_test)
    else:
        model = create_compile_model(model_type, units, learning_rate, cnn_filters, dropout)
        x_train_input = x_train
        x_test_input = x_test
    
    csv_logger = CSVLogger("model_history_log_{}.csv".format(name.replace(' ', '_')), append=False)
    
    start = time.time()
    model.fit(x_train_input, y_train, batch_size=BATCH_SIZE, epochs=TOTAL_EPOCHS, 
              validation_data=(x_test_input, y_val_scaled), callbacks=[csv_logger], verbose=0)
    stop = time.time()
    
    evaluate_model(name, model, x_test_input, y_test, start, stop, "model_history_log_{}.csv".format(name.replace(' ', '_')))

# #################################################
## M A I N E X E C U T I O N
# #################################################

if __name__ == "__main__":
    
    # 1. Run all models
    
    # Simple RNN Models
    run_fastrnn() 
    run_keras_model('LSTM_50', 'LSTM (50 Units)', 50, learning_rate=0.01)
    run_keras_model('LSTM_1', 'LSTM (1 Unit)', 1, learning_rate=0.01)

    # Hybrid Models
    run_keras_model('CNN_BiLSTM', 'CNN_BiLSTM', 50, cnn_filters=50, dropout=0.5)
    run_keras_model('FastRNN_Hybrid', 'FASTRNN_CNN_BiLSTM', 50)
    # run_keras_model('LSTM_Attention', 'LSTM_Attention (ReLU)', 50)
    run_keras_model('LSTM_CNN', 'LSTM_CNN', 50, cnn_filters=50)
    run_keras_model('CNN_2_LSTM_2', 'CNN_2_LSTM_2', units=128) # Units parameter is ignored for this specific model type
    
    # Statistical Models
    run_fb_prophet()
    run_arima() 
    

    # 2. Compile and Save Results
    print("\n--- Compiling Final Results ---")
    
    # Create final performance DataFrame
    df_final = pd.DataFrame(RESULTS)
    
    # Create final predictions DataFrame
    predicted_results = pd.DataFrame(PREDICTIONS)
    predicted_results.insert(0, 'Test_Data', np.squeeze(y_test))
    
    # Save results to CSV
    df_final.to_csv('Performance_Results.csv', index=False)
    predicted_results.to_csv('Predicted_Values.csv', index=False)
    
    print("\n✅ Execution Complete.")
    print("\nPerformance Results:")
    print(df_final)
    print("\nPredictions Head:")
    print(predicted_results.head())