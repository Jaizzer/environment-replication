import pandas as pd
import numpy as np
import math
import time
import sys # <-- NEW: Import sys for command-line arguments

# --- TensorFlow/Keras/Prophet Imports ---
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, RNN, LSTM, Dropout, Bidirectional, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
# Assuming 'rnn' is a custom file in the workspace
# from rnn import FastGRNNCellKeras, FastRNNCellKeras 
# Placeholder imports for custom cells if 'rnn.py' is not provided:
class FastGRNNCellKeras(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def call(self, inputs): return inputs[:, 0:self.units]
class FastRNNCellKeras(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def call(self, inputs): return inputs[:, 0:self.units]

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

def reshape_for_rnn_layer(data):
    # x_train is currently [samples, timesteps (10), 1]
    # Final layer reshape is to [samples, 1, input_dims (10)]
    return np.reshape(data, [data.shape[0], 1, LOOKBACK_WINDOW])


# #################################################
## FAST R N N
# #################################################
def run_fastrnn(x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims):
    print("\n--- Running FastRNN ---")
    
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
    
    return evaluate_model("FastRNN (proposed)", model, x_test_rnn, y_test, start, stop, "model_history_log_fastrnn.csv")

# #################################################
## F B P R O P H E T
# #################################################
def run_fb_prophet(df_fb, training_data_len):
    print("\n--- Running FBProphet ---")
    
    train_fb = df_fb[:training_data_len].filter(['ds', 'y'])
    valid_fb = df_fb[training_data_len:].filter(['ds', 'y'])

    # Fit the model
    model_fb = Prophet(daily_seasonality=True)
    start_fb = time.time()
    model_fb.fit(train_fb)
    stop_fb = time.time()

    # Predictions
    # NOTE: Using 'min' frequency based on previous context, but Prophet typically works best with 'D' or 'H'
    future = model_fb.make_future_dataframe(periods=len(valid_fb), freq='min')
    forecast_fb = model_fb.predict(future)
    
    # Align and slice forecast results
    forecast_valid_fb = forecast_fb['yhat'][-len(valid_fb):]
    
    # Calculate RMSE and R2 (against original unscaled 'y')
    rmse_fb = np.sqrt(np.mean(np.power((np.array(valid_fb['y'])-np.array(forecast_valid_fb)),2)))
    cod_fb = r2_score(valid_fb['y'], forecast_valid_fb)
    time_fb = stop_fb - start_fb

    # Store results
    RESULTS.append({
        'Variabel_name': 'FBProphet',
        'RMSE': rmse_fb,
        'Time(in sec)': time_fb,
        'Description': 'Daily Seasonality=True',
        'R-Square value': cod_fb
    })
    PREDICTIONS['FBProphet'] = np.squeeze(np.array(forecast_valid_fb).reshape(-1,1))
    
    print("FBProphet Results: Time: {:.3f}s, RMSE: {:.3f}, R2: {:.3f}".format(time_fb, rmse_fb, cod_fb))


# #################################################
## A R I M A
# #################################################
def run_arima(df, training_data_len):
    print("\n--- Running ARIMA ---")
    
    X = df['Close'].dropna().values
    train_arima, test_arima = X[0:training_data_len], X[training_data_len:len(X)]
    
    start_arima = time.time()
    error = np.nan
    cod = np.nan
    
    # Using fixed order (2,1,1) as per original code
    try:
        model_arima = sm.tsa.arima.model.ARIMA(train_arima, order=(2, 1, 1))
        model_fit = model_arima.fit()
        # Predictions need to be generated for the length of the test set
        predictions = model_fit.predict(start=len(train_arima), end=len(train_arima) + len(test_arima) - 1, dynamic=False)

        if not np.isnan(predictions).all():
            error = math.sqrt(mean_squared_error(test_arima, predictions))
            cod = r2_score(test_arima, predictions)
            PREDICTIONS['ARIMA'] = np.squeeze(predictions)
        
    except Exception as e:
        print("ARIMA failed to fit: {}".format(e))
        predictions = np.zeros_like(test_arima)
        PREDICTIONS['ARIMA'] = np.squeeze(predictions)
        
    time_arima = time.time() - start_arima

    # Store results
    RESULTS.append({
        'Variabel_name': 'ARIMA',
        'RMSE': error,
        'Time(in sec)': time_arima,
        'Description': 'Order (2,1,1)',
        'R-Square value': cod
    })
    
    print("ARIMA Results: Time: {:.3f}s, RMSE: {:.3f}, R2: {:.3f}".format(time_arima, error, cod))

# #################################################
## L S T M / C N N - L S T M M O D E L S
# #################################################
def create_compile_model(model_type, units, x_train, learning_rate=0.001, cnn_filters=50, dropout=0.0):
    model = Sequential()
    
    # Helper to avoid repetitive input_shape definition
    input_shape = (x_train.shape[1], 1)
    
    if model_type == 'LSTM':
        model.add(LSTM(units=units, return_sequences=False, input_shape=input_shape))
    elif model_type == 'LSTM_50': # Kept LSTM_50 for structure, will run as 'LSTM'
        model.add(LSTM(units=units, return_sequences=False, input_shape=input_shape))
    elif model_type == 'LSTM_1': # Kept LSTM_1 for structure
        model.add(LSTM(units=units, return_sequences=False, input_shape=input_shape))
    elif model_type == 'CNN_BiLSTM_ORIGINAL': # Kept original CNN_BiLSTM for reference
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='tanh', input_shape=input_shape))
        model.add(Bidirectional(LSTM(units=units, return_sequences=False)))
        model.add(Dropout(dropout))
    elif model_type == 'LSTM_Attention_ORIGINAL': # Kept original LSTM_Attention for reference
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(SeqSelfAttention(attention_activation='relu'))
        model.add(Flatten())
    elif model_type == 'LSTM_CNN_ORIGINAL': # Kept original LSTM_CNN for reference
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
    
    # --- NEW MODELS START HERE ---
    elif model_type == 'BiLSTM_Attention_CNN_BiLSTM':
        model.add(Bidirectional(LSTM(units=units, return_sequences=True), input_shape=input_shape))
        model.add(SeqSelfAttention(attention_activation='relu'))
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(LSTM(units=units, return_sequences=False)))

    elif model_type == 'CNN_LSTM_Attention_LSTM':
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(SeqSelfAttention(attention_activation='relu'))
        model.add(LSTM(units=units, return_sequences=False))

    elif model_type == 'LSTM_Attention_CNN_BiLSTM':
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(SeqSelfAttention(attention_activation='relu'))
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(LSTM(units=units, return_sequences=False)))

    elif model_type == 'LSTM_Attention_CNN_LSTM':
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(SeqSelfAttention(attention_activation='relu'))
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units=units, return_sequences=False))
    
    elif model_type == 'LSTM_Attention_LSTM':
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(SeqSelfAttention(attention_activation='relu'))
        model.add(LSTM(units=units, return_sequences=False))

    elif model_type == 'LSTM_CNN_BiLSTM':
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Bidirectional(LSTM(units=units, return_sequences=False)))
    
    # --- Existing Complex Model (Kept for compatibility) ---
    elif model_type == 'CNN_2_LSTM_2':
        # This model uses a complex hardcoded structure, ignoring the passed units parameter
        model.add(Dense(units=128, input_shape=input_shape))
        model.add(Conv1D(filters=112, kernel_size=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=1, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dense(units=100))
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def build_fastrnn_hybrid_model(units, dataDimension, inputDims, FastCell, ADAM_OPTIMIZER):
    y = inputs = Input(shape=[int(dataDimension / inputDims), inputDims], name='input')
    y = RNN(FastCell, return_sequences=True, name='rnn')(y)
    y = Conv1D(filters=300, kernel_size=3 , padding='same', activation='relu', name='Conv1D')(y)
    y = MaxPooling1D(pool_size=1,name='MaxPooling1D')(y)
    y = Bidirectional(LSTM(units=units, return_sequences=False))(y)
    out = Dense(units=1)(y)
    
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=ADAM_OPTIMIZER, loss='mean_squared_error')
    
    return model

def run_keras_model(model_type, name, units, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims, cnn_filters=50, learning_rate=LEARNING_RATE, dropout=0.0):
    print("\n--- Running {} ---".format(name))
    
    if model_type == 'FastRNN_Hybrid':
        model = build_fastrnn_hybrid_model(units, dataDimension, inputDims, FastCell, ADAM_OPTIMIZER)
        # Reshape data for FastRNN layers
        x_train_input = reshape_for_rnn_layer(x_train)
        x_test_input = reshape_for_rnn_layer(x_test)
    else:
        # Pass x_train to create_compile_model to get the correct input shape
        model = create_compile_model(model_type, units, x_train, learning_rate, cnn_filters, dropout)
        x_train_input = x_train
        x_test_input = x_test
    
    csv_logger = CSVLogger("model_history_log_{}.csv".format(name.replace(' ', '_')), append=False)
    
    start = time.time()
    model.fit(x_train_input, y_train, batch_size=BATCH_SIZE, epochs=TOTAL_EPOCHS, 
              validation_data=(x_test_input, y_val_scaled), callbacks=[csv_logger], verbose=0)
    stop = time.time()
    
    evaluate_model(name, model, x_test_input, y_test, start, stop, "model_history_log_{}.csv".format(name.replace(' ', '_')))


# --- Data Scaling and Splitting ---

# Function to create sequence data (x_train, y_train, x_test, y_test)
def create_sequences(dataset, scaled_data, training_len, lookback, scaler):
    # Training Data
    train_data_scaled = scaled_data[0:training_len, :]
    x_train, y_train = [], []
    for i in range(lookback, len(train_data_scaled)):
        x_train.append(train_data_scaled[i-lookback:i, 0])
        y_train.append(train_data_scaled[i, 0])
    
    # Test Data (Includes lookback overlap)
    test_data_scaled = scaled_data[training_len - lookback:, :]
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

    # Validation targets are the scaled data slice
    y_val_scaled = scaled_data[training_len:training_len + len(y_test_unscaled), :]

    return x_train, y_train, x_test, y_test_unscaled, y_val_scaled


# #################################################
## M A I N E X E C U T I O N
# #################################################

def main():
    
    # --- 1. Get Filename from Command Line ---
    if len(sys.argv) < 2:
        print("Usage: python stock_prediction_models.py <input_csv_file>")
        sys.exit(1)
        
    INPUT_FILE = sys.argv[1]
    
    # --- 2. Data Loading and Preparation ---

    try:
        # Load Data using the argument
        data_1 = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File not found at '{INPUT_FILE}'.")
        sys.exit(1)
        
    print(f"Initial Data Length (from {INPUT_FILE}): {len(data_1)}")

    # Preprocessing (applied once)
    df = data_1.filter(['Close']).reset_index(drop=True)
    df = df[df['Close'].notna()]
    print("Data Length after NaN removal: {}".format(len(df)))

    # Convert Time_b for FBProphet later
    df_fb = data_1.copy()
    
    # CRITICAL: Ensure 'Time_b' column exists before accessing it
    if 'Time_b' not in df_fb.columns:
        print("Error: The CSV must contain a 'Time_b' column for FBProphet and ARIMA indexing.")
        sys.exit(1)
        
    df_fb['ds'] = pd.to_datetime(df_fb['Time_b'])
    df_fb.rename(columns={'Close': 'y'}, inplace=True)
    df_fb.index = df_fb['ds']
    df_fb = df_fb.filter(['ds', 'y'])
    df_fb = df_fb[df_fb['y'].notna()]


    # --- 3. Data Scaling and Splitting ---

    dataset = df.values
    training_data_len = math.ceil(len(dataset) - TRAINING_PERCENTAGE_CUTOFF)

    # Scale data
    global scaler # Keep scaler global for use in evaluate_model
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create sequences
    x_train, y_train, x_test, y_test, y_val_scaled = create_sequences(
        dataset, scaled_data, training_data_len, LOOKBACK_WINDOW, scaler
    )

    # --- 4. Model Configuration Parameters (Global scope for RNN models) ---
    global inputDims, hiddenDims, update_non_linearity, FastCell, ADAM_OPTIMIZER, dataDimension

    inputDims = LOOKBACK_WINDOW 
    hiddenDims = 50 
    update_non_linearity = "relu"
    # Use a dummy cell if the external file is missing, otherwise use the imported one
    try:
        from rnn import FastGRNNCellKeras, FastRNNCellKeras 
    except ImportError:
        pass # Use the placeholder classes defined earlier

    FastCell = FastRNNCellKeras(hiddenDims, update_non_linearity=update_non_linearity)
    ADAM_OPTIMIZER = Adam(learning_rate=LEARNING_RATE) 
    dataDimension = LOOKBACK_WINDOW


    # 5. Run all models
    
    # --- Models based on uploaded image ---
    
    # RNN/Hybrid Models
    run_fastrnn(x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims) # FastRNN (proposed)
    run_keras_model('LSTM', 'LSTM', 50, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims, learning_rate=0.01) # LSTM
    run_keras_model('BiLSTM_Attention_CNN_BiLSTM', 'BiLSTM_Attention_CNN_BiLSTM', 50, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims)
    run_keras_model('CNN_LSTM_Attention_LSTM', 'CNN_LSTM_Attention_LSTM', 50, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims)
    run_keras_model('LSTM_Attention_CNN_BiLSTM', 'LSTM_Attention_CNN_BiLSTM', 50, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims)
    run_keras_model('LSTM_Attention_CNN_LSTM', 'LSTM_Attention_CNN_LSTM', 50, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims)
    run_keras_model('LSTM_Attention_LSTM', 'LSTM_Attention_LSTM', 50, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims)
    run_keras_model('LSTM_CNN_BiLSTM', 'LSTM_CNN_BiLSTM', 50, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims)
    run_keras_model('FastRNN_Hybrid', 'FASTRNN_CNN_BiLSTM (proposed)', 50, x_train, y_train, x_test, y_val_scaled, y_test, FastCell, ADAM_OPTIMIZER, dataDimension, inputDims)
    
    # Statistical Models
    run_fb_prophet(df_fb, training_data_len) # FBProphet
    run_arima(df, training_data_len) # ARIMA
    
    # --- All other old models are now implicitly excluded ---
    
    # 6. Compile and Save Results
    print("\n--- Compiling Final Results ---")
    
    # Create final performance DataFrame
    df_final = pd.DataFrame(RESULTS)
    
    # Create final predictions DataFrame
    predicted_results = pd.DataFrame(PREDICTIONS)
    predicted_results.insert(0, 'Test_Data', np.squeeze(y_test))
    
    # Use a proxy for TICKER based on the input filename
    TICKER = INPUT_FILE.replace('.csv', '').split('/')[-1]
    
    # Save results to CSV
    df_final.to_csv('Performance_Results_{}.csv'.format(TICKER), index=False)
    predicted_results.to_csv('Predicted_Values_{}.csv'.format(TICKER), index=False)
    
    print("\n✅ Execution Complete.")
    print("Performance Results saved to Performance_Results_{}.csv".format(TICKER))
    print("Prediction Values saved to Predicted_Values_{}.csv".format(TICKER))
    
    # Display top results
    print("\nTop 5 Results:")
    print(df_final.sort_values(by='RMSE').head())

if __name__ == "__main__":
    # Ensure environment is set up for TensorFlow
    try:
        # Increase verbosity slightly for better debugging in complex models
        tf.get_logger().setLevel('ERROR')
        main()
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")
        sys.exit(1)