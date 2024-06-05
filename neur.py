import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import os

# Function to load CSV files
def load_csv_files():
    files = {
        '1.ushape.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\1.ushape.csv',
        '2.concerticcir1.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\2.concerticcir1.csv',
        '3.concertriccir2.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\3.concertriccir2.csv',
        '4.linearsep.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\4.linearsep.csv',
        '5.outlier.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\5.outlier.csv',
        '6.overlap.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\6.overlap.csv',
        '7.xor.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\7.xor.csv',
        '8.twospirals.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\8.twospirals.csv',
        '9.random.csv': r'C:\Users\arsha\Downloads\Multiple CSV\Multiple CSV\9.random.csv'
    }
    data_dict = {}
    for name, path in files.items():
        data_dict[name] = pd.read_csv(path, header=None)
    return data_dict

# Load data
data_dict = load_csv_files()

# Display an image
image_path = r'C:\Users\arsha\Downloads\fibaaa.png'
st.image(image_path, use_column_width=True)

# Sidebar for selecting dataset and parameters
dataset_name = st.sidebar.selectbox("Select Dataset", list(data_dict.keys()))
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.03)
activation_function = st.sidebar.selectbox("Activation Function", ["tanh", "relu", "sigmoid"])
batch_size = st.sidebar.slider("Batch Size", 1, 200, 10)
epochs = st.sidebar.slider("Epochs", 10, 1000, 200)
ratio_train_test = st.sidebar.slider("Ratio of training to test data", 0.1, 0.9, 0.5)
problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])

# Hidden layers and neurons
hidden_layers = st.sidebar.number_input("Number of Hidden Layers", min_value=1, max_value=10, value=2)
neurons_per_layer = st.sidebar.number_input("Neurons per Layer", min_value=1, max_value=10, value=2)

# Add submit button
if st.sidebar.button("Submit"):
    # Get selected dataset
    data = data_dict[dataset_name]

    # Split data into training and testing
    num_train = int(len(data) * ratio_train_test)
    train_data = data.iloc[:num_train]
    test_data = data.iloc[num_train:]

    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    # Build model
    input_layer = Input(shape=(2,))
    x = input_layer
    hidden_layers_outputs = []
    for _ in range(hidden_layers):
        x = Dense(units=neurons_per_layer, activation=activation_function)(x)
        hidden_layers_outputs.append(x)
    if problem_type == "Classification":
        output_layer = Dense(units=1, activation="sigmoid")(x)
        loss_function = "binary_crossentropy"
    else:
        output_layer = Dense(units=1, activation="linear")(x)
        loss_function = "mean_squared_error"

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=SGD(learning_rate=learning_rate), loss=loss_function, metrics=["accuracy"] if problem_type == "Classification" else ["mse"])

    # Train model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=0)

    # Plot decision regions for classification problems
    if problem_type == "Classification":
        fig, ax = plt.subplots()
        plot_decision_regions(X_train.values, y_train.values.astype(int), clf=model, ax=ax)
        st.pyplot(fig)

    # Display loss and accuracy
    st.subheader("Training Results")
    if problem_type == "Classification":
        st.write(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        st.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    else:
        st.write(f"Training MSE: {history.history['mse'][-1]:.4f}")
        st.write(f"Validation MSE: {history.history['val_mse'][-1]:.4f}")
    st.write(f"Training Loss: {history.history['loss'][-1]:.4f}")
    st.write(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")

    # Plot loss curves
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    # Plot decision surface for each neuron in the hidden layers
    for layer_index, layer_output in enumerate(hidden_layers_outputs):
        intermediate_model = Model(inputs=input_layer, outputs=layer_output)
        intermediate_output = intermediate_model.predict(X_train)

        for neuron_index in range(neurons_per_layer):
            fig, ax = plt.subplots()
            # Adjust the target values for plotting decision regions
            neuron_output = intermediate_output[:, neuron_index]
            plot_decision_regions(X_train.values, (neuron_output > 0).astype(int), clf=model, ax=ax)
            st.write(f"Decision surface for hidden layer {layer_index + 1}, neuron {neuron_index + 1}")
            st.pyplot(fig)
