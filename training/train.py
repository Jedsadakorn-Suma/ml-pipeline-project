import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri("file:../mlruns")
mlflow.tensorflow.autolog(log_models=True)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

with mlflow.start_run(run_name="fashion-mnist-baseline"):
    model.fit(x_train, y_train, epochs=5, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    model.save("../models/saved_model")
