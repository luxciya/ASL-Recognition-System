import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("sign_landmarks.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, stratify=y_enc
)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(126,)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test))

model.save("models/sign_model.keras")
joblib.dump(encoder, "models/sign_encoder.pkl")
joblib.dump(scaler, "models/sign_scaler.pkl")

print("✅ Sign model saved")
