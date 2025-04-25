import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import zipfile
from pathlib import Path
import shutil

zip_path = r"C:\Users\tsvin\Downloads\archive (17).zip"
extract_path = r"C:\Users\tsvin\Downloads\extracted_dataset"

# Create the extraction folder if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete!")

dataset_path = r"C:\Users\tsvin\Downloads\extracted_dataset\dataset"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_gen.class_indices.keys())

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
output = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen, validation_data=val_gen, epochs=0)

model.save("model.h5")

import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Eye Disease Detection", layout="centered")
st.title("👁️ Eye Disease Detection with EfficientNet")
st.write("Upload a retinal image to detect eye diseases like **Cataract**, **Glaucoma**, or **Diabetic Retinopathy**.")

@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()
class_names = ['cataract', 'diabetic retinopathy', 'glaucoma', 'normal']

uploaded_file = st.file_uploader("Upload an eye image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Eye Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_name = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    if class_name == 'normal':
        st.success(f"✅ The eye is **NOT infected**. (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"⚠ The eye is **INFECTED with {class_name.upper()}**. (Confidence: {confidence:.2f}%)")

from pyngrok import ngrok

# Replace with YOUR authtoken
ngrok.set_auth_token("2wAz41kdcPzKABo6VspbZS4Ww8S_2ZFeBARFCZnqSV36BP9ky")

# Start Streamlit in background using the current file
os.system(f"streamlit run {__file__} &")

import time
time.sleep(5)

# Connect HTTP tunnel using correct `bind_tls` config
public_url = ngrok.connect(8501, "http")  # ✅ This fixes the tunnel config error
print("🚀 Your Streamlit app is live at:", public_url)

