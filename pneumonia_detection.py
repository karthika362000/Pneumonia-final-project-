import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from sklearn.utils import class_weight
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import tempfile
import shutil

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f5f9fc;
}
.stButton>button {
    background-color: #4a89dc;
    color: white;
    border-radius: 8px;
    padding: 12px 28px;
    border: none;
    font-weight: bold;
    transition: all 0.3s;
}
.stButton>button:hover {
    background-color: #3a70c2;
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.stFileUploader>div>div {
    border: 2px dashed #4a89dc;
    border-radius: 10px;
    padding: 30px;
    background-color: #f8fafc;
}
.stAlert {
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.stProgress>div>div>div {
    background-color: #4a89dc;
}
.stMarkdown h1 {
    color: #2c3e50;
    border-bottom: 2px solid #4a89dc;
    padding-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# Enhanced CNN Model
def create_enhanced_model(input_shape=(224, 224, 3)):
    model = Sequential([
        # First Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        
        # Second Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.4),
        
        # Third Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.5),
        
        # Classifier
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                 loss='binary_crossentropy',
                 metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])
    return model

# Data Preparation
def prepare_data(zip_file, extract_to):
    if not zip_file:
        return False, "No file uploaded"
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Validate folder structure
        required_folders = ['NORMAL', 'PNEUMONIA']
        for folder in required_folders:
            if not os.path.exists(os.path.join(extract_to, folder)):
                return False, f"Missing {folder} directory"
            if len(os.listdir(os.path.join(extract_to, folder))) == 0:
                return False, f"No images in {folder} directory"
                
        return True, "Data prepared successfully"
    except Exception as e:
        return False, f"Error processing ZIP file: {str(e)}"

# Training Function
def train_model(train_dir, val_dir, epochs=30, batch_size=32):
    # Advanced Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        shuffle=False
    )
    
    # Calculate class weights
    classes = train_generator.classes
    class_weights = class_weight.compute_class_weight('balanced',
                                                   classes=np.unique(classes),
                                                   y=classes)
    class_weights = dict(enumerate(class_weights))
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=8, mode='max', restore_best_weights=True),
        ModelCheckpoint('best_pneumonia_model.h5', monitor='val_auc', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]
    
    # Create and train model
    model = create_enhanced_model()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // batch_size),
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    return model, history

# Load model
@st.cache_resource
def load_saved_model():
    if os.path.exists('best_pneumonia_model.h5'):
        return load_model('best_pneumonia_model.h5')
    return None

# Preprocess image
def preprocess_image(img, target_size=(224, 224)):
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Main App
def main():
    st.title("🩺 Advanced Pneumonia Detection System")
    st.markdown("A deep learning system for accurate detection of pneumonia from chest X-ray images")
    
    tab1, tab2 = st.tabs(["Train Model", "Detect Pneumonia"])
    
    with tab1:
        st.header("Model Training")
        st.info("""
        **Instructions:**
        1. Prepare ZIP files containing:
           - `NORMAL/` folder with normal X-rays
           - `PNEUMONIA/` folder with pneumonia X-rays
        2. Upload training and validation sets
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            train_zip = st.file_uploader("Training Data (ZIP)", type="zip")
        with col2:
            val_zip = st.file_uploader("Validation Data (ZIP)", type="zip")
        
        epochs = st.slider("Number of Epochs", 10, 50, 30)
        batch_size = st.slider("Batch Size", 16, 64, 32)
        
        if st.button("Start Training") and train_zip and val_zip:
            with st.spinner("Preparing data..."):
                train_path = tempfile.mkdtemp()
                val_path = tempfile.mkdtemp()
                
                train_ok, train_msg = prepare_data(train_zip, train_path)
                val_ok, val_msg = prepare_data(val_zip, val_path)
                
                if not train_ok or not val_ok:
                    st.error(f"Data preparation failed:\nTraining: {train_msg}\nValidation: {val_msg}")
                    shutil.rmtree(train_path)
                    shutil.rmtree(val_path)
                    return
                
            with st.spinner("Training model (this may take a while)..."):
                try:
                    model, history = train_model(train_path, val_path, epochs=epochs, batch_size=batch_size)
                    
                    st.success("✅ Training completed successfully!")
                    st.subheader("Training Performance")
                    
                    # Plot training history
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Accuracy
                    ax1.plot(history.history['accuracy'], label='Train')
                    ax1.plot(history.history['val_accuracy'], label='Validation')
                    ax1.set_title('Model Accuracy')
                    ax1.set_xlabel('Epoch')
                    ax1.legend()
                    
                    # AUC
                    ax2.plot(history.history['auc'], label='Train')
                    ax2.plot(history.history['val_auc'], label='Validation')
                    ax2.set_title('Model AUC')
                    ax2.set_xlabel('Epoch')
                    ax2.legend()
                    
                    st.pyplot(fig)
                    
                    # Show final metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final Training Accuracy", f"{history.history['accuracy'][-1]:.2%}")
                    with col2:
                        st.metric("Final Validation Accuracy", f"{history.history['val_accuracy'][-1]:.2%}")
                    with col3:
                        st.metric("Validation AUC", f"{history.history['val_auc'][-1]:.3f}")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                finally:
                    shutil.rmtree(train_path)
                    shutil.rmtree(val_path)
    
    with tab2:
        st.header("Pneumonia Detection")
        
        model = load_saved_model()
        if model is None:
            st.warning("⚠️ No trained model found. Please train a model first.")
            return
        
        uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded X-ray", width=300)
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    try:
                        processed_img = preprocess_image(img)
                        prediction = model.predict(processed_img)
                        confidence = prediction[0][0] * 100
                        
                        # Display results with clinical guidance
                        if prediction[0][0] > 0.65:  # Higher threshold for pneumonia
                            st.error(f"🚨 **Pneumonia Detected** ({confidence:.1f}% confidence)")
                            st.markdown("""
                            **Clinical Recommendation:**  
                            - Immediate medical consultation recommended  
                            - Consider chest CT for confirmation  
                            - Antibiotic therapy if bacterial etiology suspected  
                            """)
                        elif prediction[0][0] > 0.4:  # Uncertain range
                            st.warning(f"⚠️ **Suspicious Findings** ({confidence:.1f}% confidence)")
                            st.markdown("""
                            **Clinical Note:**  
                            - Findings not definitive for pneumonia  
                            - Recommend clinical correlation  
                            - Follow-up imaging may be needed  
                            """)
                        else:
                            st.success(f"✅ **Normal Findings** ({(100 - confidence):.1f}% confidence)")
                            st.markdown("""
                            **Clinical Note:**  
                            - No radiographic evidence of pneumonia  
                            - Consider alternative diagnoses if symptoms persist  
                            """)
                        
                        # Confidence visualization
                        st.subheader("Diagnostic Confidence")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        
                        # Create gradient colormap
                        cmap = plt.get_cmap('RdYlGn')
                        norm = plt.Normalize(0, 100)
                        ax.barh(0, 100, color=cmap(norm(100)), alpha=0.3)
                        
                        # Add threshold markers
                        ax.axvline(40, color='orange', linestyle='--')
                        ax.axvline(65, color='red', linestyle='--')
                        
                        # Add current prediction marker
                        ax.plot(confidence, 0, 'ko', markersize=10)
                        ax.text(confidence+2, 0.1, f'{confidence:.1f}%', va='center')
                        
                        ax.set_xlim(0, 100)
                        ax.set_yticks([])
                        ax.set_xlabel('Pneumonia Probability (%)')
                        ax.set_title('Diagnostic Scale (Green: Normal, Red: Pneumonia)')
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    main()