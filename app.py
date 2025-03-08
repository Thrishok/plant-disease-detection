import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Set page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Load model and class indices
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'best_model (3).h5')
    class_indices_path = os.path.join(os.path.dirname(__file__), 'class_indices.npy')
    
    model = tf.keras.models.load_model(model_path)
    model.compile()  # Fix metric warnings
    class_indices = np.load(class_indices_path, allow_pickle=True).item()
    return model, class_indices

model, class_indices = load_model()

# Remedies database
REMEDIES = {
    'Potato___healthy': {
        'description': 'Healthy potato plant',
        'remedies': [
            'No action needed',
            'Maintain good growing conditions'
        ]
    },
    'Potato___Early_blight': {
        'description': 'Fungal disease causing dark spots on leaves',
        'remedies': [
            'Remove and destroy infected leaves',
            'Apply copper-based fungicides',
            'Practice crop rotation'
        ]
    },
    'Tomato_Septoria_leaf_spot': {
        'description': 'Fungal disease causing small dark spots with light centers on leaves',
        'remedies': [
            'Remove affected leaves',
            'Use fungicides like chlorothalonil',
            'Avoid overhead watering'
        ]
    },
    'Tomato__Tomato_mosaic_virus': {
        'description': 'Viral disease causing mottled, mosaic-like leaf patterns',
        'remedies': [
            'Remove and destroy infected plants',
            'Control aphids and other insect vectors',
            'Use resistant tomato varieties'
        ]
    },
    'Pepper__bell___healthy': {
        'description': 'Healthy bell pepper plant',
        'remedies': [
            'No action needed',
            'Ensure proper watering and nutrition'
        ]
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'description': 'Tiny pests causing yellowing and webbing on leaves',
        'remedies': [
            'Spray plants with neem oil or insecticidal soap',
            'Introduce predatory mites',
            'Increase humidity to deter mites'
        ]
    },
    'Tomato__Target_Spot': {
        'description': 'Fungal disease causing brown spots with yellow halos on leaves',
        'remedies': [
            'Apply copper-based fungicides',
            'Remove infected leaves',
            'Improve air circulation around plants'
        ]
    },
    'Tomato_healthy': {
        'description': 'Healthy tomato plant',
        'remedies': [
            'No action needed',
            'Maintain good growing conditions'
        ]
    },
    'Pepper__bell___Bacterial_spot': {
        'description': 'Bacterial disease causing water-soaked spots on leaves and fruit',
        'remedies': [
            'Remove infected plants',
            'Apply copper sprays',
            'Use disease-resistant varieties'
        ]
    },
    'Tomato_Late_blight': {
        'description': 'Serious fungal disease causing brown, water-soaked lesions',
        'remedies': [
            'Remove and destroy affected plants',
            'Apply fungicides containing chlorothalonil',
            'Ensure proper air circulation'
        ]
    },
    'Potato___Late_blight': {
        'description': 'Severe fungal disease causing dark lesions on leaves and tubers',
        'remedies': [
            'Use resistant potato varieties',
            'Apply fungicides like chlorothalonil or copper-based sprays',
            'Avoid excessive moisture'
        ]
    },
    'Tomato_Early_blight': {
        'description': 'Fungal disease causing dark, concentric rings on leaves',
        'remedies': [
            'Apply fungicides such as copper-based sprays',
            'Remove affected leaves',
            'Use mulch to prevent soil splash'
        ]
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'description': 'Viral disease causing leaf curling and yellowing',
        'remedies': [
            'Remove and destroy infected plants',
            'Control whiteflies, the main vector',
            'Use virus-resistant tomato varieties'
        ]
    },
    'Tomato_Bacterial_spot': {
        'description': 'Bacterial disease causing small, water-soaked spots on leaves and fruit',
        'remedies': [
            'Remove affected plants',
            'Apply copper-based sprays',
            'Use disease-free seeds'
        ]
    },
    'Tomato_Leaf_Mold': {
        'description': 'Fungal disease causing yellowing and mold growth on leaves',
        'remedies': [
            'Improve air circulation',
            'Apply fungicides like copper-based sprays',
            'Avoid overhead watering'
        ]
    }
}

def predict_image(img):
    """Process image and return prediction results"""
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred[0])
    class_name = list(class_indices.keys())[class_idx]
    confidence = np.max(pred[0])
    
    return class_name, confidence, REMEDIES.get(class_name, {
        'description': 'No information available',
        'remedies': ['Consult with an agricultural expert']
    })

def main():
    """Main application interface"""
    st.title("🌱 Plant Disease Detection & Remedies")
    st.markdown("---")
    
    with st.container():
        uploaded_file = st.file_uploader(
            "Upload a plant leaf image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            class_name, confidence, remedy = predict_image(image)
            
            # Display results
            st.subheader("🔍 Analysis Results")
            st.markdown(f"""
                **Predicted Disease:**  
                {class_name.replace('___', ' ').title()}
            """)
            
            # Display remedies
            st.subheader("🩺 Recommended Remedies")
            st.markdown(f"**Description:** {remedy['description']}")
            st.markdown("**Treatment Recommendations:**")
            for i, treatment in enumerate(remedy['remedies'], 1):
                st.markdown(f"{i}. {treatment}")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()