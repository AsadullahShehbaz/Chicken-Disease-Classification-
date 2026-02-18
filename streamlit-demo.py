import streamlit as st
from PIL import Image
import os
from src.cnnClassifier.pipeline.predict import PredictionPipeline

# Page config
st.set_page_config(
    page_title="Chicken Disease Classifier",
    page_icon="🐔",
    layout="centered"
)

# Title and description
st.title("🐔 Chicken Disease Classifier")
st.write("Upload an image of a chicken to detect diseases")

# Disease info
with st.expander("ℹ️ About the diseases"):
    st.write("""
    - **Coccidiosis**: Parasitic disease affecting intestines
    - **Healthy**: No disease detected
    - **New Castle Disease**: Viral infection affecting respiratory system
    - **Salmonella**: Bacterial infection
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Save uploaded file temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predict button
    if st.button("🔍 Predict Disease", type="primary"):
        with st.spinner("Analyzing image..."):
            # Run prediction
            pipeline = PredictionPipeline(filename=temp_path)
            result = pipeline.predict()
            
            # Extract prediction
            prediction = result[0]["image"]
            
            # Display result with styling
            st.success("Analysis Complete!")
            
            # Color code results
            if prediction == "Healthy":
                st.success(f"### ✅ Result: {prediction}")
            else:
                st.error(f"### ⚠️ Result: {prediction}")
            
            # Show confidence or additional info
            st.info("💡 Tip: For accurate results, ensure good lighting and clear image quality")
        
        # Cleanup temp file
        os.remove(temp_path)

else:
    st.info("👆 Please upload an image to get started")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow")