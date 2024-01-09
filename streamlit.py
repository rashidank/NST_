import streamlit as st
import tensorflow as tf
import nst  # Assuming your neural style transfer code is in nst.py

# Function to perform neural style transfer and display the result
def run_style_transfer(content_path, style_path):
    # Perform style transfer
    output_image, _ = nst.style_transfer(content_path, style_path)
    
    # Convert TensorFlow tensor to PIL Image
    output_image_pil = tf.keras.preprocessing.image.array_to_img(output_image[0])

    # Display the content and style images
    st.image([content_path, style_path], caption=['Content Image', 'Style Image'], width=300)

    # Display the resulting stylized image
    st.image(output_image_pil, caption='Stylized Image', width=300)

# Streamlit UI
st.title("Neural Style Transfer with Streamlit")

# File uploader for content image
content_image = st.file_uploader("Choose a content image", type=["jpg", "jpeg", "png"])

# File uploader for style image
style_image = st.file_uploader("Choose a style image", type=["jpg", "jpeg", "png"])

# Check if both images are uploaded
if content_image and style_image:
    # Perform style transfer and display the result
    run_style_transfer(content_image, style_image)
else:
    st.warning("Please upload both content and style images.")

# Streamlit app run
if __name__ == '__main__':
    st.run_app()
