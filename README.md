# 🎨 Neural Style Transfer with TensorFlow

## 📝 Overview
This project demonstrates **Neural Style Transfer (NST)** — a deep learning technique that blends the **content of one image** with the **style of another**, producing a unique image that looks like a piece of art.

Implemented using **TensorFlow** and **TensorFlow Hub**, the model takes a content image (e.g., a photograph) and a style image (e.g., a painting by Kandinsky) and generates a new image combining both.

---

## ✨ Key Features
✅ Load and preprocess content and style images  
✅ Apply a pre-trained NST model from TensorFlow Hub  
✅ Convert output tensors back to images for visualization  
✅ Experiment with different content-style combinations  
✅ Fully executable in Google Colab

---

## 🚀 How It Works

1. **Load Images**  
   - `content_image`: A regular photograph (e.g., Labrador dog)  
   - `style_image`: A piece of artwork (e.g., Kandinsky painting)

2. **Preprocess Images**  
   - Resize, normalize, and format for model input

3. **Apply Style Transfer**  
   - Use a pre-trained model from `tensorflow_hub` to apply style onto content

4. **Post-process & Display**  
   - Convert the output tensor into an image and display the result

---

## 🧪 Sample Output

| Content Image | Style Image | Stylized Output |
|---------------|-------------|-----------------|
| 🐶 Labrador   | 🖌️ Kandinsky | 🎨 Dog in Kandinsky Style |

---

## 🛠 Tech Stack
- Python 3  
- TensorFlow 2.x  
- TensorFlow Hub  
- Google Colab  
- NumPy & Matplotlib

---

## 📂 Usage

### 📁 Open in Google Colab
1. Upload the notebook `Task05_Style_Transfer.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run all cells in order
3. Replace or add your own images if desired

### 🖼 Change Input Images
To try different images:
```python
content_path = tf.keras.utils.get_file('your_photo.jpg', 'image_url')
style_path = tf.keras.utils.get_file('your_artwork.jpg', 'style_url')
