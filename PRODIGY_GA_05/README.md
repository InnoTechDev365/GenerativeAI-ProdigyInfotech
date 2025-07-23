# 🎨 PRODIGY INFOTECH – Task 05  
## 🎯 Neural Style Transfer

This project demonstrates **Neural Style Transfer (NST)** — a deep learning technique that blends the **content of one image** with the **artistic style of another**, producing a unique image that looks like a piece of art.

NST leverages convolutional neural networks (CNNs) to separate and recombine content and style from input images. This implementation uses a pre-trained model from **TensorFlow Hub** to apply style transfer efficiently.

---

## 🔍 Overview

This task involves:
- 🖼 Blending **content** and **style** images
- 🧠 Using a **pre-trained model** from TensorFlow Hub
- 🧩 Modular preprocessing and post-processing
- 🖼 Visualizing results with `matplotlib`
- 🧪 Testing with customizable inputs

---

## 💻 Running the Project

### Prerequisites
- Google Colab or local Jupyter environment
- Internet access for image and model downloads

### Steps
1. Open `Task05_Style_Transfer.ipynb` in Colab
2. Run all cells sequentially
3. Replace default image URLs with your own if desired
4. Observe the stylized output image

---

## 🛠 Features

| Feature                  | Description |
|--------------------------|-------------|
| ✅ Pre-trained Model     | Uses TensorFlow Hub's fast style transfer model |
| ✅ Customizable Inputs   | Supports any content and style image URLs |
| ✅ GPU Acceleration      | Automatically uses GPU in Colab |
| ✅ Image Display         | Integrated visualization |
| ✅ Lightweight           | Fast inference, minimal dependencies |

---

## 📈 Sample Results

| Content Image | Style Image | Stylized Output |
|---------------|-------------|------------------|
| ![Content](content.jpg) | ![Style](style.jpg) | ![Output](stylized_output.png) |

---

## 📚 References

- [TensorFlow Hub Style Transfer Guide](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)
- [Neural Style Transfer with TensorFlow](https://www.tensorflow.org/tutorials/generative/style_transfer)
- [Artistic Style Transfer with Magenta](https://magenta.tensorflow.org/arbitrary-image-stylization)

---

## 📝 Key Takeaways

- Neural Style Transfer is a powerful example of transfer learning in generative AI.
- TensorFlow Hub provides fast, accessible models for creative applications.
- This system can be extended to batch processing, animation, or web demos.

---

## 🧩 Future Enhancements

- Add **batch processing** for multiple image pairs
- Implement **interactive UI** with Gradio or Streamlit
- Export and deploy the model via **TensorFlow Serving**
