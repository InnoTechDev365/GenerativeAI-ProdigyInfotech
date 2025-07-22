# 🖼️ PRODIGY INFOTECH – Task 04  
## 🎯 Image-to-Image Translation with Pix2Pix (cGAN)

This project demonstrates how to perform **image-to-image translation** using a **Conditional Generative Adversarial Network (cGAN)** known as **Pix2Pix**, which learns a mapping from input images to output images [[3]](https://www.tensorflow.org/tutorials/generative/pix2pix?hl=ru).

---

## 🔍 Overview

Pix2Pix is a supervised learning method that uses **paired datasets** to train a generator and discriminator network to transform one type of image into another — such as converting edge maps into realistic photos or label maps into photorealistic scenes.

This task includes:
- 🧠 Understanding the **U-Net Generator + PatchGAN Discriminator** architecture
- 📦 Using TensorFlow Datasets for training
- 🤖 Training a **conditional GAN**
- 🖼 Visualizing results during and after training

---

## 💻 Running the Project

### Prerequisites
- Google Colab or local Jupyter environment
- Internet access for dataset download

### Steps
1. Open `Task04_cGAN.ipynb` in Colab.
2. Run all cells sequentially.
3. Observe generated outputs and loss curves.

---

## 🛠 Features

| Feature                  | Description |
|--------------------------|-------------|
| ✅ U-Net Generator       | Encoder-decoder with skip connections |
| ✅ PatchGAN Discriminator| Classifies realism at patch level |
| ✅ Paired Dataset        | Uses aligned image pairs for supervision |
| ✅ L1 + GAN Loss         | Balances realism and structural accuracy |

---

## 📈 Results & Observations

| Metric                 | Value                        |
|------------------------|------------------------------|
| Input Size             | 256x256                      |
| Generator Type         | U-Net                        |
| Discriminator Type     | PatchGAN                     |
| Dataset Used           | horse2zebra (from TFDS)      |

---

## 📚 References

- [TensorFlow Pix2Pix Tutorial]( https://www.tensorflow.org/tutorials/generative/pix2pix?hl=ru)
- [GeeksforGeeks: Conditional GANs]( https://www.geeksforgeeks.org/deep-learning/conditional-generative-adversarial-network/ )
- [AI Jobs: pix2pix Explained](https://aijobs.net/pix2pix-explained/ )

---

## 📝 Key Takeaways

- Pix2Pix achieves impressive results for many image-to-image tasks.
- Combining **L1 loss** with **adversarial loss** improves both quality and fidelity.
- This model can be extended to other domains like medical imaging, map generation, and artistic rendering.

---

## 🧩 Future Enhancements

- Add **CycleGAN** support for unpaired data
- Integrate **Wasserstein GAN** for more stable training
- Export and serve the model via **TensorFlow Serving** or **TFLite**
