# ✅ Generative AI Internship Final Project   

---

## 🌟 Overview  
This repository contains **five completed generative AI tasks** from my internship at Prodigy InfoTech. Each task demonstrates a different aspect of generative modeling, including text generation with GPT-2, image generation with pre-trained models, Markov chains, cGANs (Pix2Pix), and neural style transfer.  

**Skills Developed During Internship**:  
- **Prompt Engineering**: Crafting effective prompts for text/image generation.  
- **Manual Tuning**: Optimizing hyperparameters (e.g., beam search, learning rates).  
- **Model Deployment**: Ensuring notebooks work seamlessly in Google Colab.  

All tasks are fully functional and serve as a **starting point for future interns**, though they will need to finalize their own implementations based on project requirements.  

---

## 📁 Tasks Breakdown  

### **1. Task 01: Text Generation with GPT-2**  
**Objective**: Train and fine-tune **GPT-2** to generate contextually relevant text based on prompts.  

**Key Features**:  
✅ Pre-trained model loading from Hugging Face.  
✅ Beam search decoding (`num_beams=5`, `no_repeat_ngram_size=2`).  
✅ Prompt-based text generation (e.g., "I love GenAI Internship!").  
✅ Decoding of generated tokens into human-readable text.  

**Example Code**:  
```python  
from transformers import GPT2Tokenizer, GPT2LMHeadModel  

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  
model = GPT2LMHeadModel.from_pretrained("gpt2")  

prompt = "I love Generative AI!"  
inputs = tokenizer(prompt, return_tensors="pt")  
outputs = model.generate(**inputs, num_beams=5, no_repeat_ngram_size=2)  
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  
```  

**References**:  
- [Hugging Face Transformers](https://huggingface.co/transformers)  
- [GPT-2 Text Generation](https://www.tensorflow.org/tutorials/generative/text_generation)  

---

### **2. Task 02: Image Generation with Pre-Trained Models**  
**Objective**: Use **DALL-E-mini** and **Stable Diffusion** to create images from text prompts.  

**Key Features**:  
✅ Fast generation with DALL-E-mini.  
✅ High-quality images with Stable Diffusion.  
✅ Aspect ratio adjustments using VQGAN+CLIP.  
✅ Performance optimizations (XLA compilation, mixed precision).  

**Example Code (Stable Diffusion)**:  
```python  
from diffusers import StableDiffusionPipeline  
import torch  

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)  
pipe.to("cuda")  

image = pipe("A futuristic city at sunset").images[0]  
image.save("generated_image.png")  
```  

**References**:  
- [DALL-E Mini Colab Notebook](https://colab.research.google.com/github/robgon-art/e-dall-e/blob/main/DALL_E_Mini_Image_Generator.ipynb)  
- [Creating Digital Art with Varying Aspect Ratios](https://towardsdatascience.com/e-dall-e-creating-digital-art-with-varying-aspect-ratios-5de260f4713d)  

---

### **3. Task 03: Text Generation with Markov Chains**  
**Objective**: Build a **Markov Chain** to generate text by learning word/character transitions.  

**Key Features**:  
✅ Configurable n-gram order (unigrams, bigrams, trigrams).  
✅ Word-level and character-level modeling.  
✅ Visualization of next-token frequency distributions.  
✅ Runs seamlessly in Google Colab.  

**Example Code**:  
```python  
import markovify  

with open("shakespeare.txt", "r") as f:  
    text = f.read()  

text_model = markovify.Text(text, state_size=2)  
print(text_model.make_sentence())  # Generates a Shakespearean-style sentence  
```  

**References**:  
- [Text Generation with Markov Chains](https://towardsdatascience.com/text-generation-with-markov-chains-an-introduction-to-using-markovify-742e6680dc33)  

---

### **4. Task 04: Image-to-Image Translation with cGAN (Pix2Pix)**  
**Objective**: Implement **conditional GANs (cGANs)** using **Pix2Pix** for paired image-to-image translation (e.g., horse to zebra).  

**Key Features**:  
✅ U-Net generator and PatchGAN discriminator.  
✅ Training on the **CycleGAN horse2zebra** dataset.  
✅ Applications in style transfer, artistic filters, and domain adaptation.  

**Example Code**:  
```python  
from diffusers import StableDiffusionPipeline  

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)  
pipe.to("cuda")  

image = pipe("A mysterious dark stranger in ancient Egypt").images[0]  
image.save("generated_image.png")  
```  

**References**:  
- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004)  
- [TensorFlow Pix2Pix Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix)  

---

### **5. Task 05: Neural Style Transfer**  
**Objective**: Apply the artistic style of one image to another using **Magenta’s Arbitrary Image Stylization model**.  

**Key Features**:  
✅ Upload any image format (`.jpg`, `.png`, `.webp`).  
✅ Auto-saves and downloads the stylized output.  
✅ Non-blocking visualization (fails gracefully).  

**Example Code**:  
```python  
import tensorflow as tf  
import tensorflow_hub as hub  

hub_module = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")  

def load_image(path):  
    img = tf.io.read_file(path)  
    img = tf.image.decode_image(img, channels=3)  
    img = tf.image.resize(img, (512, 512))  
    return img[tf.newaxis, :]/255.0  

content_image = load_image("content.jpg")  
style_image = load_image("style.jpg")  
stylized_image = hub_module(content_image, style_image)[0].numpy()  
```  

**References**:  
- [TensorFlow Hub - Magenta](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)  
- [Neural Style Transfer Tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)  

---

## 🛠️ Tools & Libraries  
- **Python 3.x**  
- **TensorFlow / Keras / KerasCV**  
- **Hugging Face Transformers**  
- **Markovify**  
- **Google Colab** (for GPU acceleration)  
- **Pillow / Matplotlib** (image/text visualization)  

---

## 📂 Repository Structure  
```
GenerativeAI-ProdigyInternship/
│
├── PRODIGY_GA_01/  
│   ├── Task01_GPT2.ipynb  
│   └── README.md  
│
├── PRODIGY_GA_02/  
│   ├── Task02_ImageGeneration.ipynb  
│   └── README.md  
│
├── PRODIGY_GA_03/  
│   ├── Task03_MarkovChain.ipynb  
│   └── README.md  
│
├── PRODIGY_GA_04/  
│   ├── Task04_cGAN_Pix2Pix.ipynb  
│   └── README.md  
│
├── PRODIGY_GA_05/  
│   ├── Task05_StyleTransfer.ipynb  
│   └── README.md  
│
└── README.md (Project-level documentation)  
```  

---

## 🧩 For Future Interns  
- **Use these notebooks as templates** for understanding generative AI workflows.  
- **Finalize each task** by:  
  - Replacing placeholder datasets/paths with your own.  
  - Tuning hyperparameters (e.g., `num_beams`, `learning_rate`).  
  - Adding custom visualizations or metrics.  
- **Focus on prompt engineering** and manual tuning to improve output quality.  

---

## 📝 Key Takeaways  
- **Stable Diffusion** excels at high-quality image generation with proper prompt engineering.  
- **Markov Chains** are simple but effective for basic text generation.  
- **GPT-2** provides contextually rich text outputs with careful decoding strategies.  
- **Neural Style Transfer** and **cGANs** demonstrate the power of conditional generation in creative applications.  

---

## 🚀 Getting Started  
1. Open each notebook in **Google Colab** (ensure GPU runtime is enabled).  
2. Follow the step-by-step instructions in the `README.md` for each task.  
3. Upload your own datasets/images for customization.  

---

## 📦 Final Notes  
- **All tasks are functional and tested**, but interns must finalize their own versions.  
- **Performance optimizations** (XLA, mixed precision) are included for efficiency.  
- **Results are saved/downloadable** even if visualization fails.  

---

## 📚 References  
- [Prodigy InfoTech Internship](https://prodigyinfotech.dev/)  
- [Stable Diffusion (KerasCV)](https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion)  
- [DALL-E Mini Colab Notebook](https://colab.research.google.com/github/robgon-art/e-dall-e/blob/main/DALL_E_Mini_Image_Generator.ipynb)  
- [Text Generation with Markov Chains](https://towardsdatascience.com/text-generation-with-markov-chains-an-introduction-to-using-markovify-742e6680dc33)  
- [Neural Style Transfer (TensorFlow)](https://www.tensorflow.org/tutorials/generative/style_transfer)  

---  
Let this repository inspire your journey in generative AI! 🌟
