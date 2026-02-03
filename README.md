<meta name="google-site-verification" content="-XQC-POJtlDPD3i2KSOxbFkSBde_Uq9obAIh_4mxTkM" />

<div align="center">

<h2><a href="https://www.arxiv.org/abs/2505.10238">MTVCrafter: 4D Motion Tokenization for Open-World Human Image Animation</a></h2>

ICLR 2026

> Official project page of **MTVCrafter**, a novel framework for general and high-quality human image animation using raw 3D motion sequences.

[Yanbo Ding](https://scholar.google.com/citations?user=r_ty-f0AAAAJ&hl=zh-CN),
[Xirui Hu](https://scholar.google.com/citations?user=-C7R25QAAAAJ&hl=zh-CN&oi=ao),
[Zhizhi Guo](https://dblp.org/pid/179/1036.html),
[Yali Wang†](https://scholar.google.com/citations?user=hD948dkAAAAJ)

[![arXiv](https://img.shields.io/badge/📖%20Paper-2408.10605-b31b1b.svg)](https://www.arxiv.org/abs/2505.10238)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/yanboding/MTVCrafter)
[![ModelScope](https://img.shields.io/badge/🤖%20ModelScope-Models-blue)](https://www.modelscope.cn/models/AI-ModelScope/MTVCrafter)
[![Project Page1](https://img.shields.io/badge/🌐%20Page-CogVideoX-brightgreen)](https://dingyanb.github.io/MTVCtafter/)
[![Project Page2](https://img.shields.io/badge/🌐%20Page-Wan2.1-orange)](https://dingyanb.github.io/MTVCrafter-/)

</div>


## 📌 ToDo List

- [x] Release **global dataset statistics** (mean / std)  
- [x] Release **4D MoT** model  
- [x] Release **MV-DiT-7B** (based on *CogVideoX-T2V-5B*)  
- [x] Release **MV-DiT-17B** (based on *Wan-2.1-I2V-14B*)
- [ ] Release a Hugging Face Demo Space


## 🔍 Abstract

Human image animation has attracted increasing attention and developed rapidly due to its broad applications in digital humans. However, existing methods rely on 2D-rendered pose images for motion guidance, which limits generalization and discards essential 3D information.  
To tackle these problems, we propose **MTVCrafter (Motion Tokenization Video Crafter)**, the first framework that directly models raw 3D motion sequences for open-world human image animation beyond intermediate 2D representations.

- We introduce **4DMoT (4D motion tokenizer)** to encode raw motion data into discrete motion tokens, preserving 4D compact yet expressive spatio-temporal information.
- Then, we propose **MV-DiT (Motion-aware Video DiT)**, which integrates a motion attention module and 4D positional encodings to effectively modulate vision tokens with motion tokens.
- The overall pipeline facilitates high-quality human video generation guided by 4D motion tokens.

MTVCrafter achieves **state-of-the-art results with an FID-VID of 6.98**, outperforming the second-best by approximately **65%**. It generalizes well to diverse characters (single/multiple, full/half-body) across various styles.

## 🎯 Motivation

![Motivation](./static/images/Motivation.png)

Our motivation is that directly tokenizing 4D motion captures more faithful and expressive information than traditional 2D-rendered pose images derived from the driven video.

## 💡 Method

![Method](./static/images/4DMoT.png)

*(1) 4DMoT*:
Our 4D motion tokenizer consists of an encoder-decoder framework to learn spatio-temporal latent representations of SMPL motion sequences,
and a vector quantizer to learn discrete tokens in a unified space.
All operations are performed in 2D space along frame and joint axes.

![Method](./static/images/MV-DiT.png)

*(2) MV-DiT*:
Based on video DiT architecture,
we design a 4D motion attention module to combine motion tokens with vision tokens.
Since the tokenization and flattening disrupted positional information,
we introduce 4D RoPE to recover the spatio-temporal relationships.
To further improve the quality of generation and generalization,
we use learnable unconditional tokens for motion classifier-free guidance.

---

## 🛠️ Installation

We recommend using a clean Python environment (Python 3.10+).

```bash
git clone https://github.com/your-username/MTVCrafter.git
cd MTVCrafter

# Create virtual environment
conda create -n mtvcrafter python=3.11
conda activate mtvcrafter

# Install dependencies
pip install -r requirements.txt
```

For models regarding:

1. **NLF-Pose Estimator**  
   Download [`nlf_l_multi.torchscript`](https://github.com/isarandi/nlf/releases) from the NLF release page.

2. **MV-DiT Backbone Models**  
   - **CogVideoX**: Download the [CogVideoX-5B checkpoint](https://huggingface.co/THUDM/CogVideoX-5b).  
   - **Wan-2-1**: Download the [Wan-2-1-14B checkpoint](https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-14B-InP) and place it under the `wan2.1/` folder.

3. **MTVCrafter Checkpoints**  
   Download the MV-DiT and 4DMoT checkpoints from [MTVCrafter on Hugging Face](https://huggingface.co/yanboding/MTVCrafter).

4. *(Optional)*
   Download the enhanced LoRA for better performance of Wan2.1_I2V_14B:  
   [`Wan2.1_I2V_14B_FusionX_LoRA.safetensors`](https://huggingface.co/vrgamedevgirl84/Wan14BT2VFusioniX/blob/main/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors)  
   Place it under the `wan2.1/` folder.
   
   Note that this LoRA is used to improve the inference speed and the details of generated video, but may occur worse motion accuracy.

---

## 🚀 Usage

To animate a human image with a given 3D motion sequence,  
you first need to prepare SMPL motion-video pairs. You can either:

- Use the provided sample data: `data/sampled_data.pkl`, or  
- Extract SMPL motion sequences from your own driving video using:

```bash
python process_nlf.py "your_video_directory"
```

This will generate a motion-video `.pkl` file under `"your_video_directory"`.

---

#### ▶️ Inference of MV-DiT-7B
```bash
python infer_7b.py \
    --ref_image_path "ref_images/human.png" \
    --motion_data_path "data/sampled_data.pkl" \
    --output_dir "inference_output"
```

#### ▶️ Inference of MV-DiT-17B (with text control)
```bash
python infer_17b.py \
    --ref_image_path "ref_images/woman.png" \
    --motion_data_path "data/sampled_data.pkl" \
    --output_dir "inference_output" \
    --prompt "The woman is dancing on the beach, waves, sunset."
```

**Arguments:**

- `--ref_image_path`: Path to the reference character image.
- `--motion_data_path`: Path to the SMPL motion sequence (.pkl format).
- `--output_dir`: Directory to save the generated video.
- `--prompt` (optional): Text prompt describing the scene or style.

---

### 🏋️‍♂️ Training 4DMoT

To train the 4DMoT tokenizer on your own dataset:

```bash
accelerate launch train_vqvae.py
```

---

## 💙 Acknowledgement
MTVCrafter is largely built upon 
[CogVideoX](https://github.com/THUDM/CogVideo), 
[Wan-2-1-Fun](https://github.com/aigc-apps/VideoX-Fun).
We sincerely acknowledge these open-source codes and models.
We also appreciate the valuable insights from the researchers at Institute of Artificial Intelligence (TeleAI), China Telecom, and Shenzhen Institute of Advanced Technology.


## 📄 Citation

If you find our work useful, please consider citing:

```bibtex
@article{ding2025mtvcrafter,
  title={MTVCrafter: 4D Motion Tokenization for Open-World Human Image Animation},
  author={Ding, Yanbo and Hu, Xirui and Guo, Zhizhi and Zhang, Chi and Wang, Yali},
  journal={arXiv preprint arXiv:2505.10238},
  year={2025}
}
```

## 📬 Contact

For questions or collaboration, feel free to reach out via GitHub Issues
or email me at 📧 yb.ding@siat.ac.cn.
