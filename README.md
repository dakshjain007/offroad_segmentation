🛣️ Off-Road Semantic Segmentation using UNet (PyTorch)

A deep learning project for pixel-wise terrain understanding in off-road environments using a UNet architecture trained with Dice + CrossEntropy loss to achieve high IoU performance.

The model classifies each pixel of an image into terrain categories such as:

Background

Road / Track

Mud

Rocks / Obstacles

This project focuses on improving segmentation quality (IoU) rather than only pixel accuracy.

📌 Features

UNet based semantic segmentation

Dice + CrossEntropy hybrid loss

Class imbalance handling (weighted loss)

Automatic train/validation split

Learning rate scheduler

Best model checkpoint saving

Mean IoU metric evaluation

GPU support (CUDA) + CPU fallback

🧠 Model Architecture

UNet Encoder-Decoder Network

Key characteristics:

Skip connections preserve spatial details

Good performance on small datasets

Stable training for terrain segmentation

Produces sharp mask boundaries

📂 Project Structure
Offroad_Segmentation_Training_Dataset/
│── data/
│   ├── images/
│   ├── masks/
│
│── models/
│   └── unet_best.pth
│
│── dataset.py
│── model.py
│── train.py
│── metrics.py
│── README.md

⚙️ Installation

Create virtual environment:

python -m venv venv
venv\Scripts\activate


Install dependencies:

pip install torch torchvision torchaudio
pip install numpy pillow tqdm albumentations

🧾 Dataset Format

Images and masks must match filenames:

images/
    0001.jpg
    0002.jpg

masks/
    0001.png
    0002.png

Mask Requirements (IMPORTANT)

Masks must contain class indices, not colors:

Class	Pixel Value
Background	0
Road	1
Mud	2
Rock	3

NOT allowed:

0, 85, 170, 255  ❌
RGB colored masks ❌

🚀 Training

Run training:

python train.py


During training you will see:

Epoch 12
Train Loss: 0.43 | Train IoU: 0.72
Val   Loss: 0.39 | Val   IoU: 0.75
💾 Best model saved!


Best model saved at:

models/unet_best.pth

📊 Metrics

We evaluate using Mean Intersection over Union (mIoU):

𝐼
𝑜
𝑈
=
𝐼
𝑛
𝑡
𝑒
𝑟
𝑠
𝑒
𝑐
𝑡
𝑖
𝑜
𝑛
𝑈
𝑛
𝑖
𝑜
𝑛
IoU=
Union
Intersection
	​


Why IoU?

Pixel accuracy can be misleading

IoU measures actual shape overlap

Industry standard for segmentation

🔧 Training Details
Parameter	Value
Batch Size	4
Epochs	80
Optimizer	AdamW
Learning Rate	3e-5
Loss	Dice + CrossEntropy
Scheduler	ReduceLROnPlateau
Metric	Mean IoU
📈 Expected Performance
Epoch	IoU
5	~0.50
15	~0.65
30	~0.75
60+	~0.85+
🧪 Future Improvements

DeepLabV3+ backbone

Attention UNet

Real-time inference optimization

Temporal smoothing (video segmentation)

Domain adaptation for new terrains

👨‍💻 Author

Daksh Jain

AI / Computer Vision Project — Off-Road Terrain Understanding

📜 License

This project is open-source for educational and research purposes.

If you want, I can also write a GitHub description + project tags so your repo looks professional and searchable.