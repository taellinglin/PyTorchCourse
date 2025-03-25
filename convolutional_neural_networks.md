l🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning about **Convolution** in Neural Networks! 🧠🖼️  

## 🤔 What is Convolution?  
Convolution helps computers **understand pictures** by looking at **patterns** instead of exact positions! 🖼️🔍  

Imagine you have **two images** that look almost the same, but one is a little **moved**.  
A computer might think they are totally **different**! 😲  
**Convolution fixes this problem!** ✅  

---

## 🛠️ How Convolution Works  

We use something called a **kernel** (a small filter 🔲) that slides over an image.  
It **checks different parts** of the picture and creates a new image called an **activation map**!  

1️⃣ The **image** is a grid of numbers 🖼️  
2️⃣ The **kernel** is a small grid 🔳 that moves across the image  
3️⃣ It **multiplies** numbers in the image with the numbers in the kernel ✖️  
4️⃣ The results are **added together** ➕  
5️⃣ We move to the next spot and **repeat!** 🔄  
6️⃣ The final result is the **activation map** 🎯  

---

## 📏 How Big is the Activation Map?  

The size of the **activation map** depends on:  
- **M (image size)** 📏  
- **K (kernel size)** 🔳  
- **Stride** (how far the kernel moves) 👣  

Formula:  
```
New size = (Image size - Kernel size) + 1
```

Example:  
- **4×4 image** 📷  
- **2×2 kernel** 🔳  
- Activation map = **3×3** ✅  

---

## 👣 What is Stride?  

Stride is **how far** the kernel moves each time!  
- **Stride = 1** ➝ Moves **one step** at a time 🐢  
- **Stride = 2** ➝ Moves **two steps** at a time 🚶‍♂️  
- **Bigger stride** = **Smaller** activation map! 📏  

---

## 🛑 What is Zero Padding?  

Sometimes, the kernel **doesn’t fit** perfectly in the image. 😕  
So, we **add extra rows and columns of zeros** around the image! 0️⃣0️⃣0️⃣  

This makes sure the **kernel covers everything**! ✅  

Formula:  
```
New Image Size = Old Size + 2 × Padding
```

---

## 🎨 What About Color Images?  

For **black & white** images, we use **Conv2D** with **one channel** (grayscale). 🌑  
For **color images**, we use **three channels** (Red, Green, Blue - RGB)! 🎨🌈  

---

## 🏆 Summary  

✅ Convolution helps computers **find patterns** in images!  
✅ We use a **kernel** to create an **activation map**!  
✅ **Stride & padding** change how the convolution works!  
✅ This is how computers **"see"** images! 👀🤖  

---

🎉 **Great job!** Now, let’s try convolution in the lab! 🏗️🤖✨  

-----------------------------------------------------------------

🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning about **Activation Functions** and **Max Pooling**! 🚀🔢  

## 🤖 What is an Activation Function?  

Activation functions help a neural network **decide** what’s important! 🧠  
They change the values in the activation map to **help the model learn better**.  

---

## 🔥 Example: ReLU Activation Function  

1️⃣ We take an **input image** 🖼️  
2️⃣ We apply **convolution** to create an **activation map** 📊  
3️⃣ We apply **ReLU (Rectified Linear Unit)**:  
   - **If a value is negative** ➝ Change it to **0** ❌  
   - **If a value is positive** ➝ Keep it ✅  

### 🛠 Example Calculation  

| Before ReLU  | After ReLU  |
|-------------|------------|
| -4  | 0  |
|  0  | 0  |
|  4  | 4  |

All **negative numbers** become **zero**! ✨  

In PyTorch, we apply the ReLU function **after convolution**:  

```python
import torch.nn.functional as F

output = F.relu(conv_output)
```

---

## 🌊 What is Max Pooling?  

Max Pooling helps the network **focus on important details** while making images **smaller**! 📏🔍  

### 🏗 How It Works  

1️⃣ We **divide** the image into small regions (e.g., **2×2** squares)  
2️⃣ We **keep only the largest value** in each region  
3️⃣ We **move the window** and repeat until we’ve covered the whole image  

### 📊 Example: 2×2 Max Pooling  

| Before Pooling | After Pooling |
|--------------|--------------|
| 1, **6**, 2, 3 | **6**, **8**  |
| 5, **8**, 7, 4 | **9**, **7**  |
| **9**, 2, 3, **7** | |

**Only the biggest number** in each section is kept! ✅  

---

## 🏆 Why Use Max Pooling?  

✅ **Reduces image size** ➝ Makes training faster! 🚀  
✅ **Ignores small changes** in images ➝ More stable results! 🔄  
✅ **Helps find important features** in the picture! 🖼️  

In PyTorch, we apply **Max Pooling** like this:  

```python
import torch.nn.functional as F

output = F.max_pool2d(activation_map, kernel_size=2, stride=2)
```

---

🎉 **Great job!** Now, let’s try using activation functions and max pooling in our own models! 🏗️🤖✨  

------------------------------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning about **Convolution with Multiple Channels**! 🖼️🤖  

## 🤔 What’s a Channel?  
A **channel** is like a layer of an image! 🌈  
- **Black & White Images** ➝ **1 channel** (grayscale) 🏳️  
- **Color Images** ➝ **3 channels** (Red, Green, Blue - RGB) 🎨  

Neural networks **see** images by looking at these channels separately! 👀  

---

## 🎯 1. Multiple Output Channels  

Usually, we use **one kernel** to create **one activation map** 📊  
But what if we want to detect **different things** in an image? 🤔  
- **Solution:** Use **multiple kernels**! Each kernel **finds different features**! 🔍  

### 🔥 Example: Detecting Lines  
1️⃣ A **vertical line kernel** finds **vertical edges** 📏  
2️⃣ A **horizontal line kernel** finds **horizontal edges** 📐  

**More kernels = More ways to see the image!** 👀✅  

---

## 🎨 2. Multiple Input Channels  

Color images have **3 channels** (Red, Green, Blue).  
To process them, we use **a separate kernel for each channel**! 🎨  

1️⃣ Apply a **Red kernel** to the Red part of the image 🔴  
2️⃣ Apply a **Green kernel** to the Green part of the image 🟢  
3️⃣ Apply a **Blue kernel** to the Blue part of the image 🔵  
4️⃣ **Add the results together** to get one activation map!  

This helps the neural network understand **colors and patterns**! 🌈  

---

## 🔄 3. Multiple Input & Output Channels  

Now, let’s **combine everything**! 🚀  
- **Multiple input channels** (like RGB images)  
- **Multiple output channels** (different filters detecting different things)  

Each output channel gets its own **set of kernels** for each input channel.  
We **apply the kernels, add the results**, and get multiple **activation maps**! 🎯  

---

## 🏗 Example in PyTorch  

```python
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)  
```

This means:  
✅ **3 input channels** (Red, Green, Blue)  
✅ **5 output channels** (5 different filters detecting different things)  

---

## 🏆 Why is This Important?  

✅ Helps the neural network find **different patterns** 🎨  
✅ Works for **color images** and **complex features** 🤖  
✅ Makes the network **more powerful**! 💪  

---

🎉 **Great job!** Now, let’s try convolution with multiple channels in our own models! 🏗️🤖✨  
-----------------------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’re building a **CNN for MNIST**! 🏗️🔢  
MNIST is a dataset of **handwritten numbers (0-9)**. ✍️🖼️  

---

## 🏗 CNN Structure  

📏 **Image Size:** 16×16 (to make training faster)  
🔄 **Layers:**  
- **First Convolution Layer** ➝ 16 output channels  
- **Second Convolution Layer** ➝ 32 output channels  
- **Final Layer** ➝ 10 output neurons (one for each digit)  

---

## 🛠 Building the CNN in PyTorch  

### 📌 Step 1: Define the CNN  

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)  
        self.pool = nn.MaxPool2d(kernel_size=2)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)  
        self.fc = nn.Linear(32 * 4 * 4, 10)  # Fully connected layer (512 inputs, 10 outputs)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))  # First layer: Conv + ReLU + Pool
        x = self.pool(nn.ReLU()(self.conv2(x)))  # Second layer: Conv + ReLU + Pool
        x = x.view(-1, 512)  # Flatten the 4x4x32 output to 1D (512 elements)
        x = self.fc(x)  # Fully connected layer for classification
        return x
```

---

## 🔍 Understanding the Output Shape  

After **Max Pooling**, the image shrinks to **4×4 pixels**.  
Since we have **32 channels**, the total output is:  
```
4 × 4 × 32 = 512 elements
```
Each neuron in the final layer gets **512 inputs**, and since we have **10 digits (0-9)**, we use **10 neurons**.  

---

## 🔄 Forward Step  

1️⃣ **Apply First Convolution Layer** ➝ Activation ➝ Max Pooling  
2️⃣ **Apply Second Convolution Layer** ➝ Activation ➝ Max Pooling  
3️⃣ **Flatten the Output (4×4×32 → 512)**  
4️⃣ **Apply the Final Output Layer (10 Neurons for 10 Digits)**  

---

## 🏋️‍♂️ Training the Model  

Check the **lab** to see how we train the CNN using:  
✅ **Backpropagation**  
✅ **Stochastic Gradient Descent (SGD)**  
✅ **Loss Function & Accuracy Check**  

---

🎉 **Great job!** Now, let’s train our CNN to recognize handwritten digits! 🏗️🔢🤖  
------------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning about **Convolutional Neural Networks (CNNs)!** 🤖🖼️  

## 🤔 What is a CNN?  
A **Convolutional Neural Network (CNN)** is a special type of neural network that **understands images!** 🎨  
It learns to find patterns, like:  
✅ **Edges** (lines & shapes)  
✅ **Textures** (smooth or rough areas)  
✅ **Objects** (faces, animals, letters)  

---

## 🏗 How Does a CNN Work?  

A CNN is made of **three main steps**:  

1️⃣ **Convolution Layer** 🖼️➝🔍  
   - Uses **kernels** (small filters) to **detect patterns** in an image  
   - Creates an **activation map** that highlights important features  

2️⃣ **Pooling Layer** 🔄➝📏  
   - **Shrinks** the activation map to keep only the most important parts  
   - **Max Pooling** picks the **biggest** values in each small region  

3️⃣ **Fully Connected Layer** 🏗️➝🎯  
   - The final layer makes a **decision** (like cat 🐱 or dog 🐶)  

---

## 🎨 Example: Detecting Lines  

We train a CNN to recognize **horizontal** and **vertical** lines:  

1️⃣ **Input Image (X)**  
2️⃣ **First Convolution Layer**  
   - Uses **two kernels** to create two **activation maps**  
   - Applies **ReLU** (activation function) to remove negative values  
   - Uses **Max Pooling** to make learning easier  

3️⃣ **Second Convolution Layer**  
   - Takes **two input channels** from the first layer  
   - Uses **two new kernels** to create **one activation map**  
   - Again, applies **ReLU + Max Pooling**  

4️⃣ **Flattening** ➝ Turns the 2D image into **1D data**  
5️⃣ **Final Prediction** ➝ Uses a **fully connected layer** to decide:  
   - `0` = **Vertical Line**  
   - `1` = **Horizontal Line**  

---

## 🔄 How to Build a CNN in PyTorch  

### 🏗 CNN Constructor  
```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.fc = nn.Linear(49, 2)  # Fully connected layer (49 inputs, 2 outputs)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))  # First layer: Conv + ReLU + Pool
        x = self.pool(nn.ReLU()(self.conv2(x)))  # Second layer: Conv + ReLU + Pool
        x = x.view(-1, 49)  # Flatten to 1D
        x = self.fc(x)  # Fully connected layer
        return x
```

---

## 🏋️‍♂️ Training the CNN  

We train the CNN using **backpropagation** and **gradient descent**:  

1️⃣ **Load the dataset** (images of lines) 📊  
2️⃣ **Create a CNN model** 🏗️  
3️⃣ **Define a loss function** (to measure mistakes) ❌  
4️⃣ **Choose an optimizer** (to improve learning) 🔄  
5️⃣ **Train the model** until it **gets better**! 🚀  

As training progresses:  
📉 **Loss goes down** ➝ Model makes fewer mistakes!  
📈 **Accuracy goes up** ➝ Model gets better at predictions!  

---

## 🏆 Why Use CNNs?  

✅ **Finds patterns** in images 🔍  
✅ **Works with real-world data** (faces, animals, objects) 🖼️  
✅ **More efficient** than regular neural networks 💡  

---

🎉 **Great job!** Now, let’s build and train our own CNN! 🏗️🤖✨  
----------------------------------------------------------------------

🎵 **Music Playing**  

👋 **Welcome!** Today, we’re building a **CNN for MNIST**! 🏗️🖼️  
MNIST is a dataset of **handwritten numbers (0-9)**. ✍️🔢  

---

## 🏗 CNN Structure  

📏 **Image Size:** 16×16 (to make training faster)  
🔄 **Layers:**  
- **First Convolution Layer** ➝ 16 output channels  
- **Second Convolution Layer** ➝ 32 output channels  
- **Final Layer** ➝ 10 output neurons (one for each digit)  

---

## 🛠 Building the CNN in PyTorch  

### 🔹 Step 1: Define the CNN  

```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)  
        self.pool = nn.MaxPool2d(kernel_size=2)  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)  
        self.fc = nn.Linear(32 * 4 * 4, 10)  # Fully connected layer (512 inputs, 10 outputs)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))  # First layer: Conv + ReLU + Pool
        x = self.pool(nn.ReLU()(self.conv2(x)))  # Second layer: Conv + ReLU + Pool
        x = x.view(-1, 512)  # Flatten the 4x4x32 output to 1D (512 elements)
        x = self.fc(x)  # Fully connected layer for classification
        return x
```

---

## 🔍 Understanding the Output Shape  

After **Max Pooling**, the image shrinks to **4×4 pixels**.  
Since we have **32 channels**, the total output is:  
```
4 × 4 × 32 = 512 elements
```
Each neuron in the final layer gets **512 inputs**, and since we have **10 digits (0-9)**, we use **10 neurons**.  

---

## 🔄 Forward Step  

1️⃣ **Apply First Convolution Layer** ➝ Activation ➝ Max Pooling  
2️⃣ **Apply Second Convolution Layer** ➝ Activation ➝ Max Pooling  
3️⃣ **Flatten the Output (4×4×32 → 512)**  
4️⃣ **Apply the Final Output Layer (10 Neurons for 10 Digits)**  

---

## 🏋️‍♂️ Training the Model  

Check the **lab** to see how we train the CNN using:  
✅ **Backpropagation**  
✅ **Stochastic Gradient Descent (SGD)**  
✅ **Loss Function & Accuracy Check**  

---

🎉 **Great job!** Now, let’s train our CNN to recognize handwritten digits! 🏗️🔢🤖  
------------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning how to use **Pretrained TorchVision Models**! 🤖🖼️  

## 🤔 What is a Pretrained Model?  

A **pretrained model** is a neural network that has already been **trained by experts** on a large dataset.  
✅ **Saves time** (no need to train from scratch) ⏳  
✅ **Works better** (already optimized) 🎯  
✅ **We only train the final layer** for our own images! 🔄  

---

## 🔄 Using ResNet18 (A Pretrained Model)  

We will use **ResNet18**, a powerful model trained on **color images**. 🎨  
It has **skip connections** (we won’t go into details, but it helps learning).  

We only **replace the last layer** to match our dataset! 🔁  

---

## 🛠 Steps to Use a Pretrained Model  

### 📌 Step 1: Load the Pretrained Model  
```python
import torchvision.models as models

model = models.resnet18(pretrained=True)  # Load pretrained ResNet18
```

### 📌 Step 2: Normalize Images (Required for ResNet18)  
```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
```

### 📌 Step 3: Prepare the Dataset  
Create a **dataset object** for your own images with **training and testing data**. 📊  

### 📌 Step 4: Replace the Output Layer  
- The **last hidden layer** has **512 neurons**  
- We create a **new output layer** for **our dataset**  

Example: **If we have 7 classes**, we create a layer with **7 outputs**:  
```python
import torch.nn as nn

for param in model.parameters():
    param.requires_grad = False  # Freeze pretrained layers

model.fc = nn.Linear(512, 7)  # Replace output layer (512 inputs → 7 outputs)
```

---

## 🏋️‍♂️ Training the Model  

### 📌 Step 5: Create Data Loaders  
```python
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=15, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
```

### 📌 Step 6: Set Up Training  
```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Optimizer (only for last layer)
```

### 📌 Step 7: Train the Model  
1️⃣ **Set model to training mode** 🏋️  
```python
model.train()
```  
2️⃣ Train for **20 epochs**  
3️⃣ **Set model to evaluation mode** when predicting 📊  
```python
model.eval()
```  

---

## 🏆 Why Use Pretrained Models?  

✅ **Saves time** (no need to train from scratch)  
✅ **Works better** (pretrained on millions of images)  
✅ **We only change one layer** for our dataset!  

---

🎉 **Great job!** Now, try using a pretrained model for your own images! 🏗️🤖✨  
---------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning how to use **GPUs in PyTorch**! 🚀💻  

## 🤔 Why Use a GPU?  
A **Graphics Processing Unit (GPU)** can **train models MUCH faster** than a CPU!  
✅ Faster computation ⏩  
✅ Better for large datasets 📊  
✅ Helps train deep learning models efficiently 🤖  

---

## 🔥 What is CUDA?  
CUDA is a **special tool** made by **NVIDIA** that allows us to use **GPUs for AI tasks**. 🎮🚀  
In **PyTorch**, we use **torch.cuda** to work with GPUs.  

---

## 🛠 Step 1: Check if a GPU is Available  

```python
import torch

# Check if a GPU is available
torch.cuda.is_available()  # Returns True if a GPU is detected
```

---

## 🎯 Step 2: Set Up the GPU  

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

- `"cuda:0"` = First available GPU 🎮  
- `"cpu"` = Use the CPU if no GPU is found  

---

## 🏗 Step 3: Sending Tensors to the GPU  

In PyTorch, **data is stored in Tensors**.  
To move data to the GPU, use `.to(device)`.  

```python
tensor = torch.randn(3, 3)  # Create a random tensor
tensor = tensor.to(device)  # Move it to the GPU
```

✅ **Faster processing on the GPU!** ⚡  

---

## 🔄 Step 4: Using a GPU with a CNN  

You **don’t need to change** your CNN code! Just **move the model to the GPU** after creating it:  

```python
model = CNN()  # Create CNN model
model.to(device)  # Move the model to the GPU
```

This **converts** all layers to **CUDA tensors** for GPU computation! 🎮  

---

## 🏋️‍♂️ Step 5: Training a Model on a GPU  

Training is the same, but **you must send your data to the GPU**!  

```python
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)  # Move data to GPU
    optimizer.zero_grad()  # Clear gradients
    outputs = model(images)  # Forward pass (on GPU)
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
```

✅ **The model trains much faster!** 🚀  

---

## 🎯 Step 6: Testing the Model  

For testing, **only move the images** (not the labels) to the GPU:  

```python
for images, labels in test_loader:
    images = images.to(device)  # Move images to GPU
    outputs = model(images)  # Get predictions
```

✅ **Saves memory and speeds up testing!** ⚡  

---

## 🏆 Summary  

✅ **GPUs make training faster** 🎮  
✅ Use **torch.cuda** to work with GPUs  
✅ Move **data & models** to the GPU with `.to(device)`  
✅ Training & testing are the same, but data **must be on the GPU**  

---

🎉 **Great job!** Now, try training a model using a GPU in PyTorch! 🏗️🚀  
