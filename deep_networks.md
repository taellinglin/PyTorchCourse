🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning about **Deep Neural Networks**—a cool way computers learn! 🧠💡  

## 🤖 What is a Neural Network?  
Imagine a brain made of tiny switches called **neurons**. These neurons work together to make smart decisions!  

### 🟢 Input Layer  
This is where we give the network information, like pictures or numbers.  

### 🔵 Hidden Layers  
These layers are like **magic helpers** that figure out patterns!  
- More neurons = better learning 🤓  
- Too many neurons = can be **confusing** (overfitting) 😵  

### 🔴 Output Layer  
This is where the network **gives us answers!** 🏆  

---

## 🏗 Building a Deep Neural Network in PyTorch  

We can **build a deep neural network** using PyTorch, a tool that helps computers learn. 🖥️  

### 🛠 Layers of Our Network  
1️⃣ **First Hidden Layer:** Has `H1` neurons.  
2️⃣ **Second Hidden Layer:** Has `H2` neurons.  
3️⃣ **Output Layer:** Decides the final answer! 🎯  

---

## 🔄 How Does It Work?  
1️⃣ **Start with an input (x).**  
2️⃣ **Pass through each layer:**  
   - Apply **math functions** (like `sigmoid`, `tanh`, or `ReLU`).  
   - These help the network understand better! 🧩  
3️⃣ **Get the final answer!** ✅  

---

## 🎨 Different Activation Functions  
Activation functions help the network **think better!** 🧠  
- **Sigmoid** ➝ Good for small problems 🤏  
- **Tanh** ➝ Works better for deeper networks 🌊  
- **ReLU** ➝ Super strong for big tasks! 🚀  

---

## 🔢 Example: Recognizing Handwritten Numbers  
We train the network with **MNIST**, a dataset of handwritten numbers. 📝🔢  
- **Input:** 784 pixels (28x28 images) 📸  
- **Hidden Layers:** 50 neurons each 🤖  
- **Output:** 10 neurons (digits 0-9) 🔟  

---

## 🚀 Training the Network  
We use **Stochastic Gradient Descent (SGD)** to teach the network! 📚  
- **Loss Function:** Helps the network learn from mistakes. ❌➡✅  
- **Validation Accuracy:** Checks how well the network is doing! 🎯  

---

## 🏆 What We Learned  
✅ Deep Neural Networks have **many hidden layers**.  
✅ Different **activation functions** help improve performance.  
✅ The more layers we add, the **smarter** the network becomes! 💡  

---

🎉 **Great job!** Now, let's build and train our own deep neural networks! 🏗️🤖✨  
-----------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’ll learn how to **build a deep neural network** in PyTorch using `nn.ModuleList`. 🧠💡  

## 🤖 Why Use `nn.ModuleList`?  
Instead of adding layers **one by one** (which takes a long time ⏳), we can **automate** the process! 🚀  

---

## 🏗 Building the Neural Network  

We create a **list** called `layers` 📋:  
- **First item:** Input size (e.g., `2` features).  
- **Second item:** Neurons in the **first hidden layer** (e.g., `3`).  
- **Third item:** Neurons in the **second hidden layer** (e.g., `4`).  
- **Fourth item:** Output size (number of classes, e.g., `3`).  

---

## 🔄 Constructing the Network  

### 🔹 Step 1: Create Layers  
- We loop through the list, taking **two elements at a time**:  
  - **First element:** Input size 🎯  
  - **Second element:** Output size (number of neurons) 🧩  

### 🔹 Step 2: Connecting Layers  
- First **hidden layer** ➝ Input size = `2`, Neurons = `3`  
- Second **hidden layer** ➝ Input size = `3`, Neurons = `4`  
- **Output layer** ➝ Input size = `4`, Output size = `3`  

---

## ⚡ Forward Function  

We **pass data** through the network:  
1️⃣ **Apply linear transformation** to each layer ➝ Makes calculations 🧮  
2️⃣ **Apply activation function** (`ReLU`) ➝ Helps the network learn 📈  
3️⃣ **For the last layer**, we only apply **linear transformation** (since it's a classification task 🎯).  

---

## 🎯 Training the Network  

The **training process** is similar to before! We:  
- Use a **dataset** 📊  
- Try **different combinations** of neurons and layers 🤖  
- See which setup gives the **best performance**! 🏆  

---

🎉 **Awesome!** Now, let’s explore ways to make these networks even **better!** 🚀  
-----------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning about **weight initialization** in Neural Networks! 🧠⚡  

## 🤔 Why Does Weight Initialization Matter?  
If we **don’t** choose good starting weights, our neural network **won’t learn properly**! 🚨  
Sometimes, **all neurons** in a layer get the **same weights**, which causes problems.  

---

## 🚀 How PyTorch Handles Weights  
PyTorch **automatically** picks starting weights, but we can also set them **ourselves**! 🔧  
Let’s see what happens when we:  
- Set **all weights to 1** and **bias to 0** ➝ ❌ **Bad idea!**  
- Randomly choose weights from a **uniform distribution** ➝ ✅ **Better!**  

---

## 🔄 The Problem with Random Weights  
We use a **uniform distribution** (random values between -1 and 1). But:  
- **Too small?** ➝ Weights don’t change much 🤏  
- **Too large?** ➝ **Vanishing gradient** problem 😵  

### 📉 What’s a Vanishing Gradient?  
If weights are **too big**, activations get **too large**, and the **gradient shrinks to zero**.  
That means the network **stops learning**! 🚫  

---

## 🛠 Fixing the Problem  

### 🎯 Solution: Scale Weights Based on Neurons  
We scale the weight range based on **how many neurons** we have:  
- **2 neurons?** ➝ Scale by **1/2**  
- **4 neurons?** ➝ Scale by **1/4**  
- **100 neurons?** ➝ Scale by **1/100**  

This prevents the vanishing gradient issue! ✅  

---

## 🔬 Different Weight Initialization Methods  

### 🏗 **1. Default PyTorch Method**  
- PyTorch **automatically** picks a range:  
  - **Lower bound:** `-1 / sqrt(L_in)`  
  - **Upper bound:** `+1 / sqrt(L_in)`  

### 🔵 **2. Xavier Initialization**  
- Best for **tanh** activation  
- Uses the **number of input and output neurons**  
- We apply `xavier_uniform_()` to set the weights  

### 🔴 **3. He Initialization**  
- Best for **ReLU** activation  
- Uses the **He initialization method**  
- We apply `he_uniform_()` to set the weights  

---

## 🏆 Which One is Best?  
We compare:  
✅ **PyTorch Default**  
✅ **Xavier Method** (tanh)  
✅ **He Method** (ReLU)  

The **Xavier and He methods** help the network **learn faster**! 🚀  

---

🎉 **Great job!** Now, let’s try different weight initializations and see what works best! 🏗️🔬  
------------------------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning about **Gradient Descent with Momentum**! 🚀🔄  

## 🤔 What’s the Problem?  
Sometimes, when training a neural network, the model can get **stuck**:  
- **Saddle Points** ➝ Flat areas where learning stops 🏔️  
- **Local Minima** ➝ Not the best solution, but we get trapped 😞  

---

## 🏃‍♂️ What is Momentum?  
Momentum helps the model **keep moving** even when it gets stuck! 💨  
It’s like rolling a ball downhill:  
- **Gradient (Force)** ➝ Tells us where to go 🏀  
- **Momentum (Mass)** ➝ Helps us keep moving even on flat surfaces ⚡  

---

## 🔄 How Does It Work?  

### 🔹 Step 1: Compute Velocity  
- Velocity (`v`) = Old velocity (`v_k`) + Learning step (`gradient * learning rate`)  
- The **momentum term** (𝜌) controls how much we keep from the past.  

### 🔹 Step 2: Update Weights  
- New weight (`w_k+1`) = Old weight (`w_k`) - Learning rate * Velocity  

The bigger the **momentum**, the harder it is to stop moving! 🏃‍♂️💨  

---

## ⚠️ Why Does It Help?  

### 🏔️ **Saddle Points**  
- **Without Momentum** ➝ Model **stops** moving in flat areas ❌  
- **With Momentum** ➝ Keeps moving **past** the flat spots ✅  

### ⬇ **Local Minima**  
- **Without Momentum** ➝ Gets **stuck** in a bad spot 😖  
- **With Momentum** ➝ Pushes through and **finds a better solution!** 🎯  

---

## 🏆 Picking the Right Momentum  

- **Too Small?** ➝ Model gets **stuck** 😕  
- **Too Large?** ➝ Model **overshoots** the best answer 🚀  
- **Best Choice?** ➝ We test different values and pick what works! 🔬  

---

## 🛠 Using Momentum in PyTorch  
Just add the **momentum** value to the optimizer!  

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
```

In the lab, we test **different momentum values** on a dataset and see how they affect learning! 📊  

---

🎉 **Great job!** Now, let’s experiment with momentum and see how it helps our model! 🏗️⚡  
------------------------------------------------------------------------------------------
🎵 **Music Playing**  

👋 **Welcome!** Today, we’re learning about **Batch Normalization**! 🚀🔄  

## 🤔 What’s the Problem?  
When training a neural network, the activations (outputs) can vary a lot, making learning **slower** and **unstable**. 😖  
Batch Normalization **fixes this** by:  
✅ Making activations more consistent  
✅ Helping the network learn faster  
✅ Reducing problems like vanishing gradients  

---

## 🔄 How Does Batch Normalization Work?  

### 🏗 Step 1: Normalize Each Mini-Batch  
For each neuron in a layer:  
1️⃣ Compute the **mean** and **standard deviation** of its activations. 📊  
2️⃣ Normalize the outputs using:  
   \[
   z' = \frac{z - \text{mean}}{\text{std dev} + \epsilon}
   \]  
   (We add a **small** value `ε` to avoid division by zero.)  

### 🏗 Step 2: Scale and Shift  
- Instead of leaving activations at 0 and 1, we **scale** and **shift** them:  
  \[
  z'' = \gamma \cdot z' + \beta
  \]  
- **γ (scale) and β (shift)** are **learned** during training! 🏋️‍♂️  

---

## 🔬 Example: Normalizing Activations  

- **First Mini-Batch (X1)** ➝ Compute mean & std for each neuron, normalize, then scale & shift  
- **Second Mini-Batch (X2)** ➝ Repeat for new batch! ♻  
- **Next Layer** ➝ Apply batch normalization again! 🔄  

### 🏆 Prediction Time  
- During **training**, we compute the mean & std for **each batch**.  
- During **testing**, we use the **population mean & std** instead. 📊  

---

## 🛠 Using Batch Normalization in PyTorch  

```python
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 3)  # First layer (10 inputs, 3 neurons)
        self.bn1 = nn.BatchNorm1d(3) # Batch Norm for first layer
        self.fc2 = nn.Linear(3, 4)   # Second layer (3 inputs, 4 neurons)
        self.bn2 = nn.BatchNorm1d(4) # Batch Norm for second layer

    def forward(self, x):
        x = self.bn1(self.fc1(x))  # Apply Batch Norm
        x = self.bn2(self.fc2(x))  # Apply Batch Norm again
        return x
```

- **Training?** Set the model to **train mode** 🏋️‍♂️  
  ```python
  model.train()
  ```  
- **Predicting?** Use **evaluation mode** 📈  
  ```python
  model.eval()
  ```  

---

## 🚀 Why Does Batch Normalization Work?  

### ✅ Helps Gradient Descent Work Better  
- Normalized data = **smoother** loss function 🎯  
- Gradients point in the **right** direction = Faster learning! 🚀  

### ✅ Reduces Vanishing Gradient Problem  
- Sigmoid & Tanh activations suffer from small gradients 😢  
- Normalization **keeps activations in a good range** 📊  

### ✅ Allows Higher Learning Rates  
- Networks can **train faster** without getting unstable ⏩  

### ✅ Reduces Need for Dropout  
- Some studies show **Batch Norm can replace Dropout** 🤯  

---

🎉 **Great job!** Now, let’s try batch normalization in our own models! 🏗️📈  
