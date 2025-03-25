# 🧠 **Simple Summary of the Program**

1. **Loads and Prepares Data:**
   - Uses the **MNIST dataset**, which contains images of handwritten digits (0-9).
   - Resizes the images and converts them to tensors.
   - Creates a **data loader** to batch the images and shuffle them for training.

2. **Defines a CNN Model:**
   - The **FinalCNN** model processes the images through layers:
     - **Conv1:** Finds simple features like edges.
     - **Pool1:** Reduces the size to focus on important features.
     - **Conv2:** Finds more complex patterns.
     - **Pool2:** Reduces the size again.
     - **Flattening:** Converts the features into a single line of numbers.
     - **Fully Connected Layers:** Makes predictions about what digit is in the image.

3. **Trains the Model:**
   - Uses the **Cross-Entropy Loss** to measure how far the predictions are from the real digit labels.
   - Uses **Stochastic Gradient Descent (SGD)** to adjust the model parameters and make better predictions.
   - Runs the training for **32 epochs**, slowly improving the accuracy.

4. **Displays Predictions:**
   - Shows **6 sample images** with the model's predictions and the actual labels.
   - Prints the accuracy and loss for each epoch.

5. **GPU Acceleration:**
   - Uses **CUDA** if available, making the training faster by running on the GPU.

✅ This program is like a smart detective that learns to recognize handwritten numbers by studying lots of examples and gradually improving its guesses.
