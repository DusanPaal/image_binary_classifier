
## 1.3 Methodology

### 1.3.1 Data Preprocessing and Augmentation:
- **Resizing:** All images are resized to a fixed dimension (e.g., 150x150) to ensure consistency across architectures.
- **Normalization:** Pixel values are scaled to the range [0, 1] to improve convergence during training.
- **Data Augmentation:** Random transformations, such as horizontal flips, rotations, zooms, and shifts, are applied to the training images. This helps enhance the model’s ability to generalize by learning from varied versions of each image, especially given the relatively small dataset size.

### 1.3.2 CNN Architectures
The project evaluates several CNN architectures to determine which best captures features for this binary classification task:

- **Baseline CNN**: A simple CNN architecture with a few convolutional and pooling layers to establish baseline performance.
- **Deeper Custom CNNs**: Architectures with more convolutional layers, using dropout and batch normalization to prevent overfitting.
- **Transfer Learning Models:** Pre-trained CNN architectures, including VGG16, ResNet50, and MobileNetV2, fine-tuned on the Dogs vs Cats dataset. Transfer learning enables the use of knowledge from large-scale datasets (e.g., ImageNet) to improve performance and reduce training time.

### 1.3.3  Hyperparameter tuning
An automated hyperparameter tuner will be used to optimize parameters:

- Learning rate
- Batch size
- Number of epochs
- Optimizer type (e.g., SGD, Adam)
- Dropout rate
    
### 1.3.4  Evaluation Metrics
Each model's performance is assessed using:
- **Accuracy:** The overall percentage of correct predictions.
- **F1-Score:** The harmonic mean of precision and recall, providing a balanced view of performance, especially if misclassifying one class is more costly.
- **AUC-ROC:** Measures the model’s ability to distinguish between the two classes across all classification thresholds.
- **Precision:** The proportion of true positive predictions among all positive predictions, measuring the model's exactness.
- **Loss:** The value of the loss function (e.g., binary cross-entropy) during training, indicating how well the model fits the data.
- **Recall:** The proportion of true positive predictions among all actual positive samples, measuring the model’s ability to capture all relevant instances.

### 1.3.5  Training Process
Each model is trained on the training dataset using a fixed number of epochs, and performance is monitored on the validation set to prevent overfitting. Hyperparameters are tuned based on validation performance to ensure fair comparisons. Early stopping is implemented to terminate training if validation performance stagnates or deteriorates.

### 1.3.6 Testing and Comparison
After training, the final model for each architecture is evaluated on the test set to determine its generalization performance.
Results are analyzed and compared across architectures to identify the model that performs best by evaluating:

- **Primary Metric: F1-Score** which balances precision and recall, ensuring good performance on both classes. This is important because the model might face challenges due to the dataset’s variability (e.g., diverse lighting and poses).
- **Secondary Metrics: AUC-ROC and Accuracy** to ensure the model can effectively separate the two classes across all thresholds and the accuracy for an overall view of the model's performance.

## 1.4 Expected Outcomes
- A comprehensive comparison of CNN architectures for binary image classification on the Dogs vs Cats dataset.
- Identification of the most effective architecture in terms of both accuracy and generalization.
- Insights into the impact of data augmentation and transfer learning on small datasets in binary classification tasks.

---


# 2. System setup

This notebook was created using Python 3.12. Python versions 3.8 and newer should work as well, though they have not been specifically tested with this project.

## 2.1 Install python
Download and install Python 3.12 from python.org. It is highly recommended to create a separate virtual environment for this project to avoid potential dependency conflicts between libraries.

### 2.1.1 Creating a Python Virtual Environment on Windows (optional)
To create a virtual environment on Windows, open PowerShell or Command Prompt and run the following command: </br>
**python -m venv C:\path\to\new\virtual\environment** </br>
Replace C:\path\to\new\virtual\environment with the actual path where you would like to create the virtual environment.

### 2.1.2 Creating a Python Virtual Environment on Linux (optional)
On Linux, start by installing the venv module (if it’s not already installed) to enable virtual environment creation: </br>
**sudo apt-get install python3-venv**

Then, create the virtual environment by running: </br>
**python3 -m venv path/to/new/virtual/environment** </br>
Replace /path/to/new/virtual/environment with your preferred location for the virtual environment.

## 2.2 Install third-party libraries
If you are using a virtual anvironment on Linux, this must be activated first by running: </br>
**source /path/to/new/virtual/environment/bin/activate**

If you are using a virtual anvironment on Windows, this must be activated first by running: </br>
**path/to/new/virtual/environment/Scripts/activate**

To install the necessary third-party libraries, run the following command from the project’s root directory, where the requirements.txt file is located: </br>
**pip install -r requirements.txt**

## 2.3 Install NVIDIA Tools (Optional)
If you wish to leverage hardware-accelerated computations on an NVIDIA GPU (useful for training and optimizing dense neural networks), you will need to install the appropriate NVIDIA drivers and software suite. If an NVIDIA GPU is not available, or if you choose not to use GPU acceleration, the neural network computations will default to the CPU, though this may result in slower performance.

### 2.3.1 Install nVidia CUDA
To install CUDA, follow these steps:
1. Visit the page: https://developer.nvidia.com/cuda-downloads
2. Run the following commands to download and install CUDA 12.6.2 for Ubuntu 24.04: </br>
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin <br>
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 <br>
get https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb <br>
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb <br>
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/ <br>
sudo apt-get update <br>
sudo apt-get -y install cuda-toolkit-12-6 <br>

**Note:** CUDA must be installed before cuDNN, as cuDNN relies on CUDA during installation.

### 2.3.2 Install NVIDIA cuDNN
To enable deeper GPU optimizations for neural network training, you’ll need to install cuDNN after CUDA:

1. Visit the: https://developer.nvidia.com/cudnn-downloads
2. Run the following commands to install cuDNN 9.5.0 for Ubuntu 24.04: </br>
wget https://developer.download.nvidia.com/compute/cudnn/9.5.0/local_installers/cudnn-local-repo-ubuntu2404-9.5.0_1.0-1_amd64.deb <br>
sudo dpkg -i cudnn-local-repo-ubuntu2404-9.5.0_1.0-1_amd64.deb <br>
sudo cp /var/cudnn-local-repo-ubuntu2404-9.5.0/cudnn-*-keyring.gpg /usr/share/keyrings/ <br>
sudo apt-get update <br>
sudo apt-get -y install cudnn <br>

After completing these steps, you’ll be set up to take advantage of GPU acceleration for deep learning models with compatible NVIDIA hardware.