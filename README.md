**Urdu OCR System**

This repository contains an Optical Character Recognition (OCR) system designed for recognizing Urdu text in images. The project includes two main components:

**Data Generation:** A Python script to generate synthetic Urdu word images with augmentations (erosion, dilation, rotation, and shear).

**OCR Model Training:** A TensorFlow-based Convolutional Neural Network (CNN) model to classify Urdu words from images.

The system achieves a test accuracy of up to 94% on a dataset of 15,000 images, as demonstrated in the training scripts.

**Dataset**

The dataset consists of synthetic Urdu word images generated using the provided data generation script. The images are labeled based on Urdu characters and include augmentations to improve model robustness. The dataset is available on Google Drive:

https://drive.google.com/file/d/1lQyfEA8j4H9iOf5E2LpvxjG_VMUNbZYt/view?usp=drive_link

**Project Structure**

data_generation.py: Script to generate synthetic Urdu word images with augmentations.

urdu_ocr_model.ipynb: Jupyter notebook containing the CNN model training and evaluation code.

README.md: This file, providing an overview and instructions for the project.

**Prerequisites**

To run the scripts, ensure you have the following installed:
Python 3.8+

Libraries:
numpy
pillow
opencv-python
arabic-reshaper
python-bidi
tensorflow
scikit-learn
matplotlib

A Urdu font file (e.g., NotoNaskhArabic[wght].ttf, included in the data generation script).

Install the required libraries using:

pip install numpy pillow opencv-python arabic-reshaper python-bidi tensorflow scikit-learn matplotlib

**Setup**
Clone the Repository:

git clone https://github.com/your-username/urdu-ocr-system.git
cd urdu-ocr-system
**Download the Dataset:**

Download the dataset from the Google Drive link.

https://drive.google.com/file/d/1lQyfEA8j4H9iOf5E2LpvxjG_VMUNbZYt/view?usp=drive_link

Extract the dataset to a directory (e.g., all_words/).

**Font File:**

Ensure the Urdu font file (NotoNaskhArabic[wght].ttf) is available. Update the font_path in data_generation.py to point to the font file location.

**Usage**

1. Data Generation

The data_generation.py script generates synthetic Urdu word images by combining characters from the Urdu alphabet, applying augmentations, and saving them to a specified directory.

**To generate images:**

1.python data_generation.py
Update the output_dir and font_path variables in the script to match your system paths.
The script generates images with filenames indicating character positions and augmentation type (e.g., 5_26_02_18_01_none.png).

2. Model Training
The urdu_ocr_model.ipynb notebook contains multiple versions of the CNN model training process, with the final version achieving 94% test accuracy. The model uses a CNN architecture with convolutional layers, max-pooling, and dense layers.

To train the model:

Open the urdu_ocr_model.ipynb notebook in Jupyter Notebook or JupyterLab.
Update the image_dir path to point to the dataset directory (e.g., all_words/).
Run the notebook cells sequentially to load the dataset, preprocess images, train the model, and evaluate performance.
Alternatively, convert the notebook to a Python script and run it:
jupyter nbconvert --to script urdu_ocr_model.ipynb
python urdu_ocr_model.py

3. Model Evaluation

The trained model is evaluated on a validation set, with the final version reporting a test accuracy of 94%. The notebook includes a custom callback to print validation accuracy after each epoch.

**Results**
Data Generation: Generates thousands of synthetic Urdu word images with augmentations to simulate real-world variations.
Model Performance:
Initial model: ~80% test accuracy.
Improved model with additional convolutional layer and padding: 94% test accuracy after 60 epochs.

The model is trained on 15,000 images, split into 80% training and 20% validation sets.

**Future Improvements**
Increase dataset size for better generalization.
Experiment with advanced architectures (e.g., ResNet, Transformer-based models).
Add support for handwritten Urdu text recognition.
Optimize the model for deployment in real-time applications.

**Contributing**

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For questions or support, please open an issue on the GitHub repository.
