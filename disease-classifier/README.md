# Disease Image Classification

This project demonstrates image classification for Disease using deep learning techniques with TensorFlow.

## Dataset
The "covid-19-xray-two-proposed-databases" dataset from kaggle Datasets is used for training and evaluation. It consists of images of Chest X-ray. The dataset is split into training, validation, and test sets.

![jpg](images/images1.jpg) 
![jpg](images/images2.jpg) 
![jpg](images/images3.jpg) 

## Requirements
- TensorFlow
- TensorFlow Datasets
- Matplotlib
- Pandas

## Usage
1. Clone the repository:
git clone <repository_url>

2. Install the required dependencies:
pip install tensorflow tensorflow-datasets matplotlib pandas

3. Run the script:
python ADN_rps.py

4. Training Process
- The script loads the dataset and performs data preprocessing.
- A convolutional neural network model based on InceptionV3 is defined.
- The model is trained on the training set with validation performed on the validation set.
- Training progress, including accuracy and loss, is displayed.

5. Evaluation
- The trained model is evaluated on the test set.
- Test accuracy and loss are reported.

6. Saving the Model
- The trained model is saved to the "trained_model" directory.

## Files
- `ADN_rps.py`: The main Python script for training and evaluating the model.
- `README.md`: This file providing detailed information about the project.
- `trained_model/`: Directory containing the saved model.

## License
This project is licensed under the [MIT License](LICENSE).

## Authors
- Atul Singh: [atulnara5@gmail.com](mailto:atulnara5@gmail.com)
- Sudhir Pilaniya: [pilaniyasudhir@gmail.com](mailto:pilaniyasudhir@gmail.com)

