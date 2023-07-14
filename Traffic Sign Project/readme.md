# Traffic Sign Classification 

This project demonstrates image classification for the traffic signs using deep learning techniques with TensorFlow.

## Dataset
The "indian-traffic-signs-prediction85" dataset from Kaggle Datasets is used for training and evaluation. It consists of images which represents traffic signs . The dataset is split into training, validation, and test sets.
![jpg](images/00018.jpg)
## Requirements
- TensorFlow
- kaggle.json
- Matplotlib
- Pandas
- Operating system

## Usage
1. Clone the repository:
- git clone <repository_url>

2. Install the required dependencies:
-pip install kaggle, pip install tensorflow , kaggle datasets , matplotlib , pandas.

3. Run the script:traffic_sign_classification

4. Training Process
- The script loads the dataset and performs data preprocessing.
- A convolutional neural network model based on InceptionV3 is defined.
- The model is trained on the training set with validation performed on the validation set.
- Training progress, including accuracy and loss, is displayed.

5. Evaluation
- The trained model is evaluated on the test set.
- Test accuracy and loss are reported.

6. Saving the Model
- The trained model is saved to the "traffic_sign_classification" directory.

## Files
- `traffic_sign_classification`: The main Python script for training and evaluating the model.
- `README.md`: This file providing detailed information about the project.
- `traffic_sign_classification/`: Directory containing the saved model.

## License
This project is licensed under the [MIT License](LICENSE).

#Address info
- Uday Jain [udayjain169@gmail.com](email)
    