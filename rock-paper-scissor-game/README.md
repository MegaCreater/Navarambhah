# Rock Paper Scissors Image Classification

This project demonstrates image classification for the rock-paper-scissors game using deep learning techniques with TensorFlow.

## Dataset
The "rock_paper_scissors" dataset from TensorFlow Datasets is used for training and evaluation. It consists of images of hand gestures representing rock, paper, and scissors. The dataset is split into training, validation, and test sets.

![png](images/Rock.png) 
![png](images/Paper.png) 
![png](images/Scissor.png) 

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
- The trained model is saved to the "rock_paper_scissors_model" directory.

## Files
- `ADN_rps.py`: The main Python script for training and evaluating the model.
- `README.md`: This file providing detailed information about the project.
- `rock_paper_scissors_model/`: Directory containing the saved model.

## License
This project is licensed under the [MIT License](LICENSE).

## Authors
- Aviral Dubey: [aviralking31@gmail.com](mailto:aviralking31@gmail.com)
- Dikshant Malviya: [dikshantmalviyadm2017@gmail.com](mailto:dikshantmalviyadm2017@gmail.com)
- Neeraj Raghuwanshi: [raghuvanshineeraj007@gmail.com](mailto:raghuvanshineeraj007@gmail.com)
