# Potato Disease Classification
Our project focuses on potato disease classification using deep learning techniques. By analyzing images of potato plants,
 we aim to identify and classify various diseases, such as late blight, early blight, and potato scab. This technology plays 
 a crucial role in early disease detection, enabling farmers to take timely preventive measures and improve crop yield. We collect 
 and label a diverse dataset, train deep neural networks, and validate the model's accuracy. The ultimate goal is to deploy this 
 system in the field, allowing real-time disease detection and providing valuable insights for sustainable potato farming.

Base Paper Source: [Potato Leaf Disease Classification Using Deep Learning Approach](https://ieeexplore.ieee.org/document/9231784)

---

## Dataset - [Potato Disease](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

The potato disease dataset comprises a diverse collection of images and associated metadata, capturing various diseases that affect potato plants. 
It includes high-resolution photographs of potato foliage and tubers, showcasing common diseases like late blight, early blight, potato scab, and others.
 The dataset is meticulously labeled, indicating the disease type and severity, enabling the training of machine learning models.
Source: [Dataset Paper Name](https://link_to_dataset_paper)

## Social and Economic Impact

The social impact of the potato disease classification project extends to an important economic dimension. By preventing and mitigating the damage
 caused by potato diseases, farmers can significantly increase their crop yields and quality. This translates to higher incomes for farmers, 
 enhanced food security, and reduced reliance on costly chemical treatments. Moreover, improved potato production positively influences the supply chain,
  reducing market price fluctuations and ensuring a stable food supply. Ultimately, the project's social impact on disease management leads to economic stability, 
  sustainable agricultural practices, and long-term economic benefits for both individual farmers and the broader agricultural industry.

## Requirements

- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [TensorFlow](https://www.tensorflow.org/)
- ...  (Can be updated later on) ...

## Usage

1.Data Collection:
Gather a diverse dataset of potato plant images that contain both healthy and diseased plants. The dataset should cover various potato diseases like late blight, early blight, black scurf, etc.
2.Data Preprocessing:
    (1)Image Preprocessing:
             Resize images to a uniform size.
             Normalize pixel values to ensure consistency.
    (2)Data Split:
             Divide the dataset into three subsets: training, validation, and testing data.
    (3)Label Encoding:
             Assign labels to each image to indicate whether it is a healthy plant or affected by a specific disease.
3.Feature Extraction:
Feature extraction is a crucial step in image classification. You can use various techniques, including:
Convolutional Neural Networks (CNNs): These are widely used for image feature extraction. Pre-trained CNN models (e.g., VGG, ResNet, Inception) can be fine-tuned for this task.
Handcrafted features: You can also extract features like color histograms, texture, and shape descriptors from the images.
4.Model Selection:
Choose an appropriate machine learning or deep learning model for potato disease classification. Popular choices include:
Convolutional Neural Networks (CNNs)
Transfer learning models (pre-trained models)
Support Vector Machines (SVM)
Random Forest
K-Nearest Neighbors (K-NN)
5.Model Training:
Train the selected model using the training dataset.
Use data augmentation techniques to artificially increase the size of your training data, which helps improve model generalization.
6.Model Evaluation:
Use the validation dataset to fine-tune hyperparameters and monitor the model's performance.
Common evaluation metrics include accuracy, precision, recall, F1-score, and confusion matrices.
Conduct cross-validation to assess the model's robustness.
7.Model Testing:
After model training and validation, evaluate the model's performance using the testing dataset. This provides an estimate of how well the model will perform on new, unseen data.
8.Post-processing:
If necessary, apply post-processing techniques to the model's predictions, such as filtering out false positives or aggregating results for multiple images of the same plant.
9.Deployment:
Once you are satisfied with the model's performance, you can deploy it for real-world use. This may involve integrating the model into a mobile app, a web service, or other applications for potato disease detection.
10.Continuous Monitoring and Improvement:
Continuously monitor the model's performance in real-world applications.
Retrain the model with new data to adapt to changing disease patterns or improve its accuracy.

 (Can be updated later on) ...

## Files

List of important files and directories in the project. (Can be updated later on) ...

- [.gitignore]((https://github.com/MegaCreater/Navarambhah/blob/main/potato-disease-classification/.gitignore)
- [LICENSE](https://github.com/MegaCreater/Navarambhah/blob/main/potato-disease-classification/LICENSE)
- [README.md](https://github.com/MegaCreater/Navarambhah/blob/main/potato-disease-classification/README.md)
- ...  (Can be updated later on) ...

## License

This project is licensed under the [MIT License](https://github.com/MegaCreater/Navarambhah/blob/main/potato-disease-classification/LICENSE)


## Contributing Guidelines

Thank you for considering contributing to this project! Please take a moment to review the following guidelines.

## How to Contribute

1. Fork the repository and create your branch from `main`.
2. Clone the forked repository to your local machine.
3. Make your changes and test them thoroughly.
4. Ensure your code follows the project's coding style and conventions.
5. Commit your changes with clear and concise messages.
6. Push your commits to your fork on GitHub.
7. Submit a pull request to the main repository's `main` branch.

## Authors / Support 

- Sakshi Singh  @[Email1](sakshusingh21@gmail.com) @[LinkedIn](linkedin.com/in/sakshi-singh2)
- Tanu Yadav @[Email2](tusharyadav7455@gmail.com) @[LinkedIn](linkedin.com/in/tanu-yadav08)
- Ishika Mittal @[Email2](mittalishika03@gmail.com) @[LinkedIn](linkedin.com/in/ishika-mittal-) 

## Frequently Asked Questions

Q1: What is the objective of the potato disease classification project?
A: The project aims to develop a system that can accurately identify and classify different diseases affecting potato crops based on images.
Q2: How is the potato disease classification system trained?
A: The system is trained using a dataset of potato disease images, utilizing machine learning techniques like convolutional neural networks (CNNs).
Q3: Is this system applicable to other crops?
A: Yes, the technology can be adapted to classify diseases in other crops by training it with relevant datasets.
Q4: How is the model trained and validated?
A: The model is trained using a labeled dataset of potato disease images. It's validated using a separate dataset to assess its accuracy and generalization.
Q: What is the accuracy of the classification system?
A: The accuracy depends on the quality and size of the dataset and the chosen machine learning model, but it typically achieves high accuracy rates, often exceeding 97%.

