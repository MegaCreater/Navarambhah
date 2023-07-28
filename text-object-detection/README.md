# OCD from Scratch using TensorFlow

This repository contains a Python implementation of Optical Character Detection (OCD) made from scratch using TensorFlow. 
The OCD system is designed to extract text from images.

This repo has 2 separate approaches. One using OpenCV, other using a UNet architecture made in tensorflow from scratch

## Features

- Implementation of OCR system using TensorFlow and Keras
- Training of a deep learning model to recognize characters
- Evaluation of the OCR model on test datasets

## Installation

1. Clone the repository:

`git clone https://github.com/Yogendra019/text-object-detection`

2. Install the required dependencies:
`pip install tensorflow keras glob numpy pandas matplotlib pillow, opencv-python`

3. If you wish to use the code locally for the Unet application of OCD you will need to convert the ocr.ipynb file to a .py file. 
   The option to do the same is in colab onlt. Open the colab file, then on File tab, select download.
4. Dbnet folder contains an OpenCV detection strategy that uses pretrained model. Weights are available in the folder for the same.

5. The link to the pretrained weights for the OpenCV (Dbnet) can be found [here](https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1)

## Usage

1. Ensure installation is done properly

2. Train the OCD model

3. Evaluate the trained model

4. Use the OCD system on new images

5. If you are using the ocr.ipynb file, run the code in a sequential manner. Links to dataset are provided in the first cell of the colab file


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
