# PicDetective: AI vs Real Image Classifier

![PicDetective Walkthrough](assets/walkthrough.gif)

## Project Overview
PicDetective is a machine learning tool that can distinguish between real photographs and AI-generated images. It uses a Convolutional Neural Network (CNN) trained on the CIFAKE dataset to spot patterns that AI often leaves behind.

## Motivation
When on social media I always fall for fake contents created by AI. For this reason I wanted to create a model that would predict whether an image is AI generated or not while exploring neural networks and vision based machine learning. 

## Features
- **Real-Time Guessing:** Play against the AI to see if you can spot the fakes better than the computer.
- **Custom AI Model:** Uses a neural network built with PyTorch specifically for 32x32 image detection.
- **Confidence Scoring:** Unlike simple classifiers, PicDetective shows you exactly how "sure" the AI is about its prediction.
- **High Accuracy:** The model achieves ~94% accuracy on the CIFAKE test set.

## Tech Stack
- **Language:** Python 3
- **Deep Learning:** PyTorch & Torchvision
- **GUI:** Tkinter (for the game interface)
- **AI Assistant:** Claude (code optimization, debugging, learning)
- **Data:** CIFAKE Dataset (100,000 images)

## Project Structure
- `src/main.py`: The main game and interface.
- `notebooks/`: The training process and model evaluation.
- `models/`: The saved weights of the trained AI.
- `data/`: The processed image dataset.

## Methodology
1. **Preprocessing:** Used CIFAKE dataset, normalizing the images and resizing them into 32x32 pixels

2. **Creating Model:** Created a CNN model using pytorch, using `Conv2d`, `MaxPool2d`, `Linear`, and `Sigmoid` layers to detect patterns in images. 

3. **Training:** I trained the model for 5 epochs, using the Adam optimizer and Binary Cross-Entropy loss. The model was trained in batches of 100 images.

4. **Testing:** Tested the model on the test set to check accuracy. The model achieved **94.10% accuracy** on the testing set.

5. **Deploying** Used Tkinter to create a GUI and allow users to play the game, and allows users to upload their own images for the program to predict whether it is AI generated or not.


## Challenges and Learnings
* **The "32x32" Limitation:** A major challenge was the dataset's low resolution. While 32x32 is great for fast training, it loses fine details. During testing, I found that high-res AI images (which look fake to humans) can trick the AI because the "tells" disappear when the image is shrunk down. To fix this I decided to program into the code what confidence the model has in its prediction.

* **Tkinter UI:** Tkinter became difficult to work with because I do not have much experience with it. It took me a while to figure out how to split the screen and place images and buttons. 

* **Model Creating:** Because of the Coursera Machine Learning Specialization I took, I was able to understand the fundamentals behind created this neural network, However, with help from Claude I was able to figure it out and build a deeper understanding of how it works.

* **Organization:** Dealing with 100,000 photos taught me a lot about project management. I learned that keeping your files organized, separating the AI model from the actual game code is the only way to keep a big project from becoming a mess.

* **ML Process:** Working on this project taught me more about the stages of the machine learning: preprocessing (preparing the 100,000 images for the computer), training (building and refining the CNN architecture), and evaluation (testing the model on new data to analyze its accuracy and confidence).


## Test, uploading my own image
During testing, I uploaded a high-resolution AI-generated image of a car. Even though the image was clearly fake to a human eye, the AI was 94% sure it was "Real." 

**Why it failed:** 
The AI was trained on 32x32 pixel images. When a high-res image is shrunk down that small, many of the AI artifacts (tells) are lost. This discovery led me to add a confidence percentage to the app, so users can see exactly how sure the AI is, rather than just getting a simple answer.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download the CIFAKE dataset from Kaggle: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
3. Unzip the dataset and place it at: `data/processed/cifake_data/` (so you have `data/processed/cifake_data/test/FAKE` and `data/processed/cifake_data/test/REAL`)
4. Run the game: `python src/main.py`

> **Note:** The **"Load Your Own Photo"** button works without the dataset. You can skip steps 2-3 and still test any image from your computer!

## Data Source
The model was trained using the **CIFAKE: Real and AI-Generated Synthetic Images** dataset, which consists of 100,000 images (50,000 real and 50,000 AI-generated).
- **Dataset Link:** [Kaggle - CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)


## Future Work
- [x] **Custom Image Upload:** Allow users to upload their own images from their computer to test the AI. (Completed!)
- [ ] **Image Clarity:** Allow the user to see the original photo from CIFAKE rather than the resized one
- [ ] **Mobile Support:** Create a mobile-friendly version of the classifier.
- [ ] **Batch Processing:** Allow the user to upload a whole folder of images at once.
