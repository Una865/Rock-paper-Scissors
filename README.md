# Rock-paper-Scissors

Transfer learning with VGG-16 network on classification of:

- rock
- paper
- scissors

Additional layers are added for better classification. The Tensorflow framework was used for implementation. Additional to the rock-paper-scissors classification I wanted to provide hand-detection. Mediapipe provides trained hand detector which I incorporated into my project.

Rock-Paper-Scissors-realtime.py - applying predictions on frames of the WebCam
Rock-Paper-Scissors.py - applying predictions on uploaded images (few are shown as an example)
trainer.ipynb - notebook used for training the model
handtrack.py - provides the realtime predictions using only MediaPipe

The dataset used is from Kaggle. 
