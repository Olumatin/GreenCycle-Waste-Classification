# GreenCycle Waste Classification API

A CNN-based waste classification system developed for GreenCycle Waste Management.

## Waste Classes

The model classifies images into 10 categories:

- Battery
- Biological
- Cardboard
- Clothes
- Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash

## Model

The model is a convolutional neural network built from scratch using TensorFlow and Keras.

It uses:
- Data augmentation
- Dropout
- Class weighting
- Early stopping
- Learning-rate reduction

## API Endpoints

GET /health

POST /predict

The prediction endpoint accepts an image and returns the predicted waste class and confidence score.

## Evaluation

Test accuracy: approximately 65%.

Battery recall: approximately 84%.

The project includes the classification report and confusion matrix.

## Deployment Considerations

The model should not be trusted immediately on a real recycling line. Real conveyor-belt conditions may include poor lighting, wet or dirty objects, overlapping waste, partially hidden objects, and camera angles that differ from the training data.

Battery detection is particularly important because missing a battery can create a safety risk. Before deployment, GreenCycle should test the model using real conveyor-belt images, improve battery detection, and include appropriate human or safety verification before fully automated sorting.
