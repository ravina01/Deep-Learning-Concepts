# Deep-Learning-Concepts
---

Everything you need to know about how Neural Net works!

Deep learning is a subset of machine learning, which itself is a subset of artificial intelligence (AI). It involves training neural networks with many layers (hence the term "deep") to perform tasks such as image and speech recognition, natural language processing, and more. Here's an overview of deep learning and the typical steps involved in building and training a deep learning model:

### What is Deep Learning?
Deep learning models are inspired by the structure and function of the brain, specifically neural networks. These models learn to perform tasks by being exposed to large amounts of data and iteratively improving their performance. Key characteristics of deep learning include:

#### 1. Neural Networks: 
Composed of layers of nodes (neurons), each layer transforms the input data, progressively extracting higher-level features.
####  2. Automatic Feature Extraction: 
Unlike traditional machine learning, deep learning models can automatically discover the representations needed for feature detection.
####  3. Large Scale: 
Deep learning models often require large datasets and substantial computational power for training.

### Steps in Building a Deep Learning Model

#### 1. Data Collection:
- Gather a large and diverse dataset relevant to the task.
- Ensure the data is labeled if the task is supervised learning.

#### 2. Data Preprocessing:
- Cleaning: Remove noise and handle missing values.
- Normalization: Scale features to a similar range.
- Augmentation: Create more data by transformations (e.g., flipping, rotation for images).
- Splitting: Divide data into training, validation, and test sets.

#### 3. Model Selection:
- Choose an appropriate neural network architecture (e.g., Convolutional Neural Networks (CNNs) for images, Recurrent Neural Networks (RNNs) for sequences).
- Determine the number of layers, type of layers (dense, convolutional, recurrent), and other hyperparameters.

#### 4. Model Architecture Design:
- Define the model's architecture using a deep learning framework (e.g., TensorFlow, PyTorch, Keras).
- Specify the input shape, layer types, activation functions, and the output layer.

#### 5. Compilation:
- Select a loss function appropriate for the task (e.g., cross-entropy for classification, mean squared error for regression).
- Choose an optimizer (e.g., Adam, SGD) to update the model parameters.
- Define metrics to evaluate the model's performance (e.g., accuracy, precision, recall).

#### 6. Training:
- Feed the training data into the model in batches.
- Perform forward and backward propagation to update the weights.
- Use the validation set to monitor the model's performance and prevent overfitting.
- Adjust hyperparameters if necessary (hyperparameter tuning).

#### 7. Evaluation:
- Assess the model's performance on the test set to ensure it generalizes well to unseen data.
- Analyze metrics and create confusion matrices, ROC curves, or other visualizations to understand performance.

#### 8. Fine-Tuning and Optimization:
- Make adjustments to the model architecture or hyperparameters based on evaluation results.
- Apply techniques like dropout, batch normalization, or learning rate scheduling to improve performance.

#### 9. Deployment:
- Integrate the trained model into an application or service.
- Ensure the model can handle real-world data and scales appropriately.

### Key Concepts in Deep Learning:

#### 1. Activation Functions:
Functions like ReLU, sigmoid, and tanh introduce non-linearity into the model.
#### 2. Loss Functions: 
Quantify how well the model's predictions match the true labels.
#### 3. Optimizers: 
Algorithms that adjust the model weights to minimize the loss function (e.g., Adam, RMSprop).
#### 4. Regularization: 
Techniques to prevent overfitting, such as dropout and L2 regularization.
#### 5. Batch Size: 
Number of samples processed before the model's internal parameters are updated.
#### 6. Epochs: 
One full pass through the entire training dataset.

**Let's deep dive into each key concept of deep learning -->**
---

### 1. Activation Functions in Deep Learning

![image](https://github.com/user-attachments/assets/aa0dc93d-aa43-4057-835c-5e18e8ce1344)

Activation functions play a crucial role in neural networks by introducing non-linearity, allowing the model to learn complex patterns and representations. They determine the output of a neural network's node given an input or set of inputs. Let's delve into the details of different activation functions, their types, and specific use cases in computer vision applications.


#### a. Sigmoid
The sigmoid function maps any input to a value between 0 and 1. It is represented as:

![image](https://github.com/user-attachments/assets/fccd5a8b-4fe6-4435-8ade-5e5d457dcdbd)

Pros:
- Smooth gradient, preventing abrupt changes in direction during optimization.
- Output range between 0 and 1, useful for probabilistic interpretations.

Cons:
- Can cause vanishing gradient problems.
- Not zero-centered, leading to slower convergence.
  
Use Case:
- Used in the output layer of binary classification problems.
- Historically used in hidden layers, but less common now due to better alternatives.


#### b. Tanh (Hyperbolic Tangent)
The tanh function maps inputs to values between -1 and 1. It is represented as:

![image](https://github.com/user-attachments/assets/5c36428b-798a-4878-8714-e52538f73efa)

Pros:
- Zero-centered, leading to better convergence.
- Smoother gradient compared to sigmoid.

Cons:
- Can still suffer from the vanishing gradient problem.

Use Case:
- Used in hidden layers of neural networks, especially in RNNs.


#### c. ReLU (Rectified Linear Unit)
The ReLU function outputs the input directly if it is positive; otherwise, it outputs zero:
![image](https://github.com/user-attachments/assets/26476dde-b610-4503-8a6d-ae82fd30bda6)

Pros:
- Sparse activation, making the network efficient.
- Mitigates the vanishing gradient problem.
- Computationally efficient.
  
Cons:
- Can cause dying ReLU problem where neurons output zero for all inputs.
- Not zero-centered.

Use Case:
- Widely used in hidden layers of convolutional neural networks (CNNs) and feedforward neural networks.
- Popular in computer vision tasks like image classification, object detection, and segmentation.


#### d. Leaky ReLU
A variation of ReLU that allows a small, non-zero gradient when the input is negative:
![image](https://github.com/user-attachments/assets/6d895144-e8c9-473e-9cf5-0f9d7a117835)

Pros:
- Prevents dying ReLU problem.
- 
Cons:
- The optimal value of the slope for negative part needs tuning.

Use Case:
- Used in CNNs and other deep networks to improve training dynamics.


#### e. Parametric ReLU (PReLU)
Similar to Leaky ReLU, but the slope of the negative part is learned during training:
![image](https://github.com/user-attachments/assets/fe19cccc-ebc3-4f1d-8d06-a9322c2ab351)

Pros:
- Adaptable to the data during training.

Cons:
- Slightly more computationally expensive due to parameter learning.

Use Case:
- Used in deeper neural networks where learning the slope can provide a performance boost.

#### f. Softmax
The softmax function converts logits into probabilities, summing to 1. It is represented as:
![image](https://github.com/user-attachments/assets/56173b5f-ed4e-4e61-8f2c-8323feb5bb9d)

Pros:
- Provides a probabilistic interpretation of the output.

Cons:
- Not used in hidden layers due to computational complexity.

Use Case:
- Used in the output layer of multi-class classification problems.

  
#### Specific Use Cases in Computer Vision Applications

1. Image Classification:
- Hidden Layers: ReLU, Leaky ReLU, PReLU
- Output Layer: Softmax for multi-class classification, Sigmoid for binary classification.

2.Object Detection:
- Hidden Layers: ReLU, Leaky ReLU
- Output Layer: Sigmoid for bounding box coordinates and class probabilities in multi-label detection frameworks like YOLO.

3.Image Segmentation:
- Hidden Layers: ReLU, Leaky ReLU
- Output Layer: Sigmoid for binary segmentation, Softmax for multi-class segmentation (e.g., UNet).
  
4. Transformers:
- Attention Mechanisms: Softmax is used to calculate attention weights.
- Feedforward Layers: Typically use ReLU or variants for non-linear transformations.


#### Stage of Deep Learning Where Activation Functions are Used

Activation functions are used after each layer of a neural network to introduce non-linearity. This is crucial for the model to learn complex patterns. The specific stage is:

- Input Layer: No activation function is used here.
- Hidden Layers: Activation functions like ReLU, Leaky ReLU, and tanh are applied to introduce non-linearity.
- Output Layer: Activation functions like Softmax or Sigmoid are used to produce the final output, depending on the task.

### 2. Loss Functions in Deep Learning

Loss functions, also known as cost functions or objective functions, quantify how well a model's predictions match the true labels. They measure the difference between the predicted values and the actual values, guiding the optimization process to minimize this difference. Different types of loss functions are used depending on the nature of the task (e.g., classification, regression) and the model architecture.

#### Importance of Loss Functions - 
1. Guiding Training: Loss functions provide a way to guide the training process by quantifying prediction errors. The optimizer uses the loss function to adjust the model's weights.
2. Performance Measure: They serve as a measure of the model's performance. A lower loss indicates better performance.
3. Model Selection: Choosing the right loss function is crucial for the model's success. The choice affects how the model learns and converges.

#### Types of Loss Functions

#### 1. Mean Squared Error (MSE)
![image](https://github.com/user-attachments/assets/0bdf0f63-284f-474e-8bfb-fe581d03dfea)

- Description: Measures the average squared difference between the actual values and the predicted values.
- Use Case: Commonly used in **regression tasks**
- Pros: Penalizes larger errors more significantly, leading to more accurate predictions.
- Cons: Sensitive to outliers.

#### 2. Mean Absolute Error (MAE)
![image](https://github.com/user-attachments/assets/b1febcfd-5a82-47cd-9b71-8d2c52e677d1)

- Description: Measures the average absolute difference between the actual values and the predicted values.
- Use Case: Also used in **regression tasks**
- Pros: More robust to outliers compared to MSE.
- Cons: Does not penalize larger errors as severely as MSE.

#### 3. Binary Cross-Entropy (Log Loss)
![image](https://github.com/user-attachments/assets/10e70954-d538-4ce1-af93-1403b75965a8)

- Description: Measures the performance of a classification model whose output is a probability value between 0 and 1.
- Use Case: Used in binary classification tasks.
- Pros: Effective for probabilistic predictions.
- Cons: Can be undefined if is exactly 0 or 1.

#### 4. Categorical Cross-Entropy
![image](https://github.com/user-attachments/assets/ef91260e-c9c0-4c7f-8bda-51302c5189af)

- Description: Measures the performance of a classification model whose output is a probability distribution over multiple classes.
- Use Case: Used in multi-class classification tasks.
- Pros: Suitable for one-hot encoded targets.
- Cons: Assumes mutually exclusive classes.

#### 5. Hinge Loss
![image](https://github.com/user-attachments/assets/11459325-8871-4ff7-ba12-60f5b6fb8f29)

- Description: Used for "maximum-margin" classification, most notably for support vector machines.
- Use Case: Binary classification.
- Pros: Encourages the correct classification of samples with a margin.
- Cons: Not suitable for probabilistic outputs.

#### 6. Huber Loss

- The Huber Loss function, also known as Smooth L1 Loss, is a loss function used in regression problems. It combines the advantages of both the Mean Squared Error (MSE) and Mean Absolute Error (MAE) loss functions, making it robust to outliers while still providing smooth gradients for small errors.
 Use case: Regression problems
 
#### Applications of Loss Functions in Computer Vision

#### 1. Image Classification:
- Loss Functions: Categorical Cross-Entropy, Sparse Categorical Cross-Entropy
- Example: Classifying images of handwritten digits (e.g., MNIST dataset), identifying objects in images (e.g., ImageNet dataset).
- Description: These loss functions help in training models to assign the correct label to an image out of many possible categories.
  
#### 2. Object Detection:
- Loss Functions: Combination of Binary Cross-Entropy for object classification and Smooth L1 Loss (or Huber Loss) for bounding box regression.
- Example: Detecting and localizing objects in images (e.g., YOLO, Faster R-CNN).
- Description: The classification part of the loss function ensures that the model accurately identifies objects, while the regression part helps in predicting the precise location of the bounding boxes around objects.

#### 3. Image Segmentation:
- Loss Functions: Categorical Cross-Entropy for pixel-wise classification, Dice Loss for overlapping regions.
- Example: Segmenting regions of interest in medical images (e.g., UNet), separating foreground objects from the background (e.g., semantic segmentation).
- Description: These loss functions help in training models to assign each pixel of an image to a specific class, enabling precise segmentation of objects within an image.

#### 4. Image Denoising:
- Loss Functions: Mean Squared Error (MSE)
- Example: Removing noise from images to restore clean images.
- Description: MSE helps in training models to minimize the difference between noisy images and their clean counterparts, effectively removing noise.
---

### EfficientNet ->
- is a family of convolutional neural networks designed to achieve state-of-the-art accuracy while being computationally efficient.
- EfficientNet-B0 is the baseline model from which the other EfficientNet models are scaled.
  
#### Key Components of EfficientNet-B0

1. Stem Layer:
- Conv2d with 3 input channels (RGB) and 32 output channels, kernel size 3x3, stride 2, padding 1.
- Batch Normalization.
- Swish Activation.
  
2. MBConv Blocks:
![image](https://github.com/user-attachments/assets/344e0b67-297e-448e-b8e3-e85bd5257ca1)

3. Head Layer:
- Conv2d with 1280 output channels, kernel size 1x1.
- Batch Normalization - To normalize the outputs of the convolutional layer, ensuring that the network trains faster and more reliably.
- Swish Activation  To introduce non-linearity and enable the network to learn complex patterns.  f(x)=x⋅sigmoid(x).
- Global Average Pooling  - To reduce each feature map to a single value by averaging all the values in the feature map.
- Fully Connected Layer (Dense layer) with the number of units equal to the number of output classes. - To map the pooled features to the desired number of output classes.
- Dropout - To prevent overfitting by randomly setting a fraction of input units to zero during training.
- Softmax Activation (for classification) - To convert the logits to probabilities for classification.
  
---

will cover ->

1. Loss functions and Optimizers
2. Gradient descent Architecture
3. Neural network Architectures
4. 5 Steps to every DL model

1. Learning process - forward proppogation and back propogation.
2. from input layer to output layer
3. weighted sum of neuron * weights + bias is passed to activation functions
4. which decides weather that particular neuron can contribute in the next layer.
5. output layer is in the form of probability.
6. weight - tells us how important the neuron is
7. bias - allows for the shifting of activations function to the right/left. adding scalar value to function shifts the graph to either left/right
8. neural net spits out predictions, neural network evaluates the performance, 
9. loss function quatifies the deviation from the expected output.
10. This information is sent back to neural nets and weighst and biases are adjusted accordingly.
11. initial values of weights and biases are random.
12. use loss function to adjust weights and biases and go backwards.
13. Values are adjusted to better fit the prediction model.
14. more the data the better it will be in predicting the output.
15. But theres trade off, too much data and you will end up, with problem like overfitting.
16. once network is gone through all the data, it can be used to make predictions.
17. 

#### Learning algorithm -->

- Initialize parameters with random values - weights and biases
- feed input data to network
- Compare predicted value with expected value and calculate the loss
- perform back propogation and propogate this loss back through the network
- update the paras based on loss with gradient descent algorithm, such that the total loss is reduced and better model is obtained.
- do the same thing till we get minized loss

### terms used in DL

####  1. Activation functions ->
- introduces non-linearity in the network
- decides whether a neuron can contribute to the next layer
- how to decide neuron firing/ activation ?
- step function we can activate neuron if its above certain values, lets say 0 -> activate
- else we wont
- derivative of linear function is const. as the slope isnt changing at any given point.
- f = mx + c -> the derivative is m 
- that means the gradient has no relationaship whatsoever, with x.
- with linear layers, the output is single layer presentation on input layer.
- imagine you have connected layers and if all of them are linear in nature, the activation function of the final layer, is nothing but just a linear function of input of the first layer. If we pause and think about it, that means the entire neural network, dozens of layers can be replaced by single layer as its linear presentation of the first layer.
- combination of linear functions in linear manner is still another linear function.
- This is terrible as we just lost the ability to stack layers this way, non matter how much we stack, whole network would be still equilavent to a single layer with, single activation.
  
-  If the activation functions were linear, no matter how many layers you add to a neural network, the entire network would behave like a single linear transformation.
-  This means the network would be limited to solving only linear problems, which are not sufficient for most real-world tasks.
-  Non-linearity allows neural networks to stack multiple layers, where each layer can learn different features of the input data.
-  



**Non-linear Function: A function that does not satisfy both additivity and homogeneity is non-linear. This means the output of a non-linear function does not change in a directly proportional manner to its input.**

1. sigmoid
- The sigmoid function is non-linear because it has a characteristic S-shape, meaning small changes in input around 0 lead to significant changes in output, but for very large or very small inputs, the output saturates.


2. Tanh - 
- The tanh function is also non-linear, with outputs ranging between -1 and 1. It captures more complex relationships than a linear function.

3. ReLU (Rectified Linear Unit):
- ReLU is non-linear because it changes the output based on whether the input is positive or negative, which cannot be represented by a single straight line.
- dying relu problem.
- during back propogation, weighst will not get adjusted during descent, neurons which go into that state, will stop responding to variations in the error. simply because gradient 0 nothing changes.-> this is dying relu problem.
-  this will cause several neurons to jst die and not respond.
-  making substantial part of the network passive rather than - what we want.
-  workarounds for this would be - leaky relu and parametric relu. by adding slope in negative region.
-  usually the slope is aroun y = 0.001 x - leaky relu
-  parametric relu = y = alpha x 
-  the main idea is that the gradient should never be 0.
-  less computationally expensive than tanh and sigmoid.
-  its simple mathematical operation.
-  good point to consider when you are designing your own dl network.


#### which activation function to use ?
- Binary Classification Problems- sigmoid
- if you are unsure - Relu / modified Relu
- 
#### why activation functions are non-linear ?
- linear are polynomials with degree 1
- while non -linear are polynomials with degree more than 1
- if we use linear activation function, no matter how many layers we have, it will be always equivalent to having single layer network.
- 


### Loss functions -

- quantifies the deviation of the predicted output with expected.

1. Regression - Squared error, huber loss
2. binary classification - Binary cross entropy, hinge loss
3. Multi-class classification - multi-class cross entropy, kullback divergence,



### Optimizers -
- during the training process we tweak and change the weights/ parameters of the model.
- to try and minimize the loss function.
- make predictions as correct and optimized as possible.
- loss value tell us which direction to take step towards.
- 
### Gradient descent -
- Iterative algo that starts off at a random point on the loss function, and travels down its slope in steps until it reaches the lowest point (minimum) of the function.
- most popular, fast and robust
- 
- 
#### Algorithm -
- calculate what a small change in each individual weight would do to the loss function
- adjust each para based on its gradient
- repeats step 1 and 2, until loss function is as low as possible

#### Gradient -
- its vector of a partial derivatives with respect to all independet variable.
-  
![image](https://github.com/user-attachments/assets/25faf550-6ad1-4554-928a-473ade860669)

data point is intial weights.
- to get this data point to the min of this function, we need, to take the negative gradient, since we want to find the steepest decrease.
- this happens iteratively till the loss is minized.
- when dealing with high dimensional datasets, we want to find the global minimum of the loss function.
- its possible that you find yourself in an area where it seems like, you have reached the lowest possible value of your loss fnction, but thats just local minima.  

![image](https://github.com/user-attachments/assets/c1d1e9b8-c949-46b6-ae1b-48260c1d0192)

- to avoid getting stuck in local minima, we make sure that we use proper learning rate.
- taking steps which are too large/ too small can hinder your ability to converge/ minimize the loss functon.
-  we dont want to take longer steps so that we can miss the optimal value or too small to reach the optimal value is taking longer than expected.
-  smaller learning steps tell us, the chnages we are making to our weights are pretty small.
-  0.001
-  taking small steps mean that its taking forever to converge to the global minimum.
-  too small steps might happen that the we are converging on local minima not global minima,
-  we change the learning rate at right pace.
-  

### SGD - Stoachastic Gradient desecnt -
- 

### Hyperparas - cant be estimated from the data - specified manually
- while model paras - estimated from the data.
  
- external to the neural network, value cant be estimated right from the data.
- when dl algorithm is tuned, you are literally tuning the hyper parameters
- if you have to manually specify the parameter it is a hyperparameter
- Learning rate, activation function
- search best value by trial and error
- they are set manually and tuned accordingly
- Batch - breakdown dataset in smaller chunks, and feed those chunks to the neural network one by one.
- Epochs - when the entire dataset is passed forward and backward through the neural network only once.
- we use more than 1 epoch to help our model generalize better.
- too many epochs also can lead to overfitting, where model has essentially memorized the patterns in the training data and performs terribly on the data it has never seen before.
- there is no right number of epochs unforunately.
- iterations - no of batches needed to complete one epoch.
- No of batches = no of iterations for one epoch
- Experiment , experiment and experiment!








