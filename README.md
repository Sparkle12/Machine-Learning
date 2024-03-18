# Machine-Learning
This repository shows my journey of learning about Machine Learning and AI. I have experimented with multiple types of models and datasets (The datasets that are not imported directly from a library can be found in the "Datasets" folder)

MNIST dataset:
  - I trained a 3 layer neural network on this dataset for handwritten digit recognition(accuracy 97%):
    -The input layer is a flatten layer with the input shape 28*28 (1 neuron for each pixel)
    -The next 2 layers are dense layers that use the ReLU activation function
    -The output layer is dense layer with the softmax activation function to output the confidence level for each digit
    -I used the Adam optimization algorithm and the sparse categorical crossentropy loss function 

Diabetes dataset:
  -The diabetes dataset contains the following numerical data:
    -Number of pregnancies
    -Glucose levels
    -Blood pressure
    -Skin thickness
    -Insulin levels
    -BMI
    -Diabetes pedigree function value
    -Age
  -The target is a categorical variable that specifies whether the pacient has or dose not have diabetes.
  - On this dataset I trained and compared the following models:
    - KNN (accuracy 84%, f1 score 0.7391)
    - SVM  (accuracy 81.81%, f1 score 0.6666)
    - Logistic reggresion (accuracy 81.16%, f1 score 0.6666)

Iris datset:
  -The iris dataset contains the following numerical data:
    -Sepal length
    =Sepal width
    -Petal length
    -Petal width
  -The target is a categorical variable taht specifies the spicies of the iris flower.
  - On this dataset I trained the following models:
    - KMeansClustering (accuracy 85%)
    - Random forest (accuracy 100%, the dataset is really small)
    - Linear regression (predicted the petal length using the petal width with a mean squared error of 0.0617)

Other models:
  - Trained a linear regression model to predict the amount a person will be charged for insurance(using the insurance dataset found in the datasets folder)
  - Trained a multinomial naive bayes model to predict whether a cumstomer will make a purchase based on the type day (weekday, weekend, holyday), discount, free shipping(discount dataset found in the datasets folder)
    
