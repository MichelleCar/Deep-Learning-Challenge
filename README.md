# Neural Networks and Deep Learning
## Case Study: Alphabet Soup Charity
![How-AI-and-ML-Impact-Business-Management-and-Drive-Value-1](https://user-images.githubusercontent.com/115101031/228979010-bc3d3bab-3f3d-43dc-9426-22a8f9fbe607.jpeg)

## Neural Networks and Deep Learning Described

Deep learning defines an approach to artificial intelligence structured like the neural networks not unlike the human brain.

Neural networks are a type of machine learning, in which a computer learns to perform tasks by analyzing training examples, defined as "deep learning." 
A neural network, like the human brain, contains thousands or even millions of simple processing nodes that are densely interconnected. An individual node might be connected to several nodes in the layer beneath it, from which it receives data, and several nodes in the layer above it, to which it sends data. When a neural net is being trained, data is fed to the bottom layer — the input layer — and it passes through the succeeding layers, getting multiplied and added together in complex ways, until it finally arrives, radically transformed, at the output layer. 

“Neural network” and “deep learning” are often used interchangeably.  However, there are some nuanced differences between them. While both are subsets of machine learning, a neural network mimics the way the biological neurons in the human brain work, while a deep learning network comprises several layers of neural networks.

Neural networks can be trained to “think” and identify hidden relationships, patterns, and trends in data, within context. In the first step of the neural network process, the first layer receives the raw input data; then, each consecutive layer receives the output from the preceding layer. Each layer contains a database that stores all the network has previously learned, as well as programmed or interpreted rules. Processing continues through each layer until the data reaches the output layer, which produces the eventual result.

Deep learning, also a subset of machine learning, uses algorithms to recognize patterns in complex data and predict outputs. Unlike machine learning algorithms, which require labeled data sets, deep learning networks don't require labeled data sets to perform feature extraction with less reliance on human input. It’s called deep learning because of the number of hidden layers used in the deep learning model. While a basic neural network comprises an input, output, and hidden layer, a deep neural network has multiple hidden layers of processing. These additional layers give deep learning systems the ability to make predictions with greater accuracy, but they require millions of sample data points and hundreds of hours of training when compared to a simpler neural network.

Neural networks, and by extension deep learning, can help computers make intelligent decisions with limited human assistance. This is because they can learn and model the relationships between input and output data that are nonlinear and complex.

<sub>Sources:</sub>
<sub>1) https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414</sub>
<sub>2) https://blog.purestorage.com/purely-informational/deep-learning-vs-neural-networks/ </sub>
<sub>3) https://aws.amazon.com/what-is/neural-network/</sub>

## Case Study: Background
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special considerations for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

## Case Study: Report

### Overview of the analysis: Explain the purpose of this analysis.

**Step 1:** 
Using my knowledge of Pandas and scikit-learn’s StandardScaler(), I preprocessed the dataset, including:
* Read in the charity_data.csv to a Pandas DataFrame and removing data that won't add value to our results
* Identifying the target for the model
* Identifying the features that will work best for the model by determining the number of unique values and the number of data points for each unique value
* I used the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, Other, and then check if the binning was successful.
* I used pd.get_dummies() to encode categorical variables.
* I split the preprocessed data into a features array, X, and a target array, y, and used the arrays and the train_test_split function to split the data into training and testing datasets.
* I then scaled the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

**Step 2:** 
Using my knowledge of Google Colab, TensorFlow, I designed a neural network/deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset, taking into consideration how many inputs there are before determining the number of neurons and layers in the model. Finally, I compiled, trained, and evaluated the binary classification model to calculate the model’s loss and accuracy.  This included:
* Creating a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
* Creating the first hidden layer and choosing an appropriate activation function, and taking into consideration if a second hidden layer is necessary.
* Creating an output layer with an appropriate activation function.

**Step 3:** 
Using the structure, knowledge, and results of the previous steps, I optimized the model to achieve a target predictive accuracy higher than 75%.  This included:
* Adjusting the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    * dropping more or fewer columns
    * creating more bins for rare occurrences in columns
    * increasing or decreasing the number of values for each bin
    * adding more neurons to a hidden layer
    * adding more hidden layers
    * using different activation functions for the hidden layers
    * adding or reducing the number of epochs to the training regimen.

### Results
The purpose of this analysis is to create a deep learning neural network to predict the success rate of applicants who receive funding from Alphabet Soup Charity. The dataset contains more than 34,000 organizations, with information related to their application type, affiliated sector of industry, government organization classification, use case for funding, income classification, funding amount requested, and whether the money was used effectively. 

The analysis involves preprocessing the dataset by dropping unnecessary columns, encoding categorical variables, and splitting the data into training and testing datasets. The neural network model is then designed, trained, and evaluated to determine its loss and accuracy. Finally, the model is optimized using various methods such as adjusting input data, adding more neurons and hidden layers, using different activation functions, and adjusting the number of epochs. The ultimate goal is to achieve a predictive accuracy higher than 75% and save the optimized model as an HDF5 file.


#### Data Preprocessing
In a dataset that may hold thousands of unique values that may be difficult to handle by a neural network, a good tool would encode the meaning of the categories in some meaningful way while keeping the number of dimensions relatively low.  In essence, we are trying to develop a model that recognizes patterns in the data.  As a rule, categorical data is a set of symbols that describe a certain higher level attribute of some object, system or entity of interest. In the dataset provided, we have to determine features that can be replicated and that can provide a reliable and consistent training template for the model.  As such, a categorical variable is a variable that can take on one of a limited, and usually fixed number of possible values, assigning each individual or other unit of observation to a particular group or nominal category on the basis of some qualitative property.

After preprocessing the dataset by dropping unnecessary columns and encoding categorical variables:

*__Question: What variable(s) are the target(s) for your model?__* 
* The target variable is **IS_SUCCESSFUL**.  As such, it functions as a binary classifier, where the results determine two possible outcomes: either the decision to fund a particular organization is successful or not.

*__Question: What variable(s) are the features for your model?__*
* Reviewing the counts of unique values in each column, and focussing on those that had more than 10 unique values, we narrowed available features for our model to: **NAME, APPLICATION_TYPE, CLASSIFICATION**
* Each have unique counts greater than 10 and provide ideal qualitative properties for training the model


*__Question: What variable(s) should be removed from the input data because they are neither targets nor features?__*
* As part of the preprocessing phase for the analysis, the column labelled EIN was excluded, providing organizational IDs, and as a result, of little value to a predictive analysis
* With fewer than 10 unique values, AFFILIATION, USE_CASE, ORGANIZATION, STATUS< INCOME_AMT, SPEICAL_CONSIDERATIONS
* Due to its high variability/subjectivity, I eliminated **ASK_AMT** as a useful feature

NAME                      19568
APPLICATION_TYPE             17
AFFILIATION                   6
CLASSIFICATION               71
USE_CASE                      5
ORGANIZATION                  4
STATUS                        2
INCOME_AMT                    9
SPECIAL_CONSIDERATIONS        2
ASK_AMT                    8747
IS_SUCCESSFUL                 2


#### Compiling, Training, and Evaluating the Model

*__Question: How many neurons, layers, and activation functions did you select for your neural network model, and why?__*
* Features used: APPLICATION_TYPE, CLASSIFICATION
* Number of hidden layers: Two (with 80 and 30 neurons, respectively)
* Activation Functions: I used ReLU (activation function) for the first and second hidden layers.  ReLU is linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance. I selected Sigmoid for the output layer.  If we use a linear activation function in a neural network (such as ReLU), then this model can only learn linearly separable problems. However, with the addition of just one hidden layer and a sigmoid activation function in the hidden layer, the neural network can easily learn a non-linearly separable problem. Using a non-linear function produces non-linear boundaries and hence, the sigmoid function can be used in neural networks for learning complex decision functions. It is well-suited for binary classification tasks.
* Trainable Parameters: 5,981

*__Question: Were you able to achieve the target model performance?__*
Loss value implies how well or poorly a certain model behaves after each iteration of optimization. Typically, with neural networks, we seek to minimize the error.  Ideally, one would expect the reduction of loss after each, or several, iterations.  Accuracy is more straightforward. It measures how well our model predicts by comparing the model predictions with the true values in terms of percentage.
* After running 100 epochs, our results reflected a 0.56 loss and 73% accuracy.  We did not reach our target performance.

<img width="423" alt="Screenshot 2023-03-30 at 9 14 19 PM" src="https://user-images.githubusercontent.com/115101031/228999653-92af98f1-53b2-47d8-831c-cd419cfa86b2.png"><img width="417" alt="Screenshot 2023-03-30 at 9 14 36 PM" src="https://user-images.githubusercontent.com/115101031/228999659-6667c9ca-b81d-462d-a15e-d91aa27f21bd.png">

*__Question: What steps did you take in your attempts to increase model performance?__*

**Optimizing Model 1**
* Features used: NAME, APPLICATION_TYPE, CLASSIFICATION
* Number of hidden layers: Three (with 20, 27 and 3 neurons, respectively)
* Activation Functions: ReLU (first hidden layer), Sigmoid (second & third hidden layer, outer layer)
* Trainable Parameters: 6,155
* Epochs: 100
* Loss: 0.46
* Accuracy: 78%
* Target reached: Yes

<img width="358" alt="Screenshot 2023-03-30 at 9 44 06 PM" src="https://user-images.githubusercontent.com/115101031/229001714-d67ccb5e-d9de-4445-8d44-b9ca6c60a013.png"><img width="342" alt="Screenshot 2023-03-30 at 9 44 18 PM" src="https://user-images.githubusercontent.com/115101031/229001732-f513ae6a-08b5-4eaa-a94c-0fa0a1550fb3.png">

**Optimizing Model 2**
* Features used: NAME, APPLICATION_TYPE, CLASSIFICATION
* Number of hidden layers: Three (with 7, 14 and 21 neurons, respectively)
* Activation Functions: ReLU (first, second and third hidden layer), Sigmoid (outer layer)
* Trainable Parameters: 2,318
* Epochs: 100
* Loss: 0.46
* Accuracy: 78%
* Target reached: Yes

<img width="345" alt="Screenshot 2023-03-30 at 9 47 38 PM" src="https://user-images.githubusercontent.com/115101031/229002182-c60bab99-86fa-41bc-96f2-9da7eb28766f.png"><img width="342" alt="Screenshot 2023-03-30 at 9 47 50 PM" src="https://user-images.githubusercontent.com/115101031/229002208-511c90d1-d726-4515-8ac6-1fc05a135ed1.png">

**Optimizing Model 3**
* Features used: NAME, APPLICATION_TYPE, CLASSIFICATION
* Number of hidden layers: Two (with 80 and 30, respectively)
* Activation Functions: ReLU (first, second hidden layer), Sigmoid (outer layer)
* Trainable Parameters: 37,821
* Epochs: 100
* Loss: 0.48
* Accuracy: 79%
* Target reached: Yes

<img width="342" alt="Screenshot 2023-03-30 at 9 51 36 PM" src="https://user-images.githubusercontent.com/115101031/229002708-b8e4d5e5-a521-475a-9d78-43ba576d8b14.png"><img width="342" alt="Screenshot 2023-03-30 at 9 51 44 PM" src="https://user-images.githubusercontent.com/115101031/229002724-681c5217-431e-42ef-b80f-65fdfbcb5da1.png">

### Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
In summary, the deep learning model using TensorFlow and Keras was able to achieve a predictive accuracy of 78% in classifying the success of organizations funded by Alphabet Soup based on their features. The model underwent several optimization attempts, including dropping columns, binning categorical variables, adding hidden layers and neurons, and trying different activation functions, among other adjustments. While the target predictive accuracy of 75% was achieved, it required significant optimization attempts to reach that level.

One recommendation for solving this classification problem would be to try a different type of model, such as a Random Forest Classifier or a Support Vector Machine (SVM). These models have been shown to be effective in binary classification problems and may be able to achieve a higher accuracy without the need for extensive optimization attempts. Additionally, they can handle both numerical and categorical variables and can handle outliers and imbalanced datasets well, which may be present in this dataset. Therefore, it may be worth exploring these alternative models as a potential solution to the classification problem.
