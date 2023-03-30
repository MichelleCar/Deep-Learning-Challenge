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

Source:
* https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414
* https://blog.purestorage.com/purely-informational/deep-learning-vs-neural-networks/ 
* https://aws.amazon.com/what-is/neural-network/

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

**Step 1** 

Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.


### Results: Using bulleted lists and images to support your answers, address the following questions:

#### Data Preprocessing
Q: What variable(s) are the target(s) for your model?
Q: What variable(s) are the features for your model?
Q: What variable(s) should be removed from the input data because they are neither targets nor features?

#### Compiling, Training, and Evaluating the Model
Q: How many neurons, layers, and activation functions did you select for your neural network model, and why?
Q: Were you able to achieve the target model performance?
Q: What steps did you take in your attempts to increase model performance?

### Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
