# RECOMMENDATION-SYSTEM
**COMPANY**        : CODTECH IT SOLUTIONS
**NAME**           : EDAMALAPATI MOTHILAL CHOWDARY
**INTERN ID**      : CT12JJP
**DOMAIN**         : MACHINE LEARNING
**BATCH DURATION** : January 5th,2025 to March 5th,2025
**MENTOR NAME**    : NEELA SANTHOSH

# DESCRIPTION OF TASK :
**Movie Recommendation System Using Neural Networks with Embeddings :**
This code implements a movie recommendation system based on collaborative filtering using a neural network with embeddings. The dataset used is the MovieLens 100K dataset, which contains 100,000 user ratings for different movies. This implementation leverages TensorFlow/Keras to train a model that predicts movie ratings based on past user preferences. Additionally, it incorporates bias terms for better accuracy and evaluates the model using Root Mean Squared Error (RMSE).

**1. Loading the MovieLens Dataset :**
The dataset is obtained from GroupLens, a research lab that provides open datasets for recommender system experiments. It is loaded using pandas, and the following preprocessing steps are applied:

Dropping the timestamp column as it is not required for our model.
Mean-centering the ratings, which helps the model learn relative preferences instead of absolute ratings.
Adjusting user and item indices to start from 0, ensuring compatibility with TensorFlow’s embedding layers.
**2. Splitting the Data into Training and Test Sets :**
To evaluate the performance of the recommendation system, the dataset is divided into training (80%) and test (20%) subsets using train_test_split from sklearn. The training set is used to train the model, while the test set is used for performance evaluation.

**3. Building the Neural Network Model :**
The model is built using deep learning techniques to learn latent representations (embeddings) of users and movies. The following layers are defined:

**Input Layers:** Separate inputs for users and movies.
**Embedding Layers:**
Each user and movie is mapped to a 50-dimensional embedding vector that captures their preferences.
These embeddings are learned during training.
**Bias Layers:**
Each user and movie has an additional bias term, helping capture individual tendencies.
This is implemented using additional embedding layers of size (num_users, 1) and (num_items, 1).
**Dot Product Layer:**
The dot product of user and movie embeddings is computed to estimate user preference for the movie.
**Addition Layer:**
The bias values of users and items are added to the dot product output.
**Output Layer:**
A fully connected Dense(1, activation='linear') layer ensures a single continuous value as the predicted rating.
**Loss Function and Optimizer:**
The model is compiled with Mean Squared Error (MSE) as the loss function and Adam optimizer for efficient training.
**4. Training the Model :**
The model is trained for 10 epochs using a batch size of 64 to optimize performance. The input to the model consists of:

User IDs (as an array)
Item IDs (Movie IDs)
Actual Ratings (Target Values)
The neural network learns to adjust the embedding vectors and bias terms to minimize the error between predicted and actual ratings.

**5. Evaluating the Model (RMSE Calculation) :**
After training, the model is evaluated using Root Mean Squared Error (RMSE), a common metric for recommender systems.

Predictions are made on the test set.

RMSE is computed using the formula:

𝑅
𝑀
𝑆
𝐸
=
1
𝑁
∑
(
𝑦
𝑡
𝑟
𝑢
𝑒
−
𝑦
𝑝
𝑟
𝑒
𝑑
)
2
RMSE= 
N
1
​
 ∑(y 
true
​
 −y 
pred
​
 ) 
2
 
​
 
A lower RMSE indicates better model performance.

**6. Recommendation Function :**
A function recommend_items(user_id, num_recommendations=5) is implemented to generate recommendations:

Extracts unique movie IDs.
Filters out movies already rated by the user.
Predicts ratings for unrated movies.
Sorts the predictions in descending order.
Returns the top num_recommendations movies.
The function ensures users receive personalized movie recommendations based on learned preferences.

**7. Example Recommendation for a User :**
To test the recommendation system, we generate movie suggestions for User 1 (indexed as 0 after adjustment). The system provides a list of highly-rated movies that the user has not yet seen.
