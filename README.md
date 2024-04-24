This project is dedicated to developing an intelligent system capable of automatically generating descriptive tags for various food items. Leveraging Natural Language Processing (NLP) techniques, this project aims to streamline the process of creating engaging and contextually relevant food description tags, catering to the increasing demand for efficient content generation in the food industry.
For instance,
Idly - South Indian, Protein Rich, Breakfast, Baked Items, etc.
Ragi Dosa - South Indian, Diabetic Friendly, Millet Based, Pregnancy friendly, etc.

Data:
The project relies on a diverse dataset comprising food item names and descriptive tags about the dish. This dataset would encompass various cuisines, culinary styles, and naming conventions to ensure the model's adaptability to various food-related contexts.
The data was collected using Open AI’s API key by specifying a descriptive prompt explaining the required input features and output tags corresponding to each dish in a set of dish names.

Workflow:

Data Collection
Generate a list  of dishes (predominantly Indian and also other cuisines) to create the database.
Utilizes the OpenAI API, specifically the GPT-3.5 model, to collect comprehensive information for each dish based on a predefined prompt. The prompt specifies input features such as main ingredients, nutritional content, region of popularity, and output classifications including cuisine type, meal type, and health considerations.
After receiving responses from the API, the code stores data about each dish as a JSON file. 

About the Dataset:
Features:
Dish name
Main Ingredients (list) Nutritional value (per serve): 
Calories
Protein in g
Sugar in g
Saturated Fat in g
Glycemic index
Region where it’s famous
Cooking method
100-word description
Preparation time
Spice Level
Contains allergens

Output Categories:
Cuisine
Meal Type (Breakfast/Lunch/Dinner/Snack/etc)
Dietary preference(Veg/non-veg/vegan)
Protein-rich
Diabetic friendly
Pregnancy friendly

The JSON files are processed and loaded as a Pandas DataFrame/
Check for any missing or null values.
Analyze the distribution of numeric columns like spice level, preparation time, etc.
Check for the different classes under each categorical variable like cuisine, dietary preferences.
It was identified that the ‘Cuisine’ feature had a lot of classes with very few data points and the dishes were predominantly of Indian origin. Classes with low frequency have been combined.
Hence, the cuisines were combined to form two classes: Indian and Continental.
Visualizations: The textual features like ingredients, cooking method, description, and region were visualized using Word Clouds. Data distribution of few other columns analyzed using count plots.

Preprocessing for Text Classification Modelling:
Numeric columns: 
Standard scaler applied to numeric columns like nutritional value, preparation time, etc.

Text based columns (Sentence transformers followed by PCA):
Utilized Sentence Transformer model to generate embeddings for text columns, such as 'Description' and 'CookingMethod' to capture the semantic information and relationships between words.
Applied PCA to reduce dimensionality of the sentence embeddings.
Identified optimal number of reduced dimensions using elbow plot of cumulative variance explained by principal components.

Embedding size for Description:128 and Cooking Method: 32

Word2Vec Embeddings
Word2Vec embeddings are used to process other text-based columns with one or two words like Dish Name, Ingredients and Region.

One-hot Encoding
Performed one-hot encoding for categorical variables like Cuisine and Dietary Preference.
Overall, the preprocessing pipeline involves scaling numeric data, one-hot encoding of categorical features, transforming text data into numerical embeddings using both Sentence Transformer and Word2Vec models, reducing dimensionality, and preparing the data for model training.

Model Building
Training and Evaluation: Models are trained and evaluated using training and testing datasets. Metrics include accuracy scores and classification reports.
Initially, one-shot prediction for all tags was tried using MultiOutput Classifiers. However, the model did not perform well.
Individual Binary Classification: Various classifiers including Logistic Regression, Support Vector Classifier, Random Forest Classifier and XGBoost were trained for each tag and the best model was identified to predict individual binary labels (Protein-Rich, Diabetic-Friendly, Pregnancy-Safe).
Multi-label Classification: MultiOutputClassifiers are used for multi-label prediction (DietaryPreference, Cuisine).


Model evaluation:
Category Test Accuracy Best model
Protein-Rich 92.68% XGBoost
Pregnancy-Safe 89.02% Logistic Regression
Diabetic Friendly 90.24% XGBoost
Dietary Preference 90.24% XGBoost
Cuisine 93.90% Logistic Regression


BERT Classification
The previously built models take in various input features to generate descriptive tags. In real time, it is difficult to find such detailed information about each dish. Hence, we propose to build a BERT-based sequence classification model which takes a textual description about the dish as input and generates tags.

Implementation
Tokenizer used: ‘bert-base-cased’
Model architecture: ‘distilbert-base-uncased'

Evaluation metrics: It is observed that although accuracy was good, recall score was poor for the final checkpoint model. Hence, hyperparameter tuning was performed to find the ideal number of epochs which gave the best f1-score.

Performance Metrics:
Accuracy 87.81%
Precision 81.63%
Recall 97.61%
f1-score 88.89%

