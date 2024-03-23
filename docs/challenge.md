# exploration.ipynb

1. Originally, when using the **Seaborn barplot method**, the DS was passing the parameter as positional arguments, however, Seaborn was complaining about it, **we need to pass the "x" and "y" parameters as keyword arguments**, so I made that modification.  

2. There was a problem with the logic of the **"get_period_day" method**, the logic proposed by the DS was **excluding the following hours: 5:00, 19:00, 12:00, 00:00, 18:59, 11:59, 4:59 and 23:59**, observations (flights) with those hours on the "Fecha-I" columns were not getting any value assign, so we ended up with missing values on the 'get_period_day' column. I made some **changes to that specific method to consider those hours as well.**  

3. There was a problem with the logic of the **"is_high_season" method**, the logic proposed by the DS was **excluding the following dates as "high season" (value of 1): DEC 31, MAR 3, JUL 31 AND SEP 30.** I made some **changes to that specific method to include those dates as well as "high season".**  

4. The DS is using the **feature importance from the imbalanced XGBoost model to select the "top 10" features**, then he proceeds to train both XGBoost and LogisticRegression models using those "top 10" features, however, we need to keep in mind that those features are the top ones **only for the Imbalanced XGBoost model**, the Logistic Regression model is built and trained differently, so most likely that model will not have the same "top 10" features.  

5. Another important consideration that the DS might be missing is **the metric being used** for the feature importance plot he gets, the metric being used is the "F1-score", if we are dealing with an imbalanced problem, **I would suggest considering using some other metrics, such as the** [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html), I do not think it is the best option to use the "F1-score" to select the "top 10" features in this specific scenario.  

6. At the end of the notebook, the DS has a "Conclusions" section, he states that "Does not decrease the performance of the model by reducing the features to the 10 most important.", however, the comparison he is using to make that statement might not be appropriate, why? because he is **comparing the result of the 'All Features imbalanced model' vs the 'Top 10 features imbalanced model'**. We already know that those models (the DS is comparing) are imbalanced and thus have a **big bias towards the majority class**, at the point of sometimes predicting all observations on the test set as "0". In my opinion, in order to determine if it might be a good idea to select a subset of features for training, a more appropriate/representative exercise would be to **compare the results of the 'All Features balanced model' vs the 'top 10 features balanced model'**, but the DS did not train a single balanced model with All features.  

7. **With this, the model to be productive must be the one that is trained with the top 10 features and class balancing, but which one?**
I have chosen to use the XGBoost model since it is considered a more robust model, this model puts together multiple "weak learners" using the "ensembling" technique in order to get a "strong learner". These kinds of ensemble models commonly have a good performance when dealing with tabular data, such as in this case. Another advantage is that the XGBoost model is well-optimized from a computational point of view, normally these models train faster especially on larger datasets.  

# model.py

1. I added the functions for building the extra features period_day, high_season and min_diff. Currently the period_day and the high_season features are not being used by the models, however, we might want to consider those two features for the models in the feature, so I thought it would not hurt to have those functions available there.  

2. The 10 features that are being used by the ML model are **features which originally came from categorical columns**, those categorical
columns are being transformed using the "dummy transformation", which creates **1 new column for each category found**. On the Jupyter notebook it is an straightforward transformation, however, **when we operationalize this transformation we need to add some extra logic to handle the following case**: The caller of the model.py module passes some data for either training or predicting, the given data contains the  OPERA column, we apply the dummy transformation, but **that specific dataset does not contain observations for the "Grupo LATAM" category**, in that case the **'get_dummies' method will not create an 'OPERA_Grupo LATAM' column** and we will eventually get an error since the ML model must get that feature. In order to avoid this situation, we need to make sure that the necessary columns are created, if not, we will 'manually' add them as 0-only columns.  

3. During the exploration, model selection, and hyper-parameter tunning phases is a good practice to use a specific random seed for your ML models, for repeatability and consistency, however, **once you move into production you should remove the random seed (or random state) on your models.**  

4. The instructions of this challenge suggested **using the self.model attribute (from the DelayModel class) to save the trained model object**, however, this approach was not working for me and the **'test_model_predict' test cases were failing**. In order to pass all the cases I needed to take a **different approach by serializing the already trained model into a disk file using pickle**, then for the 'predict()' method I load the pre-trained model from the disk and proceed to get the predictions.  


# test_model.py

1. I needed to make a small change in order to avoid **conflicts when using different operating systems**. It is not a good practice to explicitly use either backslash or frontslash. It is **better to use a flexible separator.** In this case I used the separator from the os library. 



