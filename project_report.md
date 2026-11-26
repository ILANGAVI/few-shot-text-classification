# META-LEARNING FOR FEW-SHOT TEXT CLASSIFICATION
## Project Report — Demonstration MVP

## 1. Abstract
Text classification assigns a category to a text document or sentence. Conventional supervised classification often requires a large labelled dataset for every new task. Few-shot learning aims to make predictions using only a small number of labelled examples. This project presents a runnable few-shot text classification application with an episodic Prototypical Network-style learning extension. Users provide labelled support examples and a query sentence. The application converts the text into TF-IDF vectors, forms a prototype for each class by averaging its support vectors, and predicts the query class using cosine similarity. A Streamlit interface provides text input, predictions, similarity scores and a small illustrative evaluation workflow.

## 2. Problem Statement
In domains such as customer feedback, support requests, news and rapidly changing topics, collecting large labelled datasets can be time-consuming. A system that can classify new text categories from a small number of examples can reduce the amount of manual labelling required.

## 3. Objectives
- Build an easy-to-use text classification interface.
- Accept a small labelled support set for each category.
- Represent text numerically using TF-IDF.
- Build class prototypes from support examples.
- Classify new query text using cosine similarity.
- Display predictions and similarity-based confidence.
- Demonstrate evaluation using accuracy, precision, recall and F1-score.

## 4. Proposed System
The user enters a few examples for each class and a new query sentence. The application vectorizes the support texts and computes one mean vector (prototype) per class. It vectorizes the query and calculates cosine similarity between the query vector and each prototype. The class with the highest similarity is returned.

## 5. System Workflow
User labelled examples (support set)
→ Text preprocessing and TF-IDF vectorization
→ Class prototype calculation
→ User query text
→ Query vectorization
→ Cosine similarity calculation
→ Predicted class and relative confidence
→ Evaluation and feedback

## 6. Algorithm
1. Read the labelled support examples.
2. Separate examples according to their class labels.
3. Fit a TF-IDF vectorizer on the support text.
4. Transform support examples into feature vectors.
5. Calculate the mean feature vector for each class.
6. Transform the new query text using the same vectorizer.
7. Calculate cosine similarity between the query vector and every class prototype.
8. Select the class with the maximum similarity.
9. Display the predicted label and normalized relative similarity score.

## 7. Modules
### 7.1 Support Set Creation
Allows users to provide a few labelled examples for each class.

### 7.2 Text Vectorization
Uses TF-IDF to convert text into numerical feature vectors.

### 7.3 Prototype Generation
Computes the average vector for all examples belonging to a class.

### 7.4 Few-Shot Prediction
Compares a query vector with class prototypes using cosine similarity.

### 7.5 Evaluation
Shows accuracy, precision, recall and F1-score on a small illustrative test set.

### 7.6 User Interface
Streamlit provides tabs for classification, evaluation and project information.

## 8. Technology Stack
- Language: Python
- Interface: Streamlit
- Text features: Scikit-learn TF-IDF Vectorizer
- Similarity: Cosine similarity
- Data handling: Pandas and NumPy

## 9. Evaluation Metrics
- Accuracy = Correct predictions / Total predictions
- Precision = True positives / (True positives + False positives)
- Recall = True positives / (True positives + False negatives)
- F1-score = 2 × (Precision × Recall) / (Precision + Recall)

The application computes these metrics when the user runs the evaluation tab. The included test set is small and illustrative; its scores should not be presented as generalizable benchmark performance.

## 10. Results
The application demonstrates the complete support-set-to-query prediction flow. It displays the predicted class, relative similarity-based confidence, and class similarity scores. Actual evaluation values must be recorded from the running application after clicking “Run Evaluation”.

## 11. Applications
- Customer review classification
- Customer support ticket routing
- News topic categorization
- Feedback analysis
- Rapid prototyping of new text categories

## 12. Limitations and Future Enhancement
The application includes a TF-IDF baseline and an episodically trained shared linear encoder. The episodic extension samples tasks, support sets and query sets, forms class prototypes, and optimizes a prototypical loss across episodes. The demonstration uses a small hand-curated dataset, so results are educational rather than benchmark-grade. Future work can replace the TF-IDF input with a pretrained BERT or Sentence-BERT encoder, expand the number and diversity of tasks, perform rigorous held-out-domain evaluation, add authentication, store tasks in a database, and deploy to a cloud platform.

## 13. Conclusion
This project provides a working demonstration of few-shot text classification through class prototypes. It allows a user to define categories with a small number of labelled examples and classify new text through similarity comparison. The modular design provides a foundation for future development into a full meta-learning system.


## 14. Episodic Meta-Learning Extension
The project includes three task datasets: sentiment review classification, support-ticket routing and news-topic classification. During each training episode, one task is selected and three classes are sampled. Two support examples per class form the support set, while two separate examples per class form the query set. A shared trainable linear encoder maps TF-IDF vectors into a compact embedding space. The mean embedding of each support class forms its prototype. Negative squared Euclidean distance between each query embedding and the class prototypes produces the episode logits. Cross-entropy loss is calculated on query labels, and the encoder is updated using the Adam optimizer. Repeating this process across episodes trains the shared representation to support prototype-based classification.

## 15. Reproducibility and Reporting
Run the episodic training tab and record the actual held-out accuracy and per-task results shown by the application. Results can vary with episode sampling and the small dataset. Include a screenshot of the training loss chart and evaluation table in the final report. Do not invent or pre-fill performance values.
