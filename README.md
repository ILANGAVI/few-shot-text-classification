# Meta-Learning for Few-Shot Text Classification — Runnable MVP

## What this project does
This application demonstrates few-shot text classification. A user supplies a small set of labelled examples (support set), and the application predicts the class of a new text.

## Method
This MVP uses TF-IDF text vectors and a prototype-based classifier:
1. Convert each support example into a TF-IDF vector.
2. Average the vectors belonging to each class to form a class prototype.
3. Convert the query text into a vector.
4. Compare the query vector with each prototype using cosine similarity.
5. Predict the class with the highest similarity.

## Important scope note
This is a prototype-based few-shot baseline, not a fully meta-trained Prototypical Network. It does not train an encoder through episodic meta-learning. Do not claim that it does. The app is designed as a runnable demonstration that can be extended with a pretrained sentence encoder and episodic training.

## Requirements
- Windows 10/11
- Python 3.10 or 3.11
- Internet connection for installing packages

## Run on Windows
1. Extract the ZIP file.
2. Open the extracted folder.
3. Double-click `run_app.bat`, or open Command Prompt in this folder and run:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

4. A browser tab should open with the application. If not, copy the Local URL shown in Command Prompt (usually http://localhost:8501) into your browser.

## How to use
- Open the Classify Text tab.
- Edit the Positive, Negative and Neutral examples. Keep one example per line.
- Enter a new sentence.
- Click Classify Text.
- View the predicted class, relative confidence and similarity scores.
- Open Evaluate Model and click Run Evaluation to see sample metrics.

## Limitations
- TF-IDF uses word frequencies and does not fully understand meaning or context.
- The displayed confidence is normalized similarity, not a calibrated probability.
- The built-in evaluation dataset is tiny and illustrative; results are not a reliable benchmark.
- For a full research-grade meta-learning system, train an encoder episodically across multiple tasks and evaluate on held-out tasks.

## Project structure
- `app.py` — Streamlit application and classifier
- `requirements.txt` — Python dependencies
- `run_app.bat` — Windows launch script
- `project_report.md` — Draft project report
- `sample_dataset.csv` — Example labelled texts


## Episodic meta-learning extension
The **Episodic Meta-Learning** tab adds a compact Prototypical Network-style training demonstration:
- A shared linear text encoder is trained over sampled episodes from multiple tasks.
- Each episode uses a support set to calculate class prototypes and query examples to calculate prototypical loss.
- The encoder is optimized with Adam using NumPy.
- The trained encoder can classify a new query using user-provided labelled support examples.

The bundled `meta_tasks.csv` contains three small demonstration tasks: sentiment reviews, support-ticket routing and news-topic classification. The dataset is educational and intentionally small. Report the actual metrics produced by your run; do not present them as large-scale benchmark results.
