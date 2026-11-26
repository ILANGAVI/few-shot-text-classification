import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

st.set_page_config(page_title="Few-Shot Text Classifier", page_icon="🧠", layout="wide")

st.title("🧠 Meta-Learning for Few-Shot Text Classification")
st.caption("A prototype-based few-shot text classification demonstration")

st.info(
    "How it works: provide a few labelled examples (support set). "
    "The app converts text into TF-IDF vectors, builds one prototype per class, "
    "and assigns a new text to the most similar prototype."
)

DEFAULTS = {
    "Positive": "I love this product\nThe service was excellent\nThis is wonderful and useful",
    "Negative": "I hate this product\nThe service was terrible\nThis is disappointing and poor",
    "Neutral": "The product arrived today\nThe service is available\nThe package contains three items",
}

def parse_examples(raw):
    return [line.strip() for line in raw.splitlines() if line.strip()]

def build_model(examples_by_class):
    texts, labels = [], []
    for label, examples in examples_by_class.items():
        for text in examples:
            texts.append(text)
            labels.append(label)
    if len(set(labels)) < 2 or len(texts) < 2:
        raise ValueError("Please provide examples for at least two classes.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")
    matrix = vectorizer.fit_transform(texts)
    prototypes = {}
    start = 0
    for label, examples in examples_by_class.items():
        count = len(examples)
        if count:
            prototypes[label] = np.asarray(matrix[start:start+count].mean(axis=0)).ravel()
        start += count
    return vectorizer, prototypes

def predict(text, vectorizer, prototypes):
    vector = vectorizer.transform([text])
    labels = list(prototypes.keys())
    proto_matrix = np.vstack([prototypes[label] for label in labels])
    scores = cosine_similarity(vector, proto_matrix)[0]
    # Convert similarity scores into readable relative confidence values.
    shifted = scores - scores.max()
    exp_scores = np.exp(shifted * 5)
    confidence = exp_scores / exp_scores.sum()
    best = int(np.argmax(scores))
    return labels[best], float(confidence[best]), {
        label: float(score) for label, score in zip(labels, scores)
    }

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Classify Text", "📊 Evaluate Model", "🧠 Episodic Meta-Learning", "ℹ️ About Project"])

with tab1:
    st.subheader("1. Create your few-shot support set")
    st.write("Enter a few labelled text examples. Put one example on each line.")
    cols = st.columns(3)
    examples_by_class = {}
    labels = ["Positive", "Negative", "Neutral"]
    for col, label in zip(cols, labels):
        with col:
            st.markdown(f"**{label} examples**")
            raw = st.text_area(
                f"Examples for {label}",
                value=DEFAULTS[label],
                height=145,
                key=f"examples_{label}"
            )
            examples_by_class[label] = parse_examples(raw)

    st.divider()
    st.subheader("2. Enter a new text")
    query = st.text_area(
        "Text to classify",
        value="The product is really useful and I am happy with it.",
        height=90
    )
    if st.button("🚀 Classify Text", type="primary"):
        try:
            if any(len(v) == 0 for v in examples_by_class.values()):
                st.error("Please enter at least one example for each class.")
            elif not query.strip():
                st.warning("Please enter a text to classify.")
            else:
                vectorizer, prototypes = build_model(examples_by_class)
                predicted, confidence, scores = predict(query, vectorizer, prototypes)
                st.success(f"Predicted class: **{predicted}**")
                st.metric("Relative confidence", f"{confidence*100:.1f}%")
                st.caption("Confidence is a normalized similarity score, not a calibrated probability.")
                st.markdown("**Similarity with each class prototype**")
                st.bar_chart(pd.DataFrame({"Similarity": scores}))
        except Exception as exc:
            st.error(f"Could not classify text: {exc}")

with tab2:
    st.subheader("Evaluate using a small labelled test set")
    st.write("The built-in demonstration dataset is separate from the examples above.")
    test_data = pd.DataFrame([
        ("I absolutely love this excellent product", "Positive"),
        ("The experience was wonderful and amazing", "Positive"),
        ("I am happy with the great service", "Positive"),
        ("This is the worst product I have bought", "Negative"),
        ("Very poor quality and terrible support", "Negative"),
        ("I am unhappy and disappointed", "Negative"),
        ("The order contains two books", "Neutral"),
        ("The meeting is scheduled for Monday", "Neutral"),
        ("The package arrived this morning", "Neutral"),
    ], columns=["Text", "Actual label"])
    st.dataframe(test_data, use_container_width=True, hide_index=True)
    if st.button("📈 Run Evaluation"):
        try:
            eval_examples = {
                "Positive": ["I love this product", "excellent and wonderful service", "happy with this purchase"],
                "Negative": ["I hate this product", "terrible poor quality", "very disappointed and unhappy"],
                "Neutral": ["The product arrived today", "the meeting is on Monday", "package contains two items"]
            }
            vectorizer, prototypes = build_model(eval_examples)
            predictions = [predict(t, vectorizer, prototypes)[0] for t in test_data["Text"]]
            y_true = test_data["Actual label"].tolist()
            accuracy = accuracy_score(y_true, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, predictions, average="weighted", zero_division=0
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy*100:.1f}%")
            c2.metric("Precision", f"{precision*100:.1f}%")
            c3.metric("Recall", f"{recall*100:.1f}%")
            c4.metric("F1-score", f"{f1*100:.1f}%")
            result = test_data.copy()
            result["Predicted label"] = predictions
            result["Result"] = np.where(result["Actual label"] == result["Predicted label"], "Correct", "Incorrect")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.caption("These are results on a tiny illustrative dataset, not a benchmark or a guarantee of real-world performance.")
        except Exception as exc:
            st.error(f"Evaluation error: {exc}")


with tab3:
    st.subheader("Episodic Meta-Learning with Prototypical Networks")
    st.write(
        "This module trains a shared text projection across multiple classification tasks. "
        "Each training episode samples a task, a few support examples per class, and separate query examples."
    )
    st.markdown("**Training tasks:** sentiment reviews, support-ticket routing and news-topic classification.")
    st.markdown("**Episode setup:** 3-way classification, 2 labelled support examples and 2 query examples per class.")
    episodes = st.slider("Training episodes", min_value=50, max_value=500, value=250, step=50)
    if st.button("🧠 Train episodic meta-learning model", type="primary"):
        with st.spinner("Training across sampled classification episodes..."):
            try:
                from meta_learning import train_meta_model, evaluate_meta_model
                model, losses, task_df = train_meta_model(
                    "meta_tasks.csv", episodes=episodes, n_way=3, k_shot=2, q_query=2
                )
                score, per_task = evaluate_meta_model(model, task_df)
                st.session_state["episodic_model"] = model
                st.session_state["episodic_eval"] = per_task
                st.session_state["episodic_loss"] = losses
                st.success("Episodic training completed.")
                c1, c2 = st.columns(2)
                c1.metric("Training episodes", len(losses))
                c2.metric("Held-out query accuracy", f"{score*100:.1f}%")
                st.line_chart(pd.DataFrame({"Episode loss": losses}))
                st.markdown("**Held-out evaluation by task**")
                st.dataframe(per_task, use_container_width=True, hide_index=True)
                st.caption("Results depend on the small demonstration dataset and random episode sampling; they are not a general benchmark.")
            except Exception as exc:
                st.error(f"Meta-learning training failed: {exc}")
    if "episodic_model" in st.session_state:
        st.divider()
        st.markdown("**Try the trained episodic model**")
        support_raw = st.text_area(
            "Support examples (one per line, format: LABEL: example text)",
            value="Positive: I love the excellent service\nPositive: This product is wonderful\nNegative: The service was terrible\nNegative: I dislike this poor product",
            height=120
        )
        query_meta = st.text_input("New query text", value="I am very happy with this purchase")
        if st.button("Predict with episodic model"):
            try:
                from meta_learning import predict_with_meta_model
                texts, labels = [], []
                for line in support_raw.splitlines():
                    if ":" not in line:
                        continue
                    label, text = line.split(":", 1)
                    if label.strip() and text.strip():
                        labels.append(label.strip())
                        texts.append(text.strip())
                if len(set(labels)) < 2:
                    st.error("Enter at least two labels with examples.")
                else:
                    pred, conf, scores = predict_with_meta_model(
                        st.session_state["episodic_model"], texts, labels, query_meta
                    )
                    st.success(f"Predicted class: **{pred}**")
                    st.metric("Relative confidence", f"{conf*100:.1f}%")
                    st.bar_chart(pd.DataFrame({"Relative confidence": scores}))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab4:
    st.subheader("Project overview")
    st.markdown("""
    **Objective:** Build a text classification system that can adapt to new categories using only a small number of labelled examples.

    **Method used in this runnable MVP:**
    1. Collect a few labelled examples (support set).
    2. Convert text into TF-IDF feature vectors.
    3. Calculate the mean vector (prototype) for each class.
    4. Convert a query text into a vector.
    5. Compare the query with each class prototype using cosine similarity.
    6. Return the closest class and relative similarity-based confidence.

    **Important academic note:** This implementation is a prototype-based few-shot baseline. It demonstrates the support-set/prototype/query workflow, but it does not perform episodic meta-training of a neural Prototypical Network. A full meta-learning extension would train an encoder across many sampled classification episodes.
    """)
    st.markdown("**Technology:** Python, Streamlit, Pandas, NumPy and Scikit-learn.")
    st.markdown("**Possible applications:** customer reviews, support tickets, news topics and feedback categorization.")
