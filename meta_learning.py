"""Episodic Prototypical Network-style meta-learning with a trainable linear encoder.

This educational implementation:
- uses a shared TF-IDF input representation;
- samples N-way K-shot episodes from multiple tasks;
- learns a projection matrix using prototypical-network episodic loss;
- evaluates on held-out examples from each task.

It is a compact CPU-friendly demonstration, not a substitute for large-scale research training.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances

def load_tasks(csv_path):
    df = pd.read_csv(csv_path)
    required = {"task_id", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain task_id, text and label columns.")
    df = df.dropna(subset=["task_id", "text", "label"]).copy()
    df["task_id"] = df["task_id"].astype(str)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    return df

def _prepare(df, max_features=600):
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), max_features=max_features,
        sublinear_tf=True, strip_accents="unicode"
    )
    X = vectorizer.fit_transform(df["text"]).toarray().astype(np.float32)
    return vectorizer, X

def _episode_indices(df, rng, n_way=3, k_shot=2, q_query=2):
    tasks = df["task_id"].unique()
    task = rng.choice(tasks)
    subset = df[df["task_id"] == task]
    labels = subset["label"].unique()
    if len(labels) < n_way:
        raise ValueError(f"Task {task} has fewer than {n_way} classes.")
    chosen_labels = rng.choice(labels, size=n_way, replace=False)
    support_idx, query_idx, query_y = [], [], []
    for class_id, label in enumerate(chosen_labels):
        idx = subset.index[subset["label"] == label].to_numpy()
        needed = k_shot + q_query
        if len(idx) < needed:
            raise ValueError(
                f"Task {task}, class {label} needs at least {needed} examples; found {len(idx)}."
            )
        chosen = rng.choice(idx, size=needed, replace=False)
        support_idx.extend(chosen[:k_shot])
        query_idx.extend(chosen[k_shot:])
        query_y.extend([class_id] * q_query)
    return task, chosen_labels, np.asarray(support_idx), np.asarray(query_idx), np.asarray(query_y)

def _episode_loss_and_grad(Xs, Xq, yq, W):
    # Encode support and query using the shared trainable linear projection.
    Zs = Xs @ W
    Zq = Xq @ W
    n_way = int(yq.max()) + 1
    k_shot = Xs.shape[0] // n_way
    prototypes = np.vstack([
        Zs[c*k_shot:(c+1)*k_shot].mean(axis=0) for c in range(n_way)
    ])
    # Negative squared Euclidean distance is the prototypical-network logit.
    diff = Zq[:, None, :] - prototypes[None, :, :]
    logits = -np.sum(diff * diff, axis=2)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True) + 1e-12
    loss = -np.log(probs[np.arange(len(yq)), yq] + 1e-12).mean()

    dlogits = probs
    dlogits[np.arange(len(yq)), yq] -= 1.0
    dlogits /= len(yq)

    # Derivatives of -||query - prototype||^2.
    dZq = np.zeros_like(Zq)
    dProto = np.zeros_like(prototypes)
    for c in range(n_way):
        # d logit_c / d query = -2(query - prototype_c)
        dZq += dlogits[:, c:c+1] * (-2.0 * (Zq - prototypes[c]))
        # d logit_c / d prototype = 2(query - prototype_c)
        dProto[c] = np.sum(
            dlogits[:, c:c+1] * (2.0 * (Zq - prototypes[c])), axis=0
        )

    dZs = np.zeros_like(Zs)
    for c in range(n_way):
        dZs[c*k_shot:(c+1)*k_shot] = dProto[c] / k_shot

    grad_W = Xq.T @ dZq + Xs.T @ dZs
    return float(loss), grad_W

def train_meta_model(csv_path, episodes=250, n_way=3, k_shot=2, q_query=2,
                     embedding_dim=48, learning_rate=0.08, seed=42):
    df = load_tasks(csv_path)
    vectorizer, X = _prepare(df)
    rng = np.random.default_rng(seed)
    # Small random projection, shared across all sampled tasks.
    W = rng.normal(0, 0.08, size=(X.shape[1], embedding_dim)).astype(np.float32)
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    losses = []
    for step in range(1, episodes + 1):
        _, _, s_idx, q_idx, yq = _episode_indices(
            df, rng, n_way=n_way, k_shot=k_shot, q_query=q_query
        )
        loss, grad = _episode_loss_and_grad(X[s_idx], X[q_idx], yq, W)
        # Adam optimizer for stable, compact training.
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad * grad)
        m_hat = m / (1 - 0.9 ** step)
        v_hat = v / (1 - 0.999 ** step)
        W -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        losses.append(loss)
    model = {"vectorizer": vectorizer, "W": W, "n_way": n_way,
             "k_shot": k_shot, "q_query": q_query}
    return model, losses, df

def evaluate_meta_model(model, df, seed=123):
    rng = np.random.default_rng(seed)
    vectorizer, W = model["vectorizer"], model["W"]
    X = vectorizer.transform(df["text"]).toarray().astype(np.float32)
    correct, total = 0, 0
    task_results = []
    for task in df["task_id"].unique():
        subset = df[df["task_id"] == task]
        task_correct, task_total = 0, 0
        for label in subset["label"].unique():
            class_idx = subset.index[subset["label"] == label].to_numpy()
            if len(class_idx) < 3:
                continue
            # Hold out one query; remaining examples form a small support set.
            shuffled = rng.permutation(class_idx)
            support_idx, query_idx = shuffled[:-1], shuffled[-1:]
            support_labels = df.loc[support_idx, "label"].to_numpy()
            query_label = df.loc[query_idx[0], "label"]
            classes = list(dict.fromkeys(support_labels.tolist()))
            prototypes = []
            for cls in classes:
                cls_idx = support_idx[support_labels == cls]
                prototypes.append((X[cls_idx] @ W).mean(axis=0))
            q = X[query_idx] @ W
            distances = euclidean_distances(q, np.vstack(prototypes))[0]
            predicted = classes[int(np.argmin(distances))]
            task_correct += int(predicted == query_label)
            task_total += 1
        if task_total:
            acc = task_correct / task_total
            task_results.append({"Task": task, "Held-out accuracy": acc, "Test examples": task_total})
            correct += task_correct
            total += task_total
    overall = correct / total if total else 0.0
    return overall, pd.DataFrame(task_results)

def predict_with_meta_model(model, support_texts, support_labels, query_text):
    vectorizer, W = model["vectorizer"], model["W"]
    Xs = vectorizer.transform(support_texts).toarray().astype(np.float32) @ W
    Xq = vectorizer.transform([query_text]).toarray().astype(np.float32) @ W
    labels = list(dict.fromkeys(support_labels))
    prototypes = np.vstack([
        Xs[np.asarray(support_labels) == label].mean(axis=0) for label in labels
    ])
    distances = euclidean_distances(Xq, prototypes)[0]
    logits = -distances
    exp = np.exp(logits - logits.max())
    confidence = exp / exp.sum()
    best = int(np.argmin(distances))
    return labels[best], float(confidence[best]), {
        label: float(score) for label, score in zip(labels, confidence)
    }
