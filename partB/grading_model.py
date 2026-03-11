import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer


_WORDNET_POS_MAP = {
    "J": "a",  # adjective
    "V": "v",  # verb
    "N": "n",  # noun
    "R": "r",  # adverb
}


def _ensure_nltk_resources() -> None:
    """
    Ensure required NLTK resources are available.
    To avoid path issues on some systems, unconditionally
    trigger downloads into the default NLTK data directory.
    """
    for name in ("punkt", "averaged_perceptron_tagger", "wordnet"):
        nltk.download(name, quiet=True)


_ensure_nltk_resources()
_lemmatizer = WordNetLemmatizer()


def _normalize_text(text: str) -> str:
    return " ".join(word_tokenize(text.lower()))


def _tokenize(text: str) -> List[str]:
    return word_tokenize(text.lower())


def word_overlap_ratio(ref: str, stu: str) -> float:
    ref_tokens = set(_tokenize(ref))
    stu_tokens = set(_tokenize(stu))
    if not ref_tokens and not stu_tokens:
        return 1.0
    union = ref_tokens | stu_tokens
    if not union:
        return 0.0
    return len(ref_tokens & stu_tokens) / len(union)


def length_difference(ref: str, stu: str) -> float:
    ref_tokens = _tokenize(ref)
    stu_tokens = _tokenize(stu)
    max_len = max(len(ref_tokens), len(stu_tokens), 1)
    return abs(len(ref_tokens) - len(stu_tokens)) / max_len


def pos_overlap_ratio(ref: str, stu: str) -> float:
    ref_tags = [tag for _, tag in pos_tag(_tokenize(ref))]
    stu_tags = [tag for _, tag in pos_tag(_tokenize(stu))]
    if not ref_tags and not stu_tags:
        return 1.0
    ref_set = set(ref_tags)
    stu_set = set(stu_tags)
    union = ref_set | stu_set
    if not union:
        return 0.0
    return len(ref_set & stu_set) / len(union)


def _lemmatize_token(token: str, pos_tag_str: str) -> str:
    wn_pos = _WORDNET_POS_MAP.get(pos_tag_str[0], "n")
    return _lemmatizer.lemmatize(token.lower(), pos=wn_pos)


def root_verb_match(ref: str, stu: str) -> int:
    ref_pos = pos_tag(_tokenize(ref))
    stu_pos = pos_tag(_tokenize(stu))
    ref_root = next((tok for tok, tag in ref_pos if tag.startswith("V")), None)
    stu_root = next((tok for tok, tag in stu_pos if tag.startswith("V")), None)
    if not ref_root or not stu_root:
        return 0
    ref_tag = next(tag for tok, tag in ref_pos if tok == ref_root)
    stu_tag = next(tag for tok, tag in stu_pos if tok == stu_root)
    ref_lemma = _lemmatize_token(ref_root, ref_tag)
    stu_lemma = _lemmatize_token(stu_root, stu_tag)
    return int(ref_lemma == stu_lemma)


@dataclass
class FeatureConfig:
    use_semantic: bool = True
    use_structural: bool = True


class ShortAnswerGrader:
    """
    Lightweight short-answer grading model using surface, semantic, and
    lightweight structural features with an SVR backend.
    """

    def __init__(self, feature_config: FeatureConfig | None = None, random_state: int = 42):
        self.feature_config = feature_config or FeatureConfig()
        self.random_state = random_state

        # Vectorizers for TF-IDF and LSA-style sentence embeddings
        self._tfidf = TfidfVectorizer()
        self._tfidf_lsa = TfidfVectorizer()
        self._svd = TruncatedSVD(n_components=50, random_state=random_state)

        self._svr = SVR(kernel="linear", C=1.0, epsilon=0.1)

    def _fit_vectorizers(self, refs: List[str], stus: List[str]) -> None:
        all_texts = [*refs, *stus]
        self._tfidf.fit(all_texts)
        tfidf_matrix = self._tfidf_lsa.fit_transform(all_texts)
        # Ensure TruncatedSVD uses a valid number of components
        n_features = tfidf_matrix.shape[1]
        if n_features > 0 and self._svd.n_components > n_features:
            self._svd.n_components = max(1, n_features - 1)
        self._svd.fit(tfidf_matrix)

    def _pairwise_tfidf_cosine(self, refs: List[str], stus: List[str]) -> np.ndarray:
        ref_vecs = self._tfidf.transform(refs)
        stu_vecs = self._tfidf.transform(stus)
        # cosine similarity via normalized dot product
        ref_norm = np.linalg.norm(ref_vecs.toarray(), axis=1, keepdims=True) + 1e-8
        stu_norm = np.linalg.norm(stu_vecs.toarray(), axis=1, keepdims=True) + 1e-8
        ref_unit = ref_vecs.toarray() / ref_norm
        stu_unit = stu_vecs.toarray() / stu_norm
        return np.sum(ref_unit * stu_unit, axis=1)

    def _pairwise_sentence_embedding_cosine(self, refs: List[str], stus: List[str]) -> np.ndarray:
        # LSA-style low-dimensional embeddings
        all_texts = [*refs, *stus]
        tfidf_matrix = self._tfidf_lsa.transform(all_texts)
        emb = self._svd.transform(tfidf_matrix)
        n = len(refs)
        ref_emb = emb[:n]
        stu_emb = emb[n:]
        ref_norm = np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-8
        stu_norm = np.linalg.norm(stu_emb, axis=1, keepdims=True) + 1e-8
        ref_unit = ref_emb / ref_norm
        stu_unit = stu_emb / stu_norm
        return np.sum(ref_unit * stu_unit, axis=1)

    def _compute_feature_matrix(self, refs: List[str], stus: List[str]) -> np.ndarray:
        surface_feats = []
        semantic_feats = []
        structural_feats = []

        # Surface feature: TF-IDF cosine similarity (Section 3.3, Table 5)
        tfidf_cosine = self._pairwise_tfidf_cosine(refs, stus)

        for i, (r, s) in enumerate(zip(refs, stus)):
            w_overlap = word_overlap_ratio(r, s)
            len_diff = length_difference(r, s)
            surface_feats.append([tfidf_cosine[i], w_overlap, len_diff])

            if self.feature_config.use_structural:
                pos_overlap = pos_overlap_ratio(r, s)
                root_match = root_verb_match(r, s)
                structural_feats.append([pos_overlap, root_match])

        surface_feats = np.asarray(surface_feats, dtype=float)

        if self.feature_config.use_semantic:
            sem_sim = self._pairwise_sentence_embedding_cosine(refs, stus)
            semantic_feats = sem_sim.reshape(-1, 1)

        if self.feature_config.use_structural:
            structural_feats = np.asarray(structural_feats, dtype=float)

        feature_blocks: List[np.ndarray] = [surface_feats]
        if self.feature_config.use_semantic:
            feature_blocks.append(semantic_feats)
        if self.feature_config.use_structural:
            feature_blocks.append(structural_feats)

        return np.concatenate(feature_blocks, axis=1)

    def fit(self, df: pd.DataFrame, C: float = 1.0, epsilon: float = 0.1) -> None:
        refs = df["reference_answer"].tolist()
        stus = df["student_answer"].tolist()
        self._svr.C = C
        self._svr.epsilon = epsilon

        self._fit_vectorizers(refs, stus)
        X = self._compute_feature_matrix(refs, stus)
        y = df["grade"].to_numpy(dtype=float)
        self._svr.fit(X, y)

    def predict(self, refs: List[str], stus: List[str]) -> np.ndarray:
        X = self._compute_feature_matrix(refs, stus)
        return self._svr.predict(X)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        refs = df["reference_answer"].tolist()
        stus = df["student_answer"].tolist()
        y_true = df["grade"].to_numpy(dtype=float)
        y_pred = self.predict(refs, stus)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        if len(np.unique(y_true)) > 1:
            # Use numpy.corrcoef to avoid depending on SciPy
            corr_matrix = np.corrcoef(y_true, y_pred)
            corr = float(corr_matrix[0, 1])
        else:
            corr = float("nan")
        return {"rmse": float(rmse), "pearson": corr}


def build_toy_short_answer_dataset(n_per_question: int = 50, random_state: int = 42) -> pd.DataFrame:
    """
    Construct a synthetic short-answer grading dataset suitable for CPU-only
    experimentation. The grades are generated heuristically from word overlap
    and noise, so the ground truth is controlled but realistic enough for
    reproduction.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    questions_and_refs: List[Tuple[str, str]] = [
        (
            "What is the role of a prototype program in problem solving?",
            "To simulate the behavior of portions of the desired software product.",
        ),
        (
            "What are the main advantages of object-oriented programming?",
            "Abstraction, modularity, and reusability of code.",
        ),
        (
            "What is the purpose of a data structure in programming?",
            "To organize and store data efficiently for access and modification.",
        ),
    ]

    paraphrases = [
        "simulate the behaviour of some parts of the target program",
        "test ideas for the final system before full implementation",
        "reuse and extend existing components in a modular way",
        "split complex programs into smaller reusable classes",
        "provide organized containers to hold and update data quickly",
        "arrange information so that operations are efficient",
    ]

    rows: List[Dict[str, str | float]] = []

    for q_text, ref_ans in questions_and_refs:
        for _ in range(n_per_question):
            base = ref_ans
            mutation_type = random.choice(["perfect", "partial", "noisy", "off_topic"])

            if mutation_type == "perfect":
                stu = base
            elif mutation_type == "partial":
                tokens = _tokenize(base)
                keep = max(2, int(0.6 * len(tokens)))
                stu = " ".join(tokens[:keep])
            elif mutation_type == "noisy":
                base_tokens = _tokenize(base)
                noise_tokens = random.choice(paraphrases).split()
                slice_len = max(2, int(0.5 * len(base_tokens)))
                stu = " ".join(base_tokens[:slice_len] + noise_tokens[:3])
            else:  # off_topic
                stu = "This answer talks about something unrelated like cooking or sports."

            overlap = word_overlap_ratio(ref_ans, stu)
            length_penalty = length_difference(ref_ans, stu)
            raw_score = 5.0 * overlap - 2.0 * length_penalty + np.random.normal(0.0, 0.4)
            grade = float(np.clip(raw_score, 0.0, 5.0))

            rows.append(
                {
                    "question": q_text,
                    "reference_answer": ref_ans,
                    "student_answer": stu,
                    "grade": grade,
                }
            )

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_paraphrase_challenge_set(random_state: int = 123) -> pd.DataFrame:
    """
    Create a small challenge dataset where student answers are paraphrases
    of the reference answers with intentionally low lexical overlap.

    This is used in Question 3 as a failure-mode test for models that rely
    heavily on surface overlap and shallow similarity features.
    """
    random.seed(random_state)
    np.random.seed(random_state)

    questions_and_refs: List[Tuple[str, str]] = [
        (
            "What is the role of a prototype program in problem solving?",
            "To simulate the behavior of portions of the desired software product.",
        ),
        (
            "What are the main advantages of object-oriented programming?",
            "Abstraction, modularity, and reusability of code.",
        ),
        (
            "What is the purpose of a data structure in programming?",
            "To organize and store data efficiently for access and modification.",
        ),
    ]

    # Manually written paraphrases with different surface forms
    paraphrase_bank: Dict[int, List[str]] = {
        0: [
            "It lets you try out parts of the system in a stripped-down trial version before building the full application.",
            "A prototype is an early, simplified implementation used to experiment with how the software should behave.",
        ],
        1: [
            "It encourages designing programs as reusable components that can be extended and maintained more easily.",
            "Object orientation supports encapsulation and code reuse, which simplifies evolving large systems.",
        ],
        2: [
            "It provides organised ways of arranging information so that operations like lookup and updates are fast.",
            "Data structures are schemes for structuring information to make processing and retrieval efficient.",
        ],
    }

    rows: List[Dict[str, str | float]] = []

    for idx, (q_text, ref_ans) in enumerate(questions_and_refs):
        for stu in paraphrase_bank[idx]:
            # Assign high grades despite low lexical overlap
            grade = 4.5 + np.random.normal(0.0, 0.2)
            grade = float(np.clip(grade, 0.0, 5.0))
            rows.append(
                {
                    "question": q_text,
                    "reference_answer": ref_ans,
                    "student_answer": stu,
                    "grade": grade,
                }
            )

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)




