"""Medical vs non-medical question gate: TF-IDF features + logistic regression.

Scope policy (docs/ood_gate.md): in-domain = questions seeking information about human health, disease,
symptoms, treatment, medicines, prevention, nutrition for health, mental, reproductive and maternal health,
and public health. Anything else (including veterinary, health-sector jobs, prices and admin) is declined.
Only scikit-learn is needed, so the app can import this without torch.
"""
import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

FEATURE_SETS = ("word", "char", "word+char")


def build_pipeline(features: str = "word+char", C: float = 1.0, seed: int = 42) -> Pipeline:
    parts = []
    if "word" in features:
        parts.append(("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)))
    if "char" in features:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                              sublinear_tf=True, max_features=200_000)))
    return Pipeline([
        ("features", FeatureUnion(parts)),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=seed)),
    ])


class MedicalGate:
    """Wraps a fitted pipeline and the decision threshold chosen on validation data."""

    def __init__(self, pipeline: Pipeline, threshold: float, meta: dict | None = None):
        self.pipeline, self.threshold, self.meta = pipeline, threshold, meta or {}

    def prob_medical(self, texts: list[str]) -> list[float]:
        return self.pipeline.predict_proba(texts)[:, 1].tolist()

    def is_medical(self, text: str) -> tuple[bool, float]:
        p = self.prob_medical([text])[0]
        return p >= self.threshold, p

    def save(self, path) -> None:
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold,
                     "meta": {**self.meta, "sklearn_version": sklearn.__version__}}, path)

    @classmethod
    def load(cls, path) -> "MedicalGate":
        d = joblib.load(path)
        return cls(d["pipeline"], d["threshold"], d["meta"])
