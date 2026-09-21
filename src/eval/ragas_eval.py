"""
Runs the agent against a fixed testset, scores answers with RAGAS
(faithfulness + answer relevancy, both reference-free so they only need
the question + retrieved context, not a hand-labeled "correct answer"),
and logs everything to MLflow.

This is what CI runs on every push (.github/workflows/ci.yml) — a real
"does quality regress" signal, not a screenshot of one good demo run.

Run:  python -m src.eval.ragas_eval
View: mlflow ui   (then open http://localhost:5000)
"""
import json
import os
from pathlib import Path

# Must be set before any ML library (torch, numpy, etc.) is imported.
# Works around a common Windows issue where multiple packages bundling
# their own Intel OpenMP runtime (libiomp5md.dll) collide when loaded
# into the same process — harmless to set even where it isn't needed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Imported first, deliberately: this pulls in sentence-transformers/torch
# before mlflow/datasets/ragas get a chance to. On Windows the two sides
# have been fighting over a native DLL slot when torch loads second —
# controlling load order is a cheap thing to try before anything drastic.
from src.agents.graph import run_agent

import mlflow
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness

from src.config import settings

TESTSET_PATH = Path(__file__).parent / "testset.json"


def _extract_contexts(steps: list[dict]) -> list[str]:
    return [s["tool_result"] for s in steps if "tool_result" in s]


def run_eval() -> dict:
    testset = json.loads(TESTSET_PATH.read_text())

    questions, answers, contexts_list, ground_truths = [], [], [], []
    for item in testset:
        result = run_agent(item["question"], user_id=None, session_id=None)
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts_list.append(_extract_contexts(result["steps"]) or [""])
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    judge_llm = LangchainLLMWrapper(
        ChatGroq(api_key=settings.GROQ_API_KEY, model=settings.GROQ_TEXT_MODEL)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    )

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    scores_df = scores.to_pandas()

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("docsight-rag-eval")
    with mlflow.start_run():
        mlflow.log_metric("faithfulness", scores_df["faithfulness"].mean())
        mlflow.log_metric("answer_relevancy", scores_df["answer_relevancy"].mean())
        mlflow.log_param("model", settings.GROQ_TEXT_MODEL)
        mlflow.log_param("testset_size", len(testset))
        scores_df.to_csv("./ragas_scores.csv", index=False)
        mlflow.log_artifact("./ragas_scores.csv")

    return {
        "faithfulness": scores_df["faithfulness"].mean(),
        "answer_relevancy": scores_df["answer_relevancy"].mean(),
    }


if __name__ == "__main__":
    results = run_eval()
    print(f"Faithfulness:     {results['faithfulness']:.3f}")
    print(f"Answer relevancy: {results['answer_relevancy']:.3f}")
