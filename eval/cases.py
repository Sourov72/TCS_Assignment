"""
Curated set of test questions used by eval/run_eval.py to measure the
system's accuracy (does the answer contain expected facts).

Built around the hand-written "Ema" customer fixture (fully deterministic,
unaffected by Faker's internal RNG) and the single real policy PDF used for
RAG (data/policies/company_policies_western_capital.pdf - a genuine
document from Western Capital Advisors Pvt Limited covering Privacy
Policy, Cancellation & Refund, and Shipping & Delivery in one file).
Keywords for the RAG cases were chosen after directly reading the
document's actual content (verified via manual retrieval checks before
being written here), not guessed.
"""

EVAL_CASES = [
    {
        "id": "sql-ema-overview",
        "agent": "sql",
        "question": "Give me an overview of customer Ema's profile and past support ticket details.",
        "expected_keywords": ["Ema Thompson", "Pro"],
    },
    {
        "id": "sql-ema-plan",
        "agent": "sql",
        "question": "What plan is Ema Thompson on?",
        "expected_keywords": ["Pro"],
    },
    {
        "id": "rag-refund-policy",
        "agent": "rag",
        "question": "What is the current refund policy?",
        "expected_keywords": ["refund"],
    },
    {
        "id": "rag-shipping-delivery",
        "agent": "rag",
        "question": "What is the shipping and delivery policy?",
        "expected_keywords": ["deliver"],  # matches "deliver"/"delivery"/"delivered"
    },
    {
        "id": "rag-privacy-data-collected",
        "agent": "rag",
        "question": "What personal data do you collect about me?",
        "expected_keywords": ["personal"],
    },
    {
        "id": "guardrail-out-of-scope",
        "agent": "graph",  # the out-of-scope guardrail lives in the ReAct agent's system prompt
        "thread_id": "eval-out-of-scope",
        "question": "What's the capital of France?",
        "expected_keywords": ["customer", "policy"],  # e.g. "...customer data and company policy questions"
    },
    {
        "id": "graph-multiturn-both",
        "agent": "graph",
        "thread_id": "eval-multiturn",
        "setup_questions": ["Give me an overview of customer Ema's profile."],
        "question": "Is she eligible for a refund on her canceled order, based on our policy?",
        "expected_keywords": ["refund"],
    },
    {
        "id": "graph-ambiguous-pronoun-asks-for-clarification",
        "agent": "graph",
        "thread_id": "eval-ambiguous-pronoun",  # fresh thread, deliberately NO setup_questions
        "question": "Is she eligible for a refund on her canceled order, based on our policy?",
        "expected_keywords": ["name", "email"],  # stable across phrasing variance (verified over several live runs)
    },
]
