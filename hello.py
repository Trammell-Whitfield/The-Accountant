# Template

# Forensic Accountant AI System Starting Template
# Implements a basic RAG system for transaction analysis with placeholders for full functionality

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Mock knowledge base: IRS rules and fraud indicators
KNOWLEDGE_BASE = [
    {"id": "IRS1", "text": "Supplies Expense under $2500 is deductible per IRC Section 162."},
    {"id": "IRS2", "text": "Payments over $5000 to new vendors are high-risk for fraud."},
    {"id": "FRAUD1", "text": "Round-number amounts (e.g., $10000) may indicate fraud."},
]

# Mock original financial statements (simplified as totals for now)
ORIGINAL_FINANCIAL_STATEMENTS = {
    "income_statement": {"revenue": 1000000.0, "expenses": 600000.0, "net_income": 400000.0},
    "balance_sheet": {"assets": 2000000.0, "liabilities": 800000.0, "equity": 1200000.0},
}

# Mock ledger data based on provided CSV schema
LEDGER_DATA = pd.DataFrame([
    {"transaction_id": "TXN12345", "date": "2024-03-15", "description": "Office Supplies Purchase", "account": "Supplies Expense", "amount": 500.00},
    {"transaction_id": "TXN12346", "date": "2024-03-20", "description": "Consulting Fee", "account": "Consulting Expense", "amount": 10000.00},
])

# Initialize embedding model for RAG
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for knowledge base
kb_texts = [doc["text"] for doc in KNOWLEDGE_BASE]
kb_embeddings = embedder.encode(kb_texts)

# Mock LLM function (replace with real LLM like GPT via API)
def mock_llm(prompt):
    if "deductible" in prompt:
        return "The transaction is deductible per IRS rules."
    elif "fraud" in prompt:
        return "The transaction is flagged for potential fraud due to high amount and round number."
    else:
        return "Analysis complete, no specific issues found."

# Forensic Accountant AI System Class
class ForensicAccountantAI:
    def __init__(self, knowledge_base, ledger_data, original_statements):
        self.knowledge_base = knowledge_base
        self.ledger_data = ledger_data
        self.original_statements = original_statements

    # Step 1: Ingest Rules (placeholder for full ingestion)
    def ingest_rules(self):
        print("Ingesting state and federal tax rules into knowledge base...")
        # Future: Parse tax codes from official sources and update KNOWLEDGE_BASE
        return self.knowledge_base

    # Step 2: Classify Entries
    def classify_entries(self):
        print("Classifying ledger entries based on tax rules...")
        classifications = {}
        for _, row in self.ledger_data.iterrows():
            query = f"Is {row['description']} of ${row['amount']} in {row['account']} compliant with IRS rules?"
            retrieved_docs = self.retrieve(query)
            augmented_prompt = self.augment(query, retrieved_docs)
            classification = self.generate(augmented_prompt)
            classifications[row["transaction_id"]] = classification
        return classifications

    # Step 3: Scenario Modeling (placeholder)
    def scenario_modeling(self, classifications):
        print("Generating scenario-based financial statements...")
        # Future: Generate multiple financial statements by varying classifications
        scenarios = [self.original_statements]  # Placeholder: only original for now
        return scenarios

    # Step 4: Identify Discrepancies (placeholder)
    def identify_discrepancies(self, scenarios):
        print("Identifying discrepancies between scenarios and original statements...")
        # Future: Compare financial metrics and flag differences
        discrepancies = {"example_discrepancy": "Net income differs by 10% in Scenario 1"}
        return discrepancies

    # Step 5: Counterfactual Analysis (placeholder)
    def counterfactual_analysis(self, discrepancies):
        print("Performing counterfactual analysis on discrepancies...")
        # Future: Explain discrepancies with rule or data changes
        explanations = {"example_discrepancy": "Reclassifying $5000 as non-deductible resolves the difference."}
        return explanations

    # Step 6: Red Flag Analysis
    def red_flag_analysis(self):
        print("Analyzing ledger for potential fraud...")
        flags = {}
        for _, row in self.ledger_data.iterrows():
            query = f"Is {row['description']} of ${row['amount']} in {row['account']} suspicious for fraud?"
            retrieved_docs = self.retrieve(query)
            augmented_prompt = self.augment(query, retrieved_docs)
            flag_result = self.generate(augmented_prompt)
            if "fraud" in flag_result.lower():
                flags[row["transaction_id"]] = flag_result
        return flags

    # Step 7: Summary
    def generate_summary(self, classifications, discrepancies, explanations, flags):
        print("Generating summary of findings...")
        summary = {
            "classifications": classifications,
            "discrepancies": discrepancies,
            "explanations": explanations,
            "red_flags": flags
        }
        return summary

    # RAG Functions
    def retrieve(self, query, top_k=2):
        query_embedding = embedder.encode(query)
        similarities = np.dot(kb_embeddings, query_embedding) / (
            np.linalg.norm(kb_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.knowledge_base[i] for i in top_indices]

    def augment(self, query, retrieved_docs):
        context = "\n".join([doc["text"] for doc in retrieved_docs])
        return f"Query: {query}\nContext: {context}\nAnswer:"

    def generate(self, augmented_prompt):
        return mock_llm(augmented_prompt)

    # User Query Interface
    def answer_user_query(self, user_query):
        retrieved_docs = self.retrieve(user_query)
        augmented_prompt = self.augment(user_query, retrieved_docs)
        return self.generate(augmented_prompt)

# Main execution
def main():
    # Initialize the system
    forensic_ai = ForensicAccountantAI(KNOWLEDGE_BASE, LEDGER_DATA, ORIGINAL_FINANCIAL_STATEMENTS)

    # Run the AI system flow
    rules = forensic_ai.ingest_rules()
    classifications = forensic_ai.classify_entries()
    scenarios = forensic_ai.scenario_modeling(classifications)
    discrepancies = forensic_ai.identify_discrepancies(scenarios)
    explanations = forensic_ai.counterfactual_analysis(discrepancies)
    flags = forensic_ai.red_flag_analysis()
    summary = forensic_ai.generate_summary(classifications, discrepancies, explanations, flags)

    # Print results
    print("\nForensic Accounting Analysis Summary:")
    print(f"Classifications: {summary['classifications']}")
    print(f"Discrepancies: {summary['discrepancies']}")
    print(f"Explanations: {summary['explanations']}")
    print(f"Red Flags: {summary['red_flags']}")

    # Example user query
    user_query = "Is a $500 Supplies Expense deductible?"
    response = forensic_ai.answer_user_query(user_query)
    print(f"\nUser Query: {user_query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()