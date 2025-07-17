# * Set up data models and basic generators
# * Create simple RAG system

# ===== DATA MODELS =====
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Optional, Any
from enum import Enum
import uuid
import json

class TransactionType(Enum):
    INCOME = "income"
    EXPENSE = "expense"
    DEDUCTION = "deduction"
    TRANSFER = "transfer"

class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class Transaction:
    """Core transaction data model matching CSV schema."""
    transaction_id: str
    date: date
    description: str
    account: str
    amount: float
    transaction_type: TransactionType = TransactionType.EXPENSE
    payee_payer: str = ""
    account_number: str = ""
    tax_type: str = ""
    fraud_flag: bool = False
    notes: str = ""
    
    def __post_init__(self):
        """Validate transaction data."""
        if self.amount <= 0:
            raise ValueError("Transaction amount must be positive")
        if not self.transaction_id:
            self.transaction_id = f"TXN{uuid.uuid4().hex[:8].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'transaction_id': self.transaction_id,
            'date': self.date.isoformat(),
            'description': self.description,
            'account': self.account,
            'amount': self.amount,
            'transaction_type': self.transaction_type.value,
            'payee_payer': self.payee_payer,
            'account_number': self.account_number,
            'tax_type': self.tax_type,
            'fraud_flag': self.fraud_flag,
            'notes': self.notes
        }

@dataclass
class FinancialStatement:
    """Financial statement data model."""
    revenue: float
    total_expenses: float
    operating_expenses: float
    cost_of_goods_sold: float
    gross_profit: float
    net_income: float
    total_assets: float
    total_liabilities: float
    equity: float
    cash_flow: float
    statement_date: date
    
    def __post_init__(self):
        """Calculate derived values."""
        if self.gross_profit == 0:
            self.gross_profit = self.revenue - self.cost_of_goods_sold
        if self.net_income == 0:
            self.net_income = self.gross_profit - self.operating_expenses
        if self.equity == 0:
            self.equity = self.total_assets - self.total_liabilities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'revenue': self.revenue,
            'total_expenses': self.total_expenses,
            'operating_expenses': self.operating_expenses,
            'cost_of_goods_sold': self.cost_of_goods_sold,
            'gross_profit': self.gross_profit,
            'net_income': self.net_income,
            'total_assets': self.total_assets,
            'total_liabilities': self.total_liabilities,
            'equity': self.equity,
            'cash_flow': self.cash_flow,
            'statement_date': self.statement_date.isoformat()
        }

@dataclass
class AnalysisResult:
    """Analysis result data model."""
    transaction_id: str
    compliance_score: float
    fraud_probability: float
    risk_level: RiskLevel
    explanations: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'transaction_id': self.transaction_id,
            'compliance_score': self.compliance_score,
            'fraud_probability': self.fraud_probability,
            'risk_level': self.risk_level.value,
            'explanations': self.explanations,
            'red_flags': self.red_flags,
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }

# ===== BASIC DATA GENERATORS =====
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class BasicFinancialStatementGenerator:
    """Basic financial statement generator using probabilistic models."""
    
    def __init__(self, industry_type: str = "retail"):
        self.industry_type = industry_type
        self.industry_ratios = self._get_industry_ratios()
    
    def _get_industry_ratios(self) -> Dict[str, float]:
        """Get industry-specific financial ratios."""
        ratios = {
            "retail": {
                "cogs_ratio": 0.65,
                "operating_expense_ratio": 0.25,
                "asset_turnover": 1.5,
                "debt_ratio": 0.4
            },
            "manufacturing": {
                "cogs_ratio": 0.70,
                "operating_expense_ratio": 0.20,
                "asset_turnover": 0.8,
                "debt_ratio": 0.5
            },
            "services": {
                "cogs_ratio": 0.30,
                "operating_expense_ratio": 0.45,
                "asset_turnover": 2.0,
                "debt_ratio": 0.3
            }
        }
        return ratios.get(self.industry_type, ratios["retail"])
    
    def generate_statement(self, revenue_range: tuple = (1000000, 10000000)) -> FinancialStatement:
        """Generate a realistic financial statement."""
        # Generate base revenue
        revenue = np.random.uniform(revenue_range[0], revenue_range[1])
        
        # Calculate expenses based on industry ratios with some variation
        cogs_ratio = np.random.normal(self.industry_ratios["cogs_ratio"], 0.05)
        opex_ratio = np.random.normal(self.industry_ratios["operating_expense_ratio"], 0.03)
        
        cost_of_goods_sold = revenue * max(0.1, cogs_ratio)
        operating_expenses = revenue * max(0.05, opex_ratio)
        total_expenses = cost_of_goods_sold + operating_expenses
        
        # Calculate derived values
        gross_profit = revenue - cost_of_goods_sold
        net_income = gross_profit - operating_expenses
        
        # Generate balance sheet items
        total_assets = revenue / max(0.5, self.industry_ratios["asset_turnover"])
        total_liabilities = total_assets * max(0.1, self.industry_ratios["debt_ratio"])
        equity = total_assets - total_liabilities
        
        # Simple cash flow approximation
        cash_flow = net_income + (total_assets * 0.1)  # Adding depreciation estimate
        
        return FinancialStatement(
            revenue=revenue,
            total_expenses=total_expenses,
            operating_expenses=operating_expenses,
            cost_of_goods_sold=cost_of_goods_sold,
            gross_profit=gross_profit,
            net_income=net_income,
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            equity=equity,
            cash_flow=cash_flow,
            statement_date=date.today()
        )

class BasicTransactionGenerator:
    """Basic transaction generator with Benford's Law compliance."""
    
    def __init__(self):
        self.account_types = {
            "Supplies Expense": {"type": TransactionType.EXPENSE, "descriptions": [
                "Office Supplies Purchase", "Printer Supplies", "Stationery Order", 
                "Cleaning Supplies", "Computer Accessories"
            ]},
            "Consulting Expense": {"type": TransactionType.EXPENSE, "descriptions": [
                "Professional Services", "IT Consulting", "Legal Consultation", 
                "Business Advisory", "Technical Support"
            ]},
            "Travel Expense": {"type": TransactionType.EXPENSE, "descriptions": [
                "Business Travel", "Hotel Accommodation", "Flight Tickets", 
                "Car Rental", "Meals & Entertainment"
            ]},
            "Sales Revenue": {"type": TransactionType.INCOME, "descriptions": [
                "Product Sales", "Service Revenue", "Consulting Income", 
                "Subscription Revenue", "Commission Income"
            ]},
            "Interest Income": {"type": TransactionType.INCOME, "descriptions": [
                "Bank Interest", "Investment Income", "Dividend Payment", 
                "Bond Interest", "Savings Account Interest"
            ]}
        }
        
        self.vendors = [
            "ABC Office Supplies", "TechCorp Solutions", "Global Consulting Ltd",
            "Premier Services Inc", "Innovation Partners", "Strategic Advisors",
            "Digital Solutions Co", "Business Partners LLC", "Expert Consultants"
        ]
    
    def _generate_benford_amount(self, min_amount: float = 10, max_amount: float = 50000) -> float:
        """Generate amount that follows Benford's Law."""
        # Use logarithmic distribution to naturally follow Benford's Law
        log_min = np.log10(min_amount)
        log_max = np.log10(max_amount)
        log_amount = np.random.uniform(log_min, log_max)
        
        # Add some randomness to avoid perfect distribution
        amount = 10 ** log_amount
        
        # Add minor random variation
        variation = np.random.normal(1, 0.1)
        return max(min_amount, amount * variation)
    
    def _generate_fraudulent_amount(self, min_amount: float = 10, max_amount: float = 50000) -> float:
        """Generate amount that violates Benford's Law (for fraud injection)."""
        fraud_type = np.random.choice(["round", "avoid_1_2", "prefer_high"])
        
        if fraud_type == "round":
            # Round numbers
            return np.random.choice([500, 1000, 1500, 2000, 2500, 5000, 7500, 10000])
        elif fraud_type == "avoid_1_2":
            # Avoid numbers starting with 1 or 2
            first_digit = np.random.choice([3, 4, 5, 6, 7, 8, 9])
            magnitude = np.random.randint(2, 5)
            return first_digit * (10 ** magnitude)
        else:
            # Prefer numbers starting with 5 or 9
            first_digit = np.random.choice([5, 9])
            magnitude = np.random.randint(2, 4)
            return first_digit * (10 ** magnitude) + np.random.randint(0, 100)
    
    def generate_transaction_batch(self, 
                                 target_amounts: Dict[str, float], 
                                 date_range: tuple = None,
                                 fraud_rate: float = 0.05) -> List[Transaction]:
        """Generate batch of transactions that align with financial statement totals."""
        if date_range is None:
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            date_range = (start_date, end_date)
        
        transactions = []
        
        for account, target_amount in target_amounts.items():
            if account not in self.account_types:
                continue
                
            account_info = self.account_types[account]
            
            # Determine number of transactions needed
            num_transactions = max(10, int(target_amount / np.random.uniform(100, 2000)))
            
            # Generate individual transactions
            remaining_amount = target_amount
            
            for i in range(num_transactions):
                # Determine if this should be a fraudulent transaction
                is_fraudulent = np.random.random() < fraud_rate
                
                if i == num_transactions - 1:
                    # Last transaction gets remaining amount
                    amount = remaining_amount
                else:
                    # Generate amount (fraudulent or legitimate)
                    if is_fraudulent:
                        amount = self._generate_fraudulent_amount(50, min(5000, remaining_amount))
                    else:
                        amount = self._generate_benford_amount(50, min(5000, remaining_amount))
                    
                    amount = min(amount, remaining_amount * 0.8)  # Don't use too much in one transaction
                
                # Generate random date in range
                days_diff = (date_range[1] - date_range[0]).days
                random_days = np.random.randint(0, days_diff)
                transaction_date = date_range[0] + timedelta(days=random_days)
                
                # Create transaction
                transaction = Transaction(
                    transaction_id=f"TXN{uuid.uuid4().hex[:8].upper()}",
                    date=transaction_date,
                    description=np.random.choice(account_info["descriptions"]),
                    account=account,
                    amount=amount,
                    transaction_type=account_info["type"],
                    payee_payer=np.random.choice(self.vendors),
                    account_number=f"ACC-{np.random.randint(100000, 999999)}",
                    tax_type="business_expense" if account_info["type"] == TransactionType.EXPENSE else "business_income",
                    fraud_flag=is_fraudulent,
                    notes=f"Generated transaction {'(fraudulent)' if is_fraudulent else '(legitimate)'}"
                )
                
                transactions.append(transaction)
                remaining_amount -= amount
                
                if remaining_amount <= 0:
                    break
        
        return transactions

# ===== SIMPLE RAG SYSTEM =====
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

class SimpleRAGSystem:
    """Simple RAG system for forensic accounting knowledge retrieval."""
    
    def __init__(self):
        # Initialize embedding model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Knowledge base
        self.knowledge_base = self._create_knowledge_base()
        
        # Precompute embeddings
        self.kb_texts = [doc["text"] for doc in self.knowledge_base]
        self.kb_embeddings = self.embedder.encode(self.kb_texts)
    
    def _create_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create initial knowledge base with tax and fraud rules."""
        return [
            {
                "id": "IRS_1",
                "category": "tax_rules",
                "text": "Business expenses under $2,500 are generally deductible per IRC Section 162 if they are ordinary and necessary for business operations.",
                "source": "IRS Publication 535",
                "relevance_score": 0.9
            },
            {
                "id": "IRS_2", 
                "category": "tax_rules",
                "text": "Travel expenses must be substantiated with receipts for amounts over $75, and meals are generally 50% deductible.",
                "source": "IRS Publication 463",
                "relevance_score": 0.8
            },
            {
                "id": "FRAUD_1",
                "category": "fraud_indicators",
                "text": "Round number amounts (e.g., $1,000, $5,000) appearing frequently may indicate fabricated transactions or expense padding.",
                "source": "ACFE Fraud Manual",
                "relevance_score": 0.9
            },
            {
                "id": "FRAUD_2",
                "category": "fraud_indicators", 
                "text": "Payments to new vendors over $5,000 without proper documentation represent high fraud risk and require enhanced scrutiny.",
                "source": "ACFE Best Practices",
                "relevance_score": 0.8
            },
            {
                "id": "BENFORD_1",
                "category": "analytical_techniques",
                "text": "Benford's Law expects 30.1% of amounts to start with digit 1, 17.6% with digit 2. Significant deviations may indicate data manipulation.",
                "source": "Digital Analysis Guidelines",
                "relevance_score": 0.9
            },
            {
                "id": "THRESHOLD_1",
                "category": "compliance",
                "text": "Transactions just below $10,000 may indicate structuring to avoid reporting requirements (Currency Transaction Reports).",
                "source": "FinCEN Guidelines",
                "relevance_score": 0.8
            },
            {
                "id": "VENDOR_1",
                "category": "fraud_indicators",
                "text": "Multiple transactions to the same vendor on the same day, especially round amounts, may indicate invoice manipulation or duplicate payments.",
                "source": "Internal Audit Guidelines",
                "relevance_score": 0.7
            }
        ]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents for a query."""
        # Encode query
        query_embedding = self.embedder.encode([query])
        
        # Calculate cosine similarity
        similarities = np.dot(self.kb_embeddings, query_embedding.T).flatten()
        similarities = similarities / (np.linalg.norm(self.kb_embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return relevant documents with similarity scores
        results = []
        for idx in top_indices:
            doc = self.knowledge_base[idx].copy()
            doc['similarity_score'] = similarities[idx]
            results.append(doc)
        
        return results
    
    def augment_prompt(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Create augmented prompt with retrieved context."""
        context = "\n".join([
            f"Source: {doc['source']}\nRule: {doc['text']}\nRelevance: {doc['similarity_score']:.3f}"
            for doc in retrieved_docs
        ])
        
        augmented_prompt = f"""
Query: {query}

Relevant Knowledge Base Context:
{context}

Based on the above context and forensic accounting principles, provide a detailed analysis addressing the query. Include:
1. Compliance assessment
2. Fraud risk evaluation  
3. Specific recommendations
4. Confidence level in assessment

Response:"""
        
        return augmented_prompt
    
    def analyze_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        """Analyze a transaction using RAG approach."""
        # Create analysis queries
        compliance_query = f"Tax compliance for {transaction.description} amount ${transaction.amount:.2f} in {transaction.account}"
        fraud_query = f"Fraud indicators for {transaction.description} amount ${transaction.amount:.2f} to {transaction.payee_payer}"
        
        # Retrieve relevant context
        compliance_context = self.retrieve(compliance_query, top_k=2)
        fraud_context = self.retrieve(fraud_query, top_k=2)
        
        # Simple rule-based analysis (placeholder for LLM)
        analysis = self._simple_analysis(transaction, compliance_context, fraud_context)
        
        return {
            "transaction_id": transaction.transaction_id,
            "compliance_analysis": analysis["compliance"],
            "fraud_analysis": analysis["fraud"],
            "retrieved_compliance_context": compliance_context,
            "retrieved_fraud_context": fraud_context,
            "overall_risk_score": analysis["risk_score"],
            "recommendations": analysis["recommendations"]
        }
    
    def _simple_analysis(self, transaction: Transaction, compliance_context: List[Dict], fraud_context: List[Dict]) -> Dict[str, Any]:
        """Simple rule-based analysis (placeholder for LLM integration)."""
        risk_score = 0
        recommendations = []
        
        # Amount-based analysis
        if transaction.amount >= 10000:
            risk_score += 20
            recommendations.append("High-value transaction requires enhanced documentation")
        elif transaction.amount % 1000 == 0:  # Round number
            risk_score += 15
            recommendations.append("Round number amount may indicate fabrication")
        
        # Account-based analysis
        if "Consulting" in transaction.account and transaction.amount > 5000:
            risk_score += 10
            recommendations.append("High-value consulting expense requires contract verification")
        
        # Vendor analysis
        if "New" in transaction.payee_payer or transaction.payee_payer == "":
            risk_score += 25
            recommendations.append("New vendor requires enhanced due diligence")
        
        # Compliance assessment
        compliance_score = max(0, 100 - risk_score)
        
        return {
            "compliance": f"Compliance score: {compliance_score}/100. " + 
                         ("Likely compliant" if compliance_score > 70 else "Requires review"),
            "fraud": f"Fraud risk score: {risk_score}/100. " + 
                    ("High risk" if risk_score > 50 else "Low risk"),
            "risk_score": risk_score,
            "recommendations": recommendations
        }

# ===== INTEGRATION EXAMPLE =====
def create_sample_dataset():
    """Create sample dataset for testing."""
    # Generate financial statement
    statement_gen = BasicFinancialStatementGenerator("retail")
    financial_statement = statement_gen.generate_statement()
    
    # Generate transactions that align with statement
    transaction_gen = BasicTransactionGenerator()
    target_amounts = {
        "Supplies Expense": financial_statement.operating_expenses * 0.3,
        "Consulting Expense": financial_statement.operating_expenses * 0.2,
        "Travel Expense": financial_statement.operating_expenses * 0.1,
        "Sales Revenue": financial_statement.revenue * 0.8,
        "Interest Income": financial_statement.revenue * 0.05
    }
    
    transactions = transaction_gen.generate_transaction_batch(
        target_amounts=target_amounts,
        fraud_rate=0.08  # 8% fraud rate
    )
    
    return financial_statement, transactions

def main():
    """Main function demonstrating the foundation components."""
    print("=== FORENSIC ACCOUNTING FOUNDATION DEMO ===\n")
    
    # Create sample dataset
    print("1. Generating sample financial data...")
    financial_statement, transactions = create_sample_dataset()
    
    print(f"Generated financial statement with revenue: ${financial_statement.revenue:,.2f}")
    print(f"Generated {len(transactions)} transactions")
    
    # Initialize RAG system
    print("\n2. Initializing RAG system...")
    rag_system = SimpleRAGSystem()
    
    # Analyze sample transactions
    print("\n3. Analyzing transactions with RAG...")
    sample_transactions = transactions[:5]  # Analyze first 5 transactions
    
    for transaction in sample_transactions:
        print(f"\n--- Analyzing Transaction {transaction.transaction_id} ---")
        print(f"Amount: ${transaction.amount:,.2f}")
        print(f"Description: {transaction.description}")
        print(f"Account: {transaction.account}")
        print(f"Fraud Flag: {transaction.fraud_flag}")
        
        # Perform RAG analysis
        analysis = rag_system.analyze_transaction(transaction)
        print(f"Risk Score: {analysis['overall_risk_score']}/100")
        print(f"Compliance: {analysis['compliance_analysis']}")
        print(f"Fraud Analysis: {analysis['fraud_analysis']}")
        
        if analysis['recommendations']:
            print(f"Recommendations: {', '.join(analysis['recommendations'])}")
    
    # Save sample data
    print("\n4. Saving sample data...")
    
    # Save transactions to JSON
    transactions_data = [t.to_dict() for t in transactions]
    with open('sample_transactions.json', 'w') as f:
        json.dump(transactions_data, f, indent=2)
    
    # Save financial statement
    with open('sample_financial_statement.json', 'w') as f:
        json.dump(financial_statement.to_dict(), f, indent=2)
    
    print("Sample data saved to JSON files")
    print("\nFoundation components ready for integration!")
    
    return financial_statement, transactions, rag_system

if __name__ == "__main__":
    financial_statement, transactions, rag_system = main()