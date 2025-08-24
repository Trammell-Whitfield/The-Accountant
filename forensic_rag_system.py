"""
Comprehensive RAG System for Forensic Accounting Analysis
Integrates Retrieval-Augmented Generation with fraud detection capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import json
import warnings
from pathlib import Path
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import logging

warnings.filterwarnings('ignore')

class ForensicKnowledgeBase:
    """
    Knowledge base for forensic accounting rules and patterns
    """
    
    def __init__(self, db_path: str = "forensic_knowledge.db"):
        self.db_path = db_path
        self.init_database()
        self.load_default_knowledge()
        
    def init_database(self):
        """Initialize SQLite database for knowledge storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                subcategory TEXT,
                rule_text TEXT NOT NULL,
                confidence_score REAL DEFAULT 1.0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                tags TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fraud_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                pattern_description TEXT,
                detection_logic TEXT,
                risk_level TEXT,
                examples TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_default_knowledge(self):
        """Load default forensic accounting knowledge"""
        default_rules = [
            {
                'category': 'IRS_RULES',
                'subcategory': 'DEDUCTIBLE_EXPENSES',
                'rule_text': 'Business expenses under $2,500 for supplies are generally deductible per IRC Section 162',
                'confidence_score': 0.95,
                'source': 'IRC Section 162',
                'tags': 'tax,deduction,supplies'
            },
            {
                'category': 'IRS_RULES',
                'subcategory': 'REPORTING_THRESHOLDS',
                'rule_text': 'Cash transactions over $10,000 must be reported on Form 8300',
                'confidence_score': 1.0,
                'source': 'IRS Publication 1544',
                'tags': 'reporting,cash,threshold'
            },
            {
                'category': 'FRAUD_INDICATORS',
                'subcategory': 'AMOUNT_PATTERNS',
                'rule_text': 'Frequent round-number transactions may indicate fabricated data',
                'confidence_score': 0.8,
                'source': 'Forensic Accounting Standards',
                'tags': 'fraud,patterns,amounts'
            },
            {
                'category': 'FRAUD_INDICATORS',
                'subcategory': 'VENDOR_PATTERNS',
                'rule_text': 'Payments over $5,000 to new or unknown vendors are high-risk',
                'confidence_score': 0.85,
                'source': 'ACFE Guidelines',
                'tags': 'fraud,vendor,risk'
            },
            {
                'category': 'BENFORDS_LAW',
                'subcategory': 'DIGIT_DISTRIBUTION',
                'rule_text': 'First digit 1 should appear in ~30.1% of natural datasets',
                'confidence_score': 0.9,
                'source': 'Benford Law Statistical Analysis',
                'tags': 'benford,statistics,digit'
            },
            {
                'category': 'TAX_COMPLIANCE',
                'subcategory': 'DEPRECIATION',
                'rule_text': 'Equipment purchases over $2,500 must be capitalized and depreciated',
                'confidence_score': 0.95,
                'source': 'IRC Section 179',
                'tags': 'tax,depreciation,equipment'
            }
        ]
        
        fraud_patterns = [
            {
                'pattern_name': 'Round Number Bias',
                'pattern_description': 'Excessive use of round numbers like $1,000, $5,000',
                'detection_logic': 'Check percentage of amounts ending in 00',
                'risk_level': 'MEDIUM',
                'examples': '[$1000, $5000, $10000]'
            },
            {
                'pattern_name': 'Threshold Avoidance',
                'pattern_description': 'Amounts just below reporting thresholds',
                'detection_logic': 'Check for clustering around $9,999, $4,999',
                'risk_level': 'HIGH',
                'examples': '[$9999, $9950, $9800]'
            },
            {
                'pattern_name': 'Duplicate Transactions',
                'pattern_description': 'Identical amount-vendor-date combinations',
                'detection_logic': 'Group by amount, vendor, date and count duplicates',
                'risk_level': 'HIGH',
                'examples': '[Same invoice processed twice]'
            },
            {
                'pattern_name': 'Sequential Amounts',
                'pattern_description': 'Amounts in arithmetic sequence',
                'detection_logic': 'Check for sequential patterns in amounts',
                'risk_level': 'MEDIUM',
                'examples': '[$1000, $1100, $1200, $1300]'
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        
        # Insert default rules if database is empty
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM knowledge_base")
        if cursor.fetchone()[0] == 0:
            for rule in default_rules:
                cursor.execute('''
                    INSERT INTO knowledge_base 
                    (category, subcategory, rule_text, confidence_score, source, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (rule['category'], rule['subcategory'], rule['rule_text'], 
                      rule['confidence_score'], rule['source'], rule['tags']))
        
        cursor.execute("SELECT COUNT(*) FROM fraud_patterns")
        if cursor.fetchone()[0] == 0:
            for pattern in fraud_patterns:
                cursor.execute('''
                    INSERT INTO fraud_patterns 
                    (pattern_name, pattern_description, detection_logic, risk_level, examples)
                    VALUES (?, ?, ?, ?, ?)
                ''', (pattern['pattern_name'], pattern['pattern_description'],
                      pattern['detection_logic'], pattern['risk_level'], pattern['examples']))
        
        conn.commit()
        conn.close()
        
    def retrieve_knowledge(self, query: str, category: Optional[str] = None, 
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge based on query
        
        Args:
            query: Search query
            category: Optional category filter
            top_k: Number of results to return
            
        Returns:
            List of relevant knowledge items
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build SQL query
        sql = "SELECT * FROM knowledge_base WHERE 1=1"
        params = []
        
        if category:
            sql += " AND category = ?"
            params.append(category)
        
        sql += " ORDER BY confidence_score DESC"
        
        cursor = conn.cursor()
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        # Convert to dictionaries
        columns = ['id', 'category', 'subcategory', 'rule_text', 'confidence_score',
                  'created_date', 'source', 'tags']
        knowledge_items = []
        for row in results:
            item = dict(zip(columns, row))
            knowledge_items.append(item)
        
        # Simple text similarity scoring
        if query and knowledge_items:
            query_words = set(query.lower().split())
            scored_items = []
            
            for item in knowledge_items:
                text_to_search = f"{item['rule_text']} {item['tags']} {item['subcategory']}"
                text_words = set(text_to_search.lower().split())
                similarity = len(query_words.intersection(text_words)) / len(query_words.union(text_words))
                item['relevance_score'] = similarity
                scored_items.append(item)
            
            # Sort by relevance and return top_k
            scored_items.sort(key=lambda x: x['relevance_score'], reverse=True)
            return scored_items[:top_k]
        
        return knowledge_items[:top_k]

class ForensicRAGAnalyzer:
    """
    Main RAG system for forensic accounting analysis
    """
    
    def __init__(self, knowledge_base_path: str = "forensic_knowledge.db"):
        self.kb = ForensicKnowledgeBase(knowledge_base_path)
        self.analysis_history = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for analysis tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('forensic_analysis.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single transaction using RAG approach
        
        Args:
            transaction_data: Transaction information
            
        Returns:
            Analysis results with recommendations
        """
        transaction_id = transaction_data.get('transaction_id', 'UNKNOWN')
        amount = transaction_data.get('amount', 0)
        description = transaction_data.get('description', '')
        payee = transaction_data.get('payee_payer', '')
        account = transaction_data.get('account', '')
        
        self.logger.info(f"Analyzing transaction {transaction_id}")
        
        # Build analysis context
        analysis_context = {
            'transaction_id': transaction_id,
            'amount': amount,
            'description': description,
            'payee': payee,
            'account': account,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Retrieve relevant knowledge
        compliance_query = f"tax deduction compliance {account} {amount}"
        compliance_knowledge = self.kb.retrieve_knowledge(compliance_query, 'IRS_RULES')
        
        fraud_query = f"fraud patterns {amount} {payee}"
        fraud_knowledge = self.kb.retrieve_knowledge(fraud_query, 'FRAUD_INDICATORS')
        
        # Analyze compliance
        compliance_analysis = self._analyze_compliance(transaction_data, compliance_knowledge)
        
        # Analyze fraud risk
        fraud_analysis = self._analyze_fraud_risk(transaction_data, fraud_knowledge)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            transaction_data, compliance_analysis, fraud_analysis
        )
        
        # Compile results
        analysis_result = {
            'transaction_analysis': analysis_context,
            'compliance_analysis': compliance_analysis,
            'fraud_analysis': fraud_analysis,
            'recommendations': recommendations,
            'retrieved_knowledge': {
                'compliance_rules': compliance_knowledge,
                'fraud_patterns': fraud_knowledge
            },
            'overall_risk_score': self._calculate_risk_score(compliance_analysis, fraud_analysis)
        }
        
        self.analysis_history.append(analysis_result)
        return analysis_result
        
    def _analyze_compliance(self, transaction_data: Dict[str, Any], 
                           knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze tax compliance aspects"""
        amount = transaction_data.get('amount', 0)
        account = transaction_data.get('account', '').lower()
        
        compliance_result = {
            'is_compliant': True,
            'compliance_score': 0.9,
            'issues': [],
            'applicable_rules': []
        }
        
        # Apply retrieved knowledge
        for rule in knowledge:
            rule_text = rule['rule_text'].lower()
            
            # Check supplies deduction rule
            if 'supplies' in rule_text and 'supplies' in account:
                if amount > 2500:
                    compliance_result['is_compliant'] = False
                    compliance_result['compliance_score'] = 0.3
                    compliance_result['issues'].append(
                        f"Supplies expense of ${amount} exceeds $2,500 threshold - may need capitalization"
                    )
                compliance_result['applicable_rules'].append(rule['rule_text'])
                
            # Check reporting thresholds
            if 'cash' in rule_text and amount > 10000:
                compliance_result['issues'].append(
                    f"Cash transaction of ${amount} requires Form 8300 reporting"
                )
                compliance_result['applicable_rules'].append(rule['rule_text'])
        
        return compliance_result
        
    def _analyze_fraud_risk(self, transaction_data: Dict[str, Any], 
                           knowledge: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fraud risk indicators"""
        amount = transaction_data.get('amount', 0)
        payee = transaction_data.get('payee_payer', '')
        
        fraud_result = {
            'risk_level': 'LOW',
            'risk_score': 0.1,
            'detected_patterns': [],
            'risk_factors': []
        }
        
        # Check for round numbers
        if amount % 1000 == 0 and amount >= 1000:
            fraud_result['risk_score'] += 0.3
            fraud_result['detected_patterns'].append('Round Number Pattern')
            fraud_result['risk_factors'].append(f"Amount ${amount} is a round thousand")
            
        # Check threshold avoidance
        if amount in [9999, 9998, 9997, 4999, 4998]:
            fraud_result['risk_score'] += 0.5
            fraud_result['detected_patterns'].append('Threshold Avoidance')
            fraud_result['risk_factors'].append(f"Amount ${amount} appears to avoid reporting threshold")
            
        # Check vendor patterns
        if 'vendor' in payee.lower() and any(char.isdigit() for char in payee):
            fraud_result['risk_score'] += 0.2
            fraud_result['risk_factors'].append("Generic vendor name with numbers")
            
        # Determine risk level
        if fraud_result['risk_score'] >= 0.7:
            fraud_result['risk_level'] = 'HIGH'
        elif fraud_result['risk_score'] >= 0.4:
            fraud_result['risk_level'] = 'MEDIUM'
            
        return fraud_result
        
    def _generate_recommendations(self, transaction_data: Dict[str, Any], 
                                compliance_analysis: Dict[str, Any], 
                                fraud_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Compliance recommendations
        if not compliance_analysis['is_compliant']:
            recommendations.append("Review transaction for tax compliance issues")
            for issue in compliance_analysis['issues']:
                recommendations.append(f"Action required: {issue}")
                
        # Fraud risk recommendations
        if fraud_analysis['risk_level'] == 'HIGH':
            recommendations.append("URGENT: Manual review required - high fraud risk detected")
            recommendations.append("Verify supporting documentation")
            recommendations.append("Confirm vendor legitimacy")
        elif fraud_analysis['risk_level'] == 'MEDIUM':
            recommendations.append("Enhanced review recommended")
            recommendations.append("Sample additional transactions from same vendor")
            
        # General recommendations
        amount = transaction_data.get('amount', 0)
        if amount > 5000:
            recommendations.append("Verify approval authorization for large transaction")
            
        if not recommendations:
            recommendations.append("Transaction appears normal - continue standard monitoring")
            
        return recommendations
        
    def _calculate_risk_score(self, compliance_analysis: Dict[str, Any], 
                             fraud_analysis: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        compliance_weight = 0.4
        fraud_weight = 0.6
        
        compliance_risk = 1.0 - compliance_analysis['compliance_score']
        fraud_risk = fraud_analysis['risk_score']
        
        overall_risk = (compliance_risk * compliance_weight + 
                       fraud_risk * fraud_weight)
        
        return min(overall_risk, 1.0)
        
    def analyze_ledger_batch(self, ledger_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze entire ledger using RAG approach
        
        Args:
            ledger_df: Ledger dataframe
            
        Returns:
            Comprehensive analysis results
        """
        self.logger.info(f"Starting batch analysis of {len(ledger_df)} transactions")
        
        batch_results = {
            'summary': {
                'total_transactions': len(ledger_df),
                'analysis_timestamp': datetime.now().isoformat(),
                'high_risk_transactions': 0,
                'compliance_issues': 0
            },
            'transaction_analyses': [],
            'aggregate_patterns': {},
            'recommendations': []
        }
        
        # Analyze each transaction
        for idx, row in ledger_df.iterrows():
            transaction_data = row.to_dict()
            analysis = self.analyze_transaction(transaction_data)
            
            batch_results['transaction_analyses'].append(analysis)
            
            # Update summary counters
            if analysis['overall_risk_score'] > 0.7:
                batch_results['summary']['high_risk_transactions'] += 1
                
            if not analysis['compliance_analysis']['is_compliant']:
                batch_results['summary']['compliance_issues'] += 1
                
        # Analyze aggregate patterns
        batch_results['aggregate_patterns'] = self._analyze_aggregate_patterns(ledger_df)
        
        # Generate batch recommendations
        batch_results['recommendations'] = self._generate_batch_recommendations(batch_results)
        
        self.logger.info("Batch analysis completed")
        return batch_results
        
    def _analyze_aggregate_patterns(self, ledger_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns across the entire ledger"""
        patterns = {}
        
        # Amount patterns
        amounts = ledger_df['amount'].dropna()
        if not amounts.empty:
            # Round number analysis
            round_numbers = amounts[amounts % 1000 == 0]
            patterns['round_number_rate'] = len(round_numbers) / len(amounts) * 100
            
            # Benford's Law quick check
            first_digits = [int(str(abs(amount)).replace('.', '')[0]) 
                           for amount in amounts if amount > 0]
            if first_digits:
                digit_1_rate = first_digits.count(1) / len(first_digits) * 100
                patterns['benford_digit_1_rate'] = digit_1_rate
                patterns['benford_deviation'] = abs(digit_1_rate - 30.1)
                
        # Vendor patterns
        if 'payee_payer' in ledger_df.columns:
            vendor_counts = ledger_df['payee_payer'].value_counts()
            patterns['unique_vendors'] = len(vendor_counts)
            patterns['top_vendor_frequency'] = vendor_counts.iloc[0] if len(vendor_counts) > 0 else 0
            
        return patterns
        
    def _generate_batch_recommendations(self, batch_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for the entire batch"""
        recommendations = []
        
        summary = batch_results['summary']
        patterns = batch_results['aggregate_patterns']
        
        # High-level recommendations
        if summary['high_risk_transactions'] > summary['total_transactions'] * 0.05:
            recommendations.append(
                f"WARNING: {summary['high_risk_transactions']} high-risk transactions detected "
                f"({summary['high_risk_transactions']/summary['total_transactions']*100:.1f}%)"
            )
            recommendations.append("Recommend comprehensive forensic audit")
            
        if summary['compliance_issues'] > 0:
            recommendations.append(
                f"{summary['compliance_issues']} compliance issues identified - "
                "review tax treatment"
            )
            
        # Pattern-based recommendations
        if patterns.get('round_number_rate', 0) > 15:
            recommendations.append(
                f"High round number rate ({patterns['round_number_rate']:.1f}%) - "
                "investigate for potential fraud"
            )
            
        if patterns.get('benford_deviation', 0) > 10:
            recommendations.append(
                f"Significant Benford's Law deviation detected - "
                "detailed statistical analysis recommended"
            )
            
        return recommendations
        
    def generate_detailed_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        if 'transaction_analysis' in analysis_results:
            # Single transaction report
            return self._generate_transaction_report(analysis_results)
        else:
            # Batch analysis report
            return self._generate_batch_report(analysis_results)
            
    def _generate_transaction_report(self, results: Dict[str, Any]) -> str:
        """Generate report for single transaction analysis"""
        tx = results['transaction_analysis']
        comp = results['compliance_analysis']
        fraud = results['fraud_analysis']
        
        report = f"""
=== FORENSIC TRANSACTION ANALYSIS REPORT ===
Transaction ID: {tx['transaction_id']}
Amount: ${tx['amount']:,.2f}
Payee: {tx['payee']}
Account: {tx['account']}
Analysis Date: {tx['analysis_timestamp']}

=== OVERALL RISK ASSESSMENT ===
Risk Score: {results['overall_risk_score']:.2f}/1.00
Risk Level: {'HIGH' if results['overall_risk_score'] > 0.7 else 'MEDIUM' if results['overall_risk_score'] > 0.4 else 'LOW'}

=== COMPLIANCE ANALYSIS ===
Compliant: {'YES' if comp['is_compliant'] else 'NO'}
Compliance Score: {comp['compliance_score']:.2f}/1.00
Issues: {comp['issues'] if comp['issues'] else 'None detected'}

=== FRAUD RISK ANALYSIS ===
Risk Level: {fraud['risk_level']}
Risk Score: {fraud['risk_score']:.2f}/1.00
Detected Patterns: {fraud['detected_patterns'] if fraud['detected_patterns'] else 'None'}
Risk Factors: {fraud['risk_factors'] if fraud['risk_factors'] else 'None'}

=== RECOMMENDATIONS ===
"""
        for i, rec in enumerate(results['recommendations'], 1):
            report += f"{i}. {rec}\n"
            
        return report
        
    def _generate_batch_report(self, results: Dict[str, Any]) -> str:
        """Generate report for batch analysis"""
        summary = results['summary']
        patterns = results['aggregate_patterns']
        
        report = f"""
=== COMPREHENSIVE LEDGER ANALYSIS REPORT ===
Analysis Date: {summary['analysis_timestamp']}
Total Transactions: {summary['total_transactions']:,}

=== SUMMARY STATISTICS ===
High Risk Transactions: {summary['high_risk_transactions']} ({summary['high_risk_transactions']/summary['total_transactions']*100:.1f}%)
Compliance Issues: {summary['compliance_issues']} ({summary['compliance_issues']/summary['total_transactions']*100:.1f}%)

=== PATTERN ANALYSIS ===
Round Number Rate: {patterns.get('round_number_rate', 0):.1f}%
Benford's Law Digit 1 Rate: {patterns.get('benford_digit_1_rate', 0):.1f}% (Expected: 30.1%)
Benford's Law Deviation: {patterns.get('benford_deviation', 0):.1f} percentage points
Unique Vendors: {patterns.get('unique_vendors', 0)}
Top Vendor Frequency: {patterns.get('top_vendor_frequency', 0)} transactions

=== KEY RECOMMENDATIONS ===
"""
        for i, rec in enumerate(results['recommendations'], 1):
            report += f"{i}. {rec}\n"
            
        report += f"""
=== DETAILED FINDINGS ===
High-risk transactions requiring immediate review: {summary['high_risk_transactions']}
Transactions with compliance issues: {summary['compliance_issues']}

=== NEXT STEPS ===
1. Review all high-risk transactions manually
2. Verify supporting documentation for flagged items
3. Consider expanded testing if patterns suggest systematic issues
4. Update internal controls based on findings
"""
        return report

if __name__ == "__main__":
    # Example usage
    rag_analyzer = ForensicRAGAnalyzer()
    
    # Example transaction analysis
    sample_transaction = {
        'transaction_id': 'TXN12345',
        'amount': 9999,
        'description': 'Office Supplies Purchase',
        'payee_payer': 'Vendor Services 123',
        'account': 'Supplies Expense'
    }
    
    print("Analyzing sample transaction...")
    result = rag_analyzer.analyze_transaction(sample_transaction)
    print(rag_analyzer.generate_detailed_report(result))