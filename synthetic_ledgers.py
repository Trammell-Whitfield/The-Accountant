import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
from typing import List, Dict, Tuple
import json
from scipy import stats
import matplotlib.pyplot as plt

class SyntheticLedgerGenerator:
    """
    Generates synthetic accounting ledgers with realistic transaction patterns
    following Benford's Law for legitimate entries and fraud patterns for training.
    """
    
    def __init__(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Benford's Law probabilities for digits 1-9
        self.benford_probs = {
            1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
            6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
        }
        
        # Transaction categories with realistic weights
        self.categories = {
            'income': ['Sales Revenue', 'Service Income', 'Interest Income', 'Rental Income'],
            'expense': ['Supplies Expense', 'Utilities Expense', 'Rent Expense', 'Travel Expense',
                      'Marketing Expense', 'Professional Services', 'Insurance Expense'],
            'deduction': ['Charitable Contributions', 'Business Meals', 'Equipment Depreciation']
        }
        
        # Common payee/payer names
        self.payees = [
            'ABC Supplies Co', 'Tech Solutions Inc', 'Metro Utilities', 'City Properties LLC',
            'Quick Print Services', 'Professional Consulting', 'Insurance Partners',
            'Marketing Plus', 'Travel Services', 'Equipment Rental Co', 'John Doe Consulting',
            'Smith & Associates', 'Global Logistics', 'Office Depot', 'Fuel Express'
        ]
        
        # Tax types
        self.tax_types = ['Income Tax', 'Sales Tax', 'Payroll Tax', 'VAT', 'Property Tax']
    
    def generate_benford_amount(self, min_amount: float = 1.0, max_amount: float = 100000.0) -> float:
        """Generate amount following Benford's Law distribution"""
        # Generate uniform random variable
        u = np.random.uniform(0, 1)
        
        # Use logarithmic distribution to ensure Benford's Law compliance
        log_min = np.log10(min_amount)
        log_max = np.log10(max_amount)
        log_amount = np.random.uniform(log_min, log_max)
        
        # Convert back to linear scale
        amount = 10 ** log_amount
        
        # Add some random variation while maintaining Benford's Law
        variation = np.random.normal(1, 0.1)
        amount *= variation
        
        return round(max(amount, min_amount), 2)
    
    def generate_fraudulent_amount(self, fraud_type: str = 'random') -> float:
        """Generate amounts with fraudulent patterns"""
        if fraud_type == 'rounded':
            # Rounded amounts (common fraud pattern)
            base_amounts = [100, 250, 500, 1000, 2500, 5000, 10000, 25000]
            return float(random.choice(base_amounts))
        
        elif fraud_type == 'threshold_avoidance':
            # Just below reporting thresholds
            thresholds = [9999, 4999, 2999, 1999]
            return float(random.choice(thresholds))
        
        elif fraud_type == 'repeated_digits':
            # Overuse of specific digits
            preferred_digits = [5, 9, 7]
            digit = random.choice(preferred_digits)
            # Create amounts starting with preferred digit
            magnitude = random.choice([100, 1000, 10000])
            return float(digit * magnitude + random.randint(0, 99))
        
        elif fraud_type == 'sequential':
            # Sequential amounts (fabricated pattern)
            base = random.randint(100, 1000)
            increment = random.randint(50, 200)
            return float(base + increment * random.randint(0, 10))
        
        else:
            # Random fraudulent amount
            return round(random.uniform(100, 50000), 2)
    
    def generate_random_date(self) -> datetime:
        """Generate random date within the specified range"""
        time_between = self.end_date - self.start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        return self.start_date + timedelta(days=random_days)
    
    def generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        return f"TXN{str(uuid.uuid4())[:8].upper()}"
    
    def generate_account_number(self) -> str:
        """Generate synthetic account number"""
        return f"ACC-{random.randint(100000, 999999)}"
    
    def generate_legitimate_transaction(self) -> Dict:
        """Generate a legitimate transaction entry"""
        # Select category and account
        category_type = np.random.choice(['income', 'expense', 'deduction'], 
                                       p=[0.4, 0.5, 0.1])
        account = random.choice(self.categories[category_type])
        
        # Generate amount following Benford's Law
        amount = self.generate_benford_amount()
        
        # Adjust amount based on category (income typically higher)
        if category_type == 'income':
            amount *= random.uniform(2.0, 5.0)
        elif category_type == 'deduction':
            amount *= random.uniform(0.5, 1.5)
        
        return {
            'transaction_id': self.generate_transaction_id(),
            'date': self.generate_random_date().strftime('%Y-%m-%d'),
            'amount': round(amount, 2),
            'category': category_type,
            'account': account,
            'payee_payer': random.choice(self.payees),
            'account_number': self.generate_account_number(),
            'tax_type': random.choice(self.tax_types),
            'fraud_flag': 0,
            'notes': f"Legitimate {category_type} transaction"
        }
    
    def generate_fraudulent_transaction(self) -> Dict:
        """Generate a fraudulent transaction entry"""
        fraud_types = ['rounded', 'threshold_avoidance', 'repeated_digits', 'sequential']
        fraud_type = random.choice(fraud_types)
        
        # Generate fraudulent amount
        amount = self.generate_fraudulent_amount(fraud_type)
        
        # Select category and account
        category_type = np.random.choice(['expense', 'deduction'], p=[0.8, 0.2])
        account = random.choice(self.categories[category_type])
        
        # Sometimes use suspicious payee names
        if random.random() < 0.3:
            payee = f"Vendor {random.randint(1, 999)}"
        else:
            payee = random.choice(self.payees)
        
        return {
            'transaction_id': self.generate_transaction_id(),
            'date': self.generate_random_date().strftime('%Y-%m-%d'),
            'amount': amount,
            'category': category_type,
            'account': account,
            'payee_payer': payee,
            'account_number': self.generate_account_number(),
            'tax_type': random.choice(self.tax_types),
            'fraud_flag': 1,
            'notes': f"Potential fraud: {fraud_type} pattern"
        }
    
    def generate_ledger(self, num_transactions: int = 1000, fraud_percentage: float = 0.1) -> pd.DataFrame:
        """
        Generate complete synthetic ledger
        
        Args:
            num_transactions: Total number of transactions to generate
            fraud_percentage: Percentage of transactions that should be fraudulent (0.0-1.0)
        
        Returns:
            DataFrame containing the synthetic ledger
        """
        transactions = []
        num_fraudulent = int(num_transactions * fraud_percentage)
        num_legitimate = num_transactions - num_fraudulent
        
        print(f"Generating {num_legitimate} legitimate transactions...")
        # Generate legitimate transactions
        for i in range(num_legitimate):
            transactions.append(self.generate_legitimate_transaction())
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_legitimate} legitimate transactions")
        
        print(f"Generating {num_fraudulent} fraudulent transactions...")
        # Generate fraudulent transactions
        for i in range(num_fraudulent):
            transactions.append(self.generate_fraudulent_transaction())
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_fraudulent} fraudulent transactions")
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(transactions)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Add description field (combination of account and notes)
        df['description'] = df['account'] + ' - ' + df['payee_payer']
        
        # Reorder columns to match the schema from your document
        column_order = ['transaction_id', 'date', 'description', 'account', 'amount', 
                       'category', 'payee_payer', 'account_number', 'tax_type', 
                       'fraud_flag', 'notes']
        df = df[column_order]
        
        return df
    
    def validate_benford_law(self, df: pd.DataFrame, amount_column: str = 'amount') -> Dict:
        """
        Validate that the generated data follows Benford's Law
        
        Args:
            df: DataFrame containing the ledger data
            amount_column: Column name containing the amounts
        
        Returns:
            Dictionary with validation results
        """
        # Filter legitimate transactions only
        legitimate_df = df[df['fraud_flag'] == 0]
        amounts = legitimate_df[amount_column].values
        
        # Extract first digits
        first_digits = [int(str(abs(amount)).replace('.', '')[0]) for amount in amounts if amount > 0]
        
        # Calculate observed frequencies
        observed_counts = {}
        for digit in range(1, 10):
            observed_counts[digit] = first_digits.count(digit)
        
        total_count = len(first_digits)
        observed_proportions = {digit: count/total_count for digit, count in observed_counts.items()}
        
        # Calculate expected frequencies based on Benford's Law
        expected_proportions = self.benford_probs
        expected_counts = {digit: prob * total_count for digit, prob in expected_proportions.items()}
        
        # Chi-square test
        chi2_stat = sum((observed_counts[digit] - expected_counts[digit])**2 / expected_counts[digit] 
                       for digit in range(1, 10))
        
        # Degrees of freedom = 8 (digits 1-9)
        p_value = 1 - stats.chi2.cdf(chi2_stat, 8)
        
        return {
            'observed_proportions': observed_proportions,
            'expected_proportions': expected_proportions,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'follows_benford': p_value > 0.05,
            'total_legitimate_transactions': total_count
        }
    
    def plot_benford_analysis(self, df: pd.DataFrame, amount_column: str = 'amount'):
        """Plot Benford's Law analysis"""
        validation_results = self.validate_benford_law(df, amount_column)
        
        digits = list(range(1, 10))
        observed = [validation_results['observed_proportions'][d] for d in digits]
        expected = [validation_results['expected_proportions'][d] for d in digits]
        
        plt.figure(figsize=(12, 6))
        
        # Plot comparison
        plt.subplot(1, 2, 1)
        x = np.arange(len(digits))
        width = 0.35
        
        plt.bar(x - width/2, observed, width, label='Observed', alpha=0.7)
        plt.bar(x + width/2, expected, width, label='Expected (Benford)', alpha=0.7)
        
        plt.xlabel('First Digit')
        plt.ylabel('Proportion')
        plt.title('Benford\'s Law Analysis - Legitimate Transactions')
        plt.xticks(x, digits)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot fraud vs legitimate amounts distribution
        plt.subplot(1, 2, 2)
        legitimate_amounts = df[df['fraud_flag'] == 0]['amount']
        fraudulent_amounts = df[df['fraud_flag'] == 1]['amount']
        
        plt.hist(legitimate_amounts, bins=50, alpha=0.7, label='Legitimate', density=True)
        plt.hist(fraudulent_amounts, bins=50, alpha=0.7, label='Fraudulent', density=True)
        
        plt.xlabel('Amount ($)')
        plt.ylabel('Density')
        plt.title('Amount Distribution: Legitimate vs Fraudulent')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print validation results
        print(f"\nBenford's Law Validation Results:")
        print(f"Chi-square statistic: {validation_results['chi2_statistic']:.4f}")
        print(f"P-value: {validation_results['p_value']:.4f}")
        print(f"Follows Benford's Law: {validation_results['follows_benford']}")
        print(f"Total legitimate transactions analyzed: {validation_results['total_legitimate_transactions']}")
    
    def save_ledger(self, df: pd.DataFrame, filename: str = "synthetic_ledger.csv"):
        """Save ledger to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Ledger saved to {filename}")
        
        # Also save as JSON for flexibility
        json_filename = filename.replace('.csv', '.json')
        df.to_json(json_filename, orient='records', indent=2)
        print(f"Ledger also saved to {json_filename}")
class ImprovedSyntheticLedgerGenerator:
    """
    Generates synthetic accounting ledgers with realistic transaction patterns
    following Benford's Law for legitimate entries and fraud patterns for training.
    """
    
    def __init__(self, start_date: str = "2024-01-01", end_date: str = "2024-12-31", 
                 company_type: str = "manufacturing", company_size: str = "medium"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.company_type = company_type
        self.company_size = company_size
        
        # Benford's Law probabilities for digits 1-9
        self.benford_probs = {
            1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
            6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
        }
        
        # Realistic Chart of Accounts with account codes
        self.chart_of_accounts = {
            # Assets (1000-1999)
            '1010': 'Cash - Operating Account',
            '1020': 'Cash - Payroll Account', 
            '1100': 'Accounts Receivable',
            '1200': 'Inventory - Raw Materials',
            '1210': 'Inventory - Work in Process',
            '1220': 'Inventory - Finished Goods',
            '1500': 'Equipment',
            '1510': 'Accumulated Depreciation - Equipment',
            '1600': 'Building',
            '1610': 'Accumulated Depreciation - Building',
            
            # Liabilities (2000-2999)
            '2010': 'Accounts Payable',
            '2020': 'Accrued Wages Payable',
            '2030': 'Payroll Taxes Payable',
            '2040': 'Sales Tax Payable',
            '2100': 'Notes Payable - Short Term',
            '2200': 'Mortgage Payable',
            
            # Equity (3000-3999)
            '3000': 'Common Stock',
            '3100': 'Retained Earnings',
            
            # Revenue (4000-4999)
            '4010': 'Sales Revenue',
            '4020': 'Service Revenue',
            '4030': 'Interest Income',
            '4040': 'Other Income',
            
            # Expenses (5000-6999)
            '5010': 'Cost of Goods Sold',
            '5020': 'Raw Materials',
            '5030': 'Direct Labor',
            '5040': 'Manufacturing Overhead',
            '6010': 'Salaries Expense',
            '6020': 'Payroll Tax Expense',
            '6030': 'Benefits Expense',
            '6100': 'Rent Expense',
            '6110': 'Utilities Expense',
            '6120': 'Insurance Expense',
            '6130': 'Office Supplies Expense',
            '6140': 'Professional Services',
            '6150': 'Marketing Expense',
            '6160': 'Travel Expense',
            '6170': 'Depreciation Expense',
            '6180': 'Interest Expense',
            '6190': 'Bad Debt Expense'
        }
        
        # Recurring transaction patterns
        self.recurring_patterns = {
            'monthly': ['6100', '6110', '6120'],  # Rent, utilities, insurance
            'biweekly': ['6010', '6020', '6030'],  # Payroll related
            'quarterly': ['2030', '2040'],  # Tax payments
            'daily': ['4010', '1100', '2010'],  # Sales, AR, AP
            'weekly': ['5020', '1200']  # Inventory purchases
        }
        
        # Industry-specific vendors by company type
        self.vendors = {
            'manufacturing': [
                'Industrial Supply Co', 'Steel Works Inc', 'Precision Tools Ltd',
                'Manufacturing Services', 'Quality Control Systems', 'Machinery Parts Co',
                'Raw Materials Supplier', 'Packaging Solutions', 'Transportation Logistics'
            ],
            'retail': [
                'Wholesale Distributors', 'Point of Sale Systems', 'Retail Fixtures Inc',
                'Inventory Management', 'Customer Service Solutions', 'Marketing Agency',
                'Shopping Center Management', 'Security Services', 'Cleaning Services'
            ],
            'service': [
                'Professional Development', 'Software Solutions', 'Communication Systems',
                'Office Equipment Rental', 'Consulting Services', 'Training Programs',
                'Technology Support', 'Administrative Services', 'Legal Services'
            ]
        }
        
        # Common customers by industry
        self.customers = {
            'manufacturing': ['ABC Manufacturing', 'Industrial Corp', 'Factory Direct'],
            'retail': ['Walk-in Customer', 'Online Customer', 'Wholesale Buyer'],
            'service': ['Corporate Client A', 'Government Contract', 'Small Business Client']
        }
        
        # Employee names for payroll
        self.employees = [
            'John Smith', 'Mary Johnson', 'Robert Williams', 'Patricia Brown',
            'Michael Davis', 'Jennifer Miller', 'William Wilson', 'Elizabeth Moore',
            'David Taylor', 'Susan Anderson', 'James Thomas', 'Karen Jackson'
        ]
    
    def generate_benford_amount(self, min_amount: float = 1.0, max_amount: float = 100000.0) -> float:
        """Generate amount following Benford's Law distribution"""
        # Generate uniform random variable
        u = np.random.uniform(0, 1)
        
        # Use logarithmic distribution to ensure Benford's Law compliance
        log_min = np.log10(min_amount)
        log_max = np.log10(max_amount)
        log_amount = np.random.uniform(log_min, log_max)
        
        # Convert back to linear scale
        amount = 10 ** log_amount
        
        # Add some random variation while maintaining Benford's Law
        variation = np.random.normal(1, 0.1)
        amount *= variation
        
        return round(max(amount, min_amount), 2)
    
    def generate_fraudulent_amount(self, fraud_type: str = 'random') -> float:
        """Generate amounts with fraudulent patterns"""
        if fraud_type == 'rounded':
            # Rounded amounts (common fraud pattern)
            base_amounts = [100, 250, 500, 1000, 2500, 5000, 10000, 25000]
            return float(random.choice(base_amounts))
        
        elif fraud_type == 'threshold_avoidance':
            # Just below reporting thresholds
            thresholds = [9999, 4999, 2999, 1999]
            return float(random.choice(thresholds))
        
        elif fraud_type == 'repeated_digits':
            # Overuse of specific digits
            preferred_digits = [5, 9, 7]
            digit = random.choice(preferred_digits)
            # Create amounts starting with preferred digit
            magnitude = random.choice([100, 1000, 10000])
            return float(digit * magnitude + random.randint(0, 99))
        
        elif fraud_type == 'sequential':
            # Sequential amounts (fabricated pattern)
            base = random.randint(100, 1000)
            increment = random.randint(50, 200)
            return float(base + increment * random.randint(0, 10))
        
        else:
            # Random fraudulent amount
            return round(random.uniform(100, 50000), 2)
    
    def generate_random_date(self) -> datetime:
        """Generate random date within the specified range"""
        time_between = self.end_date - self.start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        return self.start_date + timedelta(days=random_days)
    
    def generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        return f"TXN{str(uuid.uuid4())[:8].upper()}"
    
    def generate_account_number(self) -> str:
        """Generate synthetic account number"""
        return f"ACC-{random.randint(100000, 999999)}"
    
    def generate_realistic_transaction_date(self, account_code: str) -> datetime:
        """Generate realistic dates based on account type and patterns"""
        base_date = self.generate_random_date()
        
        # Adjust for recurring patterns
        if account_code in self.recurring_patterns['monthly']:
            # Monthly transactions typically on 1st or last day of month
            day = random.choice([1, 28, 29, 30, 31])
            try:
                return base_date.replace(day=min(day, 28))  # Safe day
            except:
                return base_date
        elif account_code in self.recurring_patterns['biweekly']:
            # Payroll typically on 15th and 30th
            day = random.choice([15, 30])
            try:
                return base_date.replace(day=day)
            except:
                return base_date
        
        return base_date
    
    def generate_legitimate_transaction(self) -> Dict:
        """Generate a legitimate transaction entry with realistic patterns"""
        # Select account based on realistic business patterns
        account_code = random.choice(list(self.chart_of_accounts.keys()))
        account_name = self.chart_of_accounts[account_code]
        
        # Generate amount following Benford's Law with realistic ranges
        amount = self.generate_benford_amount()
        
        # Adjust amount based on account type
        if account_code.startswith('4'):  # Revenue
            amount *= random.uniform(1000, 50000)  # Higher revenue amounts
        elif account_code.startswith('6'):  # Expenses
            if account_code in ['6010', '6020', '6030']:  # Payroll
                amount = random.uniform(2000, 8000)  # Typical payroll range
            elif account_code in ['6100', '6110', '6120']:  # Fixed expenses
                amount = random.uniform(500, 5000)
            else:
                amount *= random.uniform(50, 5000)
        elif account_code.startswith('5'):  # COGS
            amount *= random.uniform(100, 10000)
        elif account_code.startswith('1'):  # Assets
            amount *= random.uniform(1000, 100000)
        elif account_code.startswith('2'):  # Liabilities
            amount *= random.uniform(500, 20000)
        
        # Select appropriate vendor/customer based on account type
        if account_code.startswith('4'):  # Revenue - use customers
            payee = random.choice(self.customers.get(self.company_type, ['Generic Customer']))
        elif account_code in ['6010', '6020', '6030']:  # Payroll - use employees
            payee = random.choice(self.employees)
        else:  # Expenses - use vendors
            payee = random.choice(self.vendors.get(self.company_type, ['Generic Vendor']))
        
        # Generate realistic date based on account type
        trans_date = self.generate_realistic_transaction_date(account_code)
        
        # Create reference numbers
        journal_entry = f"JE{random.randint(1000, 9999)}"
        reference_number = f"REF{random.randint(100000, 999999)}"
        
        return {
            'transaction_id': self.generate_transaction_id(),
            'date': trans_date.strftime('%Y-%m-%d'),
            'amount': round(amount, 2),
            'account_code': account_code,
            'account_name': account_name,
            'payee_payer': payee,
            'journal_entry': journal_entry,
            'reference_number': reference_number,
            'debit_credit': 'Debit' if self.is_debit_account(account_code) else 'Credit',
            'fraud_flag': 0,
            'notes': f"Legitimate transaction - {account_name}"
        }
    
    def is_debit_account(self, account_code: str) -> bool:
        """Determine if account normally has debit balance"""
        # Assets (1000-1999) and Expenses (5000-6999) are debit accounts
        return account_code.startswith('1') or account_code.startswith('5') or account_code.startswith('6')
    
    def generate_fraudulent_transaction(self) -> Dict:
        """Generate a fraudulent transaction entry with realistic fraud patterns"""
        fraud_types = ['rounded', 'threshold_avoidance', 'repeated_digits', 'sequential', 'duplicate_vendor']
        fraud_type = random.choice(fraud_types)
        
        # Generate fraudulent amount
        amount = self.generate_fraudulent_amount(fraud_type)
        
        # Typically fraud in expense accounts
        expense_accounts = {k: v for k, v in self.chart_of_accounts.items() if k.startswith('6')}
        account_code = random.choice(list(expense_accounts.keys()))
        account_name = expense_accounts[account_code]
        
        # Fraudulent payee patterns
        if fraud_type == 'duplicate_vendor':
            # Create fake vendor that sounds similar to real one
            real_vendor = random.choice(self.vendors.get(self.company_type, ['Generic Vendor']))
            payee = real_vendor.replace('Co', 'Corp').replace('Inc', 'LLC')
        elif random.random() < 0.4:
            # Suspicious generic vendor names
            payee = f"Vendor Services {random.randint(1, 999)}"
        else:
            payee = random.choice(self.vendors.get(self.company_type, ['Generic Vendor']))
        
        # Create reference numbers (might be suspicious)
        journal_entry = f"JE{random.randint(1000, 9999)}"
        if fraud_type == 'sequential':
            reference_number = f"REF{random.randint(100000, 100010)}"  # Sequential refs
        else:
            reference_number = f"REF{random.randint(100000, 999999)}"
        
        return {
            'transaction_id': self.generate_transaction_id(),
            'date': self.generate_random_date().strftime('%Y-%m-%d'),
            'amount': amount,
            'account_code': account_code,
            'account_name': account_name,
            'payee_payer': payee,
            'journal_entry': journal_entry,
            'reference_number': reference_number,
            'debit_credit': 'Debit',  # Fraudulent expenses are typically debits
            'fraud_flag': 1,
            'notes': f"Potential fraud: {fraud_type} pattern"
        }
    
    def generate_ledger(self, num_transactions: int = 1000, fraud_percentage: float = 0.1) -> pd.DataFrame:
        """
        Generate complete synthetic ledger
        
        Args:
            num_transactions: Total number of transactions to generate
            fraud_percentage: Percentage of transactions that should be fraudulent (0.0-1.0)
        
        Returns:
            DataFrame containing the synthetic ledger
        """
        transactions = []
        num_fraudulent = int(num_transactions * fraud_percentage)
        num_legitimate = num_transactions - num_fraudulent
        
        print(f"Generating {num_legitimate} legitimate transactions...")
        # Generate legitimate transactions
        for i in range(num_legitimate):
            transactions.append(self.generate_legitimate_transaction())
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_legitimate} legitimate transactions")
        
        print(f"Generating {num_fraudulent} fraudulent transactions...")
        # Generate fraudulent transactions
        for i in range(num_fraudulent):
            transactions.append(self.generate_fraudulent_transaction())
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_fraudulent} fraudulent transactions")
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(transactions)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Add description field (more realistic format)
        df['description'] = df['account_name'] + ' - ' + df['payee_payer']
        
        # Reorder columns to match realistic ledger format
        column_order = ['transaction_id', 'date', 'description', 'account_code', 'account_name', 
                       'amount', 'debit_credit', 'payee_payer', 'journal_entry', 'reference_number',
                       'fraud_flag', 'notes']
        df = df[column_order]
        
        return df
    
    def validate_benford_law(self, df: pd.DataFrame, amount_column: str = 'amount') -> Dict:
        """
        Validate that the generated data follows Benford's Law
        
        Args:
            df: DataFrame containing the ledger data
            amount_column: Column name containing the amounts
        
        Returns:
            Dictionary with validation results
        """
        # Filter legitimate transactions only
        legitimate_df = df[df['fraud_flag'] == 0]
        amounts = legitimate_df[amount_column].values
        
        # Extract first digits
        first_digits = [int(str(abs(amount)).replace('.', '')[0]) for amount in amounts if amount > 0]
        
        # Calculate observed frequencies
        observed_counts = {}
        for digit in range(1, 10):
            observed_counts[digit] = first_digits.count(digit)
        
        total_count = len(first_digits)
        observed_proportions = {digit: count/total_count for digit, count in observed_counts.items()}
        
        # Calculate expected frequencies based on Benford's Law
        expected_proportions = self.benford_probs
        expected_counts = {digit: prob * total_count for digit, prob in expected_proportions.items()}
        
        # Chi-square test
        chi2_stat = sum((observed_counts[digit] - expected_counts[digit])**2 / expected_counts[digit] 
                       for digit in range(1, 10))
        
        # Degrees of freedom = 8 (digits 1-9)
        p_value = 1 - stats.chi2.cdf(chi2_stat, 8)
        
        return {
            'observed_proportions': observed_proportions,
            'expected_proportions': expected_proportions,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'follows_benford': p_value > 0.05,
            'total_legitimate_transactions': total_count
        }
    
    def plot_benford_analysis(self, df: pd.DataFrame, amount_column: str = 'amount'):
        """Plot Benford's Law analysis"""
        validation_results = self.validate_benford_law(df, amount_column)
        
        digits = list(range(1, 10))
        observed = [validation_results['observed_proportions'][d] for d in digits]
        expected = [validation_results['expected_proportions'][d] for d in digits]
        
        plt.figure(figsize=(12, 6))
        
        # Plot comparison
        plt.subplot(1, 2, 1)
        x = np.arange(len(digits))
        width = 0.35
        
        plt.bar(x - width/2, observed, width, label='Observed', alpha=0.7)
        plt.bar(x + width/2, expected, width, label='Expected (Benford)', alpha=0.7)
        
        plt.xlabel('First Digit')
        plt.ylabel('Proportion')
        plt.title('Benford\'s Law Analysis - Legitimate Transactions')
        plt.xticks(x, digits)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot fraud vs legitimate amounts distribution
        plt.subplot(1, 2, 2)
        legitimate_amounts = df[df['fraud_flag'] == 0]['amount']
        fraudulent_amounts = df[df['fraud_flag'] == 1]['amount']
        
        plt.hist(legitimate_amounts, bins=50, alpha=0.7, label='Legitimate', density=True)
        plt.hist(fraudulent_amounts, bins=50, alpha=0.7, label='Fraudulent', density=True)
        
        plt.xlabel('Amount ($)')
        plt.ylabel('Density')
        plt.title('Amount Distribution: Legitimate vs Fraudulent')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print validation results
        print(f"\nBenford's Law Validation Results:")
        print(f"Chi-square statistic: {validation_results['chi2_statistic']:.4f}")
        print(f"P-value: {validation_results['p_value']:.4f}")
        print(f"Follows Benford's Law: {validation_results['follows_benford']}")
        print(f"Total legitimate transactions analyzed: {validation_results['total_legitimate_transactions']}")
    
    def save_ledger(self, df: pd.DataFrame, filename: str = "synthetic_ledger.csv"):
        """Save ledger to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Ledger saved to {filename}")
        
        # Also save as JSON for flexibility
        json_filename = filename.replace('.csv', '.json')
        df.to_json(json_filename, orient='records', indent=2)
        print(f"Ledger also saved to {json_filename}")

if __name__ == "__main__":
    print("Synthetic Ledger Generators Available:")
    print("1. SyntheticLedgerGenerator - Basic generator")
    print("2. ImprovedSyntheticLedgerGenerator - Advanced generator with realistic chart of accounts")
    
    # Example usage with improved generator
    generator = ImprovedSyntheticLedgerGenerator(
        company_type="manufacturing",
        company_size="medium"
    )
    
    print("\nGenerating realistic synthetic accounting ledger...")
    ledger_df = generator.generate_ledger(num_transactions=2000, fraud_percentage=0.08)
    
    print(f"\nLedger Statistics:")
    print(f"Total transactions: {len(ledger_df)}")
    print(f"Legitimate transactions: {len(ledger_df[ledger_df['fraud_flag'] == 0])}")
    print(f"Fraudulent transactions: {len(ledger_df[ledger_df['fraud_flag'] == 1])}")
    print(f"Total amount: ${ledger_df['amount'].sum():,.2f}")
    print(f"Average amount: ${ledger_df['amount'].mean():.2f}")
    
    # Save ledger
    generator.save_ledger(ledger_df, "realistic_accounting_ledger.csv")
