import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
class BenfordsLawAnalyzer:
    """
    A comprehensive Benford's Law analyzer for forensic accounting.
    Detects potential fraud by analyzing the distribution of leading digits
    in financial data.
    """
    
    def __init__(self):
        # Benford's Law expected probabilities for digits 1-9
        self.benford_probabilities = {
            1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
            6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
        }
        
        # Store analysis results
        self.analysis_results = {}
        
    def extract_leading_digits(self, data: pd.Series) -> List[int]:
        """
        Extract leading digits from a series of numbers.
        
        Args:
            data: Pandas Series containing numerical values
            
        Returns:
            List of leading digits (1-9)
        """
        leading_digits = []
        
        for value in data:
            if pd.isna(value) or value == 0:
                continue
                
            # Convert to string and remove decimal point and sign
            str_value = str(abs(float(value))).replace('.', '')
            
            # Find first non-zero digit
            for char in str_value:
                if char.isdigit() and char != '0':
                    leading_digits.append(int(char))
                    break
                    
        return leading_digits
    
    def calculate_observed_frequencies(self, leading_digits: List[int]) -> Dict[int, float]:
        """
        Calculate observed frequencies of leading digits.
        
        Args:
            leading_digits: List of leading digits
            
        Returns:
            Dictionary with digit frequencies
        """
        if not leading_digits:
            return {}
            
        total_count = len(leading_digits)
        observed_frequencies = {}
        
        for digit in range(1, 10):
            count = leading_digits.count(digit)
            observed_frequencies[digit] = count / total_count
            
        return observed_frequencies
    
    def chi_square_test(self, observed_frequencies: Dict[int, float], 
                       sample_size: int) -> Tuple[float, float, bool]:
        """
        Perform chi-square goodness-of-fit test.
        
        Args:
            observed_frequencies: Observed digit frequencies
            sample_size: Total number of observations
            
        Returns:
            Tuple of (chi_square_statistic, p_value, is_significant)
        """
        observed_counts = [observed_frequencies.get(digit, 0) * sample_size for digit in range(1, 10)]
        expected_counts = [self.benford_probabilities[digit] * sample_size for digit in range(1, 10)]
        
        # Perform chi-square test
        chi_square_stat, p_value = stats.chisquare(observed_counts, expected_counts)
        
        # Check significance at 95% confidence level
        critical_value = stats.chi2.ppf(0.95, df=8)  # 8 degrees of freedom
        is_significant = chi_square_stat > critical_value
        
        return chi_square_stat, p_value, is_significant
    
    def z_test_by_digit(self, observed_frequencies: Dict[int, float], 
                       sample_size: int) -> Dict[int, Tuple[float, bool]]:
        """
        Perform Z-test for each digit individually.
        
        Args:
            observed_frequencies: Observed digit frequencies
            sample_size: Total number of observations
            
        Returns:
            Dictionary with Z-scores and significance for each digit
        """
        z_scores = {}
        
        for digit in range(1, 10):
            observed_prop = observed_frequencies.get(digit, 0)
            expected_prop = self.benford_probabilities[digit]
            
            # Calculate standard error
            standard_error = np.sqrt((expected_prop * (1 - expected_prop)) / sample_size)
            
            # Calculate Z-score
            if standard_error > 0:
                z_score = (observed_prop - expected_prop) / standard_error
                is_significant = abs(z_score) > 1.96  # 95% confidence
                z_scores[digit] = (z_score, is_significant)
            else:
                z_scores[digit] = (0, False)
                
        return z_scores
    
    def detect_fraud_patterns(self, observed_frequencies: Dict[int, float], 
                            z_scores: Dict[int, Tuple[float, bool]]) -> Dict[str, any]:
        """
        Detect common fraud patterns based on digit analysis.
        
        Args:
            observed_frequencies: Observed digit frequencies
            z_scores: Z-scores for each digit
            
        Returns:
            Dictionary with fraud pattern analysis
        """
        fraud_indicators = {
            'round_number_bias': False,
            'digit_avoidance': [],
            'digit_overuse': [],
            'human_fabrication_signs': [],
            'overall_fraud_score': 0
        }
        
        # Check for round number bias (overuse of 5 and 9)
        if (observed_frequencies.get(5, 0) > self.benford_probabilities[5] * 1.5 or
            observed_frequencies.get(9, 0) > self.benford_probabilities[9] * 1.5):
            fraud_indicators['round_number_bias'] = True
            fraud_indicators['overall_fraud_score'] += 20
        
        # Check for digit avoidance (underuse of 1 and 2)
        if observed_frequencies.get(1, 0) < self.benford_probabilities[1] * 0.7:
            fraud_indicators['digit_avoidance'].append(1)
            fraud_indicators['overall_fraud_score'] += 15
            
        if observed_frequencies.get(2, 0) < self.benford_probabilities[2] * 0.7:
            fraud_indicators['digit_avoidance'].append(2)
            fraud_indicators['overall_fraud_score'] += 10
        
        # Check for significant overuse of any digit
        for digit, (z_score, is_significant) in z_scores.items():
            if is_significant and z_score > 2:
                fraud_indicators['digit_overuse'].append(digit)
                fraud_indicators['overall_fraud_score'] += 10
        
        # Human fabrication signs (specific patterns)
        if observed_frequencies.get(7, 0) > self.benford_probabilities[7] * 1.3:
            fraud_indicators['human_fabrication_signs'].append("Overuse of digit 7")
            fraud_indicators['overall_fraud_score'] += 8
            
        if observed_frequencies.get(3, 0) > self.benford_probabilities[3] * 1.3:
            fraud_indicators['human_fabrication_signs'].append("Overuse of digit 3")
            fraud_indicators['overall_fraud_score'] += 5
        
        return fraud_indicators
    
    def analyze_dataset(self, data: pd.Series, dataset_name: str = "Dataset") -> Dict[str, any]:
        """
        Perform complete Benford's Law analysis on a dataset.
        
        Args:
            data: Pandas Series containing numerical values
            dataset_name: Name for the dataset (for reporting)
            
        Returns:
            Complete analysis results
        """
        # Extract leading digits
        leading_digits = self.extract_leading_digits(data)
        
        if len(leading_digits) < 50:
            return {
                'error': f"Insufficient data: {len(leading_digits)} observations (minimum 50 required)",
                'dataset_name': dataset_name
            }
        
        # Calculate observed frequencies
        observed_frequencies = self.calculate_observed_frequencies(leading_digits)
        sample_size = len(leading_digits)
        
        # Perform statistical tests
        chi_square_stat, p_value, chi_square_significant = self.chi_square_test(
            observed_frequencies, sample_size
        )
        
        z_scores = self.z_test_by_digit(observed_frequencies, sample_size)
        
        # Detect fraud patterns
        fraud_patterns = self.detect_fraud_patterns(observed_frequencies, z_scores)
        
        # Compile results
        results = {
            'dataset_name': dataset_name,
            'sample_size': sample_size,
            'observed_frequencies': observed_frequencies,
            'expected_frequencies': self.benford_probabilities,
            'chi_square_statistic': chi_square_stat,
            'chi_square_p_value': p_value,
            'chi_square_significant': chi_square_significant,
            'z_scores': z_scores,
            'fraud_patterns': fraud_patterns,
            'compliance_score': max(0, 100 - fraud_patterns['overall_fraud_score']),
            'risk_level': self._determine_risk_level(fraud_patterns['overall_fraud_score'])
        }
        
        # Store results
        self.analysis_results[dataset_name] = results
        
        return results
    
    def _determine_risk_level(self, fraud_score: int) -> str:
        """Determine risk level based on fraud score."""
        if fraud_score >= 50:
            return "HIGH"
        elif fraud_score >= 25:
            return "MEDIUM"
        elif fraud_score >= 10:
            return "LOW"
        else:
            return "MINIMAL"
    
    def generate_report(self, analysis_results: Dict[str, any]) -> str:
        """
        Generate a comprehensive text report.
        
        Args:
            analysis_results: Results from analyze_dataset()
            
        Returns:
            Formatted report string
        """
        if 'error' in analysis_results:
            return f"Error: {analysis_results['error']}"
        
        report = f"""
=== BENFORD'S LAW ANALYSIS REPORT ===
Dataset: {analysis_results['dataset_name']}
Sample Size: {analysis_results['sample_size']:,} transactions
Compliance Score: {analysis_results['compliance_score']:.1f}/100
Risk Level: {analysis_results['risk_level']}

=== STATISTICAL TESTS ===
Chi-Square Test:
  Statistic: {analysis_results['chi_square_statistic']:.4f}
  P-Value: {analysis_results['chi_square_p_value']:.4f}
  Significant Deviation: {'YES' if analysis_results['chi_square_significant'] else 'NO'}

=== DIGIT FREQUENCY ANALYSIS ===
Digit | Observed | Expected | Difference | Z-Score | Significant
------|----------|----------|------------|---------|------------"""
        
        for digit in range(1, 10):
            obs = analysis_results['observed_frequencies'].get(digit, 0)
            exp = analysis_results['expected_frequencies'][digit]
            diff = obs - exp
            z_score, significant = analysis_results['z_scores'][digit]
            
            report += f"\n  {digit}   |  {obs:6.3f}  |  {exp:6.3f}  |   {diff:+6.3f}   |  {z_score:+6.2f}  | {'YES' if significant else 'NO'}"
        
        report += f"""

=== FRAUD PATTERN ANALYSIS ===
Round Number Bias: {'DETECTED' if analysis_results['fraud_patterns']['round_number_bias'] else 'Not Detected'}
Digit Avoidance: {analysis_results['fraud_patterns']['digit_avoidance'] if analysis_results['fraud_patterns']['digit_avoidance'] else 'None'}
Digit Overuse: {analysis_results['fraud_patterns']['digit_overuse'] if analysis_results['fraud_patterns']['digit_overuse'] else 'None'}
Human Fabrication Signs: {analysis_results['fraud_patterns']['human_fabrication_signs'] if analysis_results['fraud_patterns']['human_fabrication_signs'] else 'None'}
Overall Fraud Score: {analysis_results['fraud_patterns']['overall_fraud_score']}/100

=== RECOMMENDATIONS ==="""
        
        if analysis_results['risk_level'] == "HIGH":
            report += "\n- IMMEDIATE INVESTIGATION REQUIRED"
            report += "\n- Review all flagged transactions manually"
            report += "\n- Consider external forensic audit"
        elif analysis_results['risk_level'] == "MEDIUM":
            report += "\n- Enhanced monitoring recommended"
            report += "\n- Sample additional transactions for review"
            report += "\n- Implement additional controls"
        elif analysis_results['risk_level'] == "LOW":
            report += "\n- Minor deviations detected"
            report += "\n- Continue regular monitoring"
            report += "\n- Consider random sampling for verification"
        else:
            report += "\n- Data appears to follow Benford's Law"
            report += "\n- No immediate fraud concerns"
            report += "\n- Maintain standard monitoring procedures"
        
        return report
    
    def create_visualization(self, analysis_results: Dict[str, any], 
                           save_path: Optional[str] = None) -> None:
        """
        Create visualization comparing observed vs expected frequencies.
        
        Args:
            analysis_results: Results from analyze_dataset()
            save_path: Optional path to save the plot
        """
        if 'error' in analysis_results:
            print(f"Cannot create visualization: {analysis_results['error']}")
            return
        
        # Prepare data for plotting
        digits = list(range(1, 10))
        observed = [analysis_results['observed_frequencies'].get(d, 0) for d in digits]
        expected = [analysis_results['expected_frequencies'][d] for d in digits]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Bar chart comparison
        x = np.arange(len(digits))
        width = 0.35
        
        ax1.bar(x - width/2, observed, width, label='Observed', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, expected, width, label='Expected (Benford)', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Leading Digit')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Benford\'s Law Analysis: {analysis_results["dataset_name"]}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(digits)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Deviation analysis
        deviations = [obs - exp for obs, exp in zip(observed, expected)]
        colors = ['red' if abs(dev) > 0.02 else 'green' for dev in deviations]
        
        ax2.bar(digits, deviations, color=colors, alpha=0.7)
        ax2.set_xlabel('Leading Digit')
        ax2.set_ylabel('Deviation from Expected')
        ax2.set_title('Deviations from Benford\'s Law')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# Example usage and testing functions
def create_sample_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    
    # Generate legitimate transactions (following Benford's Law)
    legitimate_data = []
    for _ in range(800):
        # Generate amounts that naturally follow Benford's Law
        order_of_magnitude = np.random.uniform(1, 5)  # 10 to 100,000
        amount = 10 ** order_of_magnitude * np.random.exponential(1)
        legitimate_data.append(amount)
    
    # Generate fraudulent transactions (violating Benford's Law)
    fraudulent_data = []
    for _ in range(200):
        # Round numbers and specific patterns
        if np.random.random() < 0.4:
            # Round numbers
            amount = np.random.choice([500, 1000, 1500, 2000, 2500, 5000, 10000])
        elif np.random.random() < 0.3:
            # Numbers starting with 5 or 9
            base = np.random.choice([5, 9])
            amount = base * (10 ** np.random.randint(2, 4))
        else:
            # Random but avoiding 1 and 2
            first_digit = np.random.choice([3, 4, 5, 6, 7, 8, 9])
            amount = first_digit * (10 ** np.random.randint(2, 4))
        
        fraudulent_data.append(amount)
    
    return pd.Series(legitimate_data + fraudulent_data)

def main():
    """Main function to demonstrate the analyzer."""
    # Create analyzer instance
    analyzer = BenfordsLawAnalyzer()
    
    # Generate sample data
    print("Generating sample transaction data...")
    sample_data = create_sample_data()
    
    # Perform analysis
    print("Performing Benford's Law analysis...")
    results = analyzer.analyze_dataset(sample_data, "Sample Transaction Data")
    
    # Generate report
    print("\nGenerating analysis report...")
    report = analyzer.generate_report(results)
    print(report)
    
    # Create visualization
    print("\nCreating visualization...")
    analyzer.create_visualization(results)
    
    print("\nAnalysis complete!")
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()


# PART TWO!!!
# PART TWO!!!
# PART TWO!!!
# PART TWO!!!
# Advanced Analyzer 


class AdvancedLedgerAnalyzer:
    """
    Advanced accounting ledger analyzer that complements BenfordsLawAnalyzer.
    Provides duplicate detection, outlier analysis, threshold monitoring, 
    ratio analysis, and time-series features.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with ledger dataframe.
        
        Expected columns: ['amount', 'payee', 'date', 'type', 'description']
        """
        self.df = df.copy()
        self.prepare_data()
        self.analysis_results = {}
        
    def prepare_data(self):
        """Clean and prepare data for analysis."""
        # Convert date column if it exists
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            
        # Ensure amount is numeric
        self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
        self.df['abs_amount'] = abs(self.df['amount'])
        
        # Clean payee field if it exists
        if 'payee' in self.df.columns:
            self.df['payee_clean'] = self.df['payee'].str.strip().str.upper()
            
        # Clean type field if it exists
        if 'type' in self.df.columns:
            self.df['type_clean'] = self.df['type'].str.strip().str.upper()
    
    def analyze_duplicates(self) -> Dict[str, any]:
        """
        Analyze duplicate transactions and patterns.
        
        Returns:
            Dictionary with duplicate analysis results
        """
        print("Analyzing duplicates...")
        duplicate_features = {}
        
        # 1. Exact amount duplicates
        amount_duplicates = self.df[self.df.duplicated(subset=['amount'], keep=False)]
        duplicate_features['exact_amount_duplicates'] = len(amount_duplicates)
        duplicate_features['exact_amount_duplicate_rate'] = len(amount_duplicates) / len(self.df) * 100
        
        # 2. Amount-Payee duplicates
        if 'payee_clean' in self.df.columns:
            amount_payee_duplicates = self.df[self.df.duplicated(subset=['amount', 'payee_clean'], keep=False)]
            duplicate_features['amount_payee_duplicates'] = len(amount_payee_duplicates)
            duplicate_features['amount_payee_duplicate_rate'] = len(amount_payee_duplicates) / len(self.df) * 100
        
        # 3. Full transaction duplicates (amount-payee-date)
        if 'payee_clean' in self.df.columns and 'date' in self.df.columns:
            exact_duplicates = self.df[self.df.duplicated(subset=['amount', 'payee_clean', 'date'], keep=False)]
            duplicate_features['exact_transaction_duplicates'] = len(exact_duplicates)
            duplicate_features['exact_transaction_duplicate_rate'] = len(exact_duplicates) / len(self.df) * 100
            
            # Get the actual duplicate transactions for review
            if len(exact_duplicates) > 0:
                duplicate_features['duplicate_transactions'] = exact_duplicates[['amount', 'payee', 'date']].to_dict('records')
        
        # 4. Round number analysis
        amounts = self.df['amount'].dropna()
        round_numbers = amounts[amounts % 1 == 0]  # Whole numbers
        round_tens = amounts[amounts % 10 == 0]     # Tens
        round_hundreds = amounts[amounts % 100 == 0]  # Hundreds
        round_thousands = amounts[amounts % 1000 == 0]  # Thousands
        
        duplicate_features['round_number_count'] = len(round_numbers)
        duplicate_features['round_number_rate'] = len(round_numbers) / len(amounts) * 100
        duplicate_features['round_tens_rate'] = len(round_tens) / len(amounts) * 100
        duplicate_features['round_hundreds_rate'] = len(round_hundreds) / len(amounts) * 100
        duplicate_features['round_thousands_rate'] = len(round_thousands) / len(amounts) * 100
        
        # 5. Payee frequency analysis
        if 'payee_clean' in self.df.columns:
            payee_counts = self.df['payee_clean'].value_counts()
            duplicate_features['unique_payees'] = len(payee_counts)
            duplicate_features['top_payee_frequency'] = payee_counts.iloc[0] if len(payee_counts) > 0 else 0
            duplicate_features['payee_concentration'] = (payee_counts.iloc[:5].sum() / len(self.df) * 100) if len(payee_counts) > 0 else 0
            
            # Flag high-frequency payees
            high_freq_payees = payee_counts[payee_counts > len(self.df) * 0.05]  # More than 5% of transactions
            duplicate_features['high_frequency_payees'] = len(high_freq_payees)
            if len(high_freq_payees) > 0:
                duplicate_features['high_frequency_payee_list'] = high_freq_payees.to_dict()
        
        return duplicate_features
    
    def analyze_outliers(self) -> Dict[str, any]:
        """
        Analyze outlier transactions using Z-scores and IQR methods.
        
        Returns:
            Dictionary with outlier analysis results
        """
        print("Analyzing outliers...")
        outlier_features = {}
        amounts = self.df['amount'].dropna()
        
        # Z-score analysis
        z_scores = np.abs(stats.zscore(amounts))
        self.df['z_score'] = np.nan
        self.df.loc[amounts.index, 'z_score'] = z_scores
        
        z_outliers_2 = self.df[self.df['z_score'] > 2]
        z_outliers_3 = self.df[self.df['z_score'] > 3]
        
        outlier_features['z_score_outliers_2sigma'] = len(z_outliers_2)
        outlier_features['z_score_outliers_3sigma'] = len(z_outliers_3)
        outlier_features['z_score_outlier_rate_2sigma'] = len(z_outliers_2) / len(amounts) * 100
        outlier_features['z_score_outlier_rate_3sigma'] = len(z_outliers_3) / len(amounts) * 100
        
        # IQR analysis
        Q1 = amounts.quantile(0.25)
        Q3 = amounts.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = self.df[(self.df['amount'] < lower_bound) | (self.df['amount'] > upper_bound)]
        outlier_features['iqr_outliers'] = len(iqr_outliers)
        outlier_features['iqr_outlier_rate'] = len(iqr_outliers) / len(amounts) * 100
        
        # Extreme outliers (3 * IQR)
        extreme_lower = Q1 - 3 * IQR
        extreme_upper = Q3 + 3 * IQR
        extreme_outliers = self.df[(self.df['amount'] < extreme_lower) | (self.df['amount'] > extreme_upper)]
        outlier_features['extreme_outliers'] = len(extreme_outliers)
        outlier_features['extreme_outlier_rate'] = len(extreme_outliers) / len(amounts) * 100
        
        # Statistical measures
        outlier_features['amount_stats'] = {
            'mean': amounts.mean(),
            'median': amounts.median(),
            'std': amounts.std(),
            'skewness': stats.skew(amounts),
            'kurtosis': stats.kurtosis(amounts),
            'min': amounts.min(),
            'max': amounts.max(),
            'q1': Q1,
            'q3': Q3,
            'iqr': IQR
        }
        
        # Flag extreme outliers for review
        if len(extreme_outliers) > 0:
            outlier_features['extreme_outlier_transactions'] = extreme_outliers[['amount', 'payee', 'date']].to_dict('records')
        
        return outlier_features
    
    def analyze_thresholds(self) -> Dict[str, any]:
        """
        Analyze transactions near key financial reporting thresholds.
        
        Returns:
            Dictionary with threshold analysis results
        """
        print("Analyzing thresholds...")
        threshold_features = {}
        amounts = self.df['abs_amount'].dropna()
        
        # Define key thresholds
        thresholds = {
            'cash_reporting_10k': 10000,
            'structuring_9999': 9999,
            'structuring_9500': 9500,
            'suspicious_5k': 5000,
            'check_clearing_3k': 3000,
            '_daily_limit_1k': 1000
        }
        
        # Analyze each threshold
        for threshold_name, threshold_value in thresholds.items():
            # Exact hits
            exact_hits = len(amounts[amounts == threshold_value])
            threshold_features[f'{threshold_name}_exact_hits'] = exact_hits
            
            # Just under threshold (within 5%)
            just_under = amounts[(amounts >= threshold_value * 0.95) & (amounts < threshold_value)]
            threshold_features[f'{threshold_name}_just_under'] = len(just_under)
            
            # Just over threshold (within 5%)
            just, just_over = amounts[(amounts > threshold_value) & (amounts <= threshold_value * 1.05)]
            threshold_features[f'{threshold_name}_just_over'] = len(just_over)
            
            # Clustering around threshold (within 10%)
            cluster_range = amounts[(amounts >= threshold_value * 0.9) & (amounts <= threshold_value * 1.1)]
            threshold_features[f'{threshold_name}_cluster'] = len(cluster_range)
            threshold_features[f'{threshold_name}_cluster_rate'] = len(cluster_range) / len(amounts) * 100
        
        # Structuring analysis (specific focus on $9,500-$9,999 range)
        structuring_range = amounts[(amounts >= 9500) & (amounts <= 9999)]
        threshold_features['structuring_indicators'] = len(structuring_range)
        threshold_features['structuring_rate'] = len(structuring_range) / len(amounts) * 100
        
        # Get actual structuring transactions for review
        if len(structuring_range) > 0:
            structuring_transactions = self.df[self.df['abs_amount'].between(9500, 9999)]
            threshold_features['structuring_transactions'] = structuring_transactions[['amount', 'payee', 'date']].to_dict('records')
        
        return threshold_features
    
    def analyze_ratios(self) -> Dict[str, any]:
        """
        Analyze financial ratios and relationships.
        
        Returns:
            Dictionary with ratio analysis results
        """
        print("Analyzing ratios...")
        ratio_features = {}
        
        # Separate income and expenses based on type or amount sign
        if 'type_clean' in self.df.columns:
            income_keywords = ['INCOME', 'DEPOSIT', 'CREDIT', 'REVENUE', 'SALE']
            expense_keywords = ['EXPENSE', 'DEBIT', 'PAYMENT', 'WITHDRAWAL', 'PURCHASE']
            
            income_mask = self.df['type_clean'].str.contains('|'.join(income_keywords), na=False)
            expense_mask = self.df['type_clean'].str.contains('|'.join(expense_keywords), na=False)
            
            income = self.df[income_mask]
            expenses = self.df[expense_mask]
        else:
            # Fallback: assume positive amounts are income, negative are expenses
            income = self.df[self.df['amount'] > 0]
            expenses = self.df[self.df['amount'] < 0]
        
        # Calculate basic ratios
        total_income = income['amount'].sum()
        total_expenses = abs(expenses['amount'].sum())
        
        ratio_features['financial_summary'] = {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_income': total_income - total_expenses,
            'expense_to_income_ratio': total_expenses / total_income if total_income > 0 else 0
        }
        
        # Transaction count analysis
        ratio_features['transaction_counts'] = {
            'income_transactions': len(income),
            'expense_transactions': len(expenses),
            'expense_to_income_transaction_ratio': len(expenses) / len(income) if len(income) > 0 else 0
        }
        
        # Average transaction sizes
        avg_income = income['amount'].mean() if len(income) > 0 else 0
        avg_expense = abs(expenses['amount'].mean()) if len(expenses) > 0 else 0
        
        ratio_features['average_transactions'] = {
            'avg_income_transaction': avg_income,
            'avg_expense_transaction': avg_expense,
            'avg_expense_to_income_ratio': avg_expense / avg_income if avg_income > 0 else 0
        }
        
        # Concentration analysis
        if len(income) > 0:
            top_5_income = income.nlargest(5, 'amount')['amount'].sum()
            ratio_features['income_concentration'] = {
                'top_5_income_total': top_5_income,
                'top_5_income_percentage': top_5_income / total_income * 100 if total_income > 0 else 0
            }
        
        if len(expenses) > 0:
            top_5_expenses = expenses.nsmallest(5, 'amount')['amount'].sum()
            ratio_features['expense_concentration'] = {
                'top_5_expense_total': abs(top_5_expenses),
                'top_5_expense_percentage': abs(top_5_expenses) / total_expenses * 100 if total_expenses > 0 else 0
            }
        
        return ratio_features
    
    def analyze_time_series(self) -> Dict[str, any]:
        """
        Analyze time-series patterns and trends.
        
        Returns:
            Dictionary with time series analysis results
        """
        print("Analyzing time series...")
        time_features = {}
        
        if 'date' not in self.df.columns:
            time_features['error'] = "No date column available for time series analysis"
            return time_features
        
        # Monthly analysis
        monthly_data = self.df.groupby(['year', 'month']).agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'abs_amount': ['sum', 'mean']
        }).reset_index()
        
        monthly_data.columns = ['year', 'month', 'total_amount', 'avg_amount', 'transaction_count', 'amount_std', 'total_abs_amount', 'avg_abs_amount']
        
        time_features['monthly_analysis'] = {
            'total_months': len(monthly_data),
            'avg_monthly_transactions': monthly_data['transaction_count'].mean(),
            'monthly_transaction_volatility': monthly_data['transaction_count'].std(),
            'avg_monthly_amount': monthly_data['total_amount'].mean(),
            'monthly_amount_volatility': monthly_data['total_amount'].std()
        }
        
        # Quarterly analysis
        quarterly_data = self.df.groupby(['year', 'quarter']).agg({
            'amount': ['sum', 'count'],
            'abs_amount': 'sum'
        }).reset_index()
        
        quarterly_data.columns = ['year', 'quarter', 'total_amount', 'transaction_count', 'total_abs_amount']
        
        time_features['quarterly_analysis'] = {
            'total_quarters': len(quarterly_data),
            'avg_quarterly_transactions': quarterly_data['transaction_count'].mean(),
            'quarterly_transaction_volatility': quarterly_data['transaction_count'].std(),
            'avg_quarterly_amount': quarterly_data['total_amount'].mean(),
            'quarterly_amount_volatility': quarterly_data['total_amount'].std()
        }
        
        # Day of week analysis
        dow_analysis = self.df.groupby('day_of_week').agg({
            'amount': ['sum', 'count']
        }).reset_index()
        
        dow_analysis.columns = ['day_of_week', 'total_amount', 'transaction_count']
        
        # Weekend vs weekday analysis
        weekend_transactions = self.df[self.df['day_of_week'].isin([5, 6])]  # Saturday=5, Sunday=6
        weekday_transactions = self.df[~self.df['day_of_week'].isin([5, 6])]
        
        time_features['weekend_analysis'] = {
            'weekend_transaction_count': len(weekend_transactions),
            'weekday_transaction_count': len(weekday_transactions),
            'weekend_transaction_rate': len(weekend_transactions) / len(self.df) * 100,
            'weekend_amount_total': weekend_transactions['amount'].sum(),
            'weekday_amount_total': weekday_transactions['amount'].sum(),
            'weekend_amount_rate': weekend_transactions['amount'].sum() / self.df['amount'].sum() * 100
        }
        
        # Unusual activity detection
        daily_counts = self.df.groupby(self.df['date'].dt.date).size()
        daily_amounts = self.df.groupby(self.df['date'].dt.date)['amount'].sum()
        
        # Days with unusually high transaction counts
        daily_mean = daily_counts.mean()
        daily_std = daily_counts.std()
        burst_days = daily_counts[daily_counts > daily_mean + 2 * daily_std]
        
        time_features['unusual_activity'] = {
            'burst_activity_days': len(burst_days),
            'max_daily_transactions': daily_counts.max(),
            'avg_daily_transactions': daily_mean,
            'max_daily_amount': daily_amounts.max(),
            'avg_daily_amount': daily_amounts.mean()
        }
        
        return time_features
    
    def generate_comprehensive_analysis(self) -> Dict[str, any]:
        """
        Generate comprehensive analysis combining all feature types.
        
        Returns:
            Dictionary with all analysis results
        """
        print("Starting comprehensive ledger analysis...")
        
        # Basic dataset info
        analysis_results = {
            'dataset_summary': {
                'total_transactions': len(self.df),
                'unique_amounts': self.df['amount'].nunique(),
                'amount_range': {
                    'min': self.df['amount'].min(),
                    'max': self.df['amount'].max(),
                    'mean': self.df['amount'].mean(),
                    'median': self.df['amount'].median()
                }
            }
        }
        
        if 'date' in self.df.columns:
            analysis_results['dataset_summary']['date_range'] = {
                'start': self.df['date'].min(),
                'end': self.df['date'].max(),
                'days_covered': (self.df['date'].max() - self.df['date'].min()).days
            }
        
        # Run all analyses
        analysis_results['duplicates'] = self.analyze_duplicates()
        analysis_results['outliers'] = self.analyze_outliers()
        analysis_results['thresholds'] = self.analyze_thresholds()
        analysis_results['ratios'] = self.analyze_ratios()
        analysis_results['time_series'] = self.analyze_time_series()
        
        # Calculate composite risk scores
        analysis_results['risk_assessment'] = self.calculate_risk_scores(analysis_results)
        
        # Store results
        self.analysis_results = analysis_results
        
        print("Comprehensive analysis complete!")
        return analysis_results
    
    def calculate_risk_scores(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        """
        Calculate risk scores based on all analysis results.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            Dictionary with risk assessment
        """
        risk_scores = {}
        
        # Duplicate risk (0-100)
        duplicate_risk = 0
        if 'duplicates' in analysis_results:
            duplicate_risk += min(analysis_results['duplicates'].get('exact_transaction_duplicate_rate', 0), 30)
            duplicate_risk += min(analysis_results['duplicates'].get('amount_payee_duplicate_rate', 0), 20)
            duplicate_risk += min(analysis_results['duplicates'].get('round_thousands_rate', 0), 15)
            duplicate_risk += min(analysis_results['duplicates'].get('payee_concentration', 0), 10)
            duplicate_risk += min(analysis_results['duplicates'].get('high_frequency_payees', 0) * 5, 25)
        
        risk_scores['duplicate_risk'] = min(duplicate_risk, 100)
        
        # Outlier risk (0-100)
        outlier_risk = 0
        if 'outliers' in analysis_results:
            outlier_risk += min(analysis_results['outliers'].get('z_score_outlier_rate_3sigma', 0), 30)
            outlier_risk += min(analysis_results['outliers'].get('extreme_outlier_rate', 0), 25)
            skewness = abs(analysis_results['outliers']['amount_stats'].get('skewness', 0))
            outlier_risk += min(skewness * 5, 20)
            kurtosis = abs(analysis_results['outliers']['amount_stats'].get('kurtosis', 0))
            outlier_risk += min(kurtosis * 2, 15)
        
        risk_scores['outlier_risk'] = min(outlier_risk, 100)
        
        # Threshold risk (0-100) - Focus on structuring
        threshold_risk = 0
        if 'thresholds' in analysis_results:
            threshold_risk += min(analysis_results['thresholds'].get('structuring_rate', 0), 50)
            threshold_risk += analysis_results['thresholds'].get('structuring_9999_exact_hits', 0) * 10
            threshold_risk += analysis_results['thresholds'].get('cash_reporting_10k_exact_hits', 0) * 5
            threshold_risk += min(analysis_results['thresholds'].get('cash_reporting_10k_cluster_rate', 0), 20)
        
        risk_scores['threshold_risk'] = min(threshold_risk, 100)
        
        # Time series risk (0-100)
        time_risk = 0
        if 'time_series' in analysis_results and 'error' not in analysis_results['time_series']:
            weekend_rate = analysis_results['time_series']['weekend_analysis'].get('weekend_transaction_rate', 0)
            time_risk += min(weekend_rate, 30)
            burst_days = analysis_results['time_series']['unusual_activity'].get('burst_activity_days', 0)
            time_risk += min(burst_days * 5, 25)
        
        risk_scores['time_series_risk'] = min(time_risk, 100)
        
        # Overall composite risk
        risk_weights = {
            'duplicate_risk': 0.3,
            'outlier_risk': 0.25,
            'threshold_risk': 0.35,
            'time_series_risk': 0.1
        }
        
        overall_risk = sum(risk_scores[risk_type] * weight for risk_type, weight in risk_weights.items())
        risk_scores['overall_risk'] = overall_risk
        
        # Risk level classification
        if overall_risk >= 70:
            risk_scores['risk_level'] = 'HIGH'
            risk_scores['risk_description'] = 'Multiple significant fraud indicators detected'
        elif overall_risk >= 40:
            risk_scores['risk_level'] = 'MEDIUM'
            risk_scores['risk_description'] = 'Some concerning patterns identified'
        elif overall_risk >= 20:
            risk_scores['risk_level'] = 'LOW'
            risk_scores['risk_description'] = 'Minor irregularities detected'
        else:
            risk_scores['risk_level'] = 'MINIMAL'
            risk_scores['risk_description'] = 'No significant anomalies detected'
        
        return risk_scores
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report.
        
        Returns:
            Formatted text report
        """
        if not self.analysis_results:
            return "No analysis results available. Run generate_comprehensive_analysis() first."
        
        results = self.analysis_results
        
        report = f"""
=== ADVANCED ACCOUNTING LEDGER ANALYSIS REPORT ===
DATASET OVERVIEW:
- Total Transactions: {results['dataset_summary']['total_transactions']:,}
- Unique Amounts: {results['dataset_summary']['unique_amounts']:,}
- Amount Range: ${results['dataset_summary']['amount_range']['min']:,.2f} to ${results['dataset_summary']['amount_range']['max']:,.2f}
- Average Amount: ${results['dataset_summary']['amount_range']['mean']:,.2f}
- Median Amount: ${results['dataset_summary']['amount_range']['median']:,.2f}
"""
        if 'date_range' in results['dataset_summary']:
            report += f"""
- Date Range: {results['dataset_summary']['date_range']['start'].strftime('%Y-%m-%d')} to {results['dataset_summary']['date_range']['end'].strftime('%Y-%m-%d')}
- Days Covered: {results['dataset_summary']['date_range']['days_covered']:,}
"""
        report += f"""
RISK ASSESSMENT:
- Overall Risk Level: {results['risk_assessment']['risk_level']}
- Overall Risk Score: {results['risk_assessment']['overall_risk']:.1f}/100
- Risk Description: {results['risk_assessment']['risk_description']}
- Individual Risk Scores:
  - Duplicate Risk: {results['risk_assessment']['duplicate_risk']:.1f}/100
  - Outlier Risk: {results['risk_assessment']['outlier_risk']:.1f}/100
  - Threshold Risk: {results['risk_assessment']['threshold_risk']:.1f}/100
  - Time Series Risk: {results['risk_assessment']['time_series_risk']:.1f}/100

DUPLICATE ANALYSIS:
- Exact Transaction Duplicates: {results['duplicates'].get('exact_transaction_duplicates', 0):,} ({results['duplicates'].get('exact_transaction_duplicate_rate', 0):.1f}%)
- Amount-Payee Duplicates: {results['duplicates'].get('amount_payee_duplicates', 0):,} ({results['duplicates'].get('amount_payee_duplicate_rate', 0):.1f}%)
- Round Number Rate: {results['duplicates'].get('round_number_rate', 0):.1f}%
- Round Thousands Rate: {results['duplicates'].get('round_thousands_rate', 0):.1f}%
- Unique Payees: {results['duplicates'].get('unique_payees', 0):,}
- High Frequency Payees: {results['duplicates'].get('high_frequency_payees', 0):,}

OUTLIER ANALYSIS:
- Z-Score Outliers (3σ): {results['outliers'].get('z_score_outliers_3sigma', 0):,} ({results['outliers'].get('z_score_outlier_rate_3sigma', 0):.1f}%)
- IQR Outliers: {results['outliers'].get('iqr_outliers', 0):,} ({results['outliers'].get('iqr_outlier_rate', 0):.1f}%)
- Extreme Outliers: {results['outliers'].get('extreme_outliers', 0):,} ({results['outliers'].get('extreme_outlier_rate', 0):.1f}%)
- Skewness: {results['outliers']['amount_stats'].get('skewness', 0):.2f}
- Kurtosis: {results['outliers']['amount_stats'].get('kurtosis', 0):.2f}

THRESHOLD ANALYSIS:
- Structuring Indicators ($9,500-$9,999): {results['thresholds'].get('structuring_indicators', 0):,} ({results['thresholds'].get('structuring_rate', 0):.1f}%)
- $10,000 Exact Hits: {results['thresholds'].get('cash_reporting_10k_exact_hits', 0):,}
- $9,999 Exact Hits: {results['thresholds'].get('structuring_9999_exact_hits', 0):,}
- $10,000 Cluster Range: {results['thresholds'].get('cash_reporting_10k_cluster', 0):,}

FINANCIAL RATIO ANALYSIS:
- Total Income: ${results['ratios']['financial_summary'].get('total_income', 0):,.2f}
- Total Expenses: ${results['ratios']['financial_summary'].get('total_expenses', 0):,.2f}
- Net Income: ${results['ratios']['financial_summary'].get('net_income', 0):,.2f}
- Expense-to-Income Ratio: {results['ratios']['financial_summary'].get('expense_to_income_ratio', 0):.2f}
- Average Income Transaction: ${results['ratios']['average_transactions'].get('avg_income_transaction', 0):,.2f}
- Average Expense Transaction: ${results['ratios']['average_transactions'].get('avg_expense_transaction', 0):,.2f}
"""
        if 'error' not in results['time_series']:
            report += f"""
TIME SERIES ANALYSIS:
- Average Monthly Transactions: {results['time_series']['monthly_analysis'].get('avg_monthly_transactions', 0):.0f}
- Monthly Transaction Volatility: {results['time_series']['monthly_analysis'].get('monthly_transaction_volatility', 0):.1f}
- Weekend Transaction Rate: {results['time_series']['weekend_analysis'].get('weekend_transaction_rate', 0):.1f}%
- Burst Activity Days: {results['time_series']['unusual_activity'].get('burst_activity_days', 0):,}
- Max Daily Transactions: {results['time_series']['unusual_activity'].get('max_daily_transactions', 0):,}
- Average Daily Transactions: {results['time_series']['unusual_activity'].get('avg_daily_transactions', 0):.1f}
- Max Daily Amount: ${results['time_series']['unusual_activity'].get('max_daily_amount', 0):,.2f}
- Average Daily Amount: ${results['time_series']['unusual_activity'].get('avg_daily_amount', 0):,.2f}
"""
        report += """
=== END OF REPORT ===
"""
        return report
    


# Example usage (how to plug the data into the analyser)
'''
data = pd.DataFrame({
    'amount': [100, 200, 9999, 10000, -50],
    'payee': ['A', 'B', 'C', 'D', 'E'],
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'type': ['income', 'expense', 'income', 'income', 'expense']
})

analyzer = AdvancedLedgerAnalyzer(data)
results = analyzer.generate_comprehensive_analysis()
print(analyzer.generate_summary_report())
'''




# Assuming BenfordsLawAnalyzer and AdvancedLedgerAnalyzer classes are defined elsewhere

def main():
    """Main function to analyze the accounting ledger from CSV."""
    # Load the CSV, parsing the 'date' column as datetime
    df = pd.read_csv('realistic_accounting_ledger.csv', parse_dates=['date'])

    # Define required and optional columns
    required_columns = ['amount', 'date', 'account_name']
    payee_candidates = ['company', 'vendor', 'recipient', 'payee']
    available_columns = [col for col in required_columns if col in df.columns]
    payee_column = next((col for col in payee_candidates if col in df.columns), None)

    # Check for required columns
    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Select columns and handle payee
    selected_columns = available_columns
    if payee_column:
        selected_columns.append(payee_column)
    df = df[selected_columns]
    if payee_column:
        df = df.rename(columns={payee_column: 'payee'})
    else:
        df['payee'] = 'Unknown'

    # Create 'type' column based on account_name
    df['type'] = df['account_name'].apply(
        lambda x: 'expense' if 'Expense' in x else ('income' if 'Revenue' in x or 'Sales' in x else 'other')
    )

    # Benford's Law analysis
    print("Performing Benford's Law analysis...")
    benford_analyzer = BenfordsLawAnalyzer()
    amounts = df['amount']  # Pass the 'amount' column as a Series
    benford_results = benford_analyzer.analyze_dataset(amounts, "Accounting Ledger")
    benford_report = benford_analyzer.generate_report(benford_results)
    print(benford_report)
    benford_analyzer.create_visualization(benford_results)

    # Advanced ledger analysis
    print("\nPerforming advanced ledger analysis...")
    advanced_analyzer = AdvancedLedgerAnalyzer(df)
    advanced_results = advanced_analyzer.generate_comprehensive_analysis()
    advanced_report = advanced_analyzer.generate_summary_report()
    print(advanced_report)
    advanced_analyzer.create_visualizations(advanced_results)

    print("\nAnalysis complete!")
    return benford_analyzer, benford_results, advanced_analyzer, advanced_results

if __name__ == "__main__":
    benford_analyzer, benford_results, advanced_analyzer, advanced_results = main()

