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

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

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
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
            
        # Ensure amount is numeric
        self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
        self.df['abs_amount'] = self.df['amount'].abs()
        
        # Clean payee field if it exists
        if 'payee' in self.df.columns:
            self.df['payee_clean'] = self.df['payee'].astype(str).str.strip().str.upper()
            
        # Clean type field if it exists
        if 'type' in self.df.columns:
            self.df['type_clean'] = self.df['type'].astype(str).str.strip().str.upper()
    
    def analyze_duplicates(self) -> Dict[str, Any]:
        """
        Analyze duplicate transactions and patterns.
        
        Returns:
            Dictionary with duplicate analysis results
        """
        print("Analyzing duplicates...")
        duplicate_features = {}
        
        # Exact amount duplicates
        amount_duplicates = self.df[self.df.duplicated(subset=['amount'], keep=False)]
        duplicate_features['exact_amount_duplicates'] = len(amount_duplicates)
        duplicate_features['exact_amount_duplicate_rate'] = len(amount_duplicates) / len(self.df) * 100 if len(self.df) > 0 else 0
        
        # Amount-Payee duplicates
        if 'payee_clean' in self.df.columns:
            amount_payee_duplicates = self.df[self.df.duplicated(subset=['amount', 'payee_clean'], keep=False)]
            duplicate_features['amount_payee_duplicates'] = len(amount_payee_duplicates)
            duplicate_features['amount_payee_duplicate_rate'] = len(amount_payee_duplicates) / len(self.df) * 100 if len(self.df) > 0 else 0
        else:
            duplicate_features['amount_payee_duplicates'] = 0
            duplicate_features['amount_payee_duplicate_rate'] = 0
        
        # Full transaction duplicates (amount-payee-date)
        if 'payee_clean' in self.df.columns and 'date' in self.df.columns:
            exact_duplicates = self.df[self.df.duplicated(subset=['amount', 'payee_clean', 'date'], keep=False)]
            duplicate_features['exact_transaction_duplicates'] = len(exact_duplicates)
            duplicate_features['exact_transaction_duplicate_rate'] = len(exact_duplicates) / len(self.df) * 100 if len(self.df) > 0 else 0
            
            if len(exact_duplicates) > 0:
                duplicate_features['duplicate_transactions'] = exact_duplicates[['amount', 'payee', 'date']].head(10).to_dict('records')
        else:
            duplicate_features['exact_transaction_duplicates'] = 0
            duplicate_features['exact_transaction_duplicate_rate'] = 0
        
        # Round number analysis
        amounts = self.df['amount'].dropna()
        if len(amounts) > 0:
            round_numbers = amounts[amounts % 1 == 0]
            round_tens = amounts[amounts % 10 == 0]
            round_hundreds = amounts[amounts % 100 == 0]
            round_thousands = amounts[amounts % 1000 == 0]
            
            duplicate_features['round_number_count'] = len(round_numbers)
            duplicate_features['round_number_rate'] = len(round_numbers) / len(amounts) * 100
            duplicate_features['round_tens_rate'] = len(round_tens) / len(amounts) * 100
            duplicate_features['round_hundreds_rate'] = len(round_hundreds) / len(amounts) * 100
            duplicate_features['round_thousands_rate'] = len(round_thousands) / len(amounts) * 100
        else:
            duplicate_features.update({
                'round_number_count': 0,
                'round_number_rate': 0,
                'round_tens_rate': 0,
                'round_hundreds_rate': 0,
                'round_thousands_rate': 0
            })
        
        # Payee frequency analysis
        if 'payee_clean' in self.df.columns:
            payee_counts = self.df['payee_clean'].value_counts()
            duplicate_features['unique_payees'] = len(payee_counts)
            duplicate_features['top_payee_frequency'] = payee_counts.iloc[0] if len(payee_counts) > 0 else 0
            duplicate_features['payee_concentration'] = (payee_counts.iloc[:5].sum() / len(self.df) * 100) if len(payee_counts) > 0 and len(self.df) > 0 else 0
            
            high_freq_payees = payee_counts[payee_counts > len(self.df) * 0.05] if len(self.df) > 0 else pd.Series(dtype=int)
            duplicate_features['high_frequency_payees'] = len(high_freq_payees)
            if len(high_freq_payees) > 0:
                duplicate_features['high_frequency_payee_list'] = high_freq_payees.to_dict()
        else:
            duplicate_features.update({
                'unique_payees': 0,
                'top_payee_frequency': 0,
                'payee_concentration': 0,
                'high_frequency_payees': 0
            })
        
        return duplicate_features
    
    def analyze_outliers(self) -> Dict[str, Any]:
        """
        Analyze outlier transactions using Z-scores and IQR methods.
        
        Returns:
            Dictionary with outlier analysis results
        """
        print("Analyzing outliers...")
        outlier_features = {}
        amounts = self.df['amount'].dropna()
        
        if len(amounts) == 0:
            outlier_features['error'] = "No valid numeric amounts found"
            return outlier_features
        
        # Z-score analysis
        if len(amounts) > 1 and amounts.std() > 0:
            z_scores = np.abs(stats.zscore(amounts))
            self.df['z_score'] = np.nan
            self.df.loc[amounts.index, 'z_score'] = z_scores
            
            z_outliers_2 = self.df[self.df['z_score'] > 2]
            z_outliers_3 = self.df[self.df['z_score'] > 3]
            
            outlier_features['z_score_outliers_2sigma'] = len(z_outliers_2)
            outlier_features['z_score_outliers_3sigma'] = len(z_outliers_3)
            outlier_features['z_score_outlier_rate_2sigma'] = len(z_outliers_2) / len(amounts) * 100
            outlier_features['z_score_outlier_rate_3sigma'] = len(z_outliers_3) / len(amounts) * 100
        else:
            outlier_features.update({
                'z_score_outliers_2sigma': 0,
                'z_score_outliers_3sigma': 0,
                'z_score_outlier_rate_2sigma': 0,
                'z_score_outlier_rate_3sigma': 0
            })
        
        # IQR analysis
        Q1 = amounts.quantile(0.25)
        Q3 = amounts.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
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
            
            if len(extreme_outliers) > 0:
                outlier_features['extreme_outlier_transactions'] = extreme_outliers[['amount', 'payee', 'date']].head(10).to_dict('records')
        else:
            outlier_features.update({
                'iqr_outliers': 0,
                'iqr_outlier_rate': 0,
                'extreme_outliers': 0,
                'extreme_outlier_rate': 0
            })
        
        # Statistical measures
        outlier_features['amount_stats'] = {
            'mean': float(amounts.mean()) if not amounts.empty else 0,
            'median': float(amounts.median()) if not amounts.empty else 0,
            'std': float(amounts.std()) if not amounts.empty else 0,
            'skewness': float(stats.skew(amounts)) if len(amounts) > 2 else 0,
            'kurtosis': float(stats.kurtosis(amounts)) if len(amounts) > 2 else 0,
            'min': float(amounts.min()) if not amounts.empty else 0,
            'max': float(amounts.max()) if not amounts.empty else 0,
            'q1': float(Q1),
            'q3': float(Q3),
            'iqr': float(IQR)
        }
        
        return outlier_features
    
    def analyze_thresholds(self) -> Dict[str, Any]:
        """
        Analyze transactions near key financial reporting thresholds.
        
        Returns:
            Dictionary with threshold analysis results
        """
        print("Analyzing thresholds...")
        threshold_features = {}
        amounts = self.df['abs_amount'].dropna()
        
        # Validate amounts is a non-empty Series
        if not isinstance(amounts, pd.Series) or amounts.empty:
            threshold_features['error'] = "No valid numeric data in 'abs_amount' column after dropping NaNs"
            return threshold_features
        
        thresholds = {
            'cash_reporting_10k': 10000,
            'structuring_9999': 9999,
            'structuring_9500': 9500,
            'suspicious_5k': 5000,
            'check_clearing_3k': 3000,
            'daily_limit_1k': 1000
        }
        
        for threshold_name, threshold_value in thresholds.items():
            # Exact matches
            exact_hits = amounts[amounts == threshold_value]
            threshold_features[f'{threshold_name}_exact_hits'] = len(exact_hits)
            
            # Just under threshold (95% to <100%)
            just_under = amounts[(amounts >= threshold_value * 0.95) & (amounts < threshold_value)]
            threshold_features[f'{threshold_name}_just_under'] = len(just_under)
            
            # Just over threshold (>100% to 105%)
            just_over = amounts[(amounts > threshold_value) & (amounts <= threshold_value * 1.05)]
            threshold_features[f'{threshold_name}_just_over'] = len(just_over)
            
            # Cluster range (90% to 110%)
            cluster_range = amounts[(amounts >= threshold_value * 0.9) & (amounts <= threshold_value * 1.1)]
            threshold_features[f'{threshold_name}_cluster'] = len(cluster_range)
            threshold_features[f'{threshold_name}_cluster_rate'] = len(cluster_range) / len(amounts) * 100 if len(amounts) > 0 else 0
        
        # Structuring range analysis ($9,500-$9,999)
        structuring_range = amounts[(amounts >= 9500) & (amounts <= 9999)]
        threshold_features['structuring_indicators'] = len(structuring_range)
        threshold_features['structuring_rate'] = len(structuring_range) / len(amounts) * 100 if len(amounts) > 0 else 0
        if len(structuring_range) > 0:
            structuring_transactions = self.df[self.df['abs_amount'].between(9500, 9999)]
            threshold_features['structuring_transactions'] = structuring_transactions[['amount', 'payee', 'date']].head(10).to_dict('records')
        
        return threshold_features
    
    def analyze_ratios(self) -> Dict[str, Any]:
        """
        Analyze financial ratios and relationships.
        
        Returns:
            Dictionary with ratio analysis results
        """
        print("Analyzing ratios...")
        ratio_features = {}
        
        # Separate income and expenses
        if 'type_clean' in self.df.columns:
            income_keywords = ['INCOME', 'DEPOSIT', 'CREDIT', 'REVENUE', 'SALE']
            expense_keywords = ['EXPENSE', 'DEBIT', 'PAYMENT', 'WITHDRAWAL', 'PURCHASE']
            
            income_mask = self.df['type_clean'].str.contains('|'.join(income_keywords), na=False)
            expense_mask = self.df['type_clean'].str.contains('|'.join(expense_keywords), na=False)
            
            income = self.df[income_mask]
            expenses = self.df[expense_mask]
        else:
            income = self.df[self.df['amount'] > 0]
            expenses = self.df[self.df['amount'] < 0]
        
        # Financial summary
        total_income = float(income['amount'].sum()) if not income.empty else 0
        total_expenses = float(abs(expenses['amount'].sum())) if not expenses.empty else 0
        
        ratio_features['financial_summary'] = {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_income': total_income - total_expenses,
            'expense_to_income_ratio': total_expenses / total_income if total_income > 0 else 0
        }
        
        # Transaction counts
        ratio_features['transaction_counts'] = {
            'income_transactions': len(income),
            'expense_transactions': len(expenses),
            'expense_to_income_transaction_ratio': len(expenses) / len(income) if len(income) > 0 else 0
        }
        
        # Average transactions
        avg_income = float(income['amount'].mean()) if len(income) > 0 else 0
        avg_expense = float(abs(expenses['amount'].mean())) if len(expenses) > 0 else 0
        
        ratio_features['average_transactions'] = {
            'avg_income_transaction': avg_income,
            'avg_expense_transaction': avg_expense,
            'avg_expense_to_income_ratio': avg_expense / avg_income if avg_income > 0 else 0
        }
        
        # Concentration analysis
        if len(income) > 0:
            top_5_income = float(income.nlargest(5, 'amount')['amount'].sum())
            ratio_features['income_concentration'] = {
                'top_5_income_total': top_5_income,
                'top_5_income_percentage': top_5_income / total_income * 100 if total_income > 0 else 0
            }
        
        if len(expenses) > 0:
            top_5_expenses = float(expenses.nsmallest(5, 'amount')['amount'].sum())
            ratio_features['expense_concentration'] = {
                'top_5_expense_total': abs(top_5_expenses),
                'top_5_expense_percentage': abs(top_5_expenses) / total_expenses * 100 if total_expenses > 0 else 0
            }
        
        return ratio_features
    
    def analyze_time_series(self) -> Dict[str, Any]:
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
        
        valid_dates = self.df.dropna(subset=['date'])
        if len(valid_dates) == 0:
            time_features['error'] = "No valid dates found"
            return time_features
        
        # Monthly analysis
        monthly_data = valid_dates.groupby(['year', 'month']).agg({
            'amount': ['sum', 'mean', 'count', 'std'],
            'abs_amount': ['sum', 'mean']
        }).reset_index()
        monthly_data.columns = ['year', 'month', 'total_amount', 'avg_amount', 'transaction_count', 'amount_std', 'total_abs_amount', 'avg_abs_amount']
        
        time_features['monthly_analysis'] = {
            'total_months': len(monthly_data),
            'avg_monthly_transactions': float(monthly_data['transaction_count'].mean()) if not monthly_data.empty else 0,
            'monthly_transaction_volatility': float(monthly_data['transaction_count'].std()) if not monthly_data.empty else 0,
            'avg_monthly_amount': float(monthly_data['total_amount'].mean()) if not monthly_data.empty else 0,
            'monthly_amount_volatility': float(monthly_data['total_amount'].std()) if not monthly_data.empty else 0
        }
        
        # Quarterly analysis
        quarterly_data = valid_dates.groupby(['year', 'quarter']).agg({
            'amount': ['sum', 'count'],
            'abs_amount': 'sum'
        }).reset_index()
        quarterly_data.columns = ['year', 'quarter', 'total_amount', 'transaction_count', 'total_abs_amount']
        
        time_features['quarterly_analysis'] = {
            'total_quarters': len(quarterly_data),
            'avg_quarterly_transactions': float(quarterly_data['transaction_count'].mean()) if not quarterly_data.empty else 0,
            'quarterly_transaction_volatility': float(quarterly_data['transaction_count'].std()) if not quarterly_data.empty else 0,
            'avg_quarterly_amount': float(quarterly_data['total_amount'].mean()) if not quarterly_data.empty else 0,
            'quarterly_amount_volatility': float(quarterly_data['total_amount'].std()) if not quarterly_data.empty else 0
        }
        
        # Day of week analysis
        weekend_transactions = valid_dates[valid_dates['day_of_week'].isin([5, 6])]
        weekday_transactions = valid_dates[~valid_dates['day_of_week'].isin([5, 6])]
        
        total_amount = float(valid_dates['amount'].sum())
        
        time_features['weekend_analysis'] = {
            'weekend_transaction_count': len(weekend_transactions),
            'weekday_transaction_count': len(weekday_transactions),
            'weekend_transaction_rate': len(weekend_transactions) / len(valid_dates) * 100 if len(valid_dates) > 0 else 0,
            'weekend_amount_total': float(weekend_transactions['amount'].sum()) if not weekend_transactions.empty else 0,
            'weekday_amount_total': float(weekday_transactions['amount'].sum()) if not weekday_transactions.empty else 0,
            'weekend_amount_rate': weekend_transactions['amount'].sum() / total_amount * 100 if total_amount != 0 else 0
        }
        
        # Unusual activity detection
        daily_counts = valid_dates.groupby(valid_dates['date'].dt.date).size()
        daily_amounts = valid_dates.groupby(valid_dates['date'].dt.date)['amount'].sum()
        
        if len(daily_counts) > 0:
            daily_mean = float(daily_counts.mean())
            daily_std = float(daily_counts.std())
            burst_days = daily_counts[daily_counts > daily_mean + 2 * daily_std] if daily_std > 0 else pd.Series(dtype=int)
            
            time_features['unusual_activity'] = {
                'burst_activity_days': len(burst_days),
                'max_daily_transactions': int(daily_counts.max()),
                'avg_daily_transactions': daily_mean,
                'max_daily_amount': float(daily_amounts.max()),
                'avg_daily_amount': float(daily_amounts.mean())
            }
        else:
            time_features['unusual_activity'] = {
                'burst_activity_days': 0,
                'max_daily_transactions': 0,
                'avg_daily_transactions': 0,
                'max_daily_amount': 0,
                'avg_daily_amount': 0
            }
        
        return time_features
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis combining all feature types.
        
        Returns:
            Dictionary with all analysis results
        """
        print("Starting comprehensive ledger analysis...")
        
        analysis_results = {
            'dataset_summary': {
                'total_transactions': len(self.df),
                'unique_amounts': int(self.df['amount'].nunique()) if 'amount' in self.df.columns else 0,
                'amount_range': {
                    'min': float(self.df['amount'].min()) if 'amount' in self.df.columns and not self.df['amount'].empty else 0,
                    'max': float(self.df['amount'].max()) if 'amount' in self.df.columns and not self.df['amount'].empty else 0,
                    'mean': float(self.df['amount'].mean()) if 'amount' in self.df.columns and not self.df['amount'].empty else 0,
                    'median': float(self.df['amount'].median()) if 'amount' in self.df.columns and not self.df['amount'].empty else 0
                }
            }
        }
        
        if 'date' in self.df.columns:
            valid_dates = self.df['date'].dropna()
            if not valid_dates.empty:
                analysis_results['dataset_summary']['date_range'] = {
                    'start': valid_dates.min(),
                    'end': valid_dates.max(),
                    'days_covered': (valid_dates.max() - valid_dates.min()).days
                }
        
        try:
            analysis_results['duplicates'] = self.analyze_duplicates()
        except Exception as e:
            analysis_results['duplicates'] = {'error': str(e)}
            
        try:
            analysis_results['outliers'] = self.analyze_outliers()
        except Exception as e:
            analysis_results['outliers'] = {'error': str(e)}
            
        try:
            analysis_results['thresholds'] = self.analyze_thresholds()
        except Exception as e:
            analysis_results['thresholds'] = {'error': str(e)}
            
        try:
            analysis_results['ratios'] = self.analyze_ratios()
        except Exception as e:
            analysis_results['ratios'] = {'error': str(e)}
            
        try:
            analysis_results['time_series'] = self.analyze_time_series()
        except Exception as e:
            analysis_results['time_series'] = {'error': str(e)}
        
        analysis_results['risk_assessment'] = self.calculate_risk_scores(analysis_results)
        self.analysis_results = analysis_results
        
        print("Comprehensive analysis complete!")
        return analysis_results
    
    def calculate_risk_scores(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk scores based on all analysis results.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            Dictionary with risk assessment
        """
        risk_scores = {}
        
        # Duplicate risk
        duplicate_risk = 0
        if 'duplicates' in analysis_results and 'error' not in analysis_results['duplicates']:
            duplicate_risk += min(analysis_results['duplicates'].get('exact_transaction_duplicate_rate', 0), 30)
            duplicate_risk += min(analysis_results['duplicates'].get('amount_payee_duplicate_rate', 0), 20)
            duplicate_risk += min(analysis_results['duplicates'].get('round_thousands_rate', 0), 15)
            duplicate_risk += min(analysis_results['duplicates'].get('payee_concentration', 0), 10)
            duplicate_risk += min(analysis_results['duplicates'].get('high_frequency_payees', 0) * 5, 25)
        risk_scores['duplicate_risk'] = min(duplicate_risk, 100)
        
        # Outlier risk
        outlier_risk = 0
        if 'outliers' in analysis_results and 'error' not in analysis_results['outliers']:
            outlier_risk += min(analysis_results['outliers'].get('z_score_outlier_rate_3sigma', 0), 30)
            outlier_risk += min(analysis_results['outliers'].get('extreme_outlier_rate', 0), 25)
            skewness = abs(analysis_results['outliers']['amount_stats'].get('skewness', 0))
            outlier_risk += min(skewness * 5, 20)
            kurtosis = abs(analysis_results['outliers']['amount_stats'].get('kurtosis', 0))
            outlier_risk += min(kurtosis * 2, 15)
        risk_scores['outlier_risk'] = min(outlier_risk, 100)
        
        # Threshold risk
        threshold_risk = 0
        if 'thresholds' in analysis_results and 'error' not in analysis_results['thresholds']:
            threshold_risk += min(analysis_results['thresholds'].get('structuring_rate', 0), 50)
            threshold_risk += analysis_results['thresholds'].get('structuring_9999_exact_hits', 0) * 10
            threshold_risk += analysis_results['thresholds'].get('cash_reporting_10k_exact_hits', 0) * 5
            threshold_risk += min(analysis_results['thresholds'].get('cash_reporting_10k_cluster_rate', 0), 20)
        risk_scores['threshold_risk'] = min(threshold_risk, 100)
        
        # Time series risk
        time_risk = 0
        if 'time_series' in analysis_results and 'error' not in analysis_results['time_series']:
            weekend_rate = analysis_results['time_series']['weekend_analysis'].get('weekend_transaction_rate', 0)
            time_risk += min(weekend_rate, 30)
            burst_days = analysis_results['time_series']['unusual_activity'].get('burst_activity_days', 0)
            time_risk += min(burst_days * 5, 25)
        risk_scores['time_series_risk'] = min(time_risk, 100)
        
        # Overall risk
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
        
