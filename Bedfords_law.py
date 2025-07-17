import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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