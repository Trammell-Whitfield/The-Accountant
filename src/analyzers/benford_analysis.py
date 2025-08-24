"""
Enhanced Benford's Law Analysis for Forensic Accounting
Integrates with RAG system and provides comprehensive statistical analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class BenfordsLawAnalyzer:
    """
    Enhanced Benford's Law analyzer with advanced statistical methods
    and RAG system integration
    """
    
    def __init__(self):
        # Benford's Law probabilities for digits 1-9
        self.benford_probabilities = {
            1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
            6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
        }
        
        # Store analysis results
        self.analysis_results = {}
        self.visualization_paths = []
        
    def extract_leading_digits(self, data: pd.Series, position: int = 1) -> List[int]:
        """
        Extract leading digits from numerical data
        
        Args:
            data: Pandas Series containing numerical values
            position: Position of digit to extract (1=first, 2=second, etc.)
            
        Returns:
            List of extracted digits
        """
        digits = []
        
        for value in data:
            if pd.isna(value) or value == 0:
                continue
                
            # Convert to string and remove decimal point and sign
            str_value = str(abs(float(value))).replace('.', '')
            
            # Remove leading zeros and extract specified position
            str_value = str_value.lstrip('0')
            if len(str_value) >= position:
                digit = int(str_value[position - 1])
                if digit > 0:  # Only include non-zero digits
                    digits.append(digit)
                    
        return digits
    
    def calculate_frequencies(self, digits: List[int]) -> Dict[int, float]:
        """Calculate observed frequencies of digits"""
        if not digits:
            return {}
            
        total_count = len(digits)
        frequencies = {}
        
        digit_range = range(1, 10) if max(digits) <= 9 else range(0, 10)
        
        for digit in digit_range:
            count = digits.count(digit)
            frequencies[digit] = count / total_count if total_count > 0 else 0
            
        return frequencies
    
    def chi_square_test(self, observed_frequencies: Dict[int, float], 
                       sample_size: int) -> Dict[str, Any]:
        """
        Perform chi-square goodness-of-fit test
        
        Args:
            observed_frequencies: Observed digit frequencies
            sample_size: Total number of observations
            
        Returns:
            Dictionary with test results
        """
        observed_counts = []
        expected_counts = []
        
        for digit in range(1, 10):
            obs_count = observed_frequencies.get(digit, 0) * sample_size
            exp_count = self.benford_probabilities[digit] * sample_size
            
            observed_counts.append(obs_count)
            expected_counts.append(exp_count)
        
        # Perform chi-square test
        chi_square_stat, p_value = stats.chisquare(observed_counts, expected_counts)
        
        # Calculate critical values
        critical_05 = stats.chi2.ppf(0.95, df=8)  # 95% confidence
        critical_01 = stats.chi2.ppf(0.99, df=8)  # 99% confidence
        
        return {
            'chi_square_statistic': chi_square_stat,
            'p_value': p_value,
            'critical_value_05': critical_05,
            'critical_value_01': critical_01,
            'significant_05': chi_square_stat > critical_05,
            'significant_01': chi_square_stat > critical_01,
            'degrees_freedom': 8
        }
    
    def z_test_by_digit(self, observed_frequencies: Dict[int, float], 
                       sample_size: int, confidence_level: float = 0.95) -> Dict[int, Dict[str, Any]]:
        """
        Perform Z-test for each digit individually
        
        Args:
            observed_frequencies: Observed digit frequencies
            sample_size: Total number of observations
            confidence_level: Confidence level for significance testing
            
        Returns:
            Dictionary with Z-test results for each digit
        """
        z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        z_results = {}
        
        for digit in range(1, 10):
            observed_prop = observed_frequencies.get(digit, 0)
            expected_prop = self.benford_probabilities[digit]
            
            # Calculate standard error
            standard_error = np.sqrt((expected_prop * (1 - expected_prop)) / sample_size)
            
            if standard_error > 0:
                z_score = (observed_prop - expected_prop) / standard_error
                is_significant = abs(z_score) > z_critical
                
                z_results[digit] = {
                    'z_score': z_score,
                    'standard_error': standard_error,
                    'is_significant': is_significant,
                    'p_value': 2 * (1 - stats.norm.cdf(abs(z_score))),
                    'confidence_interval': (
                        observed_prop - z_critical * standard_error,
                        observed_prop + z_critical * standard_error
                    )
                }
            else:
                z_results[digit] = {
                    'z_score': 0,
                    'standard_error': 0,
                    'is_significant': False,
                    'p_value': 1.0,
                    'confidence_interval': (observed_prop, observed_prop)
                }
                
        return z_results
    
    def kolmogorov_smirnov_test(self, observed_frequencies: Dict[int, float]) -> Dict[str, Any]:
        """
        Perform Kolmogorov-Smirnov test for distribution comparison
        
        Args:
            observed_frequencies: Observed digit frequencies
            
        Returns:
            Dictionary with KS test results
        """
        # Prepare cumulative distributions
        observed_cumulative = []
        expected_cumulative = []
        cumulative_obs = 0
        cumulative_exp = 0
        
        for digit in range(1, 10):
            cumulative_obs += observed_frequencies.get(digit, 0)
            cumulative_exp += self.benford_probabilities[digit]
            observed_cumulative.append(cumulative_obs)
            expected_cumulative.append(cumulative_exp)
        
        # Calculate KS statistic (maximum difference)
        differences = [abs(obs - exp) for obs, exp in zip(observed_cumulative, expected_cumulative)]
        ks_statistic = max(differences)
        
        # Approximate p-value for KS test
        n = sum(observed_frequencies.get(digit, 0) for digit in range(1, 10))
        if n > 0:
            ks_critical_05 = 1.36 / np.sqrt(n)  # Critical value at 95% confidence
            ks_critical_01 = 1.63 / np.sqrt(n)  # Critical value at 99% confidence
        else:
            ks_critical_05 = ks_critical_01 = float('inf')
        
        return {
            'ks_statistic': ks_statistic,
            'critical_value_05': ks_critical_05,
            'critical_value_01': ks_critical_01,
            'significant_05': ks_statistic > ks_critical_05,
            'significant_01': ks_statistic > ks_critical_01,
            'observed_cumulative': observed_cumulative,
            'expected_cumulative': expected_cumulative
        }
    
    def detect_fraud_patterns(self, observed_frequencies: Dict[int, float], 
                            z_scores: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Advanced fraud pattern detection using multiple indicators
        
        Args:
            observed_frequencies: Observed digit frequencies
            z_scores: Z-test results for each digit
            
        Returns:
            Dictionary with fraud pattern analysis
        """
        fraud_indicators = {
            'round_number_bias': False,
            'digit_avoidance': [],
            'digit_overuse': [],
            'human_fabrication_patterns': [],
            'psychological_patterns': [],
            'statistical_anomalies': [],
            'fraud_score': 0,
            'risk_factors': []
        }
        
        # Round number bias (preference for 5 and 9)
        digit_5_excess = observed_frequencies.get(5, 0) - self.benford_probabilities[5]
        digit_9_excess = observed_frequencies.get(9, 0) - self.benford_probabilities[9]
        
        if digit_5_excess > 0.02 or digit_9_excess > 0.015:
            fraud_indicators['round_number_bias'] = True
            fraud_indicators['fraud_score'] += 25
            fraud_indicators['risk_factors'].append("Excessive use of digits 5 or 9 suggesting round number preference")
        
        # Digit avoidance (underuse of 1 and 2)
        for digit in [1, 2]:
            expected = self.benford_probabilities[digit]
            observed = observed_frequencies.get(digit, 0)
            
            if observed < expected * 0.7:  # More than 30% below expected
                fraud_indicators['digit_avoidance'].append(digit)
                fraud_indicators['fraud_score'] += 15 if digit == 1 else 10
                fraud_indicators['risk_factors'].append(f"Significant underuse of digit {digit}")
        
        # Digit overuse (statistical significance)
        for digit, z_data in z_scores.items():
            if z_data['is_significant'] and z_data['z_score'] > 2:
                fraud_indicators['digit_overuse'].append(digit)
                fraud_indicators['fraud_score'] += 10
                fraud_indicators['statistical_anomalies'].append(
                    f"Digit {digit} overused (Z-score: {z_data['z_score']:.2f})"
                )
        
        # Human fabrication patterns
        # Pattern 1: Overuse of middle digits (4, 5, 6)
        middle_digit_rate = sum(observed_frequencies.get(d, 0) for d in [4, 5, 6])
        expected_middle_rate = sum(self.benford_probabilities[d] for d in [4, 5, 6])
        
        if middle_digit_rate > expected_middle_rate * 1.3:
            fraud_indicators['human_fabrication_patterns'].append("Overuse of middle digits (4-6)")
            fraud_indicators['fraud_score'] += 12
        
        # Pattern 2: Avoidance of extreme digits (1, 9)
        extreme_digit_rate = observed_frequencies.get(1, 0) + observed_frequencies.get(9, 0)
        expected_extreme_rate = self.benford_probabilities[1] + self.benford_probabilities[9]
        
        if extreme_digit_rate < expected_extreme_rate * 0.7:
            fraud_indicators['human_fabrication_patterns'].append("Avoidance of extreme digits (1, 9)")
            fraud_indicators['fraud_score'] += 8
        
        # Psychological patterns
        # Pattern 1: Preference for "lucky" digit 7
        if observed_frequencies.get(7, 0) > self.benford_probabilities[7] * 1.4:
            fraud_indicators['psychological_patterns'].append("Overuse of 'lucky' digit 7")
            fraud_indicators['fraud_score'] += 8
        
        # Pattern 2: Avoidance of "unlucky" digits
        unlucky_digits = [4, 13 % 10]  # 4 is unlucky in some cultures
        for digit in unlucky_digits:
            if digit in observed_frequencies:
                if observed_frequencies[digit] < self.benford_probabilities[digit] * 0.6:
                    fraud_indicators['psychological_patterns'].append(f"Possible avoidance of digit {digit}")
                    fraud_indicators['fraud_score'] += 5
        
        # Additional statistical anomalies
        # Flat distribution indicator
        frequency_variance = np.var(list(observed_frequencies.values()))
        benford_variance = np.var(list(self.benford_probabilities.values()))
        
        if frequency_variance < benford_variance * 0.5:
            fraud_indicators['statistical_anomalies'].append("Unusually flat distribution")
            fraud_indicators['fraud_score'] += 15
        
        # Determine overall fraud assessment
        fraud_indicators['fraud_score'] = min(fraud_indicators['fraud_score'], 100)
        
        return fraud_indicators
    
    def comprehensive_analysis(self, data: pd.Series, dataset_name: str = "Dataset", 
                             min_sample_size: int = 50) -> Dict[str, Any]:
        """
        Perform comprehensive Benford's Law analysis
        
        Args:
            data: Pandas Series containing numerical values
            dataset_name: Name for the dataset
            min_sample_size: Minimum sample size required for analysis
            
        Returns:
            Complete analysis results
        """
        # Extract leading digits
        leading_digits = self.extract_leading_digits(data)
        
        if len(leading_digits) < min_sample_size:
            return {
                'error': f"Insufficient data: {len(leading_digits)} observations (minimum {min_sample_size} required)",
                'dataset_name': dataset_name,
                'sample_size': len(leading_digits)
            }
        
        # Calculate observed frequencies
        observed_frequencies = self.calculate_frequencies(leading_digits)
        sample_size = len(leading_digits)
        
        # Perform statistical tests
        chi_square_results = self.chi_square_test(observed_frequencies, sample_size)
        z_test_results = self.z_test_by_digit(observed_frequencies, sample_size)
        ks_results = self.kolmogorov_smirnov_test(observed_frequencies)
        
        # Detect fraud patterns
        fraud_analysis = self.detect_fraud_patterns(observed_frequencies, z_test_results)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            observed_frequencies, chi_square_results, fraud_analysis
        )
        
        # Compile comprehensive results
        results = {
            'dataset_info': {
                'name': dataset_name,
                'sample_size': sample_size,
                'analysis_timestamp': datetime.now().isoformat()
            },
            'frequency_analysis': {
                'observed_frequencies': observed_frequencies,
                'expected_frequencies': self.benford_probabilities,
                'deviations': {digit: observed_frequencies.get(digit, 0) - self.benford_probabilities[digit] 
                              for digit in range(1, 10)}
            },
            'statistical_tests': {
                'chi_square': chi_square_results,
                'z_tests': z_test_results,
                'kolmogorov_smirnov': ks_results
            },
            'fraud_analysis': fraud_analysis,
            'quality_metrics': quality_metrics,
            'compliance_assessment': self._assess_compliance(chi_square_results, fraud_analysis),
            'recommendations': self._generate_recommendations(fraud_analysis, quality_metrics)
        }
        
        # Store results
        self.analysis_results[dataset_name] = results
        
        return results
    
    def _calculate_quality_metrics(self, observed_frequencies: Dict[int, float], 
                                  chi_square_results: Dict[str, Any],
                                  fraud_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data quality and reliability metrics"""
        
        # Mean Absolute Deviation (MAD)
        mad = np.mean([abs(observed_frequencies.get(digit, 0) - self.benford_probabilities[digit]) 
                      for digit in range(1, 10)])
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean([(observed_frequencies.get(digit, 0) - self.benford_probabilities[digit])**2 
                               for digit in range(1, 10)]))
        
        # Euclidean Distance
        euclidean_distance = np.sqrt(sum(
            (observed_frequencies.get(digit, 0) - self.benford_probabilities[digit])**2 
            for digit in range(1, 10)
        ))
        
        # Compliance score (inverse of fraud score)
        compliance_score = max(0, 100 - fraud_analysis['fraud_score'])
        
        return {
            'mean_absolute_deviation': mad,
            'root_mean_square_error': rmse,
            'euclidean_distance': euclidean_distance,
            'compliance_score': compliance_score,
            'chi_square_p_value': chi_square_results['p_value'],
            'data_quality_grade': self._assign_quality_grade(mad, chi_square_results['p_value'])
        }
    
    def _assign_quality_grade(self, mad: float, p_value: float) -> str:
        """Assign quality grade based on statistical measures"""
        if p_value > 0.10 and mad < 0.01:
            return 'A'  # Excellent compliance
        elif p_value > 0.05 and mad < 0.02:
            return 'B'  # Good compliance
        elif p_value > 0.01 and mad < 0.05:
            return 'C'  # Acceptable compliance
        elif p_value > 0.001 and mad < 0.10:
            return 'D'  # Poor compliance
        else:
            return 'F'  # Failed compliance
    
    def _assess_compliance(self, chi_square_results: Dict[str, Any], 
                          fraud_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall compliance with Benford's Law"""
        
        risk_level = 'LOW'
        if fraud_analysis['fraud_score'] >= 50:
            risk_level = 'HIGH'
        elif fraud_analysis['fraud_score'] >= 25:
            risk_level = 'MEDIUM'
        
        return {
            'follows_benfords_law': not chi_square_results['significant_05'],
            'statistical_significance': 'SIGNIFICANT' if chi_square_results['significant_05'] else 'NOT SIGNIFICANT',
            'fraud_risk_level': risk_level,
            'overall_assessment': self._get_overall_assessment(chi_square_results, fraud_analysis),
            'confidence_level': 'HIGH' if chi_square_results['p_value'] > 0.10 else 
                               'MEDIUM' if chi_square_results['p_value'] > 0.05 else 'LOW'
        }
    
    def _get_overall_assessment(self, chi_square_results: Dict[str, Any], 
                               fraud_analysis: Dict[str, Any]) -> str:
        """Generate overall assessment text"""
        if not chi_square_results['significant_05'] and fraud_analysis['fraud_score'] < 10:
            return "Data strongly conforms to Benford's Law with minimal fraud indicators"
        elif not chi_square_results['significant_05'] and fraud_analysis['fraud_score'] < 25:
            return "Data generally conforms to Benford's Law with minor anomalies"
        elif chi_square_results['significant_05'] and fraud_analysis['fraud_score'] < 50:
            return "Data shows significant deviations from Benford's Law requiring investigation"
        else:
            return "Data shows major deviations from Benford's Law with high fraud risk"
    
    def _generate_recommendations(self, fraud_analysis: Dict[str, Any], 
                                quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        fraud_score = fraud_analysis['fraud_score']
        grade = quality_metrics['data_quality_grade']
        
        # High priority recommendations
        if fraud_score >= 75:
            recommendations.append("URGENT: Immediate forensic investigation required")
            recommendations.append("Suspend related financial processes pending review")
            recommendations.append("Engage external forensic accounting expert")
        elif fraud_score >= 50:
            recommendations.append("Conduct thorough manual review of transactions")
            recommendations.append("Implement enhanced monitoring controls")
            recommendations.append("Consider expanding sample for analysis")
        elif fraud_score >= 25:
            recommendations.append("Perform targeted sampling of suspicious patterns")
            recommendations.append("Review internal controls and authorization processes")
            recommendations.append("Monitor for recurring anomalies")
        
        # Data quality recommendations
        if grade in ['D', 'F']:
            recommendations.append("Data quality issues detected - verify data integrity")
            recommendations.append("Review data collection and processing procedures")
        
        # Pattern-specific recommendations
        if fraud_analysis['round_number_bias']:
            recommendations.append("Investigate excessive round number usage")
        
        if fraud_analysis['digit_avoidance']:
            recommendations.append(f"Review underuse of digits {fraud_analysis['digit_avoidance']}")
        
        if fraud_analysis['human_fabrication_patterns']:
            recommendations.append("Investigate potential human data fabrication")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Data appears compliant - maintain regular monitoring")
            recommendations.append("Consider periodic re-analysis with fresh data")
        
        return recommendations
    
    def create_comprehensive_visualization(self, analysis_results: Dict[str, Any], 
                                         save_path: Optional[str] = None) -> str:
        """
        Create comprehensive visualization of analysis results
        
        Args:
            analysis_results: Results from comprehensive_analysis()
            save_path: Optional path to save the visualization
            
        Returns:
            Path to saved visualization file
        """
        if 'error' in analysis_results:
            print(f"Cannot create visualization: {analysis_results['error']}")
            return ""
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 1.5, 1.5], hspace=0.3, wspace=0.3)
        
        # Extract data
        observed = [analysis_results['frequency_analysis']['observed_frequencies'].get(d, 0) for d in range(1, 10)]
        expected = [analysis_results['frequency_analysis']['expected_frequencies'][d] for d in range(1, 10)]
        deviations = [analysis_results['frequency_analysis']['deviations'][d] for d in range(1, 10)]
        digits = list(range(1, 10))
        
        # 1. Main comparison plot
        ax1 = fig.add_subplot(gs[0, :2])
        x = np.arange(len(digits))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, observed, width, label='Observed', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, expected, width, label='Expected (Benford)', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Leading Digit', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Benford\'s Law Analysis: {analysis_results["dataset_info"]["name"]}', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(digits)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # 2. Deviation analysis
        ax2 = fig.add_subplot(gs[0, 2])
        colors = ['red' if abs(dev) > 0.02 else 'orange' if abs(dev) > 0.01 else 'green' for dev in deviations]
        bars = ax2.bar(digits, deviations, color=colors, alpha=0.7)
        ax2.set_xlabel('Leading Digit')
        ax2.set_ylabel('Deviation from Expected')
        ax2.set_title('Deviations from Benford\'s Law')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. Z-scores plot
        ax3 = fig.add_subplot(gs[1, 0])
        z_scores = [analysis_results['statistical_tests']['z_tests'][d]['z_score'] for d in range(1, 10)]
        z_colors = ['red' if abs(z) > 1.96 else 'orange' if abs(z) > 1.0 else 'green' for z in z_scores]
        ax3.bar(digits, z_scores, color=z_colors, alpha=0.7)
        ax3.set_xlabel('Leading Digit')
        ax3.set_ylabel('Z-Score')
        ax3.set_title('Z-Scores by Digit')
        ax3.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='95% Confidence')
        ax3.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative distribution (KS test)
        ax4 = fig.add_subplot(gs[1, 1])
        ks_results = analysis_results['statistical_tests']['kolmogorov_smirnov']
        ax4.plot(digits, ks_results['observed_cumulative'], 'b-o', label='Observed Cumulative', linewidth=2)
        ax4.plot(digits, ks_results['expected_cumulative'], 'r--o', label='Expected Cumulative', linewidth=2)
        ax4.set_xlabel('Leading Digit')
        ax4.set_ylabel('Cumulative Frequency')
        ax4.set_title('Cumulative Distribution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Quality metrics radar chart (simplified)
        ax5 = fig.add_subplot(gs[1, 2])
        metrics = analysis_results['quality_metrics']
        
        # Create a simple quality summary
        quality_labels = ['Compliance\nScore', 'Chi-Square\np-value', 'Data Quality\nGrade']
        quality_values = [
            metrics['compliance_score'] / 100,
            min(metrics['chi_square_p_value'], 1.0),
            {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4, 'F': 0.2}[metrics['data_quality_grade']]
        ]
        
        ax5.bar(quality_labels, quality_values, color=['green', 'blue', 'purple'], alpha=0.7)
        ax5.set_ylim(0, 1)
        ax5.set_ylabel('Score (0-1)')
        ax5.set_title('Quality Metrics Summary')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Statistical summary text
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        chi_sq = analysis_results['statistical_tests']['chi_square']
        fraud = analysis_results['fraud_analysis']
        
        summary_text = f"""
STATISTICAL SUMMARY:
• Sample Size: {analysis_results['dataset_info']['sample_size']:,} transactions
• Chi-Square: {chi_sq['chi_square_statistic']:.4f} (p-value: {chi_sq['p_value']:.4f})
• Fraud Score: {fraud['fraud_score']}/100
• Risk Level: {analysis_results['compliance_assessment']['fraud_risk_level']}
• Overall Assessment: {analysis_results['compliance_assessment']['overall_assessment']}

DETECTED PATTERNS:
• Round Number Bias: {'YES' if fraud['round_number_bias'] else 'NO'}
• Digit Avoidance: {fraud['digit_avoidance'] if fraud['digit_avoidance'] else 'None'}
• Digit Overuse: {fraud['digit_overuse'] if fraud['digit_overuse'] else 'None'}
• Fabrication Patterns: {len(fraud['human_fabrication_patterns'])} detected
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 7. Recommendations
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        recommendations = analysis_results['recommendations'][:6]  # Show top 6
        rec_text = "KEY RECOMMENDATIONS:\n" + "\n".join([f"• {rec}" for rec in recommendations])
        
        ax7.text(0.05, 0.95, rec_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', wrap=True,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Save the visualization
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"benford_analysis_{analysis_results['dataset_info']['name']}_{timestamp}.png"
        
        plt.suptitle(f'Comprehensive Benford\'s Law Analysis Report', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        self.visualization_paths.append(save_path)
        return save_path
    
    def generate_detailed_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive text report"""
        if 'error' in analysis_results:
            return f"Error in analysis: {analysis_results['error']}"
        
        dataset_info = analysis_results['dataset_info']
        freq_analysis = analysis_results['frequency_analysis']
        stats_tests = analysis_results['statistical_tests']
        fraud_analysis = analysis_results['fraud_analysis']
        quality_metrics = analysis_results['quality_metrics']
        compliance = analysis_results['compliance_assessment']
        
        report = f"""
{'='*80}
COMPREHENSIVE BENFORD'S LAW ANALYSIS REPORT
{'='*80}

DATASET INFORMATION:
  Dataset Name: {dataset_info['name']}
  Sample Size: {dataset_info['sample_size']:,} transactions
  Analysis Date: {dataset_info['analysis_timestamp']}

EXECUTIVE SUMMARY:
  Overall Assessment: {compliance['overall_assessment']}
  Fraud Risk Level: {compliance['fraud_risk_level']}
  Data Quality Grade: {quality_metrics['data_quality_grade']}
  Compliance Score: {quality_metrics['compliance_score']:.1f}/100

{'='*80}
FREQUENCY ANALYSIS
{'='*80}

Digit | Observed | Expected | Deviation | Z-Score | Significant
------|----------|----------|-----------|---------|------------"""
        
        for digit in range(1, 10):
            obs = freq_analysis['observed_frequencies'].get(digit, 0)
            exp = freq_analysis['expected_frequencies'][digit]
            dev = freq_analysis['deviations'][digit]
            z_score = stats_tests['z_tests'][digit]['z_score']
            significant = 'YES' if stats_tests['z_tests'][digit]['is_significant'] else 'NO'
            
            report += f"\n  {digit}   |  {obs:6.3f}  |  {exp:6.3f}  |   {dev:+6.3f}   |  {z_score:+6.2f}  | {significant:>3}"
        
        report += f"""

{'='*80}
STATISTICAL TESTS
{'='*80}

CHI-SQUARE GOODNESS-OF-FIT TEST:
  Chi-Square Statistic: {stats_tests['chi_square']['chi_square_statistic']:.6f}
  Degrees of Freedom: {stats_tests['chi_square']['degrees_freedom']}
  P-Value: {stats_tests['chi_square']['p_value']:.6f}
  Critical Value (95%): {stats_tests['chi_square']['critical_value_05']:.6f}
  Critical Value (99%): {stats_tests['chi_square']['critical_value_01']:.6f}
  Significant (95%): {'YES' if stats_tests['chi_square']['significant_05'] else 'NO'}
  Significant (99%): {'YES' if stats_tests['chi_square']['significant_01'] else 'NO'}

KOLMOGOROV-SMIRNOV TEST:
  KS Statistic: {stats_tests['kolmogorov_smirnov']['ks_statistic']:.6f}
  Critical Value (95%): {stats_tests['kolmogorov_smirnov']['critical_value_05']:.6f}
  Critical Value (99%): {stats_tests['kolmogorov_smirnov']['critical_value_01']:.6f}
  Significant (95%): {'YES' if stats_tests['kolmogorov_smirnov']['significant_05'] else 'NO'}
  Significant (99%): {'YES' if stats_tests['kolmogorov_smirnov']['significant_01'] else 'NO'}

{'='*80}
FRAUD PATTERN ANALYSIS
{'='*80}

OVERALL FRAUD SCORE: {fraud_analysis['fraud_score']}/100

DETECTED PATTERNS:
  Round Number Bias: {'DETECTED' if fraud_analysis['round_number_bias'] else 'Not Detected'}
  Digit Avoidance: {fraud_analysis['digit_avoidance'] if fraud_analysis['digit_avoidance'] else 'None'}
  Digit Overuse: {fraud_analysis['digit_overuse'] if fraud_analysis['digit_overuse'] else 'None'}
  Human Fabrication Patterns: {fraud_analysis['human_fabrication_patterns'] if fraud_analysis['human_fabrication_patterns'] else 'None'}
  Psychological Patterns: {fraud_analysis['psychological_patterns'] if fraud_analysis['psychological_patterns'] else 'None'}

RISK FACTORS:"""
        
        for factor in fraud_analysis['risk_factors']:
            report += f"\n  • {factor}"
        
        report += f"""

STATISTICAL ANOMALIES:"""
        for anomaly in fraud_analysis['statistical_anomalies']:
            report += f"\n  • {anomaly}"
        
        report += f"""

{'='*80}
QUALITY METRICS
{'='*80}

  Mean Absolute Deviation: {quality_metrics['mean_absolute_deviation']:.6f}
  Root Mean Square Error: {quality_metrics['root_mean_square_error']:.6f}
  Euclidean Distance: {quality_metrics['euclidean_distance']:.6f}
  Data Quality Grade: {quality_metrics['data_quality_grade']}

{'='*80}
RECOMMENDATIONS
{'='*80}"""
        
        for i, recommendation in enumerate(analysis_results['recommendations'], 1):
            report += f"\n{i:2d}. {recommendation}"
        
        report += f"""

{'='*80}
TECHNICAL DETAILS
{'='*80}

COMPLIANCE ASSESSMENT:
  Follows Benford's Law: {'YES' if compliance['follows_benfords_law'] else 'NO'}
  Statistical Significance: {compliance['statistical_significance']}
  Confidence Level: {compliance['confidence_level']}

ANALYSIS METHODOLOGY:
  • First digit extraction from absolute values
  • Chi-square goodness-of-fit test (df=8)
  • Individual Z-tests for each digit (95% confidence)
  • Kolmogorov-Smirnov distribution comparison
  • Multi-factor fraud pattern detection
  • Quality metrics calculation (MAD, RMSE, Euclidean distance)

{'='*80}
END OF REPORT
{'='*80}
"""
        
        return report

# Example usage and testing
def create_test_data():
    """Create test data with known Benford's Law properties"""
    np.random.seed(42)
    
    # Generate data that follows Benford's Law
    legitimate_data = []
    for _ in range(1500):
        magnitude = np.random.uniform(1, 6)
        amount = 10 ** magnitude * np.random.exponential(0.5)
        legitimate_data.append(amount)
    
    # Add some fraudulent data
    fraudulent_data = []
    for _ in range(300):
        if np.random.random() < 0.4:
            # Round numbers
            amount = np.random.choice([500, 1000, 2500, 5000, 9999])
        else:
            # Fabricated amounts avoiding 1 and 2
            first_digit = np.random.choice([3, 4, 5, 6, 7, 8, 9], p=[0.2, 0.15, 0.25, 0.15, 0.1, 0.1, 0.05])
            amount = first_digit * (10 ** np.random.randint(2, 5))
        
        fraudulent_data.append(amount)
    
    return pd.Series(legitimate_data + fraudulent_data)

if __name__ == "__main__":
    # Create and test the enhanced analyzer
    analyzer = BenfordsLawAnalyzer()
    
    print("Creating test data with fraud patterns...")
    test_data = create_test_data()
    
    print("Performing comprehensive Benford's Law analysis...")
    results = analyzer.comprehensive_analysis(test_data, "Test Financial Dataset")
    
    if 'error' not in results:
        print("Generating detailed report...")
        report = analyzer.generate_detailed_report(results)
        print(report[:2000], "...\n[Report truncated for display]")
        
        print("Creating comprehensive visualization...")
        viz_path = analyzer.create_comprehensive_visualization(results)
        print(f"Visualization saved to: {viz_path}")
        
        # Save detailed report
        with open(f"benford_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
            f.write(report)
        
        print("Analysis complete! Check generated files for detailed results.")
    else:
        print(f"Analysis failed: {results['error']}")