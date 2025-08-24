import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BenfordsLawAnalyzer:
    def __init__(self):
        self.benford_probabilities = {
            1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
            6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
        }
        self.analysis_results = {}
    
    def extract_leading_digits(self, data: pd.Series) -> List[int]:
        leading_digits = []
        for value in data:
            if pd.isna(value) or value == 0:
                continue
            str_value = str(abs(float(value))).replace('.', '')
            for char in str_value:
                if char.isdigit() and char != '0':
                    leading_digits.append(int(char))
                    break
        return leading_digits
    
    def calculate_observed_frequencies(self, leading_digits: List[int]) -> Dict[int, float]:
        if not leading_digits:
            return {}
        total_count = len(leading_digits)
        observed_frequencies = {}
        for digit in range(1, 10):
            count = leading_digits.count(digit)
            observed_frequencies[digit] = count / total_count
        return observed_frequencies
    
    def chi_square_test(self, observed_frequencies: Dict[int, float], sample_size: int) -> Tuple[float, float, bool]:
        observed_counts = [observed_frequencies.get(digit, 0) * sample_size for digit in range(1, 10)]
        expected_counts = [self.benford_probabilities[digit] * sample_size for digit in range(1, 10)]
        chi_square_stat, p_value = stats.chisquare(observed_counts, expected_counts)
        critical_value = stats.chi2.ppf(0.95, df=8)
        is_significant = chi_square_stat > critical_value
        return chi_square_stat, p_value, is_significant
    
    def z_test_by_digit(self, observed_frequencies: Dict[int, float], sample_size: int) -> Dict[int, Tuple[float, bool]]:
        z_scores = {}
        for digit in range(1, 10):
            observed_prop = observed_frequencies.get(digit, 0)
            expected_prop = self.benford_probabilities[digit]
            standard_error = np.sqrt((expected_prop * (1 - expected_prop)) / sample_size)
            z_score = (observed_prop - expected_prop) / standard_error if standard_error > 0 else 0
            is_significant = abs(z_score) > 1.96
            z_scores[digit] = (z_score, is_significant)
        return z_scores
    
    def detect_fraud_patterns(self, observed_frequencies: Dict[int, float], z_scores: Dict[int, Tuple[float, bool]]) -> Dict[str, any]:
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
        
        # Check for human fabrication patterns
        if observed_frequencies.get(7, 0) > self.benford_probabilities[7] * 1.3:
            fraud_indicators['human_fabrication_signs'].append("Overuse of digit 7")
            fraud_indicators['overall_fraud_score'] += 8
        
        if observed_frequencies.get(3, 0) > self.benford_probabilities[3] * 1.3:
            fraud_indicators['human_fabrication_signs'].append("Overuse of digit 3")
            fraud_indicators['overall_fraud_score'] += 5
        
        return fraud_indicators
    
    def analyze_dataset(self, data: pd.Series, dataset_name: str = "Dataset") -> Dict[str, any]:
        leading_digits = self.extract_leading_digits(data)
        
        if len(leading_digits) < 50:
            return {'error': f"Insufficient data: {len(leading_digits)} observations (minimum 50 required)", 'dataset_name': dataset_name}
        
        observed_frequencies = self.calculate_observed_frequencies(leading_digits)
        sample_size = len(leading_digits)
        
        chi_square_stat, p_value, chi_square_significant = self.chi_square_test(observed_frequencies, sample_size)
        z_scores = self.z_test_by_digit(observed_frequencies, sample_size)
        fraud_patterns = self.detect_fraud_patterns(observed_frequencies, z_scores)
        
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
        
        self.analysis_results[dataset_name] = results
        return results
    
    def _determine_risk_level(self, fraud_score: int) -> str:
        if fraud_score >= 50:
            return "HIGH"
        elif fraud_score >= 25:
            return "MEDIUM"
        elif fraud_score >= 10:
            return "LOW"
        else:
            return "MINIMAL"
    
    def generate_report(self, analysis_results: Dict[str, any]) -> str:
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
            report += "\n- IMMEDIATE INVESTIGATION REQUIRED\n- Review all flagged transactions manually\n- Consider external forensic audit"
        elif analysis_results['risk_level'] == "MEDIUM":
            report += "\n- Enhanced monitoring recommended\n- Sample additional transactions for review\n- Implement additional controls"
        elif analysis_results['risk_level'] == "LOW":
            report += "\n- Minor deviations detected\n- Continue regular monitoring\n- Consider random sampling for verification"
        else:
            report += "\n- Data appears to follow Benford's Law\n- No immediate fraud concerns\n- Maintain standard monitoring procedures"
        
        return report
    
    def create_visualization(self, analysis_results: Dict[str, any]) -> None:
        if 'error' in analysis_results:
            print(f"Cannot create visualization: {analysis_results['error']}")
            return
        
        digits = list(range(1, 10))
        observed = [analysis_results['observed_frequencies'].get(d, 0) for d in digits]
        expected = [analysis_results['expected_frequencies'][d] for d in digits]
        deviations = [obs - exp for obs, exp in zip(observed, expected)]
        
        print("\n=== BENFORD'S LAW VISUALIZATION DATA ===")
        print("Digit | Observed | Expected | Deviation")
        print("------|----------|----------|----------")
        for i, digit in enumerate(digits):
            print(f"  {digit}   |   {observed[i]:.3f}  |   {expected[i]:.3f}  |   {deviations[i]:+.3f}")
        print("\n")


class AdvancedLedgerAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.prepare_data()
        self.analysis_results = {}
    
    def prepare_data(self):
        # Date processing
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['year'] = self.df['date'].dt.year
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            self.df['day_of_week'] = self.df['date'].dt.dayofweek
        
        # Amount processing
        self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
        self.df['abs_amount'] = abs(self.df['amount'])
        
        # Clean payee and type columns
        if 'payee' in self.df.columns:
            self.df['payee_clean'] = self.df['payee'].str.strip().str.upper()
        
        if 'type' in self.df.columns:
            self.df['type_clean'] = self.df['type'].str.strip().str.upper()
    
    def analyze_duplicates(self) -> Dict[str, any]:
        print("Analyzing duplicates...")
        duplicate_features = {}
        
        # Exact amount duplicates
        amount_duplicates = self.df[self.df.duplicated(subset=['amount'], keep=False)]
        duplicate_features['exact_amount_duplicates'] = len(amount_duplicates)
        duplicate_features['exact_amount_duplicate_rate'] = len(amount_duplicates) / len(self.df) * 100
        
        # Amount-payee duplicates
        if 'payee_clean' in self.df.columns:
            amount_payee_duplicates = self.df[self.df.duplicated(subset=['amount', 'payee_clean'], keep=False)]
            duplicate_features['amount_payee_duplicates'] = len(amount_payee_duplicates)
            duplicate_features['amount_payee_duplicate_rate'] = len(amount_payee_duplicates) / len(self.df) * 100
        
        # Exact transaction duplicates
        if 'payee_clean' in self.df.columns and 'date' in self.df.columns:
            exact_duplicates = self.df[self.df.duplicated(subset=['amount', 'payee_clean', 'date'], keep=False)]
            duplicate_features['exact_transaction_duplicates'] = len(exact_duplicates)
            duplicate_features['exact_transaction_duplicate_rate'] = len(exact_duplicates) / len(self.df) * 100
            
            if len(exact_duplicates) > 0:
                duplicate_features['duplicate_transactions'] = exact_duplicates[['amount', 'payee', 'date']].to_dict('records')
        
        # Round number analysis
        amounts = self.df['amount'].dropna()
        round_numbers = amounts[amounts % 1 == 0]
        round_tens = amounts[amounts % 10 == 0]
        round_hundreds = amounts[amounts % 100 == 0]
        round_thousands = amounts[amounts % 1000 == 0]
        
        duplicate_features['round_number_count'] = len(round_numbers)
        duplicate_features['round_number_rate'] = len(round_numbers) / len(amounts) * 100
        duplicate_features['round_tens_rate'] = len(round_tens) / len(amounts) * 100
        duplicate_features['round_hundreds_rate'] = len(round_hundreds) / len(amounts) * 100
        duplicate_features['round_thousands_rate'] = len(round_thousands) / len(amounts) * 100
        
        # Payee analysis
        if 'payee_clean' in self.df.columns:
            payee_counts = self.df['payee_clean'].value_counts()
            duplicate_features['unique_payees'] = len(payee_counts)
            duplicate_features['top_payee_frequency'] = payee_counts.iloc[0] if len(payee_counts) > 0 else 0
            duplicate_features['payee_concentration'] = (payee_counts.iloc[:5].sum() / len(self.df) * 100) if len(payee_counts) > 0 else 0
            
            high_freq_payees = payee_counts[payee_counts > len(self.df) * 0.05]
            duplicate_features['high_frequency_payees'] = len(high_freq_payees)
            if len(high_freq_payees) > 0:
                duplicate_features['high_frequency_payee_list'] = high_freq_payees.to_dict()
        
        return duplicate_features
    
    def analyze_outliers(self) -> Dict[str, any]:
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
        
        # Extreme outliers
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
        
        if len(extreme_outliers) > 0:
            outlier_features['extreme_outlier_transactions'] = extreme_outliers[['amount', 'payee', 'date']].to_dict('records')
        
        return outlier_features
    
    def analyze_thresholds(self) -> Dict[str, any]:
        print("Analyzing thresholds...")
        threshold_features = {}
        
        amounts = self.df['abs_amount'].dropna()
        
        # Key regulatory thresholds
        thresholds = {
            'cash_reporting_10k': 10000,
            'structuring_9999': 9999,
            'structuring_9500': 9500,
            'suspicious_5k': 5000,
            'check_clearing_3k': 3000,
            'daily_limit_1k': 1000
        }
        
        for threshold_name, threshold_value in thresholds.items():
            # Exact hits
            exact_hits = len(amounts[amounts == threshold_value])
            threshold_features[f'{threshold_name}_exact_hits'] = exact_hits
            
            # Just under threshold
            just_under = amounts[(amounts >= threshold_value * 0.95) & (amounts < threshold_value)]
            threshold_features[f'{threshold_name}_just_under'] = len(just_under)
            
            # Just over threshold
            just_over = amounts[(amounts > threshold_value) & (amounts <= threshold_value * 1.05)]
            threshold_features[f'{threshold_name}_just_over'] = len(just_over)
            
            # Cluster analysis around threshold
            cluster_range = amounts[(amounts >= threshold_value * 0.9) & (amounts <= threshold_value * 1.1)]
            threshold_features[f'{threshold_name}_cluster'] = len(cluster_range)
            threshold_features[f'{threshold_name}_cluster_rate'] = len(cluster_range) / len(amounts) * 100
        
        # Structuring analysis
        structuring_range = amounts[(amounts >= 9500) & (amounts <= 9999)]
        threshold_features['structuring_indicators'] = len(structuring_range)
        threshold_features['structuring_rate'] = len(structuring_range) / len(amounts) * 100
        
        if len(structuring_range) > 0:
            structuring_transactions = self.df[self.df['abs_amount'].between(9500, 9999)]
            threshold_features['structuring_transactions'] = structuring_transactions[['amount', 'payee', 'date']].to_dict('records')
        
        return threshold_features
    
    def analyze_ratios(self) -> Dict[str, any]:
        print("Analyzing ratios...")
        ratio_features = {}
        
        # Income vs expense classification
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
        total_income = income['amount'].sum()
        total_expenses = abs(expenses['amount'].sum())
        
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
        
        # Average transaction analysis
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
        weekend_transactions = self.df[self.df['day_of_week'].isin([5, 6])]
        weekday_transactions = self.df[~self.df['day_of_week'].isin([5, 6])]
        
        time_features['weekend_analysis'] = {
            'weekend_transaction_count': len(weekend_transactions),
            'weekday_transaction_count': len(weekday_transactions),
            'weekend_transaction_rate': len(weekend_transactions) / len(self.df) * 100,
            'weekend_amount_total': weekend_transactions['amount'].sum(),
            'weekday_amount_total': weekday_transactions['amount'].sum(),
            'weekend_amount_rate': weekend_transactions['amount'].sum() / self.df['amount'].sum() * 100
        }
        
        # Daily activity analysis
        daily_counts = self.df.groupby(self.df['date'].dt.date).size()
        daily_amounts = self.df.groupby(self.df['date'].dt.date)['amount'].sum()
        
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
        
        # Store daily counts for visualization
        time_features['daily_counts'] = daily_counts.to_dict()
        
        return time_features
    
    def generate_comprehensive_analysis(self) -> Dict[str, any]:
        print("Starting comprehensive ledger analysis...")
        
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
        analysis_results['risk_assessment'] = self.calculate_risk_scores(analysis_results)
        
        self.analysis_results = analysis_results
        print("Comprehensive analysis complete!")
        return analysis_results
    
    def calculate_risk_scores(self, analysis_results: Dict[str, any]) -> Dict[str, any]:
        risk_scores = {}
        
        # Duplicate risk scoring
        duplicate_risk = 0
        if 'duplicates' in analysis_results:
            duplicate_risk += min(analysis_results['duplicates'].get('exact_transaction_duplicate_rate', 0), 30)
            duplicate_risk += min(analysis_results['duplicates'].get('amount_payee_duplicate_rate', 0), 20)
            duplicate_risk += min(analysis_results['duplicates'].get('round_thousands_rate', 0), 15)
            duplicate_risk += min(analysis_results['duplicates'].get('payee_concentration', 0), 10)
            duplicate_risk += min(analysis_results['duplicates'].get('high_frequency_payees', 0) * 5, 25)
        
        risk_scores['duplicate_risk'] = min(duplicate_risk, 100)    
# Outlier risk scoring
        outlier_risk = 0
        if 'outliers' in analysis_results:
            outlier_risk += min(analysis_results['outliers'].get('z_score_outlier_rate_3sigma', 0), 30)
            outlier_risk += min(analysis_results['outliers'].get('extreme_outlier_rate', 0), 25)
            skewness = abs(analysis_results['outliers']['amount_stats'].get('skewness', 0))
            outlier_risk += min(skewness * 5, 20)
            kurtosis = abs(analysis_results['outliers']['amount_stats'].get('kurtosis', 0))
            outlier_risk += min(kurtosis * 2, 15)
        
        risk_scores['outlier_risk'] = min(outlier_risk, 100)
        
        # Threshold risk scoring
        threshold_risk = 0
        if 'thresholds' in analysis_results:
            threshold_risk += min(analysis_results['thresholds'].get('structuring_rate', 0), 50)
            threshold_risk += analysis_results['thresholds'].get('structuring_9999_exact_hits', 0) * 10
            threshold_risk += analysis_results['thresholds'].get('cash_reporting_10k_exact_hits', 0) * 5
            threshold_risk += min(analysis_results['thresholds'].get('cash_reporting_10k_cluster_rate', 0), 20)
        
        risk_scores['threshold_risk'] = min(threshold_risk, 100)
        
        # Time series risk scoring
        time_risk = 0
        if 'time_series' in analysis_results and 'error' not in analysis_results['time_series']:
            weekend_rate = analysis_results['time_series']['weekend_analysis'].get('weekend_transaction_rate', 0)
            time_risk += min(weekend_rate, 30)
            burst_days = analysis_results['time_series']['unusual_activity'].get('burst_activity_days', 0)
            time_risk += min(burst_days * 5, 25)
        
        risk_scores['time_series_risk'] = min(time_risk, 100)
        
        # Calculate overall risk
        risk_weights = {
            'duplicate_risk': 0.3,
            'outlier_risk': 0.25,
            'threshold_risk': 0.35,
            'time_series_risk': 0.1
        }
        
        overall_risk = sum(risk_scores[risk_type] * weight for risk_type, weight in risk_weights.items())
        risk_scores['overall_risk'] = overall_risk
        
        # Determine risk level
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
    
    def create_visualizations(self, analysis_results: Dict[str, any]) -> None:
        if not analysis_results:
            print("No analysis results available for visualization.")
            return
        
        print("\n=== VISUALIZATION DATA ===")
        print("Duplicate Analysis:")
        print(f"- Exact Transaction Duplicates: {analysis_results['duplicates'].get('exact_transaction_duplicate_rate', 0):.1f}%")
        print(f"- Amount-Payee Duplicates: {analysis_results['duplicates'].get('amount_payee_duplicate_rate', 0):.1f}%")
        print(f"- Round Thousands: {analysis_results['duplicates'].get('round_thousands_rate', 0):.1f}%")
        
        print("\nOutlier Analysis:")
        print(f"- Z-Score Outliers (3σ): {analysis_results['outliers'].get('z_score_outliers_3sigma', 0):,}")
        print(f"- IQR Outliers: {analysis_results['outliers'].get('iqr_outliers', 0):,}")
        print(f"- Extreme Outliers: {analysis_results['outliers'].get('extreme_outliers', 0):,}")