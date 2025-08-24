"""
Statistical Fraud Detection Methods for Forensic Accounting
Comprehensive suite of statistical tests and anomaly detection methods
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class StatisticalFraudDetector:
    """
    Comprehensive statistical fraud detection using multiple methodologies
    """
    
    def __init__(self):
        self.detection_methods = {
            'outlier_detection': self._detect_outliers,
            'digit_analysis': self._analyze_digits,
            'pattern_analysis': self._analyze_patterns,
            'time_series_analysis': self._analyze_time_series,
            'clustering_analysis': self._clustering_analysis,
            'threshold_analysis': self._threshold_analysis,
            'duplicates_analysis': self._duplicates_analysis,
            'ratio_analysis': self._ratio_analysis
        }
        self.results_history = []
        
    def comprehensive_fraud_analysis(self, df: pd.DataFrame, 
                                   amount_col: str = 'amount',
                                   date_col: str = 'date',
                                   vendor_col: str = 'payee_payer',
                                   account_col: str = 'account') -> Dict[str, Any]:
        """
        Perform comprehensive fraud analysis using all available methods
        
        Args:
            df: DataFrame containing transaction data
            amount_col: Column name for transaction amounts
            date_col: Column name for transaction dates
            vendor_col: Column name for vendor/payee information
            account_col: Column name for account information
            
        Returns:
            Dictionary containing all analysis results
        """
        print("Starting comprehensive fraud analysis...")
        
        # Prepare data
        analysis_df = self._prepare_data(df, amount_col, date_col, vendor_col, account_col)
        
        # Initialize results
        results = {
            'summary': {
                'total_transactions': len(df),
                'analysis_timestamp': datetime.now().isoformat(),
                'columns_analyzed': {
                    'amount': amount_col,
                    'date': date_col,
                    'vendor': vendor_col,
                    'account': account_col
                },
                'overall_fraud_score': 0,
                'risk_level': 'LOW'
            },
            'detailed_results': {},
            'flagged_transactions': [],
            'recommendations': [],
            'statistical_summary': {}
        }
        
        # Run all detection methods
        for method_name, method_func in self.detection_methods.items():
            try:
                print(f"Running {method_name}...")
                method_result = method_func(analysis_df)
                results['detailed_results'][method_name] = method_result
            except Exception as e:
                print(f"Error in {method_name}: {str(e)}")
                results['detailed_results'][method_name] = {'error': str(e)}
        
        # Calculate overall fraud score and risk assessment
        results['summary']['overall_fraud_score'] = self._calculate_overall_fraud_score(results['detailed_results'])
        results['summary']['risk_level'] = self._determine_risk_level(results['summary']['overall_fraud_score'])
        
        # Generate consolidated recommendations
        results['recommendations'] = self._generate_comprehensive_recommendations(results)
        
        # Identify highest risk transactions
        results['flagged_transactions'] = self._identify_flagged_transactions(analysis_df, results['detailed_results'])
        
        # Statistical summary
        results['statistical_summary'] = self._generate_statistical_summary(analysis_df, results['detailed_results'])
        
        self.results_history.append(results)
        print("Comprehensive fraud analysis completed!")
        
        return results
    
    def _prepare_data(self, df: pd.DataFrame, amount_col: str, date_col: str, 
                     vendor_col: str, account_col: str) -> pd.DataFrame:
        """Prepare and clean data for analysis"""
        analysis_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            amount_col: 'amount',
            date_col: 'date',
            vendor_col: 'vendor',
            account_col: 'account'
        }
        
        # Only rename columns that exist
        existing_mapping = {old: new for old, new in column_mapping.items() if old in analysis_df.columns}
        analysis_df = analysis_df.rename(columns=existing_mapping)
        
        # Convert data types
        if 'amount' in analysis_df.columns:
            analysis_df['amount'] = pd.to_numeric(analysis_df['amount'], errors='coerce')
            analysis_df['abs_amount'] = analysis_df['amount'].abs()
        
        if 'date' in analysis_df.columns:
            analysis_df['date'] = pd.to_datetime(analysis_df['date'], errors='coerce')
            analysis_df['year'] = analysis_df['date'].dt.year
            analysis_df['month'] = analysis_df['date'].dt.month
            analysis_df['day_of_week'] = analysis_df['date'].dt.dayofweek
        
        # Clean text fields
        for col in ['vendor', 'account']:
            if col in analysis_df.columns:
                analysis_df[f'{col}_clean'] = analysis_df[col].astype(str).str.strip().str.upper()
        
        return analysis_df
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive outlier detection using multiple methods"""
        if 'amount' not in df.columns:
            return {'error': 'Amount column not found'}
        
        amounts = df['amount'].dropna()
        if len(amounts) == 0:
            return {'error': 'No valid amounts found'}
        
        outlier_results = {
            'methods': {},
            'outlier_indices': set(),
            'summary': {}
        }
        
        # Method 1: Z-Score based outliers
        if len(amounts) > 1 and amounts.std() > 0:
            z_scores = np.abs(stats.zscore(amounts))
            z_outliers_2 = np.where(z_scores > 2)[0]
            z_outliers_3 = np.where(z_scores > 3)[0]
            
            outlier_results['methods']['z_score'] = {
                'outliers_2sigma': len(z_outliers_2),
                'outliers_3sigma': len(z_outliers_3),
                'outlier_rate_2sigma': len(z_outliers_2) / len(amounts) * 100,
                'outlier_rate_3sigma': len(z_outliers_3) / len(amounts) * 100,
                'indices': z_outliers_2.tolist()
            }
            outlier_results['outlier_indices'].update(amounts.index[z_outliers_2])
        
        # Method 2: IQR based outliers
        Q1 = amounts.quantile(0.25)
        Q3 = amounts.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = amounts[(amounts < lower_bound) | (amounts > upper_bound)]
            outlier_results['methods']['iqr'] = {
                'outliers_count': len(iqr_outliers),
                'outlier_rate': len(iqr_outliers) / len(amounts) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'indices': iqr_outliers.index.tolist()
            }
            outlier_results['outlier_indices'].update(iqr_outliers.index)
        
        # Method 3: Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_predictions = iso_forest.fit_predict(amounts.values.reshape(-1, 1))
            iso_outliers = amounts.index[outlier_predictions == -1]
            
            outlier_results['methods']['isolation_forest'] = {
                'outliers_count': len(iso_outliers),
                'outlier_rate': len(iso_outliers) / len(amounts) * 100,
                'anomaly_scores': iso_forest.decision_function(amounts.values.reshape(-1, 1)).tolist(),
                'indices': iso_outliers.tolist()
            }
            outlier_results['outlier_indices'].update(iso_outliers)
        except Exception as e:
            outlier_results['methods']['isolation_forest'] = {'error': str(e)}
        
        # Method 4: Modified Z-Score (using median)
        median = amounts.median()
        mad = np.median(np.abs(amounts - median))
        
        if mad > 0:
            modified_z_scores = 0.6745 * (amounts - median) / mad
            mod_z_outliers = amounts.index[np.abs(modified_z_scores) > 3.5]
            
            outlier_results['methods']['modified_z_score'] = {
                'outliers_count': len(mod_z_outliers),
                'outlier_rate': len(mod_z_outliers) / len(amounts) * 100,
                'indices': mod_z_outliers.tolist()
            }
            outlier_results['outlier_indices'].update(mod_z_outliers)
        
        # Summary
        outlier_results['summary'] = {
            'total_unique_outliers': len(outlier_results['outlier_indices']),
            'outlier_rate': len(outlier_results['outlier_indices']) / len(amounts) * 100,
            'methods_agreeing': len([m for m in outlier_results['methods'].values() 
                                   if isinstance(m, dict) and 'outliers_count' in m and m['outliers_count'] > 0])
        }
        
        return outlier_results
    
    def _analyze_digits(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive digit analysis for fraud detection"""
        if 'amount' not in df.columns:
            return {'error': 'Amount column not found'}
        
        amounts = df['amount'].dropna()
        if len(amounts) == 0:
            return {'error': 'No valid amounts found'}
        
        digit_results = {
            'first_digit_analysis': {},
            'last_digit_analysis': {},
            'digit_patterns': {},
            'fraud_indicators': {}
        }
        
        # First digit analysis (Benford's Law)
        first_digits = []
        for amount in amounts:
            if amount > 0:
                first_digit = int(str(int(abs(amount)))[0])
                if first_digit > 0:
                    first_digits.append(first_digit)
        
        if first_digits:
            first_digit_freq = {d: first_digits.count(d) / len(first_digits) for d in range(1, 10)}
            benford_expected = {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079,
                              6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046}
            
            # Chi-square test for Benford's Law
            observed = [first_digit_freq.get(d, 0) * len(first_digits) for d in range(1, 10)]
            expected = [benford_expected[d] * len(first_digits) for d in range(1, 10)]
            
            chi2_stat, p_value = stats.chisquare(observed, expected)
            
            digit_results['first_digit_analysis'] = {
                'frequencies': first_digit_freq,
                'benford_expected': benford_expected,
                'chi_square_stat': chi2_stat,
                'p_value': p_value,
                'follows_benford': p_value > 0.05,
                'sample_size': len(first_digits)
            }
        
        # Last digit analysis (should be roughly uniform for natural data)
        last_digits = [int(str(int(abs(amount) * 100))[-1]) for amount in amounts if amount > 0]
        if last_digits:
            last_digit_freq = {d: last_digits.count(d) / len(last_digits) for d in range(10)}
            
            # Chi-square test for uniformity
            observed_last = [last_digit_freq[d] * len(last_digits) for d in range(10)]
            expected_last = [len(last_digits) / 10] * 10
            
            chi2_last, p_last = stats.chisquare(observed_last, expected_last)
            
            digit_results['last_digit_analysis'] = {
                'frequencies': last_digit_freq,
                'chi_square_stat': chi2_last,
                'p_value': p_last,
                'is_uniform': p_last > 0.05,
                'sample_size': len(last_digits)
            }
        
        # Digit patterns analysis
        digit_results['digit_patterns'] = self._analyze_digit_patterns(amounts)
        
        # Fraud indicators based on digit analysis
        fraud_score = 0
        indicators = []
        
        if 'first_digit_analysis' in digit_results and not digit_results['first_digit_analysis'].get('follows_benford', True):
            fraud_score += 30
            indicators.append("Significant deviation from Benford's Law")
        
        if 'last_digit_analysis' in digit_results and not digit_results['last_digit_analysis'].get('is_uniform', True):
            fraud_score += 20
            indicators.append("Non-uniform last digit distribution")
        
        patterns = digit_results['digit_patterns']
        if patterns.get('round_number_rate', 0) > 15:
            fraud_score += 25
            indicators.append(f"High round number rate: {patterns['round_number_rate']:.1f}%")
        
        if patterns.get('repeated_digits_rate', 0) > 10:
            fraud_score += 15
            indicators.append(f"High repeated digits rate: {patterns['repeated_digits_rate']:.1f}%")
        
        digit_results['fraud_indicators'] = {
            'fraud_score': min(fraud_score, 100),
            'indicators': indicators
        }
        
        return digit_results
    
    def _analyze_digit_patterns(self, amounts: pd.Series) -> Dict[str, Any]:
        """Analyze specific digit patterns that may indicate fraud"""
        patterns = {}
        
        # Round numbers
        round_1000 = sum(1 for amt in amounts if amt % 1000 == 0 and amt >= 1000)
        round_100 = sum(1 for amt in amounts if amt % 100 == 0 and amt >= 100)
        round_10 = sum(1 for amt in amounts if amt % 10 == 0 and amt >= 10)
        
        patterns['round_number_rate'] = (round_1000 + round_100 + round_10) / len(amounts) * 100
        patterns['round_1000_count'] = round_1000
        patterns['round_100_count'] = round_100
        patterns['round_10_count'] = round_10
        
        # Repeated digits (e.g., 1111, 2222, 5555)
        repeated_digits = 0
        for amount in amounts:
            amt_str = str(int(abs(amount)))
            if len(set(amt_str)) == 1 and len(amt_str) >= 2:
                repeated_digits += 1
        
        patterns['repeated_digits_rate'] = repeated_digits / len(amounts) * 100
        patterns['repeated_digits_count'] = repeated_digits
        
        # Sequential numbers (e.g., 1234, 5678)
        sequential = 0
        for amount in amounts:
            amt_str = str(int(abs(amount)))
            if len(amt_str) >= 3:
                digits = [int(d) for d in amt_str]
                is_sequential = all(digits[i] + 1 == digits[i + 1] for i in range(len(digits) - 1))
                if is_sequential:
                    sequential += 1
        
        patterns['sequential_rate'] = sequential / len(amounts) * 100
        patterns['sequential_count'] = sequential
        
        return patterns
    
    def _analyze_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transaction patterns for fraud indicators"""
        pattern_results = {
            'amount_patterns': {},
            'vendor_patterns': {},
            'temporal_patterns': {},
            'account_patterns': {}
        }
        
        # Amount patterns
        if 'amount' in df.columns:
            amounts = df['amount'].dropna()
            
            # Clustering analysis on amounts
            if len(amounts) > 10:
                amount_clusters = self._analyze_amount_clusters(amounts)
                pattern_results['amount_patterns'] = amount_clusters
        
        # Vendor patterns
        if 'vendor_clean' in df.columns:
            vendor_analysis = self._analyze_vendor_patterns(df)
            pattern_results['vendor_patterns'] = vendor_analysis
        
        # Temporal patterns
        if 'date' in df.columns:
            temporal_analysis = self._analyze_temporal_patterns(df)
            pattern_results['temporal_patterns'] = temporal_analysis
        
        # Account patterns
        if 'account_clean' in df.columns:
            account_analysis = self._analyze_account_patterns(df)
            pattern_results['account_patterns'] = account_analysis
        
        return pattern_results
    
    def _analyze_amount_clusters(self, amounts: pd.Series) -> Dict[str, Any]:
        """Analyze clustering patterns in transaction amounts"""
        # Prepare data for clustering
        X = amounts.values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cluster_results = {}
        
        # DBSCAN clustering to find dense regions
        try:
            dbscan = DBSCAN(eps=0.3, min_samples=5)
            cluster_labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            cluster_results['dbscan'] = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'noise_rate': n_noise / len(amounts) * 100,
                'cluster_labels': cluster_labels.tolist()
            }
            
            # Analyze clusters
            if n_clusters > 0:
                cluster_analysis = {}
                for cluster_id in set(cluster_labels):
                    if cluster_id != -1:
                        cluster_points = amounts[cluster_labels == cluster_id]
                        cluster_analysis[cluster_id] = {
                            'size': len(cluster_points),
                            'mean_amount': float(cluster_points.mean()),
                            'std_amount': float(cluster_points.std()),
                            'min_amount': float(cluster_points.min()),
                            'max_amount': float(cluster_points.max())
                        }
                
                cluster_results['cluster_analysis'] = cluster_analysis
        
        except Exception as e:
            cluster_results['dbscan'] = {'error': str(e)}
        
        return cluster_results
    
    def _analyze_vendor_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze vendor-related fraud patterns"""
        vendor_results = {}
        
        if 'vendor_clean' in df.columns:
            vendors = df['vendor_clean'].dropna()
            vendor_counts = vendors.value_counts()
            
            vendor_results['basic_stats'] = {
                'unique_vendors': len(vendor_counts),
                'total_transactions': len(vendors),
                'avg_transactions_per_vendor': len(vendors) / len(vendor_counts) if len(vendor_counts) > 0 else 0
            }
            
            # High-frequency vendors (potential red flag)
            high_freq_threshold = max(5, len(vendors) * 0.05)  # 5% of transactions or minimum 5
            high_freq_vendors = vendor_counts[vendor_counts >= high_freq_threshold]
            
            vendor_results['high_frequency'] = {
                'count': len(high_freq_vendors),
                'vendors': high_freq_vendors.to_dict(),
                'concentration_rate': high_freq_vendors.sum() / len(vendors) * 100
            }
            
            # Generic vendor names (fraud indicator)
            generic_patterns = ['VENDOR', 'SUPPLIER', 'COMPANY', 'INC', 'LLC', 'CORP']
            generic_vendors = []
            
            for vendor in vendors:
                if any(pattern in str(vendor) for pattern in generic_patterns):
                    if any(char.isdigit() for char in str(vendor)):  # Contains numbers
                        generic_vendors.append(vendor)
            
            vendor_results['generic_vendors'] = {
                'count': len(set(generic_vendors)),
                'rate': len(set(generic_vendors)) / len(vendor_counts) * 100 if len(vendor_counts) > 0 else 0,
                'examples': list(set(generic_vendors))[:10]
            }
        
        return vendor_results
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns for fraud indicators"""
        temporal_results = {}
        
        if 'date' in df.columns:
            valid_dates = df.dropna(subset=['date'])
            
            if len(valid_dates) > 0:
                # Weekend transactions
                weekend_mask = valid_dates['day_of_week'].isin([5, 6])  # Saturday, Sunday
                weekend_count = weekend_mask.sum()
                
                temporal_results['weekend_activity'] = {
                    'weekend_transactions': int(weekend_count),
                    'weekend_rate': weekend_count / len(valid_dates) * 100,
                    'is_suspicious': weekend_count / len(valid_dates) > 0.15  # More than 15% on weekends
                }
                
                # End-of-month clustering
                end_of_month = valid_dates['date'].dt.day > 25
                eom_count = end_of_month.sum()
                
                temporal_results['end_of_month'] = {
                    'end_of_month_transactions': int(eom_count),
                    'end_of_month_rate': eom_count / len(valid_dates) * 100,
                    'is_suspicious': eom_count / len(valid_dates) > 0.25  # More than 25% at end of month
                }
                
                # Daily transaction patterns
                daily_counts = valid_dates.groupby(valid_dates['date'].dt.date).size()
                
                if len(daily_counts) > 1:
                    temporal_results['daily_patterns'] = {
                        'avg_daily_transactions': float(daily_counts.mean()),
                        'max_daily_transactions': int(daily_counts.max()),
                        'daily_std': float(daily_counts.std()),
                        'burst_days': int((daily_counts > daily_counts.mean() + 2 * daily_counts.std()).sum())
                    }
        
        return temporal_results
    
    def _analyze_account_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze account-related patterns"""
        account_results = {}
        
        if 'account_clean' in df.columns:
            accounts = df['account_clean'].dropna()
            account_counts = accounts.value_counts()
            
            account_results['distribution'] = {
                'unique_accounts': len(account_counts),
                'total_transactions': len(accounts),
                'top_account_frequency': int(account_counts.iloc[0]) if len(account_counts) > 0 else 0,
                'account_concentration': account_counts.iloc[0] / len(accounts) * 100 if len(accounts) > 0 else 0
            }
            
            # Unusual account usage
            if 'amount' in df.columns:
                account_amounts = df.groupby('account_clean')['amount'].agg(['count', 'sum', 'mean', 'std'])
                
                # Find accounts with unusual patterns
                unusual_accounts = []
                for account, stats in account_amounts.iterrows():
                    if stats['count'] == 1 and stats['sum'] > 10000:  # Single large transaction
                        unusual_accounts.append(f"{account}: Single transaction of ${stats['sum']:,.2f}")
                    elif stats['std'] / stats['mean'] > 2 if stats['mean'] > 0 else False:  # High variability
                        unusual_accounts.append(f"{account}: High amount variability (CV={stats['std']/stats['mean']:.2f})")
                
                account_results['unusual_patterns'] = {
                    'count': len(unusual_accounts),
                    'patterns': unusual_accounts[:10]  # Top 10
                }
        
        return account_results
    
    def _analyze_time_series(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced time series analysis for fraud detection"""
        if 'date' not in df.columns:
            return {'error': 'Date column not found'}
        
        time_results = {
            'seasonal_patterns': {},
            'trend_analysis': {},
            'anomaly_detection': {}
        }
        
        valid_dates = df.dropna(subset=['date'])
        if len(valid_dates) == 0:
            return {'error': 'No valid dates found'}
        
        # Monthly aggregation
        monthly_data = valid_dates.groupby([valid_dates['date'].dt.year, valid_dates['date'].dt.month]).agg({
            'amount': ['count', 'sum', 'mean']
        }).reset_index()
        
        monthly_data.columns = ['year', 'month', 'transaction_count', 'total_amount', 'avg_amount']
        
        if len(monthly_data) > 3:
            # Seasonal patterns
            monthly_counts = monthly_data['transaction_count']
            seasonal_cv = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
            
            time_results['seasonal_patterns'] = {
                'monthly_coefficient_variation': float(seasonal_cv),
                'is_highly_seasonal': seasonal_cv > 0.5,
                'peak_month': int(monthly_data.loc[monthly_data['transaction_count'].idxmax(), 'month']),
                'low_month': int(monthly_data.loc[monthly_data['transaction_count'].idxmin(), 'month'])
            }
            
            # Trend analysis
            if len(monthly_data) > 6:
                months = np.arange(len(monthly_data))
                
                # Linear trend in transaction counts
                count_slope, count_intercept, count_r, count_p, _ = stats.linregress(months, monthly_counts)
                
                # Linear trend in amounts
                amount_slope, amount_intercept, amount_r, amount_p, _ = stats.linregress(months, monthly_data['total_amount'])
                
                time_results['trend_analysis'] = {
                    'transaction_count_trend': {
                        'slope': float(count_slope),
                        'r_squared': float(count_r ** 2),
                        'p_value': float(count_p),
                        'is_significant': count_p < 0.05
                    },
                    'amount_trend': {
                        'slope': float(amount_slope),
                        'r_squared': float(amount_r ** 2),
                        'p_value': float(amount_p),
                        'is_significant': amount_p < 0.05
                    }
                }
        
        return time_results
    
    def _clustering_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Multi-dimensional clustering analysis for anomaly detection"""
        if 'amount' not in df.columns:
            return {'error': 'Amount column required for clustering analysis'}
        
        clustering_results = {}
        
        # Prepare features for clustering
        features = []
        feature_names = []
        
        # Amount features
        amounts = df['amount'].fillna(0)
        features.append(amounts.values)
        feature_names.append('amount')
        
        # Log amount (to handle scale differences)
        log_amounts = np.log1p(np.abs(amounts))
        features.append(log_amounts.values)
        feature_names.append('log_amount')
        
        # Day of week (if available)
        if 'day_of_week' in df.columns:
            features.append(df['day_of_week'].fillna(0).values)
            feature_names.append('day_of_week')
        
        # Month (if available)
        if 'month' in df.columns:
            features.append(df['month'].fillna(0).values)
            feature_names.append('month')
        
        if len(features) >= 2:
            # Combine features
            X = np.column_stack(features)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Isolation Forest for anomaly detection
            try:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_scores = iso_forest.fit_predict(X_scaled)
                anomaly_indices = np.where(anomaly_scores == -1)[0]
                
                clustering_results['isolation_forest'] = {
                    'n_anomalies': len(anomaly_indices),
                    'anomaly_rate': len(anomaly_indices) / len(X) * 100,
                    'anomaly_indices': anomaly_indices.tolist(),
                    'feature_names': feature_names
                }
                
            except Exception as e:
                clustering_results['isolation_forest'] = {'error': str(e)}
            
            # Elliptic Envelope for outlier detection
            try:
                envelope = EllipticEnvelope(contamination=0.1, random_state=42)
                envelope_pred = envelope.fit_predict(X_scaled)
                envelope_outliers = np.where(envelope_pred == -1)[0]
                
                clustering_results['elliptic_envelope'] = {
                    'n_outliers': len(envelope_outliers),
                    'outlier_rate': len(envelope_outliers) / len(X) * 100,
                    'outlier_indices': envelope_outliers.tolist()
                }
                
            except Exception as e:
                clustering_results['elliptic_envelope'] = {'error': str(e)}
        
        return clustering_results
    
    def _threshold_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze transactions near critical thresholds"""
        if 'amount' not in df.columns:
            return {'error': 'Amount column required for threshold analysis'}
        
        amounts = df['abs_amount'].dropna() if 'abs_amount' in df.columns else df['amount'].abs().dropna()
        
        # Key financial thresholds
        thresholds = {
            'cash_reporting': 10000,
            'structuring_avoidance': 9999,
            'check_clearing': 3000,
            'petty_cash': 500,
            'approval_limit': 5000
        }
        
        threshold_results = {}
        
        for threshold_name, threshold_value in thresholds.items():
            # Exact matches
            exact_matches = (amounts == threshold_value).sum()
            
            # Near misses (within 1% below threshold)
            near_misses = ((amounts >= threshold_value * 0.99) & (amounts < threshold_value)).sum()
            
            # Just over threshold (within 1% above)
            just_over = ((amounts > threshold_value) & (amounts <= threshold_value * 1.01)).sum()
            
            # Clustering around threshold (within 5%)
            cluster_range = ((amounts >= threshold_value * 0.95) & (amounts <= threshold_value * 1.05)).sum()
            
            threshold_results[threshold_name] = {
                'threshold_value': threshold_value,
                'exact_matches': int(exact_matches),
                'near_misses': int(near_misses),
                'just_over': int(just_over),
                'cluster_count': int(cluster_range),
                'cluster_rate': cluster_range / len(amounts) * 100 if len(amounts) > 0 else 0,
                'suspicion_score': self._calculate_threshold_suspicion(exact_matches, near_misses, just_over, len(amounts))
            }
        
        return threshold_results
    
    def _calculate_threshold_suspicion(self, exact: int, near: int, over: int, total: int) -> float:
        """Calculate suspicion score for threshold analysis"""
        if total == 0:
            return 0
        
        score = 0
        score += exact * 30  # Exact matches are highly suspicious
        score += near * 20   # Near misses are suspicious
        score += over * 10   # Just over threshold is moderately suspicious
        
        return min(score / total * 100, 100)  # Normalize to 0-100 scale
    
    def _duplicates_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive duplicate transaction analysis"""
        duplicate_results = {
            'exact_duplicates': {},
            'near_duplicates': {},
            'pattern_duplicates': {}
        }
        
        # Exact duplicates - same amount and vendor
        if 'amount' in df.columns and 'vendor_clean' in df.columns:
            exact_dups = df[df.duplicated(subset=['amount', 'vendor_clean'], keep=False)]
            
            duplicate_results['exact_duplicates'] = {
                'count': len(exact_dups),
                'rate': len(exact_dups) / len(df) * 100,
                'unique_pairs': len(exact_dups.groupby(['amount', 'vendor_clean'])),
                'examples': exact_dups[['amount', 'vendor_clean']].head(10).to_dict('records') if len(exact_dups) > 0 else []
            }
        
        # Amount-only duplicates
        if 'amount' in df.columns:
            amount_dups = df[df.duplicated(subset=['amount'], keep=False)]
            
            duplicate_results['amount_duplicates'] = {
                'count': len(amount_dups),
                'rate': len(amount_dups) / len(df) * 100,
                'unique_amounts': df['amount'].nunique() if len(df) > 0 else 0,
                'duplication_ratio': len(amount_dups) / df['amount'].nunique() if df['amount'].nunique() > 0 else 0
            }
        
        # Date-based duplicates (same day, same vendor, different amounts)
        if all(col in df.columns for col in ['date', 'vendor_clean', 'amount']):
            df_temp = df.dropna(subset=['date', 'vendor_clean'])
            df_temp['date_only'] = df_temp['date'].dt.date
            
            same_day_vendor = df_temp.groupby(['date_only', 'vendor_clean']).size()
            multiple_same_day = same_day_vendor[same_day_vendor > 1]
            
            duplicate_results['same_day_vendor'] = {
                'pairs_with_multiple': len(multiple_same_day),
                'total_transactions': multiple_same_day.sum(),
                'rate': multiple_same_day.sum() / len(df_temp) * 100 if len(df_temp) > 0 else 0
            }
        
        return duplicate_results
    
    def _ratio_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Financial ratio analysis for fraud detection"""
        if 'amount' not in df.columns:
            return {'error': 'Amount column required for ratio analysis'}
        
        ratio_results = {}
        
        # Basic amount ratios
        amounts = df['amount'].dropna()
        positive_amounts = amounts[amounts > 0]
        negative_amounts = amounts[amounts < 0]
        
        ratio_results['basic_ratios'] = {
            'positive_to_negative_ratio': len(positive_amounts) / len(negative_amounts) if len(negative_amounts) > 0 else float('inf'),
            'amount_concentration': {
                'top_10_percent_share': positive_amounts.nlargest(max(1, len(positive_amounts) // 10)).sum() / positive_amounts.sum() * 100 if len(positive_amounts) > 0 else 0,
                'bottom_10_percent_share': positive_amounts.nsmallest(max(1, len(positive_amounts) // 10)).sum() / positive_amounts.sum() * 100 if len(positive_amounts) > 0 else 0
            }
        }
        
        # Vendor-based ratios
        if 'vendor_clean' in df.columns:
            vendor_amounts = df.groupby('vendor_clean')['amount'].sum().abs()
            
            ratio_results['vendor_ratios'] = {
                'vendor_concentration': vendor_amounts.nlargest(5).sum() / vendor_amounts.sum() * 100 if len(vendor_amounts) > 0 else 0,
                'single_vendor_max_share': vendor_amounts.max() / vendor_amounts.sum() * 100 if len(vendor_amounts) > 0 else 0,
                'vendors_above_threshold': (vendor_amounts > vendor_amounts.mean() * 3).sum()
            }
        
        # Account-based ratios
        if 'account_clean' in df.columns:
            account_amounts = df.groupby('account_clean')['amount'].sum().abs()
            
            ratio_results['account_ratios'] = {
                'account_concentration': account_amounts.nlargest(3).sum() / account_amounts.sum() * 100 if len(account_amounts) > 0 else 0,
                'account_imbalance': account_amounts.std() / account_amounts.mean() if account_amounts.mean() > 0 else 0
            }
        
        return ratio_results
    
    def _calculate_overall_fraud_score(self, detailed_results: Dict[str, Any]) -> float:
        """Calculate overall fraud score from all analysis methods"""
        fraud_score = 0
        weight_sum = 0
        
        # Weight each analysis method based on reliability and importance
        method_weights = {
            'outlier_detection': 0.15,
            'digit_analysis': 0.25,
            'pattern_analysis': 0.20,
            'threshold_analysis': 0.20,
            'duplicates_analysis': 0.15,
            'clustering_analysis': 0.05
        }
        
        for method, weight in method_weights.items():
            if method in detailed_results and 'error' not in detailed_results[method]:
                method_score = self._extract_method_score(method, detailed_results[method])
                fraud_score += method_score * weight
                weight_sum += weight
        
        # Normalize by actual weights used
        if weight_sum > 0:
            fraud_score = fraud_score / weight_sum
        
        return min(fraud_score, 100)
    
    def _extract_method_score(self, method: str, result: Dict[str, Any]) -> float:
        """Extract fraud score from individual method results"""
        if method == 'outlier_detection':
            return min(result.get('summary', {}).get('outlier_rate', 0) * 2, 100)
        
        elif method == 'digit_analysis':
            return result.get('fraud_indicators', {}).get('fraud_score', 0)
        
        elif method == 'pattern_analysis':
            score = 0
            # Amount clustering anomalies
            if 'amount_patterns' in result:
                dbscan = result['amount_patterns'].get('dbscan', {})
                if 'noise_rate' in dbscan:
                    score += min(dbscan['noise_rate'], 25)
            
            # Vendor patterns
            if 'vendor_patterns' in result:
                generic_rate = result['vendor_patterns'].get('generic_vendors', {}).get('rate', 0)
                score += min(generic_rate * 2, 25)
            
            # Temporal patterns
            if 'temporal_patterns' in result:
                weekend = result['temporal_patterns'].get('weekend_activity', {})
                if weekend.get('is_suspicious', False):
                    score += 20
                    
                eom = result['temporal_patterns'].get('end_of_month', {})
                if eom.get('is_suspicious', False):
                    score += 15
            
            return min(score, 100)
        
        elif method == 'threshold_analysis':
            total_score = 0
            count = 0
            for threshold_data in result.values():
                if isinstance(threshold_data, dict) and 'suspicion_score' in threshold_data:
                    total_score += threshold_data['suspicion_score']
                    count += 1
            return total_score / count if count > 0 else 0
        
        elif method == 'duplicates_analysis':
            score = 0
            if 'exact_duplicates' in result:
                score += min(result['exact_duplicates'].get('rate', 0) * 3, 40)
            if 'same_day_vendor' in result:
                score += min(result['same_day_vendor'].get('rate', 0) * 2, 30)
            return min(score, 100)
        
        elif method == 'clustering_analysis':
            score = 0
            if 'isolation_forest' in result:
                score += min(result['isolation_forest'].get('anomaly_rate', 0) * 2, 50)
            if 'elliptic_envelope' in result:
                score += min(result['elliptic_envelope'].get('outlier_rate', 0) * 2, 50)
            return min(score, 100)
        
        return 0
    
    def _determine_risk_level(self, fraud_score: float) -> str:
        """Determine risk level based on fraud score"""
        if fraud_score >= 70:
            return 'CRITICAL'
        elif fraud_score >= 50:
            return 'HIGH'
        elif fraud_score >= 30:
            return 'MEDIUM'
        elif fraud_score >= 15:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _generate_comprehensive_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all analysis results"""
        recommendations = []
        fraud_score = results['summary']['overall_fraud_score']
        risk_level = results['summary']['risk_level']
        
        # High-level recommendations based on risk level
        if risk_level == 'CRITICAL':
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: Critical fraud indicators detected",
                "Suspend all related financial processes pending investigation",
                "Engage external forensic accounting firm",
                "Preserve all documentation and system logs",
                "Notify appropriate authorities as required by regulations"
            ])
        
        elif risk_level == 'HIGH':
            recommendations.extend([
                "Urgent investigation required within 24-48 hours",
                "Conduct detailed review of all flagged transactions",
                "Implement additional authorization controls",
                "Consider engaging forensic accounting expertise"
            ])
        
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "Conduct thorough review within one week",
                "Implement enhanced monitoring procedures",
                "Review and strengthen internal controls",
                "Perform expanded sampling of similar transactions"
            ])
        
        # Specific recommendations based on analysis results
        detailed_results = results.get('detailed_results', {})
        
        # Outlier recommendations
        if 'outlier_detection' in detailed_results:
            outlier_rate = detailed_results['outlier_detection'].get('summary', {}).get('outlier_rate', 0)
            if outlier_rate > 10:
                recommendations.append(f"High outlier rate ({outlier_rate:.1f}%) - investigate unusual amounts")
        
        # Digit analysis recommendations
        if 'digit_analysis' in detailed_results:
            digit_fraud_score = detailed_results['digit_analysis'].get('fraud_indicators', {}).get('fraud_score', 0)
            if digit_fraud_score > 30:
                recommendations.append("Significant Benford's Law violations detected - investigate data manipulation")
        
        # Threshold recommendations
        if 'threshold_analysis' in detailed_results:
            for threshold_name, threshold_data in detailed_results['threshold_analysis'].items():
                if isinstance(threshold_data, dict) and threshold_data.get('suspicion_score', 0) > 20:
                    recommendations.append(f"Suspicious activity near {threshold_name} threshold - investigate structuring")
        
        # Duplicate recommendations
        if 'duplicates_analysis' in detailed_results:
            exact_dup_rate = detailed_results['duplicates_analysis'].get('exact_duplicates', {}).get('rate', 0)
            if exact_dup_rate > 5:
                recommendations.append(f"High duplicate rate ({exact_dup_rate:.1f}%) - review for duplicate payments")
        
        return recommendations
    
    def _identify_flagged_transactions(self, df: pd.DataFrame, detailed_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific transactions that should be flagged for review"""
        flagged_transactions = []
        
        # Collect indices of suspicious transactions from various analyses
        suspicious_indices = set()
        
        # From outlier detection
        if 'outlier_detection' in detailed_results:
            outlier_results = detailed_results['outlier_detection']
            if 'outlier_indices' in outlier_results:
                suspicious_indices.update(outlier_results['outlier_indices'])
        
        # From clustering analysis
        if 'clustering_analysis' in detailed_results:
            clustering = detailed_results['clustering_analysis']
            if 'isolation_forest' in clustering:
                iso_indices = clustering['isolation_forest'].get('anomaly_indices', [])
                suspicious_indices.update(df.index[iso_indices] if iso_indices else [])
        
        # From threshold analysis - find transactions near thresholds
        if 'threshold_analysis' in detailed_results and 'amount' in df.columns:
            amounts = df['amount'].abs()
            for threshold_name, threshold_data in detailed_results['threshold_analysis'].items():
                if isinstance(threshold_data, dict) and threshold_data.get('suspicion_score', 0) > 10:
                    threshold_value = threshold_data['threshold_value']
                    near_threshold = amounts[(amounts >= threshold_value * 0.95) & (amounts <= threshold_value * 1.05)]
                    suspicious_indices.update(near_threshold.index)
        
        # Create flagged transaction records
        for idx in suspicious_indices:
            if idx in df.index:
                transaction = df.loc[idx]
                flagged_record = {
                    'index': idx,
                    'transaction_data': transaction.to_dict(),
                    'flag_reasons': self._identify_flag_reasons(idx, detailed_results)
                }
                flagged_transactions.append(flagged_record)
        
        # Sort by number of flag reasons (most suspicious first)
        flagged_transactions.sort(key=lambda x: len(x['flag_reasons']), reverse=True)
        
        # Return top 50 most suspicious transactions
        return flagged_transactions[:50]
    
    def _identify_flag_reasons(self, transaction_idx: int, detailed_results: Dict[str, Any]) -> List[str]:
        """Identify specific reasons why a transaction was flagged"""
        reasons = []
        
        # Check each analysis method for this transaction
        if 'outlier_detection' in detailed_results:
            outlier_results = detailed_results['outlier_detection']
            if transaction_idx in outlier_results.get('outlier_indices', set()):
                reasons.append("Statistical outlier detected")
        
        if 'clustering_analysis' in detailed_results:
            clustering = detailed_results['clustering_analysis']
            if 'isolation_forest' in clustering:
                iso_indices = clustering['isolation_forest'].get('anomaly_indices', [])
                if transaction_idx in iso_indices:
                    reasons.append("Anomalous pattern in multi-dimensional analysis")
        
        # Add more specific reasons based on other analyses
        # This would be expanded based on the specific transaction characteristics
        
        return reasons
    
    def _generate_statistical_summary(self, df: pd.DataFrame, detailed_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical summary of all analyses"""
        summary = {
            'data_quality': {},
            'analysis_coverage': {},
            'key_statistics': {}
        }
        
        # Data quality metrics
        if 'amount' in df.columns:
            amounts = df['amount'].dropna()
            summary['data_quality'] = {
                'completeness_rate': len(amounts) / len(df) * 100,
                'amount_range': {
                    'min': float(amounts.min()) if len(amounts) > 0 else 0,
                    'max': float(amounts.max()) if len(amounts) > 0 else 0,
                    'mean': float(amounts.mean()) if len(amounts) > 0 else 0,
                    'median': float(amounts.median()) if len(amounts) > 0 else 0,
                    'std': float(amounts.std()) if len(amounts) > 0 else 0
                }
            }
        
        # Analysis coverage
        methods_run = len([m for m in detailed_results.values() if 'error' not in m])
        total_methods = len(self.detection_methods)
        
        summary['analysis_coverage'] = {
            'methods_completed': methods_run,
            'total_methods': total_methods,
            'coverage_rate': methods_run / total_methods * 100
        }
        
        # Key statistics from analyses
        summary['key_statistics'] = {
            'total_outliers_detected': self._count_total_outliers(detailed_results),
            'total_anomalies_detected': self._count_total_anomalies(detailed_results),
            'highest_risk_method': self._identify_highest_risk_method(detailed_results)
        }
        
        return summary
    
    def _count_total_outliers(self, detailed_results: Dict[str, Any]) -> int:
        """Count total unique outliers across all methods"""
        outlier_indices = set()
        
        if 'outlier_detection' in detailed_results:
            outlier_indices.update(detailed_results['outlier_detection'].get('outlier_indices', set()))
        
        if 'clustering_analysis' in detailed_results:
            clustering = detailed_results['clustering_analysis']
            if 'isolation_forest' in clustering:
                outlier_indices.update(clustering['isolation_forest'].get('anomaly_indices', []))
        
        return len(outlier_indices)
    
    def _count_total_anomalies(self, detailed_results: Dict[str, Any]) -> int:
        """Count total anomalies detected across all methods"""
        anomaly_count = 0
        
        # Count various types of anomalies
        if 'duplicates_analysis' in detailed_results:
            anomaly_count += detailed_results['duplicates_analysis'].get('exact_duplicates', {}).get('count', 0)
        
        if 'threshold_analysis' in detailed_results:
            for threshold_data in detailed_results['threshold_analysis'].values():
                if isinstance(threshold_data, dict):
                    anomaly_count += threshold_data.get('exact_matches', 0)
                    anomaly_count += threshold_data.get('near_misses', 0)
        
        return anomaly_count
    
    def _identify_highest_risk_method(self, detailed_results: Dict[str, Any]) -> str:
        """Identify which analysis method detected the highest risk"""
        method_scores = {}
        
        for method in detailed_results:
            if 'error' not in detailed_results[method]:
                method_scores[method] = self._extract_method_score(method, detailed_results[method])
        
        if method_scores:
            return max(method_scores, key=method_scores.get)
        else:
            return 'none'
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive fraud detection report"""
        summary = analysis_results['summary']
        detailed = analysis_results['detailed_results']
        recommendations = analysis_results['recommendations']
        flagged = analysis_results['flagged_transactions']
        stats = analysis_results['statistical_summary']
        
        report = f"""
{'='*100}
COMPREHENSIVE STATISTICAL FRAUD DETECTION REPORT
{'='*100}

EXECUTIVE SUMMARY:
  Analysis Date: {summary['analysis_timestamp']}
  Total Transactions Analyzed: {summary['total_transactions']:,}
  Overall Fraud Score: {summary['overall_fraud_score']:.1f}/100
  Risk Level: {summary['risk_level']}
  
  Analysis Coverage: {stats['analysis_coverage']['coverage_rate']:.1f}% ({stats['analysis_coverage']['methods_completed']}/{stats['analysis_coverage']['total_methods']} methods)
  
  Key Findings:
  • Total Outliers Detected: {stats['key_statistics']['total_outliers_detected']:,}
  • Total Anomalies Detected: {stats['key_statistics']['total_anomalies_detected']:,}
  • Highest Risk Method: {stats['key_statistics']['highest_risk_method'].title()}
  • Transactions Flagged for Review: {len(flagged):,}

{'='*100}
DETAILED ANALYSIS RESULTS
{'='*100}
"""
        
        # Add detailed results for each method
        for method_name, method_result in detailed.items():
            if 'error' not in method_result:
                report += f"\n{method_name.upper().replace('_', ' ')}:\n"
                report += self._format_method_results(method_name, method_result)
                report += "\n" + "-" * 80 + "\n"
        
        # Add recommendations
        report += f"""
{'='*100}
RECOMMENDATIONS
{'='*100}

Priority Actions ({len(recommendations)} total):
"""
        
        for i, rec in enumerate(recommendations, 1):
            priority = "HIGH" if i <= 3 else "MEDIUM" if i <= 6 else "LOW"
            report += f"{i:2d}. [{priority:6s}] {rec}\n"
        
        # Add flagged transactions summary
        if flagged:
            report += f"""
{'='*100}
FLAGGED TRANSACTIONS SUMMARY
{'='*100}

Top {min(10, len(flagged))} highest-risk transactions requiring immediate review:

"""
            for i, transaction in enumerate(flagged[:10], 1):
                tx_data = transaction['transaction_data']
                reasons = transaction['flag_reasons']
                
                report += f"{i:2d}. Transaction Index {transaction['index']}:\n"
                if 'amount' in tx_data:
                    report += f"    Amount: ${tx_data['amount']:,.2f}\n"
                if 'vendor' in tx_data:
                    report += f"    Vendor: {tx_data['vendor']}\n"
                if 'date' in tx_data:
                    report += f"    Date: {tx_data['date']}\n"
                report += f"    Flag Reasons: {', '.join(reasons)}\n\n"
        
        report += f"""
{'='*100}
STATISTICAL SUMMARY
{'='*100}

Data Quality:
  Completeness Rate: {stats['data_quality'].get('completeness_rate', 0):.1f}%

Amount Statistics:
  Range: ${stats['data_quality']['amount_range']['min']:,.2f} to ${stats['data_quality']['amount_range']['max']:,.2f}
  Mean: ${stats['data_quality']['amount_range']['mean']:,.2f}
  Median: ${stats['data_quality']['amount_range']['median']:,.2f}
  Standard Deviation: ${stats['data_quality']['amount_range']['std']:,.2f}

Analysis Performance:
  Methods Successfully Completed: {stats['analysis_coverage']['methods_completed']}/{stats['analysis_coverage']['total_methods']}
  Overall Analysis Coverage: {stats['analysis_coverage']['coverage_rate']:.1f}%

{'='*100}
END OF REPORT
{'='*100}
"""
        
        return report
    
    def _format_method_results(self, method_name: str, results: Dict[str, Any]) -> str:
        """Format results for a specific analysis method"""
        formatted = ""
        
        if method_name == 'outlier_detection':
            summary = results.get('summary', {})
            formatted = f"  • Total Unique Outliers: {summary.get('total_unique_outliers', 0)}\n"
            formatted += f"  • Outlier Rate: {summary.get('outlier_rate', 0):.2f}%\n"
            formatted += f"  • Methods in Agreement: {summary.get('methods_agreeing', 0)}\n"
            
        elif method_name == 'digit_analysis':
            fraud_indicators = results.get('fraud_indicators', {})
            formatted = f"  • Fraud Score: {fraud_indicators.get('fraud_score', 0)}/100\n"
            formatted += f"  • Indicators: {', '.join(fraud_indicators.get('indicators', []))}\n"
            
            first_digit = results.get('first_digit_analysis', {})
            if 'follows_benford' in first_digit:
                formatted += f"  • Follows Benford's Law: {'Yes' if first_digit['follows_benford'] else 'No'}\n"
                formatted += f"  • Chi-square p-value: {first_digit.get('p_value', 0):.4f}\n"
        
        elif method_name == 'threshold_analysis':
            for threshold_name, threshold_data in results.items():
                if isinstance(threshold_data, dict) and 'suspicion_score' in threshold_data:
                    formatted += f"  • {threshold_name}: Suspicion Score {threshold_data['suspicion_score']:.1f}\n"
                    if threshold_data.get('exact_matches', 0) > 0:
                        formatted += f"    - Exact matches: {threshold_data['exact_matches']}\n"
        
        elif method_name == 'duplicates_analysis':
            exact_dups = results.get('exact_duplicates', {})
            if 'count' in exact_dups:
                formatted += f"  • Exact Duplicates: {exact_dups['count']} ({exact_dups.get('rate', 0):.1f}%)\n"
            
            amount_dups = results.get('amount_duplicates', {})
            if 'count' in amount_dups:
                formatted += f"  • Amount Duplicates: {amount_dups['count']} ({amount_dups.get('rate', 0):.1f}%)\n"
        
        # Add more formatting for other methods as needed
        
        return formatted if formatted else "  • Analysis completed (see detailed results)\n"

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    sample_data = []
    for i in range(1000):
        # Most transactions are legitimate
        if np.random.random() < 0.9:
            amount = np.random.exponential(1000)
            vendor = np.random.choice(['ABC Corp', 'XYZ Ltd', 'Supply Co', 'Tech Inc', 'Service Group'])
            date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        else:
            # Some fraudulent patterns
            if np.random.random() < 0.5:
                amount = np.random.choice([9999, 4999, 5000, 10000])  # Threshold gaming
            else:
                amount = np.random.uniform(100, 50000)
            vendor = f"Vendor {np.random.randint(1, 100)}"
            date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        
        sample_data.append({
            'transaction_id': f"TXN{i+1:04d}",
            'amount': amount,
            'payee_payer': vendor,
            'date': date,
            'account': np.random.choice(['Expense', 'Revenue', 'Asset'])
        })
    
    sample_df = pd.DataFrame(sample_data)
    
    # Run comprehensive analysis
    detector = StatisticalFraudDetector()
    print("Running comprehensive statistical fraud analysis...")
    
    results = detector.comprehensive_fraud_analysis(sample_df)
    
    print(f"\nAnalysis completed!")
    print(f"Overall Fraud Score: {results['summary']['overall_fraud_score']:.1f}/100")
    print(f"Risk Level: {results['summary']['risk_level']}")
    print(f"Transactions Flagged: {len(results['flagged_transactions'])}")
    print(f"Recommendations: {len(results['recommendations'])}")
    
    # Generate and save report
    report = detector.generate_comprehensive_report(results)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"fraud_analysis_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_filename}")
    print("\nFirst 2000 characters of report:")
    print(report[:2000])
    print("\n... [Report continues]")