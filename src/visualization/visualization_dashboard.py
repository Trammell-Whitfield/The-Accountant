"""
Comprehensive Visualization Dashboard for Forensic Accounting Analysis
Creates interactive and static visualizations for all analysis results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class ForensicVisualizationDashboard:
    """
    Comprehensive visualization system for forensic accounting analysis
    """
    
    def __init__(self):
        """Initialize the visualization dashboard"""
        # Set visualization style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color schemes
        self.color_schemes = {
            'risk_levels': {
                'MINIMAL': '#2ecc71',     # Green
                'LOW': '#f39c12',         # Orange
                'MEDIUM': '#e74c3c',      # Red
                'HIGH': '#c0392b',        # Dark Red
                'CRITICAL': '#8e44ad'     # Purple
            },
            'analysis_methods': {
                'benford': '#3498db',      # Blue
                'statistical': '#e67e22', # Orange
                'rag': '#9b59b6',         # Purple
                'integrated': '#1abc9c'   # Teal
            },
            'fraud_indicators': {
                'compliant': '#27ae60',   # Green
                'suspicious': '#f39c12',  # Orange
                'fraudulent': '#e74c3c'   # Red
            }
        }
        
        self.visualization_paths = []
    
    def create_comprehensive_dashboard(self, analysis_results: Dict[str, Any], 
                                     output_dir: str = "visualizations",
                                     create_interactive: bool = True) -> Dict[str, List[str]]:
        """
        Create comprehensive visualization dashboard
        
        Args:
            analysis_results: Results from integrated forensic analysis
            output_dir: Directory to save visualizations
            create_interactive: Whether to create interactive plots
            
        Returns:
            Dictionary mapping visualization types to file paths
        """
        print("Creating comprehensive visualization dashboard...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        visualization_files = {
            'static_plots': [],
            'interactive_plots': [],
            'summary_dashboard': []
        }
        
        # 1. Executive Summary Dashboard
        print("  Creating executive summary dashboard...")
        try:
            exec_files = self._create_executive_summary_dashboard(
                analysis_results, output_dir, create_interactive
            )
            visualization_files['summary_dashboard'].extend(exec_files)
        except Exception as e:
            print(f"    Error creating executive summary: {str(e)}")
        
        # 2. Risk Assessment Visualizations
        print("  Creating risk assessment visualizations...")
        try:
            risk_files = self._create_risk_assessment_plots(
                analysis_results, output_dir, create_interactive
            )
            visualization_files['static_plots'].extend(risk_files)
        except Exception as e:
            print(f"    Error creating risk assessments: {str(e)}")
        
        # 3. Benford's Law Visualizations
        print("  Creating Benford's Law visualizations...")
        try:
            benford_files = self._create_benford_visualizations(
                analysis_results, output_dir, create_interactive
            )
            visualization_files['static_plots'].extend(benford_files)
            if create_interactive:
                benford_interactive = self._create_benford_interactive(
                    analysis_results, output_dir
                )
                visualization_files['interactive_plots'].extend(benford_interactive)
        except Exception as e:
            print(f"    Error creating Benford's Law plots: {str(e)}")
        
        # 4. Statistical Analysis Visualizations
        print("  Creating statistical analysis visualizations...")
        try:
            stats_files = self._create_statistical_visualizations(
                analysis_results, output_dir, create_interactive
            )
            visualization_files['static_plots'].extend(stats_files)
        except Exception as e:
            print(f"    Error creating statistical plots: {str(e)}")
        
        # 5. Time Series and Pattern Analysis
        print("  Creating time series and pattern visualizations...")
        try:
            pattern_files = self._create_pattern_visualizations(
                analysis_results, output_dir, create_interactive
            )
            visualization_files['static_plots'].extend(pattern_files)
        except Exception as e:
            print(f"    Error creating pattern plots: {str(e)}")
        
        # 6. Transaction Analysis Visualizations
        print("  Creating transaction analysis visualizations...")
        try:
            transaction_files = self._create_transaction_visualizations(
                analysis_results, output_dir
            )
            visualization_files['static_plots'].extend(transaction_files)
        except Exception as e:
            print(f"    Error creating transaction plots: {str(e)}")
        
        # 7. Interactive Comprehensive Dashboard
        if create_interactive:
            print("  Creating interactive comprehensive dashboard...")
            try:
                interactive_dashboard = self._create_interactive_dashboard(
                    analysis_results, output_dir
                )
                visualization_files['interactive_plots'].append(interactive_dashboard)
            except Exception as e:
                print(f"    Error creating interactive dashboard: {str(e)}")
        
        print(f"✓ Visualization dashboard created in '{output_dir}' directory")
        
        # Print summary
        total_files = sum(len(files) for files in visualization_files.values())
        print(f"  Total files created: {total_files}")
        for category, files in visualization_files.items():
            if files:
                print(f"    {category}: {len(files)} files")
        
        return visualization_files
    
    def _create_executive_summary_dashboard(self, analysis_results: Dict[str, Any], 
                                          output_dir: str, create_interactive: bool = True) -> List[str]:
        """Create executive summary dashboard"""
        files_created = []
        
        # Extract data
        executive = analysis_results.get('executive_summary', {})
        integrated = analysis_results.get('integrated_assessment', {})
        
        # Static summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Forensic Analysis Executive Summary', fontsize=16, fontweight='bold')
        
        # 1. Overall Risk Gauge
        overall_score = executive.get('overall_assessment', {}).get('fraud_score', 0)
        risk_level = executive.get('overall_assessment', {}).get('risk_level', 'UNKNOWN')
        
        # Create risk gauge
        angles = np.linspace(0, np.pi, 100)
        risk_color = self.color_schemes['risk_levels'].get(risk_level, '#gray')
        
        ax1.fill_between(angles, 0, 1, alpha=0.3, color='lightgray')
        score_angle = overall_score / 100 * np.pi
        ax1.fill_between(angles[angles <= score_angle], 0, 1, alpha=0.7, color=risk_color)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, np.pi)
        ax1.set_title(f'Overall Fraud Score: {overall_score:.1f}/100\nRisk Level: {risk_level}', 
                     fontweight='bold')
        ax1.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax1.set_xticklabels(['0', '25', '50', '75', '100'])
        ax1.set_ylabel('Risk Level')
        
        # 2. Method Performance Comparison
        method_performance = executive.get('method_performance', {})
        if method_performance:
            methods = list(method_performance.keys())
            scores = [method_performance[method].get('fraud_score', 0) for method in methods]
            method_names = [method.replace('_', ' ').title() for method in methods]
            
            colors = [self.color_schemes['analysis_methods'].get(method.split('_')[0], '#gray') 
                     for method in methods]
            
            bars = ax2.bar(method_names, scores, color=colors, alpha=0.7)
            ax2.set_title('Analysis Method Performance', fontweight='bold')
            ax2.set_ylabel('Fraud Score')
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax2.annotate(f'{score:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom')
        
        # 3. Key Findings Summary
        key_findings = executive.get('key_findings', [])
        critical_alerts = executive.get('critical_alerts', [])
        
        findings_data = {
            'Critical Alerts': len(critical_alerts),
            'Key Findings': len(key_findings),
            'Total Issues': len(critical_alerts) + len(key_findings)
        }
        
        if findings_data['Total Issues'] > 0:
            labels = list(findings_data.keys())[:2]  # Exclude 'Total Issues'
            values = [findings_data[label] for label in labels]
            colors = ['#e74c3c', '#f39c12']
            
            ax3.pie(values, labels=labels, colors=colors, autopct='%1.0f', startangle=90)
            ax3.set_title('Issues Identified', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Critical Issues\nDetected', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=14, color='green', fontweight='bold')
            ax3.set_title('Issues Identified', fontweight='bold')
        
        # 4. Confidence and Data Quality
        confidence = executive.get('overall_assessment', {}).get('confidence', 'UNKNOWN')
        data_quality = executive.get('data_quality_assessment', 'UNKNOWN')
        transactions = executive.get('overall_assessment', {}).get('total_transactions_analyzed', 0)
        
        quality_data = {
            'Confidence': {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(confidence, 0),
            'Data Quality': {'GOOD': 3, 'FAIR': 2, 'POOR': 1}.get(data_quality, 0),
        }
        
        categories = list(quality_data.keys())
        values = list(quality_data.values())
        
        ax4.barh(categories, values, color=['#2ecc71', '#3498db'])
        ax4.set_xlim(0, 3)
        ax4.set_title(f'Analysis Quality\n({transactions:,} transactions)', fontweight='bold')
        ax4.set_xticks([1, 2, 3])
        ax4.set_xticklabels(['Low', 'Medium', 'High'])
        
        plt.tight_layout()
        
        # Save static summary
        summary_path = f"{output_dir}/executive_summary_dashboard.png"
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        files_created.append(summary_path)
        
        return files_created
    
    def _create_risk_assessment_plots(self, analysis_results: Dict[str, Any], 
                                    output_dir: str, create_interactive: bool = True) -> List[str]:
        """Create risk assessment visualizations"""
        files_created = []
        
        integrated = analysis_results.get('integrated_assessment', {})
        
        # Risk Score Comparison Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Risk Assessment Analysis', fontsize=14, fontweight='bold')
        
        # 1. Method Risk Scores
        method_scores = integrated.get('method_scores', {})
        if method_scores:
            methods = list(method_scores.keys())
            scores = list(method_scores.values())
            method_names = [method.replace('_analysis', '').replace('_', ' ').title() for method in methods]
            
            colors = [self.color_schemes['analysis_methods'].get(method.split('_')[0], '#gray') 
                     for method in methods]
            
            bars = ax1.barh(method_names, scores, color=colors, alpha=0.7)
            ax1.set_title('Risk Scores by Analysis Method')
            ax1.set_xlabel('Risk Score (0-100)')
            ax1.set_xlim(0, 100)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                width = bar.get_width()
                ax1.annotate(f'{score:.1f}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(5, 0), textcoords="offset points",
                           ha='left', va='center')
        
        # 2. Risk Level Distribution
        overall_risk = integrated.get('overall_fraud_score', 0)
        risk_level = integrated.get('overall_risk_level', 'MINIMAL')
        
        # Risk level thresholds
        risk_thresholds = {
            'MINIMAL': (0, 20),
            'LOW': (20, 40),
            'MEDIUM': (40, 60),
            'HIGH': (60, 80),
            'CRITICAL': (80, 100)
        }
        
        # Create risk level visualization
        levels = list(risk_thresholds.keys())
        colors = [self.color_schemes['risk_levels'][level] for level in levels]
        
        # Create segments
        for i, (level, (start, end)) in enumerate(risk_thresholds.items()):
            alpha = 0.8 if level == risk_level else 0.3
            ax2.barh(0, end - start, left=start, color=colors[i], alpha=alpha, 
                    label=level if level == risk_level else None)
        
        # Add current score marker
        ax2.axvline(x=overall_risk, color='black', linewidth=3, label=f'Current Score: {overall_risk:.1f}')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('Risk Score')
        ax2.set_title(f'Risk Level: {risk_level}')
        ax2.set_yticks([])
        ax2.legend()
        
        # Add risk level labels
        for level, (start, end) in risk_thresholds.items():
            ax2.text((start + end) / 2, 0, level, ha='center', va='center', 
                    fontweight='bold', color='white' if level == risk_level else 'black')
        
        plt.tight_layout()
        
        risk_path = f"{output_dir}/risk_assessment_analysis.png"
        plt.savefig(risk_path, dpi=300, bbox_inches='tight')
        plt.close()
        files_created.append(risk_path)
        
        return files_created
    
    def _create_benford_visualizations(self, analysis_results: Dict[str, Any], 
                                     output_dir: str, create_interactive: bool = True) -> List[str]:
        """Create Benford's Law visualizations"""
        files_created = []
        
        individual = analysis_results.get('individual_analyses', {})
        benford_results = individual.get('benford_analysis', {})
        
        if 'error' in benford_results:
            return files_created
        
        # Extract Benford's data
        freq_analysis = benford_results.get('frequency_analysis', {})
        observed = freq_analysis.get('observed_frequencies', {})
        expected = freq_analysis.get('expected_frequencies', {})
        deviations = freq_analysis.get('deviations', {})
        
        if not observed or not expected:
            return files_created
        
        # Create comprehensive Benford's visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        
        # 1. Main frequency comparison
        ax1 = fig.add_subplot(gs[0, :2])
        digits = list(range(1, 10))
        observed_vals = [observed.get(d, 0) for d in digits]
        expected_vals = [expected.get(d, 0) for d in digits]
        
        x = np.arange(len(digits))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, observed_vals, width, label='Observed', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, expected_vals, width, label='Expected (Benford)', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Leading Digit')
        ax1.set_ylabel('Frequency')
        ax1.set_title("Benford's Law Analysis: Observed vs Expected Frequencies")
        ax1.set_xticks(x)
        ax1.set_xticklabels(digits)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        # 2. Deviations plot
        ax2 = fig.add_subplot(gs[0, 2])
        deviation_vals = [deviations.get(d, 0) for d in digits]
        colors = ['red' if abs(dev) > 0.02 else 'orange' if abs(dev) > 0.01 else 'green' 
                 for dev in deviation_vals]
        
        bars = ax2.bar(digits, deviation_vals, color=colors, alpha=0.7)
        ax2.set_xlabel('Leading Digit')
        ax2.set_ylabel('Deviation')
        ax2.set_title('Deviations from Expected')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistical test results
        ax3 = fig.add_subplot(gs[1, 0])
        statistical_tests = benford_results.get('statistical_tests', {})
        chi_square = statistical_tests.get('chi_square', {})
        
        test_results = {
            'Chi-Square\nStatistic': chi_square.get('chi_square_statistic', 0),
            'P-Value': chi_square.get('p_value', 0),
            'Critical Value\n(95%)': chi_square.get('critical_value_05', 0)
        }
        
        bars = ax3.bar(test_results.keys(), test_results.values(), 
                      color=['blue', 'green', 'red'], alpha=0.7)
        ax3.set_title('Statistical Test Results')
        ax3.set_ylabel('Value')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, value in zip(bars, test_results.values()):
            height = bar.get_height()
            ax3.annotate(f'{value:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        # 4. Z-scores by digit
        ax4 = fig.add_subplot(gs[1, 1])
        z_tests = statistical_tests.get('z_tests', {})
        z_scores = [z_tests.get(d, {}).get('z_score', 0) for d in digits]
        z_colors = ['red' if abs(z) > 1.96 else 'orange' if abs(z) > 1.0 else 'green' 
                   for z in z_scores]
        
        ax4.bar(digits, z_scores, color=z_colors, alpha=0.7)
        ax4.set_xlabel('Leading Digit')
        ax4.set_ylabel('Z-Score')
        ax4.set_title('Z-Scores by Digit')
        ax4.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, label='95% Confidence')
        ax4.axhline(y=-1.96, color='red', linestyle='--', alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Quality metrics
        ax5 = fig.add_subplot(gs[1, 2])
        quality_metrics = benford_results.get('quality_metrics', {})
        
        metrics = {
            'Compliance\nScore': quality_metrics.get('compliance_score', 0) / 100,
            'Data Quality\nGrade': {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4, 'F': 0.2}.get(
                quality_metrics.get('data_quality_grade', 'F'), 0.2),
            'Chi-Square\np-value': min(quality_metrics.get('chi_square_p_value', 0), 1.0)
        }
        
        bars = ax5.bar(metrics.keys(), metrics.values(), 
                      color=['green', 'blue', 'purple'], alpha=0.7)
        ax5.set_ylim(0, 1)
        ax5.set_ylabel('Score (0-1)')
        ax5.set_title('Quality Metrics')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Summary text
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        fraud_analysis = benford_results.get('fraud_analysis', {})
        compliance = benford_results.get('compliance_assessment', {})
        
        summary_text = f"""
BENFORD'S LAW ANALYSIS SUMMARY:
• Sample Size: {benford_results.get('dataset_info', {}).get('sample_size', 0):,} transactions
• Fraud Score: {fraud_analysis.get('fraud_score', 0)}/100
• Compliance: {'PASS' if compliance.get('follows_benfords_law', False) else 'FAIL'}
• Risk Level: {compliance.get('fraud_risk_level', 'UNKNOWN')}
• Round Number Bias: {'DETECTED' if fraud_analysis.get('round_number_bias', False) else 'Not Detected'}
• Digit Avoidance: {fraud_analysis.get('digit_avoidance', 'None')}
• Overall Assessment: {compliance.get('overall_assessment', 'No assessment available')}
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.suptitle("Comprehensive Benford's Law Analysis", fontsize=16, fontweight='bold', y=0.98)
        
        benford_path = f"{output_dir}/benfords_law_comprehensive.png"
        plt.savefig(benford_path, dpi=300, bbox_inches='tight')
        plt.close()
        files_created.append(benford_path)
        
        return files_created
    
    def _create_benford_interactive(self, analysis_results: Dict[str, Any], 
                                   output_dir: str) -> List[str]:
        """Create interactive Benford's Law visualization"""
        files_created = []
        
        individual = analysis_results.get('individual_analyses', {})
        benford_results = individual.get('benford_analysis', {})
        
        if 'error' in benford_results:
            return files_created
        
        # Extract data
        freq_analysis = benford_results.get('frequency_analysis', {})
        observed = freq_analysis.get('observed_frequencies', {})
        expected = freq_analysis.get('expected_frequencies', {})
        
        if not observed or not expected:
            return files_created
        
        # Create interactive plot
        digits = list(range(1, 10))
        observed_vals = [observed.get(d, 0) for d in digits]
        expected_vals = [expected.get(d, 0) for d in digits]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Frequency Comparison', 'Deviations', 'Cumulative Distribution', 'Z-Scores'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Frequency comparison
        fig.add_trace(
            go.Bar(name='Observed', x=digits, y=observed_vals, marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Expected', x=digits, y=expected_vals, marker_color='lightcoral'),
            row=1, col=1
        )
        
        # 2. Deviations
        deviations = freq_analysis.get('deviations', {})
        deviation_vals = [deviations.get(d, 0) for d in digits]
        
        fig.add_trace(
            go.Bar(name='Deviations', x=digits, y=deviation_vals, 
                  marker_color=['red' if abs(d) > 0.02 else 'orange' if abs(d) > 0.01 else 'green' 
                               for d in deviation_vals]),
            row=1, col=2
        )
        
        # 3. Cumulative distribution
        observed_cum = np.cumsum(observed_vals)
        expected_cum = np.cumsum(expected_vals)
        
        fig.add_trace(
            go.Scatter(name='Observed Cumulative', x=digits, y=observed_cum, mode='lines+markers'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(name='Expected Cumulative', x=digits, y=expected_cum, mode='lines+markers'),
            row=2, col=1
        )
        
        # 4. Z-scores
        statistical_tests = benford_results.get('statistical_tests', {})
        z_tests = statistical_tests.get('z_tests', {})
        z_scores = [z_tests.get(d, {}).get('z_score', 0) for d in digits]
        
        fig.add_trace(
            go.Bar(name='Z-Scores', x=digits, y=z_scores,
                  marker_color=['red' if abs(z) > 1.96 else 'orange' if abs(z) > 1.0 else 'green' 
                               for z in z_scores]),
            row=2, col=2
        )
        
        # Add horizontal lines for Z-score significance
        fig.add_hline(y=1.96, line_dash="dash", line_color="red", row=2, col=2)
        fig.add_hline(y=-1.96, line_dash="dash", line_color="red", row=2, col=2)
        
        fig.update_layout(
            title_text="Interactive Benford's Law Analysis",
            showlegend=True,
            height=800
        )
        
        # Save interactive plot
        interactive_path = f"{output_dir}/benfords_law_interactive.html"
        fig.write_html(interactive_path)
        files_created.append(interactive_path)
        
        return files_created
    
    def _create_statistical_visualizations(self, analysis_results: Dict[str, Any], 
                                         output_dir: str, create_interactive: bool = True) -> List[str]:
        """Create statistical analysis visualizations"""
        files_created = []
        
        individual = analysis_results.get('individual_analyses', {})
        statistical_results = individual.get('statistical_analysis', {})
        
        if 'error' in statistical_results or not statistical_results:
            return files_created
        
        detailed_results = statistical_results.get('detailed_results', {})
        
        # Create statistical analysis dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Fraud Detection Analysis', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        plot_idx = 0
        
        # 1. Outlier Detection Results
        if 'outlier_detection' in detailed_results:
            outlier_data = detailed_results['outlier_detection']
            summary = outlier_data.get('summary', {})
            
            if 'total_unique_outliers' in summary:
                ax = axes[plot_idx]
                plot_idx += 1
                
                methods = []
                outlier_counts = []
                
                for method, method_data in outlier_data.get('methods', {}).items():
                    if isinstance(method_data, dict) and 'outliers_count' in method_data:
                        methods.append(method.replace('_', ' ').title())
                        outlier_counts.append(method_data['outliers_count'])
                
                if methods:
                    bars = ax.bar(methods, outlier_counts, color='orange', alpha=0.7)
                    ax.set_title('Outliers Detected by Method')
                    ax.set_ylabel('Number of Outliers')
                    ax.tick_params(axis='x', rotation=45)
                    
                    for bar, count in zip(bars, outlier_counts):
                        height = bar.get_height()
                        ax.annotate(f'{count}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha='center', va='bottom')
        
        # 2. Digit Analysis Results
        if 'digit_analysis' in detailed_results:
            digit_data = detailed_results['digit_analysis']
            fraud_indicators = digit_data.get('fraud_indicators', {})
            
            if fraud_indicators and plot_idx < len(axes):
                ax = axes[plot_idx]
                plot_idx += 1
                
                # Create fraud score gauge
                fraud_score = fraud_indicators.get('fraud_score', 0)
                
                # Simple fraud score visualization
                categories = ['Fraud Score', 'Compliance Score']
                values = [fraud_score, 100 - fraud_score]
                colors = ['red', 'green']
                
                wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors, 
                                                autopct='%1.1f%%', startangle=90)
                ax.set_title(f'Digit Analysis Results\nFraud Score: {fraud_score}/100')
        
        # 3. Threshold Analysis
        if 'threshold_analysis' in detailed_results:
            threshold_data = detailed_results['threshold_analysis']
            
            if threshold_data and plot_idx < len(axes):
                ax = axes[plot_idx]
                plot_idx += 1
                
                threshold_names = []
                suspicion_scores = []
                
                for threshold_name, threshold_info in threshold_data.items():
                    if isinstance(threshold_info, dict) and 'suspicion_score' in threshold_info:
                        threshold_names.append(threshold_name.replace('_', ' ').title())
                        suspicion_scores.append(threshold_info['suspicion_score'])
                
                if threshold_names:
                    bars = ax.barh(threshold_names, suspicion_scores, color='red', alpha=0.7)
                    ax.set_title('Threshold Suspicion Scores')
                    ax.set_xlabel('Suspicion Score')
                    ax.set_xlim(0, 100)
                    
                    for bar, score in zip(bars, suspicion_scores):
                        width = bar.get_width()
                        ax.annotate(f'{score:.1f}',
                                   xy=(width, bar.get_y() + bar.get_height() / 2),
                                   xytext=(3, 0), textcoords="offset points",
                                   ha='left', va='center')
        
        # 4. Duplicate Analysis
        if 'duplicates_analysis' in detailed_results:
            duplicate_data = detailed_results['duplicates_analysis']
            
            if duplicate_data and plot_idx < len(axes):
                ax = axes[plot_idx]
                plot_idx += 1
                
                duplicate_types = []
                duplicate_counts = []
                
                type_mapping = {
                    'exact_duplicates': 'Exact Duplicates',
                    'amount_duplicates': 'Amount Duplicates',
                    'same_day_vendor': 'Same Day Vendor'
                }
                
                for dup_type, display_name in type_mapping.items():
                    if dup_type in duplicate_data and isinstance(duplicate_data[dup_type], dict):
                        count = duplicate_data[dup_type].get('count', 0)
                        if count > 0:
                            duplicate_types.append(display_name)
                            duplicate_counts.append(count)
                
                if duplicate_types:
                    bars = ax.bar(duplicate_types, duplicate_counts, color='purple', alpha=0.7)
                    ax.set_title('Duplicate Transactions Detected')
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                    
                    for bar, count in zip(bars, duplicate_counts):
                        height = bar.get_height()
                        ax.annotate(f'{count}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha='center', va='bottom')
        
        # 5. Overall Risk Assessment
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            plot_idx += 1
            
            overall_score = statistical_results.get('summary', {}).get('overall_fraud_score', 0)
            risk_level = statistical_results.get('summary', {}).get('risk_level', 'LOW')
            
            # Risk level pie chart
            risk_scores = {
                'Current Risk': overall_score,
                'Remaining': 100 - overall_score
            }
            
            colors = [self.color_schemes['risk_levels'].get(risk_level, 'gray'), 'lightgray']
            
            wedges, texts, autotexts = ax.pie(risk_scores.values(), labels=risk_scores.keys(), 
                                            colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Overall Risk Assessment\nLevel: {risk_level}')
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        stats_path = f"{output_dir}/statistical_analysis_dashboard.png"
        plt.savefig(stats_path, dpi=300, bbox_inches='tight')
        plt.close()
        files_created.append(stats_path)
        
        return files_created
    
    def _create_pattern_visualizations(self, analysis_results: Dict[str, Any], 
                                     output_dir: str, create_interactive: bool = True) -> List[str]:
        """Create pattern analysis visualizations"""
        files_created = []
        
        # This would create visualizations for time series patterns,
        # vendor patterns, account patterns, etc.
        # Implementation would depend on the specific pattern data available
        
        return files_created
    
    def _create_transaction_visualizations(self, analysis_results: Dict[str, Any], 
                                         output_dir: str) -> List[str]:
        """Create transaction-specific visualizations"""
        files_created = []
        
        flagged_transactions = analysis_results.get('flagged_transactions', [])
        
        if not flagged_transactions:
            return files_created
        
        # Create flagged transactions summary
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Flagged Transactions Analysis', fontsize=14, fontweight='bold')
        
        # 1. Flagged transactions by source method
        source_counts = {}
        for transaction in flagged_transactions:
            source = transaction.get('source_method', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        if source_counts:
            methods = list(source_counts.keys())
            counts = list(source_counts.values())
            
            bars = ax1.bar(methods, counts, color='red', alpha=0.7)
            ax1.set_title('Flagged Transactions by Detection Method')
            ax1.set_ylabel('Number of Transactions')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax1.annotate(f'{count}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom')
        
        # 2. Risk factors frequency
        risk_factor_counts = {}
        for transaction in flagged_transactions:
            risk_factors = transaction.get('risk_factors', [])
            for factor in risk_factors:
                factor_str = str(factor)[:30]  # Truncate long factors
                risk_factor_counts[factor_str] = risk_factor_counts.get(factor_str, 0) + 1
        
        if risk_factor_counts:
            # Get top 10 risk factors
            sorted_factors = sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            factors = [item[0] for item in sorted_factors]
            counts = [item[1] for item in sorted_factors]
            
            bars = ax2.barh(factors, counts, color='orange', alpha=0.7)
            ax2.set_title('Top Risk Factors')
            ax2.set_xlabel('Frequency')
            
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax2.annotate(f'{count}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(3, 0), textcoords="offset points",
                           ha='left', va='center')
        
        plt.tight_layout()
        
        transactions_path = f"{output_dir}/flagged_transactions_analysis.png"
        plt.savefig(transactions_path, dpi=300, bbox_inches='tight')
        plt.close()
        files_created.append(transactions_path)
        
        return files_created
    
    def _create_interactive_dashboard(self, analysis_results: Dict[str, Any], 
                                    output_dir: str) -> str:
        """Create comprehensive interactive dashboard"""
        
        # This would create a comprehensive interactive dashboard
        # using Plotly Dash or similar framework
        # For now, create a simple HTML dashboard
        
        executive = analysis_results.get('executive_summary', {})
        overall_assessment = executive.get('overall_assessment', {})
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Forensic Analysis Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: white; border-radius: 5px; min-width: 200px; }}
        .risk-high {{ border-left: 5px solid #e74c3c; }}
        .risk-medium {{ border-left: 5px solid #f39c12; }}
        .risk-low {{ border-left: 5px solid #27ae60; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Forensic Accounting Analysis Dashboard</h1>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metric risk-{overall_assessment.get('risk_level', 'low').lower()}">
            <h3>Overall Fraud Score</h3>
            <p><strong>{overall_assessment.get('fraud_score', 0):.1f}/100</strong></p>
        </div>
        <div class="metric">
            <h3>Risk Level</h3>
            <p><strong>{overall_assessment.get('risk_level', 'UNKNOWN')}</strong></p>
        </div>
        <div class="metric">
            <h3>Confidence</h3>
            <p><strong>{overall_assessment.get('confidence', 'UNKNOWN')}</strong></p>
        </div>
        <div class="metric">
            <h3>Transactions Analyzed</h3>
            <p><strong>{overall_assessment.get('total_transactions_analyzed', 0):,}</strong></p>
        </div>
    </div>
    
    <div class="summary">
        <h2>Analysis Methods Performance</h2>
"""
        
        method_performance = executive.get('method_performance', {})
        for method, performance in method_performance.items():
            method_name = method.replace('_', ' ').title()
            score = performance.get('fraud_score', 0)
            status = performance.get('status', 'UNKNOWN')
            
            html_content += f"""
        <div class="metric">
            <h4>{method_name}</h4>
            <p>Score: <strong>{score:.1f}/100</strong></p>
            <p>Status: <strong>{status}</strong></p>
        </div>
"""
        
        html_content += """
    </div>
    
    <div class="summary">
        <h2>Key Findings</h2>
        <ul>
"""
        
        key_findings = executive.get('key_findings', [])
        for finding in key_findings:
            html_content += f"            <li>{finding}</li>\n"
        
        html_content += """
        </ul>
    </div>
    
    <div class="summary">
        <h2>Critical Alerts</h2>
        <ul>
"""
        
        critical_alerts = executive.get('critical_alerts', [])
        for alert in critical_alerts:
            html_content += f"            <li style='color: red;'><strong>{alert}</strong></li>\n"
        
        if not critical_alerts:
            html_content += "            <li style='color: green;'>No critical alerts detected</li>\n"
        
        html_content += """
        </ul>
    </div>
</body>
</html>
"""
        
        dashboard_path = f"{output_dir}/interactive_dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
        
        return dashboard_path

def main():
    """Example usage of the visualization dashboard"""
    
    # This would typically be called with real analysis results
    # For demonstration, create mock results
    mock_results = {
        'executive_summary': {
            'overall_assessment': {
                'fraud_score': 45.7,
                'risk_level': 'MEDIUM',
                'confidence': 'HIGH',
                'total_transactions_analyzed': 1500
            },
            'method_performance': {
                'benford_analysis': {'fraud_score': 52.3, 'status': 'HIGH_RISK'},
                'statistical_analysis': {'fraud_score': 38.9, 'status': 'MEDIUM_RISK'},
                'rag_analysis': {'fraud_score': 46.2, 'status': 'MEDIUM_RISK'}
            },
            'key_findings': [
                'Significant Benford\'s Law violations detected',
                'High duplicate transaction rate identified',
                'Unusual clustering around reporting thresholds'
            ],
            'critical_alerts': [
                'Multiple fraud indicators detected across methods'
            ],
            'data_quality_assessment': 'GOOD'
        },
        'integrated_assessment': {
            'overall_fraud_score': 45.7,
            'overall_risk_level': 'MEDIUM',
            'method_scores': {
                'benford_analysis': 52.3,
                'statistical_analysis': 38.9,
                'rag_analysis': 46.2
            },
            'consensus_indicators': [
                'Round number bias detected by multiple methods'
            ],
            'conflicting_indicators': []
        },
        'individual_analyses': {
            'benford_analysis': {
                'dataset_info': {'sample_size': 1500},
                'frequency_analysis': {
                    'observed_frequencies': {1: 0.25, 2: 0.18, 3: 0.13, 4: 0.10, 5: 0.12, 6: 0.07, 7: 0.06, 8: 0.05, 9: 0.04},
                    'expected_frequencies': {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046},
                    'deviations': {1: -0.051, 2: 0.004, 3: 0.005, 4: 0.003, 5: 0.041, 6: 0.003, 7: 0.002, 8: -0.001, 9: -0.006}
                },
                'statistical_tests': {
                    'chi_square': {'chi_square_statistic': 15.7, 'p_value': 0.047, 'critical_value_05': 15.51},
                    'z_tests': {d: {'z_score': np.random.normal(0, 1)} for d in range(1, 10)}
                },
                'fraud_analysis': {'fraud_score': 52.3, 'round_number_bias': True},
                'compliance_assessment': {'follows_benfords_law': False, 'fraud_risk_level': 'MEDIUM'},
                'quality_metrics': {'compliance_score': 47.7, 'data_quality_grade': 'C', 'chi_square_p_value': 0.047}
            }
        },
        'flagged_transactions': [
            {'source_method': 'Statistical Analysis', 'risk_factors': ['High amount', 'Round number']},
            {'source_method': 'Benford Analysis', 'risk_factors': ['Digit pattern anomaly']},
        ]
    }
    
    # Create visualization dashboard
    dashboard = ForensicVisualizationDashboard()
    
    print("Creating comprehensive visualization dashboard...")
    visualization_files = dashboard.create_comprehensive_dashboard(
        mock_results,
        output_dir="demo_visualizations",
        create_interactive=True
    )
    
    print("\nVisualization files created:")
    for category, files in visualization_files.items():
        if files:
            print(f"  {category}:")
            for file_path in files:
                print(f"    - {file_path}")

if __name__ == "__main__":
    main()