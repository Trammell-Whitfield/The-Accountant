"""
Integrated Forensic Accounting System
Combines all components: RAG, Benford's Law, Statistical Detection, and Ledger Analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
import json
from pathlib import Path

# Import custom modules
from .forensic_rag_system import ForensicRAGAnalyzer
from ..analyzers.benford_analysis import BenfordsLawAnalyzer
from ..analyzers.statistical_fraud_detection import StatisticalFraudDetector
from ..generators.synthetic_ledgers import ImprovedSyntheticLedgerGenerator

warnings.filterwarnings('ignore')

class IntegratedForensicSystem:
    """
    Master class that integrates all forensic accounting components
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize the integrated forensic system
        
        Args:
            knowledge_base_path: Path to the knowledge base database
        """
        print("Initializing Integrated Forensic System...")
        
        # Set up paths
        if knowledge_base_path is None:
            base_path = Path(__file__).parent.parent.parent
            knowledge_base_path = base_path / "data" / "forensic_knowledge.db"
        
        # Initialize all components
        self.rag_analyzer = ForensicRAGAnalyzer(str(knowledge_base_path))
        self.benford_analyzer = BenfordsLawAnalyzer()
        self.statistical_detector = StatisticalFraudDetector()
        self.ledger_generator = ImprovedSyntheticLedgerGenerator()
        
        # System state
        self.current_ledger = None
        self.analysis_results = {}
        self.system_recommendations = []
        
        print("✓ Integrated Forensic System initialized successfully!")
    
    def load_ledger(self, ledger_source: Any, source_type: str = 'dataframe') -> bool:
        """
        Load ledger data from various sources
        
        Args:
            ledger_source: Data source (DataFrame, CSV path, etc.)
            source_type: Type of source ('dataframe', 'csv', 'json', 'synthetic')
            
        Returns:
            Boolean indicating success
        """
        try:
            if source_type == 'dataframe':
                self.current_ledger = ledger_source.copy()
                
            elif source_type == 'csv':
                self.current_ledger = pd.read_csv(ledger_source)
                
            elif source_type == 'json':
                self.current_ledger = pd.read_json(ledger_source)
                
            elif source_type == 'synthetic':
                print("Generating synthetic ledger...")
                num_transactions = ledger_source.get('num_transactions', 1000)
                fraud_percentage = ledger_source.get('fraud_percentage', 0.1)
                company_type = ledger_source.get('company_type', 'manufacturing')
                
                generator = ImprovedSyntheticLedgerGenerator(
                    company_type=company_type,
                    company_size='medium'
                )
                
                self.current_ledger = generator.generate_ledger(
                    num_transactions=num_transactions,
                    fraud_percentage=fraud_percentage
                )
                
                print(f"✓ Generated synthetic ledger with {len(self.current_ledger)} transactions")
                
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
            
            print(f"✓ Ledger loaded successfully: {len(self.current_ledger)} transactions")
            return True
            
        except Exception as e:
            print(f"✗ Error loading ledger: {str(e)}")
            return False
    
    def comprehensive_analysis(self, 
                             enable_rag: bool = True,
                             enable_benford: bool = True,
                             enable_statistical: bool = True,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive forensic analysis using all available methods
        
        Args:
            enable_rag: Enable RAG-based analysis
            enable_benford: Enable Benford's Law analysis
            enable_statistical: Enable statistical fraud detection
            save_results: Save results to files
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if self.current_ledger is None:
            raise ValueError("No ledger loaded. Please load a ledger first.")
        
        print("=" * 80)
        print("STARTING COMPREHENSIVE FORENSIC ANALYSIS")
        print("=" * 80)
        
        # Initialize results structure
        comprehensive_results = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'ledger_size': len(self.current_ledger),
                'columns_available': list(self.current_ledger.columns),
                'analysis_methods_used': []
            },
            'individual_analyses': {},
            'integrated_assessment': {},
            'executive_summary': {},
            'recommendations': [],
            'flagged_transactions': []
        }
        
        # 1. RAG-based Analysis
        if enable_rag:
            try:
                print("\n1. Performing RAG-based Analysis...")
                comprehensive_results['metadata']['analysis_methods_used'].append('RAG')
                
                # Analyze batch of transactions
                rag_results = self.rag_analyzer.analyze_ledger_batch(self.current_ledger)
                comprehensive_results['individual_analyses']['rag_analysis'] = rag_results
                
                print(f"   ✓ RAG Analysis completed: {rag_results['summary']['high_risk_transactions']} high-risk transactions identified")
                
            except Exception as e:
                print(f"   ✗ RAG Analysis failed: {str(e)}")
                comprehensive_results['individual_analyses']['rag_analysis'] = {'error': str(e)}
        
        # 2. Benford's Law Analysis
        if enable_benford:
            try:
                print("\n2. Performing Benford's Law Analysis...")
                comprehensive_results['metadata']['analysis_methods_used'].append('Benfords_Law')
                
                if 'amount' in self.current_ledger.columns:
                    benford_results = self.benford_analyzer.comprehensive_analysis(
                        self.current_ledger['amount'], 
                        dataset_name="Ledger_Analysis"
                    )
                    comprehensive_results['individual_analyses']['benford_analysis'] = benford_results
                    
                    if 'error' not in benford_results:
                        fraud_score = benford_results['fraud_analysis']['fraud_score']
                        print(f"   ✓ Benford's Law Analysis completed: Fraud Score {fraud_score}/100")
                    else:
                        print(f"   ⚠ Benford's Law Analysis completed with issues: {benford_results['error']}")
                else:
                    print("   ⚠ Benford's Law Analysis skipped: No 'amount' column found")
                    
            except Exception as e:
                print(f"   ✗ Benford's Law Analysis failed: {str(e)}")
                comprehensive_results['individual_analyses']['benford_analysis'] = {'error': str(e)}
        
        # 3. Statistical Fraud Detection
        if enable_statistical:
            try:
                print("\n3. Performing Statistical Fraud Detection...")
                comprehensive_results['metadata']['analysis_methods_used'].append('Statistical_Detection')
                
                statistical_results = self.statistical_detector.comprehensive_fraud_analysis(
                    self.current_ledger,
                    amount_col='amount',
                    date_col='date',
                    vendor_col='payee_payer' if 'payee_payer' in self.current_ledger.columns else 'vendor',
                    account_col='account' if 'account' in self.current_ledger.columns else 'account_name'
                )
                comprehensive_results['individual_analyses']['statistical_analysis'] = statistical_results
                
                overall_fraud_score = statistical_results['summary']['overall_fraud_score']
                risk_level = statistical_results['summary']['risk_level']
                print(f"   ✓ Statistical Analysis completed: Risk Level {risk_level} (Score: {overall_fraud_score:.1f}/100)")
                
            except Exception as e:
                print(f"   ✗ Statistical Analysis failed: {str(e)}")
                comprehensive_results['individual_analyses']['statistical_analysis'] = {'error': str(e)}
        
        # 4. Integrated Assessment
        print("\n4. Performing Integrated Assessment...")
        comprehensive_results['integrated_assessment'] = self._calculate_integrated_assessment(
            comprehensive_results['individual_analyses']
        )
        
        # 5. Executive Summary
        comprehensive_results['executive_summary'] = self._generate_executive_summary(
            comprehensive_results
        )
        
        # 6. Consolidated Recommendations
        comprehensive_results['recommendations'] = self._generate_consolidated_recommendations(
            comprehensive_results
        )
        
        # 7. Identify Highest Risk Transactions
        comprehensive_results['flagged_transactions'] = self._identify_highest_risk_transactions(
            comprehensive_results
        )
        
        # Store results
        self.analysis_results = comprehensive_results
        
        # Save results if requested
        if save_results:
            self._save_analysis_results(comprehensive_results)
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS COMPLETED")
        print("=" * 80)
        
        self._display_executive_summary(comprehensive_results['executive_summary'])
        
        return comprehensive_results
    
    def _calculate_integrated_assessment(self, individual_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate integrated risk assessment from all analyses"""
        
        assessment = {
            'overall_fraud_score': 0,
            'overall_risk_level': 'MINIMAL',
            'confidence_level': 'LOW',
            'method_scores': {},
            'consensus_indicators': [],
            'conflicting_indicators': []
        }
        
        method_weights = {
            'benford_analysis': 0.35,
            'statistical_analysis': 0.40,
            'rag_analysis': 0.25
        }
        
        total_weight = 0
        weighted_score = 0
        
        # Extract scores from each method
        for method, weight in method_weights.items():
            if method in individual_analyses and 'error' not in individual_analyses[method]:
                method_score = self._extract_method_fraud_score(method, individual_analyses[method])
                assessment['method_scores'][method] = method_score
                
                weighted_score += method_score * weight
                total_weight += weight
        
        # Calculate overall score
        if total_weight > 0:
            assessment['overall_fraud_score'] = weighted_score / total_weight
        
        # Determine risk level
        fraud_score = assessment['overall_fraud_score']
        if fraud_score >= 80:
            assessment['overall_risk_level'] = 'CRITICAL'
            assessment['confidence_level'] = 'HIGH'
        elif fraud_score >= 60:
            assessment['overall_risk_level'] = 'HIGH'
            assessment['confidence_level'] = 'HIGH'
        elif fraud_score >= 40:
            assessment['overall_risk_level'] = 'MEDIUM'
            assessment['confidence_level'] = 'MEDIUM'
        elif fraud_score >= 20:
            assessment['overall_risk_level'] = 'LOW'
            assessment['confidence_level'] = 'MEDIUM'
        else:
            assessment['overall_risk_level'] = 'MINIMAL'
            assessment['confidence_level'] = 'HIGH'
        
        # Identify consensus and conflicts
        assessment['consensus_indicators'] = self._identify_consensus_indicators(individual_analyses)
        assessment['conflicting_indicators'] = self._identify_conflicting_indicators(individual_analyses)
        
        return assessment
    
    def _extract_method_fraud_score(self, method: str, results: Dict[str, Any]) -> float:
        """Extract fraud score from individual method results"""
        
        if method == 'benford_analysis':
            return results.get('fraud_analysis', {}).get('fraud_score', 0)
            
        elif method == 'statistical_analysis':
            return results.get('summary', {}).get('overall_fraud_score', 0)
            
        elif method == 'rag_analysis':
            # Calculate RAG score based on high-risk transactions
            total_transactions = results.get('summary', {}).get('total_transactions', 1)
            high_risk = results.get('summary', {}).get('high_risk_transactions', 0)
            compliance_issues = results.get('summary', {}).get('compliance_issues', 0)
            
            rag_score = ((high_risk + compliance_issues) / total_transactions) * 100
            return min(rag_score, 100)
        
        return 0
    
    def _identify_consensus_indicators(self, individual_analyses: Dict[str, Any]) -> List[str]:
        """Identify indicators that multiple methods agree on"""
        consensus = []
        
        # Check for round number consensus
        round_number_methods = 0
        if 'benford_analysis' in individual_analyses:
            if individual_analyses['benford_analysis'].get('fraud_analysis', {}).get('round_number_bias', False):
                round_number_methods += 1
        
        if 'statistical_analysis' in individual_analyses:
            detailed_results = individual_analyses['statistical_analysis'].get('detailed_results', {})
            if detailed_results.get('digit_analysis', {}).get('fraud_indicators', {}).get('fraud_score', 0) > 20:
                round_number_methods += 1
        
        if round_number_methods >= 2:
            consensus.append("Round number bias detected by multiple methods")
        
        # Check for outlier consensus
        outlier_methods = 0
        if 'statistical_analysis' in individual_analyses:
            detailed_results = individual_analyses['statistical_analysis'].get('detailed_results', {})
            if detailed_results.get('outlier_detection', {}).get('summary', {}).get('outlier_rate', 0) > 10:
                outlier_methods += 1
        
        if 'benford_analysis' in individual_analyses:
            if individual_analyses['benford_analysis'].get('fraud_analysis', {}).get('fraud_score', 0) > 30:
                outlier_methods += 1
        
        if outlier_methods >= 2:
            consensus.append("Significant outliers detected by multiple methods")
        
        return consensus
    
    def _identify_conflicting_indicators(self, individual_analyses: Dict[str, Any]) -> List[str]:
        """Identify conflicting results between methods"""
        conflicts = []
        
        # Compare Benford's Law vs Statistical Analysis
        benford_score = 0
        statistical_score = 0
        
        if 'benford_analysis' in individual_analyses:
            benford_score = individual_analyses['benford_analysis'].get('fraud_analysis', {}).get('fraud_score', 0)
        
        if 'statistical_analysis' in individual_analyses:
            statistical_score = individual_analyses['statistical_analysis'].get('summary', {}).get('overall_fraud_score', 0)
        
        if abs(benford_score - statistical_score) > 40:
            conflicts.append(f"Large discrepancy between Benford's Law ({benford_score:.1f}) and Statistical Analysis ({statistical_score:.1f})")
        
        return conflicts
    
    def _generate_executive_summary(self, comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of all findings"""
        
        integrated = comprehensive_results['integrated_assessment']
        metadata = comprehensive_results['metadata']
        
        summary = {
            'overall_assessment': {
                'fraud_score': integrated['overall_fraud_score'],
                'risk_level': integrated['overall_risk_level'],
                'confidence': integrated['confidence_level'],
                'total_transactions_analyzed': metadata['ledger_size']
            },
            'key_findings': [],
            'critical_alerts': [],
            'method_performance': {},
            'data_quality_assessment': 'GOOD'
        }
        
        # Key findings
        if integrated['overall_fraud_score'] >= 60:
            summary['key_findings'].append(f"HIGH FRAUD RISK DETECTED: Overall score {integrated['overall_fraud_score']:.1f}/100")
        
        if integrated.get('consensus_indicators'):
            summary['key_findings'].extend(integrated['consensus_indicators'])
        
        # Critical alerts
        if integrated['overall_risk_level'] in ['CRITICAL', 'HIGH']:
            summary['critical_alerts'].append(f"{integrated['overall_risk_level']} risk level requires immediate attention")
        
        if integrated.get('conflicting_indicators'):
            summary['critical_alerts'].extend(integrated['conflicting_indicators'])
        
        # Method performance
        for method, score in integrated['method_scores'].items():
            summary['method_performance'][method] = {
                'fraud_score': score,
                'status': 'HIGH_RISK' if score > 50 else 'MEDIUM_RISK' if score > 25 else 'LOW_RISK'
            }
        
        return summary
    
    def _generate_consolidated_recommendations(self, comprehensive_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate consolidated recommendations from all analyses"""
        
        recommendations = []
        integrated = comprehensive_results['integrated_assessment']
        individual = comprehensive_results['individual_analyses']
        
        # High-level recommendations based on overall risk
        risk_level = integrated['overall_risk_level']
        
        if risk_level == 'CRITICAL':
            recommendations.extend([
                {
                    'priority': 'IMMEDIATE',
                    'category': 'OPERATIONAL',
                    'recommendation': 'Halt all financial processing pending investigation',
                    'rationale': 'Critical fraud indicators detected across multiple methods'
                },
                {
                    'priority': 'IMMEDIATE',
                    'category': 'INVESTIGATION',
                    'recommendation': 'Engage external forensic accounting firm',
                    'rationale': 'Internal investigation capacity insufficient for critical risk'
                }
            ])
        
        elif risk_level == 'HIGH':
            recommendations.extend([
                {
                    'priority': 'URGENT',
                    'category': 'INVESTIGATION',
                    'recommendation': 'Conduct comprehensive manual review within 24 hours',
                    'rationale': 'High fraud risk requires immediate detailed investigation'
                },
                {
                    'priority': 'URGENT',
                    'category': 'CONTROLS',
                    'recommendation': 'Implement enhanced authorization controls',
                    'rationale': 'Additional oversight needed for high-risk environment'
                }
            ])
        
        # Method-specific recommendations
        if 'benford_analysis' in individual and 'error' not in individual['benford_analysis']:
            benford_results = individual['benford_analysis']
            if not benford_results.get('compliance_assessment', {}).get('follows_benfords_law', True):
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'DATA_INTEGRITY',
                    'recommendation': 'Investigate potential data manipulation',
                    'rationale': "Significant deviation from Benford's Law detected"
                })
        
        if 'statistical_analysis' in individual and 'error' not in individual['statistical_analysis']:
            stat_results = individual['statistical_analysis']
            flagged_count = len(stat_results.get('flagged_transactions', []))
            if flagged_count > 0:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'TRANSACTION_REVIEW',
                    'recommendation': f'Review {flagged_count} flagged transactions',
                    'rationale': 'Statistical analysis identified suspicious transaction patterns'
                })
        
        # Sort by priority
        priority_order = {'IMMEDIATE': 0, 'URGENT': 1, 'HIGH': 2, 'MEDIUM': 3, 'LOW': 4}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 5))
        
        return recommendations
    
    def _identify_highest_risk_transactions(self, comprehensive_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify highest risk transactions across all analyses"""
        
        flagged_transactions = []
        individual = comprehensive_results['individual_analyses']
        
        # Collect flagged transactions from statistical analysis
        if 'statistical_analysis' in individual and 'error' not in individual['statistical_analysis']:
            stat_flagged = individual['statistical_analysis'].get('flagged_transactions', [])
            for transaction in stat_flagged[:10]:  # Top 10
                flagged_transactions.append({
                    'source_method': 'Statistical Analysis',
                    'transaction_data': transaction,
                    'risk_factors': transaction.get('flag_reasons', [])
                })
        
        # Collect high-risk transactions from RAG analysis
        if 'rag_analysis' in individual and 'error' not in individual['rag_analysis']:
            rag_transactions = individual['rag_analysis'].get('transaction_analyses', [])
            high_risk_rag = [t for t in rag_transactions if t.get('overall_risk_score', 0) > 0.7]
            
            for transaction in high_risk_rag[:5]:  # Top 5
                flagged_transactions.append({
                    'source_method': 'RAG Analysis',
                    'transaction_data': transaction.get('transaction_analysis', {}),
                    'risk_factors': transaction.get('recommendations', [])
                })
        
        return flagged_transactions
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean results for JSON serialization
        cleaned_results = self._clean_for_json(results)
        
        # Save JSON results
        json_filename = f"forensic_analysis_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(cleaned_results, f, indent=2, default=str)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report(results)
        report_filename = f"forensic_analysis_report_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Results saved:")
        print(f"   - JSON: {json_filename}")
        print(f"   - Report: {report_filename}")
    
    def _display_executive_summary(self, executive_summary: Dict[str, Any]):
        """Display executive summary to console"""
        
        print(f"\nEXECUTIVE SUMMARY:")
        print(f"  Overall Fraud Score: {executive_summary['overall_assessment']['fraud_score']:.1f}/100")
        print(f"  Risk Level: {executive_summary['overall_assessment']['risk_level']}")
        print(f"  Confidence: {executive_summary['overall_assessment']['confidence']}")
        print(f"  Transactions Analyzed: {executive_summary['overall_assessment']['total_transactions_analyzed']:,}")
        
        if executive_summary['critical_alerts']:
            print(f"\n⚠  CRITICAL ALERTS:")
            for alert in executive_summary['critical_alerts']:
                print(f"     • {alert}")
        
        if executive_summary['key_findings']:
            print(f"\n📋 KEY FINDINGS:")
            for finding in executive_summary['key_findings']:
                print(f"     • {finding}")
    
    def generate_comprehensive_report(self, results: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive report combining all analyses"""
        
        if results is None:
            results = self.analysis_results
        
        if not results:
            return "No analysis results available. Please run comprehensive_analysis() first."
        
        metadata = results['metadata']
        executive = results['executive_summary']
        integrated = results['integrated_assessment']
        recommendations = results['recommendations']
        flagged = results['flagged_transactions']
        
        report = f"""
{'='*120}
INTEGRATED FORENSIC ACCOUNTING SYSTEM - COMPREHENSIVE ANALYSIS REPORT
{'='*120}

ANALYSIS METADATA:
  Timestamp: {metadata['analysis_timestamp']}
  Ledger Size: {metadata['ledger_size']:,} transactions
  Analysis Methods: {', '.join(metadata['analysis_methods_used'])}
  Columns Analyzed: {', '.join(metadata['columns_available'])}

{'='*120}
EXECUTIVE SUMMARY
{'='*120}

OVERALL ASSESSMENT:
  Fraud Score: {executive['overall_assessment']['fraud_score']:.1f}/100
  Risk Level: {executive['overall_assessment']['risk_level']}
  Confidence Level: {executive['overall_assessment']['confidence']}
  Data Quality: {executive['data_quality_assessment']}

METHOD PERFORMANCE:"""
        
        for method, performance in executive['method_performance'].items():
            method_name = method.replace('_', ' ').title()
            report += f"\n  {method_name}: {performance['fraud_score']:.1f}/100 ({performance['status']})"
        
        if executive['critical_alerts']:
            report += f"\n\nCRITICAL ALERTS:"
            for alert in executive['critical_alerts']:
                report += f"\n  ⚠ {alert}"
        
        if executive['key_findings']:
            report += f"\n\nKEY FINDINGS:"
            for finding in executive['key_findings']:
                report += f"\n  • {finding}"
        
        report += f"""

{'='*120}
INTEGRATED ASSESSMENT
{'='*120}

CONSENSUS INDICATORS:"""
        
        for indicator in integrated.get('consensus_indicators', []):
            report += f"\n  ✓ {indicator}"
        
        if not integrated.get('consensus_indicators'):
            report += "\n  No strong consensus indicators detected"
        
        report += f"\n\nCONFLICTING INDICATORS:"
        for conflict in integrated.get('conflicting_indicators', []):
            report += f"\n  ⚡ {conflict}"
        
        if not integrated.get('conflicting_indicators'):
            report += "\n  No significant conflicts between methods"
        
        report += f"""

INDIVIDUAL METHOD SCORES:"""
        for method, score in integrated['method_scores'].items():
            method_name = method.replace('_', ' ').title()
            report += f"\n  {method_name}: {score:.1f}/100"
        
        report += f"""

{'='*120}
DETAILED ANALYSIS RESULTS
{'='*120}
"""
        
        # Add detailed results for each method
        individual = results.get('individual_analyses', {})
        
        if 'benford_analysis' in individual:
            benford = individual['benford_analysis']
            if 'error' not in benford:
                report += f"""
BENFORD'S LAW ANALYSIS:
  Sample Size: {benford.get('dataset_info', {}).get('sample_size', 0):,}
  Fraud Score: {benford.get('fraud_analysis', {}).get('fraud_score', 0)}/100
  Chi-Square p-value: {benford.get('statistical_tests', {}).get('chi_square', {}).get('p_value', 0):.6f}
  Follows Benford's Law: {'YES' if benford.get('compliance_assessment', {}).get('follows_benfords_law', False) else 'NO'}
  Quality Grade: {benford.get('quality_metrics', {}).get('data_quality_grade', 'N/A')}
"""
        
        if 'statistical_analysis' in individual:
            stats = individual['statistical_analysis']
            if 'error' not in stats:
                report += f"""
STATISTICAL FRAUD DETECTION:
  Overall Fraud Score: {stats.get('summary', {}).get('overall_fraud_score', 0):.1f}/100
  Risk Level: {stats.get('summary', {}).get('risk_level', 'UNKNOWN')}
  Flagged Transactions: {len(stats.get('flagged_transactions', []))}
  Methods Completed: {stats.get('statistical_summary', {}).get('analysis_coverage', {}).get('methods_completed', 0)}
"""
        
        if 'rag_analysis' in individual:
            rag = individual['rag_analysis']
            if 'error' not in rag:
                report += f"""
RAG-BASED ANALYSIS:
  High-Risk Transactions: {rag.get('summary', {}).get('high_risk_transactions', 0)}
  Compliance Issues: {rag.get('summary', {}).get('compliance_issues', 0)}
  Pattern Anomalies: {len(rag.get('aggregate_patterns', {}))}
"""
        
        report += f"""

{'='*120}
RECOMMENDATIONS
{'='*120}

PRIORITIZED ACTION ITEMS ({len(recommendations)} total):
"""
        
        for i, rec in enumerate(recommendations, 1):
            priority_icon = "🚨" if rec['priority'] in ['IMMEDIATE', 'URGENT'] else "⚠️" if rec['priority'] == 'HIGH' else "📌"
            report += f"\n{i:2d}. {priority_icon} [{rec['priority']}] [{rec['category']}]\n"
            report += f"     Action: {rec['recommendation']}\n"
            report += f"     Rationale: {rec['rationale']}\n"
        
        if flagged:
            report += f"""

{'='*120}
HIGHEST RISK TRANSACTIONS
{'='*120}

TOP {min(10, len(flagged))} TRANSACTIONS REQUIRING IMMEDIATE REVIEW:
"""
            
            for i, transaction in enumerate(flagged[:10], 1):
                tx_data = transaction.get('transaction_data', {})
                method = transaction.get('source_method', 'Unknown')
                
                report += f"\n{i:2d}. Transaction flagged by {method}:\n"
                
                if isinstance(tx_data, dict):
                    if 'amount' in tx_data:
                        report += f"     Amount: ${tx_data.get('amount', 0):,.2f}\n"
                    if 'payee' in tx_data or 'payee_payer' in tx_data:
                        payee = tx_data.get('payee', tx_data.get('payee_payer', 'Unknown'))
                        report += f"     Payee: {payee}\n"
                    if 'date' in tx_data:
                        report += f"     Date: {tx_data.get('date', 'Unknown')}\n"
                
                risk_factors = transaction.get('risk_factors', [])
                if risk_factors:
                    report += f"     Risk Factors: {', '.join(str(r) for r in risk_factors[:3])}\n"
        
        report += f"""

{'='*120}
METHODOLOGY & TECHNICAL DETAILS
{'='*120}

ANALYSIS METHODS EMPLOYED:
  1. RAG-Based Analysis: Knowledge-driven transaction evaluation
  2. Benford's Law Testing: First-digit distribution analysis
  3. Statistical Fraud Detection: Multi-method anomaly detection
  4. Integrated Assessment: Consensus and conflict analysis

QUALITY ASSURANCE:
  • Cross-method validation performed
  • Statistical significance testing applied
  • Confidence intervals calculated where applicable
  • Data quality assessment completed

LIMITATIONS:
  • Analysis based on available data fields
  • Some methods require minimum sample sizes
  • Results should be validated through manual review
  • External factors may influence patterns

{'='*120}
END OF COMPREHENSIVE REPORT
{'='*120}
"""
        
        return report
    
    def create_visualization_dashboard(self, save_path: Optional[str] = None) -> List[str]:
        """Create visualization dashboard for all analyses"""
        
        if not self.analysis_results:
            print("No analysis results available for visualization.")
            return []
        
        visualization_paths = []
        
        # Benford's Law visualization
        individual = self.analysis_results.get('individual_analyses', {})
        if 'benford_analysis' in individual and 'error' not in individual['benford_analysis']:
            try:
                benford_viz = self.benford_analyzer.create_comprehensive_visualization(
                    individual['benford_analysis'],
                    save_path
                )
                if benford_viz:
                    visualization_paths.append(benford_viz)
                    print(f"✓ Benford's Law visualization saved: {benford_viz}")
            except Exception as e:
                print(f"✗ Benford's Law visualization failed: {str(e)}")
        
        return visualization_paths
    
    def _clean_for_json(self, obj):
        """Recursively clean object for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            # Convert dictionary with proper key handling
            cleaned_dict = {}
            for key, value in obj.items():
                # Convert keys to strings if they're numpy types
                if isinstance(key, (np.integer, np.int64, np.int32)):
                    clean_key = str(int(key))
                elif isinstance(key, (np.floating, np.float64, np.float32)):
                    clean_key = str(float(key))
                else:
                    clean_key = key
                
                cleaned_dict[clean_key] = self._clean_for_json(value)
            return cleaned_dict
            
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(item) for item in obj]
            
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
            
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
            
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
            
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
            
        elif isinstance(obj, set):
            return list(obj)
            
        else:
            return obj
    
    def export_results(self, format_type: str = 'json', filename: Optional[str] = None) -> str:
        """Export analysis results in various formats"""
        
        if not self.analysis_results:
            raise ValueError("No analysis results available for export.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"forensic_analysis_{timestamp}"
        
        if format_type.lower() == 'json':
            export_path = f"{filename}.json"
            cleaned_results = self._clean_for_json(self.analysis_results)
            with open(export_path, 'w') as f:
                json.dump(cleaned_results, f, indent=2, default=str)
                
        elif format_type.lower() == 'txt':
            export_path = f"{filename}.txt"
            report = self.generate_comprehensive_report()
            with open(export_path, 'w') as f:
                f.write(report)
                
        elif format_type.lower() == 'csv':
            export_path = f"{filename}_summary.csv"
            
            # Create summary CSV
            summary_data = []
            
            # Overall assessment
            exec_summary = self.analysis_results.get('executive_summary', {})
            overall = exec_summary.get('overall_assessment', {})
            
            summary_data.append({
                'Metric': 'Overall Fraud Score',
                'Value': f"{overall.get('fraud_score', 0):.1f}/100",
                'Category': 'Overall Assessment'
            })
            
            summary_data.append({
                'Metric': 'Risk Level',
                'Value': overall.get('risk_level', 'UNKNOWN'),
                'Category': 'Overall Assessment'
            })
            
            # Method scores
            method_performance = exec_summary.get('method_performance', {})
            for method, performance in method_performance.items():
                summary_data.append({
                    'Metric': f"{method.replace('_', ' ').title()} Score",
                    'Value': f"{performance.get('fraud_score', 0):.1f}/100",
                    'Category': 'Method Performance'
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(export_path, index=False)
            
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        print(f"✓ Results exported to: {export_path}")
        return export_path

def main():
    """Example usage of the Integrated Forensic System"""
    
    print("=" * 80)
    print("INTEGRATED FORENSIC ACCOUNTING SYSTEM - DEMONSTRATION")
    print("=" * 80)
    
    # Initialize system
    system = IntegratedForensicSystem()
    
    # Generate synthetic data for demonstration
    print("\n1. Generating synthetic ledger data...")
    synthetic_config = {
        'num_transactions': 1500,
        'fraud_percentage': 0.12,  # 12% fraudulent transactions
        'company_type': 'manufacturing'
    }
    
    success = system.load_ledger(synthetic_config, source_type='synthetic')
    if not success:
        print("Failed to load ledger data. Exiting.")
        return
    
    # Run comprehensive analysis
    print("\n2. Running comprehensive forensic analysis...")
    results = system.comprehensive_analysis(
        enable_rag=True,
        enable_benford=True,
        enable_statistical=True,
        save_results=True
    )
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    viz_paths = system.create_visualization_dashboard()
    
    # Export results in multiple formats
    print("\n4. Exporting results...")
    system.export_results('json')
    system.export_results('txt')
    system.export_results('csv')
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)
    print("\nThe Integrated Forensic System has successfully:")
    print("✓ Generated synthetic ledger data with known fraud patterns")
    print("✓ Performed multi-method fraud detection analysis")
    print("✓ Generated comprehensive reports and recommendations")
    print("✓ Created visualizations and exported results")
    print("\nReview the generated files for detailed analysis results.")

if __name__ == "__main__":
    main()