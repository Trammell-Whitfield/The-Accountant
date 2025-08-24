#!/usr/bin/env python3
"""
Main runner for The Forensic Accountant System
Run this file to execute the complete analysis pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.core.integrated_forensic_system import IntegratedForensicSystem

def main():
    """Main execution function"""
    print("🔍 Starting The Forensic Accountant System...")
    
    # Initialize the integrated system
    forensic_system = IntegratedForensicSystem()
    
    # Example usage - you can modify this section
    print("\n📊 Running sample analysis...")
    
    # Load sample data or generate synthetic data
    try:
        # Try to load existing ledger data
        import pandas as pd
        sample_data = pd.read_csv('data/realistic_accounting_ledger.csv')
        forensic_system.load_ledger(sample_data, 'dataframe')
        print(f"✓ Loaded {len(sample_data)} transactions from existing data")
    except FileNotFoundError:
        # Generate synthetic data if no existing data
        print("📝 Generating synthetic ledger data...")
        synthetic_data = forensic_system.ledger_generator.generate_ledger(
            num_transactions=1000,
            fraud_percentage=0.15
        )
        forensic_system.load_ledger(synthetic_data, 'dataframe')
        print(f"✓ Generated {len(synthetic_data)} synthetic transactions")
    
    # Run comprehensive analysis
    results = forensic_system.comprehensive_analysis()
    
    # Display summary
    print("\n📋 Analysis Summary:")
    print(f"Risk Score: {results.get('overall_risk_score', 'N/A')}")
    print(f"Fraud Indicators: {len(results.get('fraud_flags', []))}")
    print(f"Benford Violations: {results.get('benford_violations', 'N/A')}")
    
    print("\n✅ Analysis complete! Check outputs/ directory for detailed results.")
    return results

if __name__ == "__main__":
    main()