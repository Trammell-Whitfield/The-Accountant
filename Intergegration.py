import pandas as pd 
from synthetic_ledgers import SyntheticLedgerGenerator  # Your provided class
from Bedfords_law import AdvancedLedgerAnalyzer

# Transformation function
def transform_ledger_for_analyzer(df):
    transformed_df = df.copy()
    transformed_df['payee'] = transformed_df['payee_payer']
    transformed_df['description'] = transformed_df['description']
    
    def get_type(account_code):
        if account_code.startswith('4'):
            return 'income'
        elif account_code.startswith('5') or account_code.startswith('6'):
            return 'expense'
        else:
            return 'other'
    
    transformed_df['type'] = transformed_df['account_code'].apply(get_type)
    required_columns = ['amount', 'payee', 'date', 'type', 'description']
    return transformed_df[required_columns]

# Initialize the generator
generator = SyntheticLedgerGenerator(
    start_date="2024-01-01",
    end_date="2024-12-31",
    company_type="manufacturing",
    company_size="medium"
)

# Generate the synthetic ledger
print("Generating synthetic ledger...")
ledger_df = generator.generate_ledger(num_transactions=1000, fraud_percentage=0.1)

# Transform the DataFrame
analyzer_df = transform_ledger_for_analyzer(ledger_df)

# Initialize and run the analyzer
analyzer = AdvancedLedgerAnalyzer(analyzer_df)
results = analyzer.generate_comprehensive_analysis()

# Display results
print("\nAnalysis Summary:")
print(analyzer.generate_summary_report())

# Generate visualizations
print("Generating visualizations...")
analyzer.visualize_analysis()    