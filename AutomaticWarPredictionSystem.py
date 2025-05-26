"""
automatic_war_prediction_system.py

Fully automated war prediction system that:
1. Automatically loads data from internet sources
2. Processes and engineers features
3. Trains models
4. Makes predictions
5. Generates reports

No human input required!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Import our data loader
from war_data_loader import WarDataLoader

# Import our prediction model
from war_prediction_model_improved import WarPredictionModel


class AutomaticWarPredictionSystem:
    """
    Fully automated war prediction system
    """
    
    def __init__(self, auto_run: bool = True):
        """
        Initialize the automatic system
        
        Args:
            auto_run: If True, runs the entire pipeline automatically
        """
        self.timestamp = datetime.now()
        self.results_dir = f"war_predictions_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.predictor = None
        self.data = None
        self.predictions = None
        
        print("="*80)
        print("AUTOMATIC WAR PREDICTION SYSTEM")
        print("="*80)
        print(f"Started at: {self.timestamp}")
        print(f"Results will be saved to: {self.results_dir}/")
        print("="*80)
        
        if auto_run:
            self.run_complete_pipeline()
    
    def run_complete_pipeline(self):
        """Run the complete prediction pipeline automatically"""
        try:
            # Step 1: Load data
            print("\nSTEP 1: LOADING DATA FROM INTERNET SOURCES")
            print("-" * 60)
            self.load_data()
            
            # Step 2: Process data
            print("\nSTEP 2: PROCESSING AND ENGINEERING FEATURES")
            print("-" * 60)
            self.process_data()
            
            # Step 3: Train models
            print("\nSTEP 3: TRAINING PREDICTION MODELS")
            print("-" * 60)
            self.train_models()
            
            # Step 4: Make predictions
            print("\nSTEP 4: GENERATING PREDICTIONS")
            print("-" * 60)
            self.make_predictions()
            
            # Step 5: Generate reports
            print("\nSTEP 5: GENERATING REPORTS AND VISUALIZATIONS")
            print("-" * 60)
            self.generate_reports()
            
            # Step 6: Alert on high-risk countries
            print("\nSTEP 6: HIGH-RISK COUNTRY ALERTS")
            print("-" * 60)
            self.generate_alerts()
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"All results saved to: {self.results_dir}/")
            print("="*80)
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed - {e}")
            raise
    
    def load_data(self):
        """Automatically load data from internet sources"""
        # Initialize data loader
        self.data_loader = WarDataLoader(start_year=2000, end_year=2023)
        
        # Load all data
        self.data = self.data_loader.load_all_data()
        
        # Save raw data
        data_file = os.path.join(self.results_dir, 'raw_data.csv')
        self.data.to_csv(data_file, index=False)
        print(f"Raw data saved to: {data_file}")
        
        # Generate data quality report
        self.generate_data_quality_report()
    
    def generate_data_quality_report(self):
        """Generate a report on data quality"""
        report = {
            'timestamp': self.timestamp.isoformat(),
            'data_shape': list(self.data.shape),
            'countries': self.data['country'].nunique(),
            'years': f"{self.data['year'].min()}-{self.data['year'].max()}",
            'missing_values': self.data.isnull().sum().to_dict(),
            'war_rate': float(self.data['war_occurrence'].mean()),
            'summary_stats': {}
        }
        
        # Add summary statistics
        for col in ['gini', 'unemployment', 'trade_connectivity', 'geopolitical_tension']:
            if col in self.data.columns:
                report['summary_stats'][col] = {
                    'mean': float(self.data[col].mean()),
                    'std': float(self.data[col].std()),
                    'min': float(self.data[col].min()),
                    'max': float(self.data[col].max())
                }
        
        # Save report
        report_file = os.path.join(self.results_dir, 'data_quality_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Data quality report saved to: {report_file}")
    
    def process_data(self):
        """Process and engineer features"""
        # Initialize predictor model
        self.predictor = WarPredictionModel(random_state=42)
        
        # Create features
        self.processed_data = self.predictor.create_features(self.data)
        
        # Save processed data
        processed_file = os.path.join(self.results_dir, 'processed_data.csv')
        self.processed_data.to_csv(processed_file, index=False)
        print(f"Processed data saved to: {processed_file}")
        
        # Generate feature statistics
        self.generate_feature_report()
    
    def generate_feature_report(self):
        """Generate report on engineered features"""
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in ['country', 'year', 'war_occurrence', 'war_intensity']]
        
        # Calculate feature correlations with war
        correlations = {}
        for col in feature_cols:
            if col in self.processed_data.columns:
                corr = self.processed_data[col].corr(self.processed_data['war_occurrence'])
                correlations[col] = float(corr)
        
        # Sort by absolute correlation
        sorted_corr = dict(sorted(correlations.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True))
        
        report = {
            'total_features': len(feature_cols),
            'feature_names': feature_cols,
            'correlations_with_war': sorted_corr,
            'top_5_features': list(sorted_corr.keys())[:5]
        }
        
        # Save report
        report_file = os.path.join(self.results_dir, 'feature_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Feature report saved to: {report_file}")
    
    def train_models(self):
        """Train prediction models"""
        # Prepare data
        X, y_clf, y_reg = self.predictor.prepare_data(self.processed_data)
        
        # Train models
        self.predictor.train_models(X, y_clf, y_reg)
        
        # Save model performance
        self.save_model_performance()
    
    def save_model_performance(self):
        """Save model performance metrics"""
        performance = {
            'timestamp': datetime.now().isoformat(),
            'classification': self.predictor.results.get('classification', {}),
            'regression': self.predictor.results.get('regression', {}),
            'feature_importance': self.predictor.results.get('feature_importance', []).tolist()
                                if hasattr(self.predictor.results.get('feature_importance', None), 'tolist')
                                else []
        }
        
        # Save performance metrics
        perf_file = os.path.join(self.results_dir, 'model_performance.json')
        with open(perf_file, 'w') as f:
            json.dump(performance, f, indent=2)
        
        print(f"Model performance saved to: {perf_file}")
    
    def make_predictions(self):
        """Generate predictions for all countries"""
        # Get latest data for predictions
        latest_data = self.data_loader.get_latest_data()
        
        # Process latest data
        latest_processed = self.predictor.create_features(latest_data)
        
        # Make predictions
        X_latest = latest_processed[self.predictor.feature_names].values
        X_scaled = self.predictor.scale_features(X_latest, fit=False)
        
        # Get predictions
        war_probabilities = self.predictor.models['logistic'].predict_proba(X_scaled)[:, 1]
        war_predictions = self.predictor.models['logistic'].predict(X_scaled)
        
        # Create predictions dataframe
        self.predictions = pd.DataFrame({
            'country': latest_processed['country'],
            'year': latest_processed['year'],
            'war_probability': war_probabilities,
            'war_prediction': war_predictions,
            'risk_level': pd.cut(war_probabilities, 
                               bins=[0, 0.3, 0.7, 1.0],
                               labels=['LOW', 'MEDIUM', 'HIGH']),
            'gini': latest_processed['gini'],
            'unemployment': latest_processed['unemployment'],
            'trade_connectivity': latest_processed['trade_connectivity'],
            'geopolitical_tension': latest_processed['geopolitical_tension']
        })
        
        # Sort by risk
        self.predictions = self.predictions.sort_values('war_probability', ascending=False)
        
        # Save predictions
        pred_file = os.path.join(self.results_dir, 'predictions.csv')
        self.predictions.to_csv(pred_file, index=False)
        print(f"Predictions saved to: {pred_file}")
    
    def generate_reports(self):
        """Generate comprehensive reports and visualizations"""
        # 1. Executive Summary
        self.generate_executive_summary()
        
        # 2. Risk Dashboard
        self.create_risk_dashboard()
        
        # 3. Detailed Analysis
        self.create_detailed_analysis()
        
        # 4. Time Series Analysis
        self.create_time_series_analysis()
    
    def generate_executive_summary(self):
        """Generate executive summary report"""
        summary = {
            'report_date': datetime.now().isoformat(),
            'analysis_period': f"{self.data['year'].min()}-{self.data['year'].max()}",
            'countries_analyzed': self.data['country'].nunique(),
            'total_observations': len(self.data),
            'model_accuracy': self.predictor.results['classification']['accuracy'],
            'high_risk_countries': self.predictions[self.predictions['risk_level'] == 'HIGH']['country'].tolist(),
            'medium_risk_countries': self.predictions[self.predictions['risk_level'] == 'MEDIUM']['country'].tolist(),
            'key_findings': [
                f"Identified {len(self.predictions[self.predictions['risk_level'] == 'HIGH'])} high-risk countries",
                f"Overall war prediction accuracy: {self.predictor.results['classification']['accuracy']:.1%}",
                f"Most important risk factor: {self.predictor.feature_names[np.argmax(self.predictor.results['feature_importance'])]}",
                f"Current global average war risk: {self.predictions['war_probability'].mean():.1%}"
            ]
        }
        
        # Save summary
        summary_file = os.path.join(self.results_dir, 'executive_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Executive summary saved to: {summary_file}")
    
    def create_risk_dashboard(self):
        """Create visual risk dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. World Risk Map (simplified bar chart)
        ax = axes[0, 0]
        top_10 = self.predictions.head(10)
        colors = ['red' if x == 'HIGH' else 'orange' if x == 'MEDIUM' else 'green' 
                 for x in top_10['risk_level']]
        ax.barh(top_10['country'], top_10['war_probability'], color=colors)
        ax.set_xlabel('War Probability')
        ax.set_title('Top 10 Highest Risk Countries')
        ax.set_xlim(0, 1)
        
        # 2. Risk Distribution
        ax = axes[0, 1]
        risk_counts = self.predictions['risk_level'].value_counts()
        ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=['green', 'orange', 'red'])
        ax.set_title('Global Risk Distribution')
        
        # 3. Risk Factors
        ax = axes[1, 0]
        if hasattr(self.predictor, 'feature_names') and 'feature_importance' in self.predictor.results:
            importance = self.predictor.results['feature_importance']
            top_features_idx = np.argsort(importance)[::-1][:5]
            top_features = [self.predictor.feature_names[i] for i in top_features_idx]
            top_importance = [importance[i] for i in top_features_idx]
            
            ax.bar(range(5), top_importance)
            ax.set_xticks(range(5))
            ax.set_xticklabels(top_features, rotation=45, ha='right')
            ax.set_ylabel('Importance')
            ax.set_title('Top 5 Risk Factors')
        
        # 4. Historical Trend
        ax = axes[1, 1]
        yearly_risk = self.data.groupby('year')['war_occurrence'].mean()
        ax.plot(yearly_risk.index, yearly_risk.values, linewidth=2)
        ax.set_xlabel('Year')
        ax.set_ylabel('War Rate')
        ax.set_title('Historical War Trends')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        dashboard_file = os.path.join(self.results_dir, 'risk_dashboard.png')
        plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Risk dashboard saved to: {dashboard_file}")
    
    def create_detailed_analysis(self):
        """Create detailed analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Feature correlations
        ax = axes[0, 0]
        features = ['gini', 'unemployment', 'trade_connectivity', 'geopolitical_tension']
        corr_matrix = self.processed_data[features + ['war_occurrence']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        
        # 2. Risk by region
        ax = axes[0, 1]
        # Group countries by risk level
        risk_by_country = self.predictions.groupby('risk_level')['country'].apply(list).to_dict()
        
        # Create a simple visualization
        risk_summary = pd.DataFrame({
            'Risk Level': ['HIGH', 'MEDIUM', 'LOW'],
            'Count': [len(risk_by_country.get('HIGH', [])), 
                     len(risk_by_country.get('MEDIUM', [])), 
                     len(risk_by_country.get('LOW', []))]
        })
        ax.bar(risk_summary['Risk Level'], risk_summary['Count'], 
               color=['red', 'orange', 'green'])
        ax.set_ylabel('Number of Countries')
        ax.set_title('Countries by Risk Level')
        
        # 3. Economic factors vs war risk
        ax = axes[1, 0]
        ax.scatter(self.predictions['gini'], self.predictions['war_probability'], 
                  alpha=0.6, label='Gini')
        ax.set_xlabel('Gini Coefficient')
        ax.set_ylabel('War Probability')
        ax.set_title('Inequality vs War Risk')
        ax.grid(True, alpha=0.3)
        
        # 4. Trade vs war risk
        ax = axes[1, 1]
        ax.scatter(self.predictions['trade_connectivity'], 
                  self.predictions['war_probability'], 
                  alpha=0.6, color='green')
        ax.set_xlabel('Trade Connectivity')
        ax.set_ylabel('War Probability')
        ax.set_title('Trade Connectivity vs War Risk')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        analysis_file = os.path.join(self.results_dir, 'detailed_analysis.png')
        plt.savefig(analysis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed analysis saved to: {analysis_file}")
    
    def create_time_series_analysis(self):
        """Create time series analysis"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. War occurrences over time
        ax = axes[0]
        yearly_wars = self.data.groupby('year')['war_occurrence'].agg(['sum', 'mean'])
        ax.bar(yearly_wars.index, yearly_wars['sum'], alpha=0.6, label='Total Wars')
        ax2 = ax.twinx()
        ax2.plot(yearly_wars.index, yearly_wars['mean'], 'r-', linewidth=2, label='War Rate')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Wars')
        ax2.set_ylabel('War Rate', color='red')
        ax.set_title('War Occurrences Over Time')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 2. Risk factors evolution
        ax = axes[1]
        yearly_avg = self.data.groupby('year').agg({
            'gini': 'mean',
            'unemployment': 'mean',
            'trade_connectivity': 'mean',
            'geopolitical_tension': 'mean'
        })
        
        ax.plot(yearly_avg.index, yearly_avg['gini'], label='Inequality (Gini)')
        ax.plot(yearly_avg.index, yearly_avg['unemployment']/30, label='Unemployment (normalized)')
        ax.plot(yearly_avg.index, yearly_avg['trade_connectivity'], label='Trade Connectivity')
        ax.plot(yearly_avg.index, yearly_avg['geopolitical_tension']/5, label='Geopolitical Tension (normalized)')
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Risk Factors Evolution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timeseries_file = os.path.join(self.results_dir, 'time_series_analysis.png')
        plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Time series analysis saved to: {timeseries_file}")
    
    def generate_alerts(self):
        """Generate alerts for high-risk countries"""
        high_risk = self.predictions[self.predictions['risk_level'] == 'HIGH']
        
        if len(high_risk) > 0:
            print("\n" + "="*60)
            print("⚠️  HIGH RISK ALERTS ⚠️")
            print("="*60)
            
            for _, country in high_risk.iterrows():
                print(f"\n🚨 {country['country']}:")
                print(f"   - War Probability: {country['war_probability']:.1%}")
                print(f"   - Risk Factors:")
                print(f"     • Inequality (Gini): {country['gini']:.3f}")
                print(f"     • Unemployment: {country['unemployment']:.1f}%")
                print(f"     • Trade Connectivity: {country['trade_connectivity']:.3f}")
                print(f"     • Geopolitical Tension: {country['geopolitical_tension']:.2f}")
            
            # Save alerts to file
            alerts_file = os.path.join(self.results_dir, 'high_risk_alerts.json')
            alerts_data = high_risk.to_dict('records')
            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
            print(f"\nAlerts saved to: {alerts_file}")
        else:
            print("\n✅ No high-risk countries detected.")
    
    def run_continuous_monitoring(self, interval_hours: int = 24):
        """
        Run continuous monitoring - checks for new data periodically
        
        Args:
            interval_hours: Hours between checks
        """
        print(f"\nStarting continuous monitoring (checking every {interval_hours} hours)")
        print("Press Ctrl+C to stop")
        
        while True:
            try:
                # Run the pipeline
                self.run_complete_pipeline()
                
                # Wait for next check
                next_run = datetime.now() + timedelta(hours=interval_hours)
                print(f"\nNext check scheduled for: {next_run}")
                time.sleep(interval_hours * 3600)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user.")
                break
            except Exception as e:
                print(f"Error in monitoring: {e}")
                print("Retrying in 1 hour...")
                time.sleep(3600)


# Run the system
if __name__ == "__main__":
    # Create and run the automatic system
    system = AutomaticWarPredictionSystem(auto_run=True)
    
    # Optionally, start continuous monitoring
    # system.run_continuous_monitoring(interval_hours=24)