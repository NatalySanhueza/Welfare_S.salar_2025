"""
Automated analysis script for Oxidative Stress Assessment.

All statistical comparisons and visualizations are performed systematically without manual intervention.
Group labels (RTR, WTR) are processed as categorical variables without prior interpretation,
ensuring operational blinding and minimizing manual bias in the analysis pipeline.
"""

import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.Oxidative_Stress import Oxidative_stress_Data, Data_pre_analysis, Oxidative_Stress_analysis
file_path = repo_root / 'Data' / 'Oxidative_Stress'
colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}

ox_stress_data = Oxidative_stress_Data(str(file_path))
ox_stress_data.colors = colors
ox_stress_data.fills = fills
ox_stress_data.process_Oxidative_stress_Data()

output_dir = repo_root / 'Output' / 'Results_Oxidative_Stress'
output_dir.mkdir(parents=True, exist_ok=True)
ox_stress_data.Intensity_df.to_excel(output_dir / 'intensitydata.xlsx', index=False)
ox_stress_data.Ox_Stress_df.to_excel(output_dir / 'X_DMPO_Gene_relative_expression.xlsx', index=False)

########################################################################################################################
pre_analysis = Data_pre_analysis(ox_stress_data)
pre_analysis.output_dir = output_dir
pre_analysis.data_description()

print("\nOutliers detection")
outliers_variables = ['X_DMPO_AU_g_FC','Rel_Expression_sod1', 'Rel_Expression_gpx1']
outliers_summary = pre_analysis.outliers_detection(outliers_variables)
print(outliers_summary)
outliers_summary.to_excel(output_dir / 'outliers_detection.xlsx', index=False)

print("\nNormalidad y homocedasticidad analysis")
distribution_variables = ['X_DMPO_AU_g_FC','Rel_Expression_sod1', 'Rel_Expression_gpx1']
categorical_variable = 'Group'
pre_analysis.distribution_analysis(distribution_variables, categorical_variable)

relationship_numeric_variables = ['Time','X_DMPO_AU_g_FC','Rel_Expression_sod1', 'Rel_Expression_gpx1']
relationship_categorical_variable = 'Group'
pre_analysis.variable_relationship_analysis(relationship_numeric_variables, relationship_categorical_variable)

###########################################################################################################################
ox_stress_analysis= Oxidative_Stress_analysis(ox_stress_data, pre_analysis)
ox_stress_analysis.output_dir = output_dir
ox_stress_analysis.process_Oxidative_Stress_analysis()