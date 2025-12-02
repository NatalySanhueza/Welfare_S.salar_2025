"""
Automated analysis script for Thermoregulatory Behavior Assessment.

All statistical comparisons and visualizations are performed systematically without manual intervention.
Group labels (RTR, WTR) are processed as categorical variables without prior interpretation,
ensuring operational blinding and minimizing manual bias in the analysis pipeline.

Clase ThermoregulationDataProcessor carga y muestra los datos de entrada. Luego establece nuevas variables independientes
categoricas y calcula las variables dependientes.
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.Thermoregulatory_Behavior import ThermoregulationDataProcessor, Data_pre_analysis, termoregulation_analysis
file_path = repo_root / 'Data' / 'Thermoregulatory_Behavior' / 'Thermoregulatory_Bahavior.xlsx'

processor = ThermoregulationDataProcessor(str(file_path))
colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}
processor.colors = colors
processor.fills = fills
 

print("\n---------------------------------Datos entrada (origen)-----------------------------------------")
print("\n")
processor.process_data()

######################################################################################################################
output_dir = repo_root / 'Output' / 'Results_Thermoregulatory_Behavior'
output_dir.mkdir(parents=True, exist_ok=True)



print("\n-----------------------------------Descripción Variables-------------------------------------------")
print("\n")
pre_analysis = Data_pre_analysis(processor)
pre_analysis.output_dir = output_dir
pre_analysis.data_description()

print("\nOutliers detection")
print("\n")
outliers_variables = ['X_cm', 'Y_cm', 'Jacobs_Index', 'Thermal_Preference_Index' ]
outliers_summary = pre_analysis.outliers_detection(outliers_variables)
print(outliers_summary)
outliers_summary.to_excel(output_dir / 'outliers_detection.xlsx', index=False)

print("\nNormalidad y homocedasticidad analysis")
print("\n")
distribution_variables = ['X_cm', 'Y_cm', 'Jacobs_Index', 'Thermal_Preference_Index' ]
categorical_variable = 'Treatment'
pre_analysis.distribution_analysis(distribution_variables, categorical_variable)

print("\nVariable_relationship")
print("\n")
relationship_numeric_variables = ['X_cm', 'Y_cm', 'Jacobs_Index', 'Thermal_Preference_Index', 'Hora']
relationship_categorical_variable = 'Treatment'
#pre_analysis.variable_relationship_analysis(relationship_numeric_variables, relationship_categorical_variable)

##########################################################################################################################

analysis = termoregulation_analysis(processor, pre_analysis)
analysis.output_dir = output_dir

print("\n------------------------Análisis estadístico Jacobs Index (RTR vs WTR)-----------------------------------")
print("\n")
analysis.mwu_test_jacobs_index()
#analysis.analyze_jacobs_index()

print("\n------------------------Correlación IPT vs tiempo (horas del día)-----------------------------------")
print("\n")
analysis.preprocess_IPT()
analysis.fit_IPT()

print("\n------------------------Análisis estadístico IPT day vs night-----------------------------------")
print("\n")
analysis.mwu_test_IPT()


#Kernel
fig_kernel, _ = analysis.estimate_kernel_density()
fig_kernel.savefig(output_dir / 'estimate_kernel_density.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close(fig_kernel)

#Jacobs
fig_jacobs, _ = analysis.analyze_jacobs_index()
fig_jacobs.savefig(output_dir / 'analyze_jacobs_index.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close(fig_jacobs)

#IPT
fig_ipt, _ = analysis.analyze_IPT_with_best_fit()
fig_ipt.savefig(output_dir / 'analyze_IPT_with_best_fit.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close(fig_ipt)

#Outdata to excel (inputdata, variables)
processor.chamber_df.to_excel(output_dir / 'Thermorregulatory_Behavior_outdata.xlsx', index=False)

# Save statistical results
analysis.save_statistical_results()