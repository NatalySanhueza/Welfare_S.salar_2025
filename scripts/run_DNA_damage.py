"""
Automated analysis script for DNA Damage Assessment.

All statistical comparisons and visualizations are performed systematically without manual intervention.
Group labels (RTR, WTR) are processed as categorical variables without prior interpretation,
ensuring operational blinding and minimizing manual bias in the analysis pipeline.
"""



from pathlib import Path
import sys


repo_root = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(repo_root))
from src.DNA_Damage import DNA_Data, Long_welfare_measures, Data_pre_analysis, Long_welfare_analysis

file_path = repo_root / "Data" / "DNA_Damage"
colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}

dna_data = DNA_Data(str(file_path))
dna_data.colors = colors
dna_data.fills = fills
dna_data.process_DNA_data()

#######################################################################################################################

wf_measures = Long_welfare_measures(dna_data)
wf_measures.process_Long_welfare_measures()

output_dir = repo_root / "Output" / "Results_DNA_damage"
output_dir.mkdir(parents=True, exist_ok=True)
wf_measures.df.to_excel(output_dir / 'DNA_Damage_Data_Full.xlsx', index=False)

########################################################################################################################

pre_analysis = Data_pre_analysis(dna_data, wf_measures)
pre_analysis.data_description()

print("\nOutliers detection")
outliers_variables = ['Cq_Gap8_Mean','Cq_Tel1b_Mean','OD_450_nm_mc_Mean','OD_450_nm_ohdg_Mean','RTL','5_mC_ng','5_mC_%',
                      '8_OHdG_ng','8_OHdG_%','RTL_FC', '5_mC_ng_FC', '5_mC_%_FC', '8_OHdG_ng_FC', '8_OHdG_%_FC']
print(pre_analysis.outliers_detection(outliers_variables))

print("\nNormalidad y homocedasticidad analysis")
distribution_variables = ['RTL_FC', '5_mC_ng_FC','5_mC_%_FC','8_OHdG_ng_FC', '8_OHdG_%_FC']
categorical_variable = 'Group'
pre_analysis.distribution_analysis(distribution_variables, categorical_variable)

relationship_numeric_variables = ['Time','RTL_FC', '5_mC_ng_FC', '5_mC_%_FC', '8_OHdG_ng_FC', '8_OHdG_%_FC']
relationship_categorical_variable = 'Group'
pre_analysis.variable_relationship_analysis(relationship_numeric_variables, relationship_categorical_variable)

###########################################################################################################################

wf_analyze = Long_welfare_analysis(dna_data, wf_measures, pre_analysis)
wf_analyze.process_Long_welfare()