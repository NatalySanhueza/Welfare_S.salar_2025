"""
Automated analysis script for Mortality and Survival Assessment.

All statistical comparisons and visualizations are performed systematically without manual intervention.
Group labels (RTR, WTR) are processed as categorical variables without prior interpretation,
ensuring operational blinding and minimizing manual bias in the analysis pipeline.
"""


from pathlib import Path
import sys

# Path handling: repository-root relative
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Import classes from src
from src.Mortality import MortalityAnalysis

# Fixed paths
file_path_mortality = str(repo_root / 'Data' / 'Mortality' / 'Mortality.xlsx')
output_dir = repo_root / 'Output' / 'Results_Mortality'
initial_population = {'RTR': 700, 'WTR': 700}

# Crea una instancia de la clase MortalityAnalysis
mortality_analysis = MortalityAnalysis(file_path_mortality, initial_population)

# Genera y guarda la figura de Kaplan-Meier
fig, ax = mortality_analysis.plot_kaplan_meier()

# Guardar figura
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / 'Kaplan_Meier_Mortality.pdf', dpi=300, bbox_inches='tight')
print(f"Figura guardada en {output_dir / 'Kaplan_Meier_Mortality.pdf'}")



