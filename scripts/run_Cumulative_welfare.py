"""
Automated analysis script for Cumulative Welfare Assessment.

All statistical comparisons and visualizations are performed systematically without manual intervention.
Group labels (RTR, WTR) are processed as categorical variables without prior interpretation,
ensuring operational blinding and minimizing manual bias in the analysis pipeline.

"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Path handling: build paths relative to repository root
repo_root = Path(__file__).resolve().parent.parent
# Make sure src/ is importable when running scripts directly
sys.path.insert(0, str(repo_root))

# Fixed paths
file_path = repo_root / 'Data' / 'Cumulative_welfare' / 'Cumulative_Welfare.xlsx'
output_dir = repo_root / 'Output' / 'Results_Cumulative_welfare'
colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}
tissue_colors = {'brain': '#A882DD', 'muscle': '#D83F87', 'all_data':'#AD8CFF'}

# Import processing classes from src and ensure module-level color variables exist
from src import Cumulative_welfare as cw_mod

# set module-level color globals expected by the classes in src/Cumulative_welfare.py
cw_mod.colors = colors
cw_mod.fills = fills
cw_mod.tissue_colors = tissue_colors

#Instancia de la clase data_processing
pre_analysis = cw_mod.data_processing(str(file_path))

print("\n--------------------------Datos entrada (origen) y Descripción Variables-----------------------------------------")
print("\n")

# Load data
pre_analysis.load_data()

#Parámetros métodos 
variables = ['Time', 'Group', 'Tissue', 'Replicate', 'Survival_Probability', 'Cumulative_Mortality', 'Body_Mass', 'RTL', 'mC', 'OHdG']
var_numeric = ['Time', 'Survival_Probability', 'Cumulative_Mortality', 'Body_Mass', 'RTL', 'mC', 'OHdG']
z_score = 3
dependent_variables = ['Survival_Probability', 'Cumulative_Mortality', 'Body_Mass', 'RTL', 'mC', 'OHdG'] 
independent_continuous_variable = 'Time'
categorical_variables = ['Tissue', 'Group']
Hue = 'Group'

print(pre_analysis.columns_name)
pre_analysis.data_description()

pre_analysis.outliers_duplicate_null_detection(variables, 
                                               categorical_variables = ['Tissue'],     #Detectar solo con respecto a cada tejido
                                               z_score=z_score)
pre_analysis.data_clean(variables,
                        categorical_variables= ['Tissue'],                   # Limpiar solo con respecto al tejido
                        handle_outliers=False,                               # por defecto True
                        outlier_method='trim',                               # trim or remove
                        z_score=z_score,
                        remove_na=False,                                     #por defecto True
                        remove_duplicates=False)                             #por defecto True

pre_analysis.scatter_plots(
    x=independent_continuous_variable,                                                # Variable en el eje X
    y_list=dependent_variables,                                                       # Lista de variables en el eje Y
    hue= Hue,                                                                         # Variable para distinguir (colorear) entre grupos tratados
    #categorical_variables = 'Tissue',                                                 # Variable para separar el data set de ser necesario
    title='Scatter plots',                                                            # Título común para los subplots
    colors=colors,                                                                    # Paleta de colores opcional
    theme='white',                                                                    # Estilo de la gráfica
    font='Arial',                                                                     # Fuente del gráfico
    font_size=6,                                                                      # Tamaño de fuente
    name_axis_x='Time (days)',                                                        # Nombre del eje X
    y_um_var='unidad',                                                                # Nombre del eje Y (se debe asignar un valor)
    scale_x='linear',                                                                 # Escala del eje X
    scale_y='linear',                                                                 # Escala del eje Y
    fig_size=(8, 8),                                                                  # Tamaño de la figura
    nrows = 3,
    ncols = 2
)

print("\nNormalidad y homocedasticidad analysis")  

pre_analysis.norm_homo_analysis(dependent_variables = dependent_variables,
                                categorical_variables = [categorical_variables[0]],
                                independent_continuous_variable = independent_continuous_variable,
                                norm_test= 'Shapiro'                      #"Shapiro" o "D’Agostino and Pearson"
                               )
print("\nVariable_relationship")
pre_analysis.variable_relationship_analysis(relationship_numeric_variables = var_numeric, 
                                            relationship_categorical_variable = Hue, 
                                            categorical_variables = [categorical_variables[0]]
                                           )
                            
print("\nAncova analysis")
pre_analysis.ancova_analysis(numeric_variables = dependent_variables,
                             independent_continuous_variable = independent_continuous_variable,
                             categorical_variables = categorical_variables
                             )


data = pre_analysis.df_clean if pre_analysis.df_clean is not None else pre_analysis.df
#data = data[data['Tissue'] == 'muscle']

colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}

tissue_colors = {'brain': '#A882DD', 'muscle': '#D83F87', 'all_data':'#AD8CFF'}

#Instancia de la clase
var_import = ['Cumulative_Mortality', 'RTL', 'mC', 'OHdG'] 
# use the class from the imported module
cw = cw_mod.cumulative_welfare(pre_analysis, data, colors, fills, var_import)

#Parámetros métodos
variables = ['Time', 'Group', 'Tissue', 'Replicate', 'Survival_Probability', 'Cumulative_Mortality', 'Body_Mass', 'RTL', 'mC', 'OHdG']
variables_dependientes = ['Survival_Probability', 'Cumulative_Mortality', 'Body_Mass', 'RTL', 'mC', 'OHdG']
Variables_independientes = ['Time', 'Group', 'Tissue']
variables_independientes_categoricas = ['Group', 'Tissue']
variables_independientes_continuas = ['Time']
treatment_var = 'Group'

pca_results, variance_fig, biplot_fig,  index_data, index_fig, fig1, pca_contribution  = cw.process_analysis(
    group=False,                                 #usar sobre data completa o agrupada by_variable
    by_variable='Tissue',                       #variable para agrupar la data si group == True
    rotate= True,                              #rotar scores and loading = True
    treatment_column='Group',
    continuous_category= 'Time',
    post_hoc_method='conover',                  #conover o wilcoxon
    post_hoc_correction='holm',                 # 'bonferroni', 'holm','sidak' Holm-Sidak ('hs'), 'hommel' 
    corr_method = 'spearman',                    #'pearson', 'spearman'
    scale_data=True,                           #For corelation
    test_size=0.2, n_estimators=100, random_state=42,
    dependent_vars=['Cumulative_Mortality', 'Body_Mass', 'OHdG', 'RTL', 'mC'],
    independent_var='PCA Index',
    cv=5,
    use_scaled_data=True,
    n_components=5,
    output_dir=str(output_dir),
    show_plots=False
)

output_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving results to: {output_dir}")

print(len(index_data))
index_data.items()


# Save PCA contributions if available
try:
    if pca_contribution is not None:
        pca_contribution.to_excel(output_dir / 'PCA_contributions.xlsx', index=False)
        print(f"Saved PCA contributions to {output_dir / 'PCA_contributions.xlsx'}")
except Exception as e:
    print(f"Warning: could not save PCA contributions: {e}")




