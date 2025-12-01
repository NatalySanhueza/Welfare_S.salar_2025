"""
Automated analysis script for Growth Assessment using Non-Linear Models.

All statistical comparisons and visualizations are performed systematically without manual intervention.
Group labels (RTR, WTR) are processed as categorical variables without prior interpretation,
ensuring operational blinding and minimizing manual bias in the analysis pipeline.

Clase data_processing carga y muestra los datos de entrada. Luego, realiza una descripción y analisis exploratorio
(normalidad, homocedasticidad y relationship) a las variables dependientes
"""


from pathlib import Path
import sys

# Path handling: repository-root relative
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# Import classes from src
from src.Growth import data_processing, NLM_Analysis

# Fixed paths
file_path = repo_root / 'Data' / 'Growth' / 'Growth_SD.xlsx'
output_dir = repo_root / 'Output' / 'Results_Growth'

colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}
variables = ['Time','Group','Body_mass']
var_numeric = ['Time', 'Body_mass']
z_score = 3
var_y = ['Body_mass']
var_x = 'Time'
sd_y = 'Body_mass_sd'
var_categorical = 'Group'
 

print("\n--------------------------Datos entrada (origen) y Descripción Variables-----------------------------------------")
print("\n")

pre_analysis = data_processing(str(file_path))
pre_analysis.load_data()

pre_analysis.data_description()

pre_analysis.outliers_duplicate_null_detection(variables, categorical_variable=var_categorical, z_score=z_score)
pre_analysis.data_clean(variables,
                        categorical_variable=var_categorical,
                        handle_outliers=True,                        #por defecto True
                        outlier_method='trim',                       # trim or remove
                        z_score=z_score,
                        remove_na=False,                             #por defecto True
                        remove_duplicates=False)                     #por defecto True

pre_analysis.scatter_plots(
    x=var_x,                                                # Variable en el eje X
    y_list=var_y,                                           # Lista de variables en el eje Y
    hue=var_categorical,                                    # Variable para distinguir los grupos
    title='Growth scatter plots',                           # Título común para los subplots
    colors=colors,                                          # Paleta de colores opcional
    theme='white',                                          # Estilo de la gráfica
    font='Arial',                                           # Fuente del gráfico
    font_size=12,                                           # Tamaño de fuente
    name_axis_x='Time (days)',                              # Nombre del eje X
    y_um_var='cm or g',                                     # Nombre del eje Y (se debe asignar un valor)
    scale_x='linear',                                       # Escala del eje X
    scale_y='linear',                                       # Escala del eje Y
    fig_size=(9, 4),                                        # Tamaño de la figura
    nrows = 1,
    ncols = 1
)

print("\nNormalidad y homocedasticidad analysis")  

pre_analysis.norm_homo_analysis(dependent_variables = var_y,
                                categorical_variable = var_categorical,
                                time_variable = var_x)
print("\nVariable_relationship")
pre_analysis.variable_relationship_analysis(relationship_numeric_variables = var_numeric,
                                            relationship_categorical_variable = var_categorical)

print("\nAncova analysis")
pre_analysis.ancova_analysis(numeric_variables = var_y, 
                             categorical_variable = var_categorical,
                             independent_variable= var_x)

nlm = NLM_Analysis(pre_analysis)

nlm.process_analysis(dependent_var='Body_mass', 
                     y_sd= 'Body_mass_sd',
                     categorical_var=var_categorical, 
                     time_var= var_x , 
                     name_xaxis='Time (days)', 
                     name_yaxis='Body mass (g)',
                     #selected_models = ['Lineal','Exponential', 'Power','Simplified_logistic','Logistic', 'Gompertz', 'von Bertalanffy', 'Brody'],
                     selected_models = ['Gompertz']
                     #normalization_method='minmax'                        #'max', 'minmax','standard' or 'robust'
                    )

