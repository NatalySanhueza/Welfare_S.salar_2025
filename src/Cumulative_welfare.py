import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.api import OLS
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import shapiro, levene
from scipy.stats import kruskal
from itertools import combinations
import scikit_posthocs as sp
import string
from pathlib import Path



class data_processing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.column_names = []
        self.df_clean = None
        self.preprocessing_summary = {}
        self.plots = []
        self.colors = colors
        self.fills = fills
        self.show_plots = False
       
    def load_data(self, sheet_name=0):
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            print("Data loaded successfully.")
            print("\nVariables:")
            self.column_names = self.df.columns.tolist()
        except Exception as e:
            print(f"Error loading file: {e}")
            
        self.columns_name = self.df.columns.tolist()
        return self.columns_name
        
            
    def data_description(self, df=None):
        if df is None:
            df = self.df

        print("\nFirst DataFrame rows:")
        print(df.head())
        
        print("\nDataFrame Information:")
        print("\n")
        print(df.info())

        print("\nStatistical summary:")
        print(df.describe(include='all'))
        
    def outliers_duplicate_null_detection(self, variables, categorical_variables, z_score=3):
        df = self.df.copy()
        analysis_list = []
    
        # Agrupar por las variables categóricas seleccionadas
        for group, group_df in df.groupby(categorical_variables):
            
            # Convertir group a tupla si es una sola categoría
            if not isinstance(group, tuple):
                group = (group,)
            
            # Iterar sobre las variables independientes
            for var in variables:
                num_outliers = None
                outliers_values = None
    
                if pd.api.types.is_numeric_dtype(group_df[var]):
                    var_zscore = var + '_zscore'
                    group_df[var_zscore] = np.abs(stats.zscore(group_df[var].dropna()))
                    group_df[var + '_is_outlier'] = group_df[var_zscore] > z_score
                    num_outliers = group_df[var + '_is_outlier'].sum()
                    outliers_values = group_df.loc[group_df[var + '_is_outlier'], var].tolist()
    
                num_duplicates = group_df.duplicated(subset=[var], keep=False).sum()
                duplicate_values = group_df[group_df.duplicated(subset=[var], keep=False)][var].unique().tolist()
    
                num_nulls = group_df[var].isnull().sum()
                null_values = group_df.loc[group_df[var].isnull(), var].tolist()
    
                # Crear un diccionario para almacenar el resultado
                result = {'Variable': var, 'Num_Outliers': num_outliers, 'Outliers_Values': outliers_values,
                          'Num_Duplicates': num_duplicates, 'Duplicate_Values': duplicate_values,
                          'Num_Nulls': num_nulls, 'Null_Values': null_values}
    
                # Añadir las variables categóricas al diccionario
                for i, cat_var in enumerate(categorical_variables):
                    result[cat_var] = group[i]
    
                analysis_list.append(result)
    
        # Crear el DataFrame resumen de análisis
        analysis_summary = pd.DataFrame(analysis_list)
    
        # Reordenar las columnas en el orden deseado
        column_order = categorical_variables + ['Variable', 'Num_Outliers', 'Outliers_Values', 
                                                'Num_Duplicates', 'Duplicate_Values', 
                                                'Num_Nulls', 'Null_Values']
        
        analysis_summary = analysis_summary[column_order]
        
        print("\nOutliers, Duplicates, and Nulls Summary by Group:")
        print(analysis_summary)
        
        return analysis_summary


    def data_clean(self, variables, categorical_variables, handle_outliers=True, outlier_method='remove', 
                   z_score=3, remove_na=True, remove_duplicates=True):
        self.df_clean = self.df.copy()
    
        # Agrupar por las variables categóricas seleccionadas
        for group, group_df in self.df_clean.groupby(categorical_variables):
            
            # Convertir group a tupla si es una sola categoría
            if not isinstance(group, tuple):
                group = (group,)
            
            group_mask = np.all([self.df_clean[cat_var] == val for cat_var, val in zip(categorical_variables, group)], axis=0)
            
            # Si se solicita manejar los outliers
            if handle_outliers and variables is not None:
                df_outliers = group_df.copy()
                
                for num_var in variables:
                    if pd.api.types.is_numeric_dtype(group_df[num_var]):
                        df_outliers[num_var + '_zscore'] = np.abs(stats.zscore(group_df[num_var].dropna()))
                        df_outliers[num_var + '_is_outlier'] = df_outliers[num_var + '_zscore'] > z_score
    
                        if outlier_method == 'remove':
                            group_df = group_df[~df_outliers[num_var + '_is_outlier']]
                            print(f"Outliers in {num_var} for group {group} were removed based on a z-score threshold of {z_score}.")
                        elif outlier_method == 'trim':
                            group_df.loc[df_outliers[num_var + '_is_outlier'], num_var] = group_df[num_var].clip(
                                lower=group_df[num_var].quantile(0.01),
                                upper=group_df[num_var].quantile(0.99)
                            )
                            print(f"Outliers in {num_var} for group {group} were trimmed to the 1st and 99th percentiles.")
                        else:
                            raise ValueError("Invalid outlier_method. Use 'remove' or 'trim'.")
                    else:
                        print(f"Skipping outlier detection for non-numeric variable: {num_var} in group {group}")
            else:
                print(f"Outlier handling was skipped for group {group}.")
        
            # Eliminar valores NaN
            if remove_na:
                group_df = group_df.dropna()
                print(f"Missing values (NaN) were removed for group {group}.")
            else:
                print(f"Skipping removal of missing values for group {group}.")
        
            # Eliminar duplicados
            if remove_duplicates:
                group_df = group_df.drop_duplicates()
                print(f"Duplicate rows were removed for group {group}.")
            else:
                print(f"Skipping removal of duplicate rows for group {group}.")
            
            # Actualizar la DataFrame limpia con los cambios del grupo actual
            self.df_clean.loc[group_mask] = group_df
    
        return self.df_clean

    def scatter_plots(self, x, y_list, hue, title, categorical_variables=None, colors=None, theme='white', 
                      font='Arial', font_size=12, name_axis_x=None, name_axis_y=None, y_um_var=None, 
                      scale_x='linear', scale_y='linear', nrows=1, ncols=1, fig_size=(12, 8)):
        """
        Crea subplots de gráficos de dispersión para una lista de variables Y, replicando las figuras 
        por combinaciones de variables categóricas si se especifican.
        
        Parámetros:
        - x: Variable en el eje X (variable independiente continua).
        - y_list: Lista de variables en el eje Y (lista de strings).
        - hue: Variable que distingue los grupos con colores (string).
        - title: Título de la figura (string).
        - categorical_variables: Lista de variables categóricas para replicar gráficos (lista de strings).
        - colors: Paleta de colores opcional (dict o lista).
        - theme: Estilo del gráfico (string, e.g. 'darkgrid').
        - font: Tipo de letra (string).
        - font_size: Tamaño de la fuente (int).
        - name_axis_x: Nombre personalizado para el eje X (string).
        - name_axis_y: Nombre personalizado para el eje Y (string, se aplica a todos los subplots).
        - scale_x: Escala del eje X ('linear' o 'log').
        - scale_y: Escala del eje Y ('linear' o 'log', se aplica a todos los subplots).
        - fig_size: Tamaño de la figura en (width, height).
        """
        data = self.df_clean if self.df_clean is not None else self.df
        sns.set_theme(style=theme)
         
        # Si no hay variables categóricas especificadas, se utiliza todo el DataFrame
        if categorical_variables:
            group_data = data.groupby(categorical_variables)
        else:
            group_data = [(None, data)]  # Un solo grupo con todo el DataFrame
    
        for group, group_df in group_data:
            # Crear una figura con subplots para cada grupo
            sns.reset_defaults()
            plt.rcdefaults()
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size[0], fig_size[1]))
            
            # Aplanar los ejes en caso de que sea una matriz para evitar errores
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
            for i, y in enumerate(y_list):
                ax = axes[i]
                if colors:
                    sns.scatterplot(data=group_df, x=x, y=y, hue=hue, palette=colors, ax=ax)
                else:
                    sns.scatterplot(data=group_df, x=x, y=y, hue=hue, ax=ax)
    
                # Etiquetas de los ejes
                ax.set_xlabel(name_axis_x if name_axis_x else x, fontsize=font_size, fontname=font)
                ax.set_ylabel(f"{y} {y_um_var if y_um_var else ''}", fontsize=font_size, fontname=font)
    
                # Ajustar la escala de los ejes
                ax.set_xscale(scale_x)
                ax.set_yscale(scale_y)
    
                # Asegurar que los ticks de ambos ejes sean visibles y ajustados
                ax.tick_params(axis='both', which='both', labelsize=font_size, colors='black')
                ax.set_axisbelow(True)
    
                # Crear el string del título para las variables categóricas
                if categorical_variables:
                    # Si hay más de una variable categórica, unir con '-'. Si es una sola, usar directamente su valor.
                    group_str = '-'.join(map(str, group)) if isinstance(group, tuple) else str(group)
                else:
                    group_str = ""
    
                # Asignar el título del subplot
                ax.set_title(f"{title} - {y} - {group_str}", fontsize=font_size + 2, fontname=font)

            plt.legend(prop={'size': 6, 'family': font})
            plt.tight_layout()
            self.plots.append(fig)
            # Save scatter plot figure
            try:
                output_dir = Path("Output") / "Results_Cumulative_welfare"
                output_dir.mkdir(parents=True, exist_ok=True)
                group_str = '-'.join(map(str, group)) if categorical_variables and isinstance(group, tuple) else (str(group) if group is not None else "all")
                fname = output_dir / f"scatter_{title.replace(' ', '_')}_{group_str}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Saved scatter plot to {fname}")
            except Exception as e:
                print(f"Warning: could not save scatter plot: {e}")
            plt.close(fig)
    
        
    def norm_homo_analysis(self, dependent_variables, categorical_variables, independent_continuous_variable, norm_test='Shapiro'):
        data = self.df_clean if self.df_clean is not None else self.df
        results_tests = []
    
        # Asegurar que categorical_variables sea una lista
        if not isinstance(categorical_variables, list):
            categorical_variables = [categorical_variables]
    
        # Iterar sobre las combinaciones de variables categóricas
        for group, group_df in data.groupby(categorical_variables):
            # Convertir group a tupla si es una sola categoría
            if not isinstance(group, tuple):
                group = (group,)
            
            group_name = " - ".join([f"{cat}: {val}" for cat, val in zip(categorical_variables, group)])
    
            for var in dependent_variables:
                sns.reset_defaults()
                plt.rcdefaults()
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                
                # Histograma y densidad
                sns.histplot(data=group_df, x=var, kde=True, ax=ax1)
                ax1.set_title(f'Distribution of {var} ({group_name})')
                
                # Q-Q plot
                stats.probplot(group_df[var], dist="norm", plot=ax2)
                ax2.set_title(f'Q-Q Plot of {var} ({group_name})')
                
                # Modelo OLS
                formula = f'{var} ~ {independent_continuous_variable}'
                model = ols(formula, data=group_df).fit()
                fitted_values = model.fittedvalues
                residuals = model.resid
                
                # Residual plot
                sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws={'color': 'red'}, ax=ax3)
                ax3.axhline(0, linestyle='--', color='black', linewidth=2)
                ax3.set_title(f'Residual Plot of {var} ({group_name})')
                ax3.set_xlabel('Fitted Values')
                ax3.set_ylabel('Residuals')
                
                plt.tight_layout()
                # Save norm/homo analysis figure
                try:
                    output_dir = Path("Output") / "Results_Cumulative_welfare"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    safe_group = group_name.replace(' ', '_').replace(':', '')
                    fname = output_dir / f"norm_homo_{var}_{safe_group}.png"
                    plt.gcf().savefig(fname, dpi=300, bbox_inches='tight')
                    print(f"Saved norm/homo figure to {fname}")
                except Exception as e:
                    print(f"Warning: could not save norm/homo figure: {e}")
                plt.close()
                
                # Test de normalidad
                if norm_test == 'Shapiro':
                    _, p_val_norm = stats.shapiro(group_df[var])
                elif norm_test == "D'Agostino and Pearson":
                    _, p_val_norm = stats.normaltest(group_df[var])
                else:
                    raise ValueError('Ingresar opción correcta "Shapiro" o "D\'Agostino and Pearson"')
                
                is_normal = p_val_norm > 0.05
    
                # Test de homocedasticidad
                _, p_val_homo = stats.levene(*[subgroup[var].values for _, subgroup in group_df.groupby(independent_continuous_variable)])
                is_homogeneous = p_val_homo > 0.05
    
                results_tests.append({
                    'Group': group_name,
                    'Variable': var,
                    'p_val_normality': p_val_norm,
                    'Is_normal': is_normal,
                    'p_val_homocedasticity': p_val_homo,
                    'Is_homogeneous': is_homogeneous
                })
    
        # Crear tabla de resultados
        df_results = pd.DataFrame(results_tests)
        print(f"\n{norm_test} test & Levene test summary")
        print(df_results)
        self.df_results_tests = df_results
        
    def variable_relationship_analysis(self, relationship_numeric_variables, relationship_categorical_variable, categorical_variables=None):
        # Si no se pasan variables categóricas, usa un único DataFrame sin agrupamiento
        if categorical_variables is None:
            categorical_variables = []
        
        # Generar combinaciones de niveles de las variables categóricas
        if categorical_variables:
            # Obtener los valores únicos de todas las variables categóricas
            unique_combinations = pd.DataFrame(
                [dict(zip(categorical_variables, combination)) for combination in 
                 pd.MultiIndex.from_product([self.df[var].unique() for var in categorical_variables])]
            )
        else:
            # Si no hay variables categóricas, se genera una única combinación sin agrupamiento
            unique_combinations = pd.DataFrame([{}])  # DataFrame vacío para manejar un solo caso
            
        for _, combination in unique_combinations.iterrows():
            # Filtrar el DataFrame por la combinación actual de niveles categóricos
            filtered_df = self.df.copy()
            for cat_var, level in combination.items():
                filtered_df = filtered_df[filtered_df[cat_var] == level]
            
            # Verificar que el DataFrame filtrado no esté vacío
            if filtered_df.empty:
                continue
            
            # Verificar que todas las categorías en el subset están en el diccionario de colores
            unique_categories = filtered_df[relationship_categorical_variable].unique()
            missing_colors = [category for category in unique_categories if category not in colors]
            
            if missing_colors:
                raise ValueError(f"Colors dictionary does not have entries for the following categories: {', '.join(missing_colors)}")
            
            # Matriz de correlación
            numeric_vars = [var for var in relationship_numeric_variables if var in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[var])]
            corr_matrix = filtered_df[numeric_vars].corr(numeric_only=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            
            # Generar el título usando las combinaciones de niveles categóricos
            title_suffix = ", ".join([f"{var}: {combination[var]}" for var in combination.index])
            plt.title(f'Correlation Matrix {title_suffix}')
            # Save correlation matrix
            try:
                output_dir = Path("Output") / "Results_Cumulative_welfare"
                output_dir.mkdir(parents=True, exist_ok=True)
                safe_suffix = title_suffix.replace(' ', '_').replace(':', '').replace(',', '')
                fname = output_dir / f"corr_matrix_{safe_suffix}.png"
                plt.gcf().savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Saved correlation matrix to {fname}")
            except Exception as e:
                print(f"Warning: could not save correlation matrix: {e}")
            plt.close()
    
            # Pairplot
            try:
                pairplot = sns.pairplot(filtered_df, vars=numeric_vars, hue=relationship_categorical_variable, palette=colors)
                pairplot.fig.suptitle(f'Pairplot {title_suffix}', y=1.02)
                # Save pairplot
                output_dir = Path("Output") / "Results_Cumulative_welfare"
                output_dir.mkdir(parents=True, exist_ok=True)
                safe_suffix = title_suffix.replace(' ', '_').replace(':', '').replace(',', '')
                fname = output_dir / f"pairplot_{safe_suffix}.png"
                pairplot.fig.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Saved pairplot to {fname}")
                plt.close(pairplot.fig)
            except Exception as e:
                print(f"Warning: could not save pairplot: {e}")
    
        
    def ancova_analysis(self, numeric_variables, independent_continuous_variable, categorical_variables):
        data = self.df_clean if self.df_clean is not None else self.df
        results_ancova = {}
        
        for var in numeric_variables:
            # Crear la fórmula del modelo ANCOVA con las variables categóricas e independientes continuas
            # Incluir interacciones entre las variables categóricas y la variable continua
            formula = f'{var} ~ {independent_continuous_variable} * ' + ' * '.join([f'C({cat})' for cat in categorical_variables])
            
            # Ajustar el modelo ANCOVA
            model = ols(formula, data=data).fit()
            anova_table = anova_lm(model)
            
            # Agregar una columna con el nombre de la fuente para identificar los resultados
            anova_table['Source'] = anova_table.index
            
            # Guardar cada resultado en un DataFrame, con el nombre de la variable como clave
            results_ancova[var] = anova_table[['Source', 'df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]
        
        # Guardar los DataFrames en un atributo de la clase para acceso posterior
        self.results_ancova_dict = results_ancova
        
        # Mostrar los DataFrames por cada variable
        for var, df in self.results_ancova_dict.items():
            print(f"\nANCOVA for {var}")
            print(df)

class cumulative_welfare:
    def __init__(self, pre_analysis, data, colors, fills, var_import):
        self.pre_analysis = pre_analysis  # Preanálisis de datos
        self.data = data.copy()           # Dataset (copiamos para no alterar el original)
        self.colors = colors              # Colores para visualización
        self.fills = fills                # Rellenos para visualización
        self.tissue_colors = tissue_colors
        self.cld_results = None
        self.var_import = var_import
        
    def perform_analysis(self, analysis_func, group=False, by_variable=None, **kwargs):
        """
        Aplica una función de análisis a los datos (agrupados o no agrupados).
        
        Parámetros:
        - analysis_func: Función de análisis a aplicar.
        - group (bool): Si es True, agrupa los datos por la variable seleccionada.
        - by_variable: La variable por la que se agrupan los datos si group=True.
        - **kwargs: Argumentos adicionales para la función de análisis.
        """
        if group and by_variable is not None:
            # Agrupar los datos y aplicar el análisis a cada grupo
            grouped_data = self.data.groupby(by_variable)
            results = {group_name: analysis_func(group_data, **kwargs) 
                       for group_name, group_data in grouped_data}
        else:
            # Aplicar el análisis al conjunto de datos completo
            results = {'all_data': analysis_func(self.data, **kwargs)}
        
        return results

    def pca_analysis(self, data, rotate= False, **kwargs):
        """
        Realiza PCA en los datos (agrupados o no agrupados) y devuelve un DataFrame con los resultados y las
        columnas categóricas relevantes. También guarda la varianza explicada para cada grupo.
        
        Parámetros:
        - data: El conjunto de datos a analizar.
        - independent_vars: Lista de variables independientes a utilizar en el análisis.
        - categorical_vars: Lista de variables categóricas a incluir en el resultado.
        - n_components: Número de componentes principales a calcular.
        
        Retorna:
        - DataFrame con resultados del PCA y columnas categóricas relevantes.
        - DataFrame con el loading de cada variable para cada componente.
        """
        dependent_vars = kwargs.get('dependent_vars', None)
        independen_vars = kwargs.get('independen_vars', None)
        n_components = kwargs.get('n_components', 2)

        X = data[dependent_vars].dropna()  # Filtrar datos con las variables independientes
        if X.empty:
            print("No hay suficientes datos para realizar PCA.")
            return None

        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components)
        principal_components = -pca.fit_transform(X_scaled) if rotate else pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_  # Capturamos la varianza explicada
        loadings = -pca.components_.T * np.sqrt(pca.explained_variance_) if rotate else pca.components_.T * np.sqrt(pca.explained_variance_)
        pca_df = pd.DataFrame(data=principal_components, 
                              columns=[f'PC{i+1}' for i in range(n_components)])
        
        
        # Construir el DataFrame de loadings
        loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_components)], index=X.columns)
        loadings_df = loadings_df.reset_index().melt(id_vars='index', var_name='componente', value_name='loading')
        loadings_df = loadings_df.rename(columns={'index': 'nombre_variable'})
        print(loadings_df)
        return pca_df, explained_variance, pca, loadings

    
    def plot_explained_variance(self, pca_results):
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 8
        plt.rcParams['font.weight'] = 'bold'
        
        fig, axes = plt.subplots(nrows=1, ncols=len(pca_results), 
                                 figsize=(2.5*len(pca_results), 2.5), squeeze=False)
        axes = axes.flatten()

        for i, (group_name, result) in enumerate(pca_results.items()):
            ax = axes[i]
            
            if result[0] is None:  # Check if PCA was performed successfully
                ax.text(0.5, 0.5, f"Not enough data for PCA in group {group_name}", 
                        ha='center', va='center')
                continue
            
            _, explained_variance, _ , _= result
            components = range(1, len(explained_variance) + 1)
            color = self.tissue_colors.get(group_name, 'black')
            ax.bar(components, explained_variance, color=color, alpha=1)
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title(f'Explained Variance by PC - {group_name}')
            ax.set_xticks(components)
            ax.set_ylim(0, 1)
        plt.tight_layout()
        return fig

    def plot_pca_biplot(self, pca_results, treatment_column, dependent_vars):
        # Definir el diccionario de nombres de variables
        variable_name = {
            'OHdG': '8-OHdG',
            'mC': '5-mC',
            'RTL': 'T/S ratio',
            'Cumulative_Mortality': 'Mortality',
            'Survival_Probability': 'Survival',
            'Body_Mass': 'Body mass'
        }
        
        fig, axes = plt.subplots(nrows=1, ncols=len(pca_results), figsize=(3.5, 3.5), squeeze=False)
                                 #figsize=(2.5*len(pca_results), 2.5), squeeze=False)
    
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 8
        plt.rcParams['font.weight'] = 'bold'
        axes = axes.flatten()
    
        for i, (group_name, result) in enumerate(pca_results.items()):
            ax = axes[i]
            
            if result[0] is None or result[0].empty:
                ax.text(0.5, 0.5, f"Not enough data for PCA in group {group_name}", 
                        ha='center', va='center')
                continue
    
            pca_df, explained_variance, pca, loadings = result
            
            # Get the original data for this group
            if group_name == 'all_data':
                group_data = self.data
            else:
                group_data = self.data[self.data[self.group_column] == group_name]
            # Reset index for both DataFrames to ensure alignment
            pca_df = pca_df.reset_index(drop=True)
            group_data = group_data.reset_index(drop=True)
    
            # Plot scores
            treatments = group_data[treatment_column].unique()
            for treatment in treatments:
                mask = group_data[treatment_column] == treatment
                scores = pca_df[mask]
                
                ax.scatter(scores['PC1'], scores['PC2'], 
                           label=treatment, alpha=1, color=self.colors[treatment], s=15)
                ax.scatter(scores['PC1'].mean(), scores['PC2'].mean(), 
                           label=treatment, alpha=1, color=self.colors[treatment], s=50)
    
            # Plot loadings con los nombres actualizados
            for j, var in enumerate(dependent_vars):
                ax.arrow(0, 0, loadings[j, 0], loadings[j, 1], color='r', alpha=0.5)
                # Usar el diccionario para obtener el nombre correcto, si existe
                display_name = variable_name.get(var, var)  # Si no existe en el diccionario, usa el nombre original
                ax.text(loadings[j, 0]*1.15, loadings[j, 1]*1.15, display_name, 
                        color='r', ha='center', va='center')
    
            # Add labels and title
            ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})', fontweight='bold')
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})', fontweight='bold')
            #ax.set_title(f'PCA Biplot - {group_name}')
            ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2)
            ax.axvline(x=0, color='black', linestyle='dotted', linewidth=2)
            ax.set_ylim(-4, 4)
            ax.set_xlim(-4, 4)
            ax.xaxis.set_tick_params(width=2, length=5)
            ax.yaxis.set_tick_params(width=2, length=5) 
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            # Add a grid for better readability
            ax.grid(True, linestyle='--', alpha=0.6)
    
            # Make sure the aspect ratio is equal
            ax.set_aspect('equal')

        return fig
    def calculate_pca_index(self, pca_results, treatment_column, continuous_category, index_name='PCA Index'):
        index_data = {}
        for group_name, result in pca_results.items():
            if result[0] is None or result[0].empty:
                index_data[group_name] = None
                continue

            pca_df, explained_variance, pca, loadings = result
            
            # Get the original data for this group
            if group_name == 'all_data':
                group_data = self.data
            else:
                group_data = self.data[self.data[self.group_column] == group_name]

            # Reset index for both DataFrames to ensure alignment
            pca_df = pca_df.reset_index(drop=True)
            group_data = group_data.reset_index(drop=True)

            # Calculate PCA index
            pca_index = pca_df['PC1'] * np.sqrt(explained_variance[0]) + pca_df['PC2'] * np.sqrt(explained_variance[1])
            
            # Add PCA index to group_data
            group_data[index_name] = pca_index
            
            #print(group_data.head())

            shapiro_stat, p_val_norm = stats.shapiro(group_data['PCA Index'])
            print(f'{group_name} Shapiro test: {shapiro_stat} | p-value: {p_val_norm}')
            levene_stat, p_val_homo = stats.levene(*[group_data[group_data[treatment_column] == group]['PCA Index'].values for group in group_data[treatment_column].unique()])
            print(f'{group_name} Levene test: {levene_stat} | p-value: {p_val_homo}')

            index_data[group_name] = {
                'group_data': group_data,
                'index_name': index_name,
                'treatment_column': treatment_column,
                'continuous_category': continuous_category
            }

        
        return index_data
        
    def statistical_analysis_pca_index(self, index_data, post_hoc_method='conover', post_hoc_correction='holm', output_dir=None):
        """
        Realiza análisis estadístico de Kruskal-Wallis para el índice PCA.
        
        Parámetros:
        - index_data (dict): Diccionario que contiene la información procesada de cada grupo (resultado de `calculate_pca_index`)
        """
        
        # Importar la clase StatisticalAnalysis desde el archivo .py
        from src.kruskal_wallis import StatisticalAnalysis
        
        # Preparar los datos y variables. 
        """ group_name clave del diccionario que agrupa los datos y variables por una de las variables categoricas (Tissue: brain, muscle)
            data son los valores del diccionario    
        """
        kw_results_all = {}
        posthoc_results_all = {}
        cld_results_all = {}

        for group_name, data in index_data.items(): #itera por cada Tissue para filtrar (agrupar los datos por Tissue)
            group_data = data['group_data']      #se obtienen todos los datos desde index_data (datos completos de entrada)
            index_name = data['index_name']      #Contiene la variable dependiente de interes filtrada desde index_data
            treatment_column = data['treatment_column']    #Contiene la variable treatment_column (RTR WTR) filtrada desde index_data
            continuous_category = data['continuous_category']   #Contiene la variable continuous_category (tiempos) filtrada desde index_data

            print(f"\nResultados del análisis de Kruskal-Wallis para el grupo: {group_name}")
            # Crear instancia de la clase y realizar análisis de Kruskal-Wallis
            kw_ph = StatisticalAnalysis(group_data)
            kw_results, post_hoc_results, cld_results = kw_ph.run_full_analysis(
                dependent_vars = [index_name], 
                independent_vars = [treatment_column, continuous_category], 
                post_hoc_method=post_hoc_method, 
                post_hoc_correction=post_hoc_correction                
            )

            # Store results in dicts keyed by sanitized group name
            safe_name = str(group_name).replace(' ', '_')
            kw_results_all[safe_name] = kw_results
            posthoc_results_all[safe_name] = post_hoc_results
            cld_results_all[safe_name] = cld_results

            # Concatenate CLD into self.cld_results later; but save files now if output_dir provided
            if output_dir is not None:
                outp = Path(output_dir)
                outp.mkdir(parents=True, exist_ok=True)
                try:
                    # Save Kruskal-Wallis results
                    kw_results.to_excel(outp / f'kruskal_wallis_{safe_name}.xlsx', index=False)
                    print(f"Saved Kruskal-Wallis results to {outp / f'kruskal_wallis_{safe_name}.xlsx'}")
                except Exception as e:
                    try:
                        kw_results.to_csv(outp / f'kruskal_wallis_{safe_name}.csv', index=False)
                        print(f"Saved Kruskal-Wallis results CSV to {outp / f'kruskal_wallis_{safe_name}.csv'}")
                    except Exception as e2:
                        print(f"Warning: could not save Kruskal-Wallis results for {group_name}: {e2}")

                try:
                    post_hoc_results.to_excel(outp / f'posthoc_{safe_name}.xlsx', index=True)
                    print(f"Saved post-hoc results to {outp / f'posthoc_{safe_name}.xlsx'}")
                except Exception as e:
                    try:
                        post_hoc_results.to_csv(outp / f'posthoc_{safe_name}.csv')
                        print(f"Saved post-hoc results CSV to {outp / f'posthoc_{safe_name}.csv'}")
                    except Exception as e2:
                        print(f"Warning: could not save post-hoc results for {group_name}: {e2}")

                try:
                    cld_results.to_excel(outp / f'cld_{safe_name}.xlsx', index=False)
                    print(f"Saved CLD results to {outp / f'cld_{safe_name}.xlsx'}")
                except Exception as e:
                    try:
                        cld_results.to_csv(outp / f'cld_{safe_name}.csv', index=False)
                        print(f"Saved CLD results CSV to {outp / f'cld_{safe_name}.csv'}")
                    except Exception as e2:
                        print(f"Warning: could not save CLD results for {group_name}: {e2}")

        # After loop, concatenate CLD results for plotting convenience
        try:
            if len(cld_results_all) > 0:
                self.cld_results = pd.concat(list(cld_results_all.values()), ignore_index=True)
            else:
                self.cld_results = None
        except Exception:
            self.cld_results = None

        return kw_results_all, posthoc_results_all, self.cld_results
            

    
    def plot_pca_index(self, index_data):
        fig, axes = plt.subplots(nrows=1, ncols=len(index_data), figsize=(3.5, 3.5), squeeze=False)
                                 #figsize=(2.5*len(index_data), 2.5), squeeze=False)
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 8
        plt.rcParams['font.weight'] = 'bold'
        axes = axes.flatten()

        for i, (group_name, data) in enumerate(index_data.items()):
            ax = axes[i]
            
            if data is None:
                ax.text(0.5, 0.5, f"Not enough data for PCA in group {group_name}", 
                        ha='center', va='center')
                continue

            group_data = data['group_data']
            index_name = data['index_name']
            treatment_column = data['treatment_column']
            continuous_category = data['continuous_category']

            categories = sorted(group_data[continuous_category].unique())
            treatments = sorted(group_data[treatment_column].unique())
            
            # Set positions and width for box plots
            positions = np.arange(len(categories))*2
            width = 0.8
            
            # Create box plots for each treatment
            for j, treatment in enumerate(treatments):
                treatment_data = [group_data[(group_data[continuous_category] == cat) & 
                                             (group_data[treatment_column] == treatment)][index_name] 
                                  for cat in categories]
                
                bp = ax.boxplot(treatment_data, positions=positions + (j-0.5)*width, 
                                widths=width, 
                                boxprops=dict(facecolor=self.fills[treatment], color=self.colors[treatment], linewidth=2),
                                whiskerprops=dict(color=self.colors[treatment], linewidth=2),
                                medianprops=dict(color=self.colors[treatment], linewidth=2),
                                capprops=dict(color=self.colors[treatment], linewidth=2),
                                flierprops=dict(markerfacecolor=self.colors[treatment], marker='o', markersize=2, linestyle='none'),
                                meanprops=dict(marker='o', markerfacecolor='black', markersize=2, markeredgecolor='black'),
                                patch_artist=True, showmeans=True)
                
                for k, cat in enumerate(categories):
                    mean_value = group_data[(group_data[continuous_category] == cat) & 
                                            (group_data[treatment_column] == treatment)][index_name].mean()
                    #ax.plot(positions[k] + (j - 0.5) * width, mean_value, 
                    #        marker='o', color='black', markersize=2, label='_nolegend_')

                    # Extraer la etiqueta CLD para este grupo y categoría
                    cld_results = self.cld_results
                    cld_label = cld_results[
                        (cld_results[treatment_column] == treatment) & 
                        (cld_results[continuous_category] == str(cat))
                    ]['labels'].values
                    
                    # Agregar la etiqueta al gráfico si existe
                    max_value = group_data[(group_data[continuous_category] == cat) & 
                                            (group_data[treatment_column] == treatment)][index_name].max()
                    if len(cld_label) > 0:
                        ax.text(positions[k] + (j - 0.5) * width, max_value + 0.2, 
                                cld_label[0], ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

            # Customize the plot
            #ax.set_title(f'{index_name} by {continuous_category} - {group_name}')
            ax.set_xlabel('Time (days)', fontsize=8, fontweight='bold')
            ax.set_ylabel(f'Cumulative welfare index', fontsize=8, fontweight='bold')
            ax.set_xticks(positions)
            ax.set_xticklabels(categories, fontsize=8, fontweight='bold')
            ax.set_ylim(-3.5, 3.5)
            ax.set_xlim(min(positions) - 1, max(positions) + 1)
            ax.xaxis.set_tick_params(width=2, length=5)
            ax.yaxis.set_tick_params(width=2, length=5) 
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            #ax.grid(True, linestyle='--', alpha=0.6)
            #ax.set_aspect('equal')
        plt.tight_layout()
        return fig 

   


    def plot_combined_figures(self, pca_results, treatment_column, dependent_vars, index_data):
        # Crear una figura con 1 fila y 2 columnas
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
        
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10  # Cambiado a 10
        plt.rcParams['font.weight'] = 'bold'

        # Código para el primer subplot (PCA biplot)
        # Definir el diccionario de nombres de variables
        variable_name = {
            'OHdG': '8-OHdG %',
            'mC': '5-mC %',
            'RTL': 'T/S ratio',
            'Cumulative_Mortality': 'Mortality',
            'Survival_Probability': 'Survival',
            'Body_Mass': 'Body mass (g)'
        }
        
        # Tomar solo el primer resultado del PCA
        group_name, result = list(pca_results.items())[0]
        
        if result[0] is not None and not result[0].empty:
            pca_df, explained_variance, pca, loadings = result
            
            if group_name == 'all_data':
                group_data = self.data
            else:
                group_data = self.data[self.data[self.group_column] == group_name]
            
            pca_df = pca_df.reset_index(drop=True)
            group_data = group_data.reset_index(drop=True)

            treatments = group_data[treatment_column].unique()
            for treatment in treatments:
                mask = group_data[treatment_column] == treatment
                scores = pca_df[mask]
                
                ax1.scatter(scores['PC1'], scores['PC2'], 
                         label=treatment, alpha=1, color=self.colors[treatment], s=15)
                ax1.scatter(scores['PC1'].mean(), scores['PC2'].mean(), 
                         label=treatment, alpha=1, color=self.colors[treatment], s=50)

            for j, var in enumerate(dependent_vars):
                ax1.arrow(0, 0, loadings[j, 0], loadings[j, 1], color='r', alpha=0.5)
                display_name = variable_name.get(var, var)
                ax1.text(loadings[j, 0]*1.15, loadings[j, 1]*1.15, display_name, 
                      color='r', ha='center', va='center', fontsize=8)  # Añadido fontsize=10

            ax1.set_xlabel(f'PC1 ({explained_variance[0]:.2%})', fontweight='bold', fontsize=10)
            ax1.set_ylabel(f'PC2 ({explained_variance[1]:.2%})', fontweight='bold', fontsize=10)
            ax1.axhline(y=0, color='black', linestyle='dotted', linewidth=2)
            ax1.axvline(x=0, color='black', linestyle='dotted', linewidth=2)
            ax1.set_ylim(-4, 4)
            ax1.set_xlim(-4, 4)
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.set_aspect('equal')
            ax1.tick_params(axis='both', labelsize=10)  # Añadido para los ticks

        # Código para el segundo subplot (Index plot)
        group_name, data = list(index_data.items())[0]
        
        if data is not None:
            group_data = data['group_data']
            index_name = data['index_name']
            treatment_column = data['treatment_column']
            continuous_category = data['continuous_category']

            categories = sorted(group_data[continuous_category].unique())
            treatments = sorted(group_data[treatment_column].unique())
            
            positions = np.arange(len(categories))*2
            width = 0.8
            
            for j, treatment in enumerate(treatments):
                treatment_data = [group_data[(group_data[continuous_category] == cat) & 
                                         (group_data[treatment_column] == treatment)][index_name] 
                              for cat in categories]
                
                bp = ax2.boxplot(treatment_data, positions=positions + (j-0.5)*width, 
                             widths=width, 
                             boxprops=dict(facecolor=self.fills[treatment], color=self.colors[treatment], linewidth=2),
                             whiskerprops=dict(color=self.colors[treatment], linewidth=2),
                             medianprops=dict(color=self.colors[treatment], linewidth=2),
                             capprops=dict(color=self.colors[treatment], linewidth=2),
                             flierprops=dict(markerfacecolor=self.colors[treatment], marker='o', markersize=2, linestyle='none'),
                             meanprops=dict(marker='o', markerfacecolor='black', markersize=2, markeredgecolor='black'),
                             patch_artist=True, showmeans=True)
                
                for k, cat in enumerate(categories):
                    max_value = group_data[(group_data[continuous_category] == cat) & 
                                       (group_data[treatment_column] == treatment)][index_name].max()
                    
                    cld_results = self.cld_results
                    cld_label = cld_results[
                        (cld_results[treatment_column] == treatment) & 
                        (cld_results[continuous_category] == str(cat))
                    ]['labels'].values
                    
                    if len(cld_label) > 0:
                        ax2.text(positions[k] + (j - 0.5) * width, max_value + 0.2, 
                             cld_label[0], ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')  # Cambiado a 10

            ax2.set_xlabel('Time (days)', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Cumulative welfare index', fontsize=10, fontweight='bold')
            ax2.set_xticks(positions)
            ax2.set_xticklabels(categories, fontsize=10, fontweight='bold')
            ax2.set_ylim(-3.5, 3.5)
            ax2.set_xlim(min(positions) - 1, max(positions) + 1)
            ax2.tick_params(axis='both', labelsize=10)  # Añadido para los ticks

        # Configuración común para ambos subplots
        for ax in [ax1, ax2]:
            ax.xaxis.set_tick_params(width=2, length=5)
            ax.yaxis.set_tick_params(width=2, length=5)
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        plt.tight_layout()
        return fig

    
        
    def calculate_variable_contributions(self, pca_results, dependent_vars, n_components=2):
        """
        Calcula la contribución de cada variable original al índice PCA usando:
        (i) loadings,
        (ii) partial communalities (h²),
        (iii) coeficientes del índice ponderado.
    
        Parámetros
        ----------
        pca_results : dict
            Resultados del PCA retornados por run_pca() dentro de process_analysis().
        dependent_vars : list
            Lista con los nombres de las variables originales usadas en el PCA.
        n_components : int
            Número de componentes principales retenidas para definir el índice.
    
        Retorna
        -------
        pandas.DataFrame
            Tabla con variable, loadings, h² y coeficientes del índice.
        """
    
        results_list = []
    
        for group_name, result in pca_results.items():
    
            if result[0] is None or result[0].empty:
                continue
    
            pca_df, explained_variance, pca, loadings = result
    
            # -------------------------------
            # Subset de loadings para PC1 y PC2
            # loadings shape = (n_variables, n_PCs)
            # -------------------------------
            loadings_subset = loadings[:, :n_components]
    
            # -------------------------------
            # 1. Partial communalities (h²)
            # -------------------------------
            partial_communalities = np.sum(loadings_subset**2, axis=1)
    
            # -------------------------------
            # 2. Coeficientes del índice PCA:
            #    coef = ∑ loading_i,j * sqrt(var_j)
            # -------------------------------
            index_coeff = np.dot(loadings_subset, np.sqrt(explained_variance[:n_components]))
    
            # -------------------------------
            # 3. Construcción del DF
            # -------------------------------
            for i, var in enumerate(dependent_vars):
                results_list.append({
                    'variable': var,
                    'PC1_loading': loadings_subset[i, 0],
                    'PC2_loading': loadings_subset[i, 1] if n_components > 1 else np.nan,
                    'partial_communalities': partial_communalities[i],
                    'index_coeff': index_coeff[i]
                })
    
        contributions_df = pd.DataFrame(results_list)
        print(contributions_df)
        return contributions_df
         
    def process_analysis(self, group=False, by_variable=None, rotate= False, treatment_column=None, continuous_category=None, 
                         post_hoc_method='conover', post_hoc_correction='holm', corr_method = 'pearson', scale_data=False,
                         test_size=0.2, n_estimators=100, random_state=42, cv=5, use_scaled_data=False, n_bootstrap=1000,
                         output_dir=None, show_plots=True, **kwargs):
        
        # Propagate plotting flag so that pre-analysis shows figures too
        self.group_column = by_variable  # Store the grouping column for later use
        self.show_plots = bool(show_plots)
        try:
            # also ensure pre_analysis uses same flag
            self.pre_analysis.show_plots = bool(show_plots)
        except Exception:
            pass
        pca_results = self.perform_analysis(self.pca_analysis, 
                                            group=group, 
                                            by_variable=by_variable,
                                            rotate=rotate,
                                            **kwargs)
        
        # Create the explained variance plot
        variance_fig = self.plot_explained_variance(pca_results)
        
        # Create the biplot
        dependent_vars = kwargs.get('dependent_vars', [])
        biplot_fig = self.plot_pca_biplot(pca_results, treatment_column, dependent_vars)
        
        # Calculate PCA index
        index_data = self.calculate_pca_index(pca_results, treatment_column, continuous_category)
        # Calculate Statistical test (Kruskal-Wallis + post-hoc + CLD)
        kw_results_all, posthoc_results_all, cld_results = self.statistical_analysis_pca_index(
            index_data, 
            post_hoc_method=post_hoc_method,
            post_hoc_correction=post_hoc_correction,
            output_dir=output_dir
        )
        # Create the PCA index plot
        index_fig = self.plot_pca_index(index_data)

        fig1 = self.plot_combined_figures(pca_results, treatment_column, dependent_vars, index_data)

        pca_contribution = self.calculate_variable_contributions(pca_results, dependent_vars, n_components=2)
        # Ensure output directory exists and save results
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save PCA contributions
            try:
                pca_contribution.to_excel(output_dir / 'PCA_contributions.xlsx', index=False)
            except Exception:
                pass

            # Collect figures: include pre-analysis plots if present
            figs = []
            try:
                figs.extend(self.pre_analysis.plots)
            except Exception:
                pass
            for fig in [variance_fig, biplot_fig, index_fig, fig1]:
                if fig is not None:
                    figs.append(fig)

            
            try:
                if variance_fig is not None:
                    try:
                        variance_fig.savefig(output_dir / 'explained_variance.pdf')
                        print(f"Saved explained variance to {output_dir / 'explained_variance.pdf'}")
                    except Exception:
                        try:
                            variance_fig.savefig(output_dir / 'explained_variance.png')
                            print(f"Saved explained variance PNG to {output_dir / 'explained_variance.png'}")
                        except Exception as e:
                            print(f"Warning: could not save explained variance figure: {e}")

                if fig1 is not None:
                    try:
                        fig1.savefig(output_dir / 'PCA_biplot_index.pdf')
                        print(f"Saved PCA biplot+index to {output_dir / 'PCA_biplot_index.pdf'}")
                    except Exception:
                        try:
                            fig1.savefig(output_dir / 'PCA_biplot_index.png')
                            print(f"Saved PCA biplot+index PNG to {output_dir / 'PCA_biplot_index.png'}")
                        except Exception as e:
                            print(f"Warning: could not save PCA biplot+index figure: {e}")

            except Exception as e:
                print(f"Warning: error saving individual key figures: {e}")

        print("Saved all cumulative welfare figures to output directory.")

        return pca_results, variance_fig, biplot_fig,  index_data, index_fig, fig1, pca_contribution