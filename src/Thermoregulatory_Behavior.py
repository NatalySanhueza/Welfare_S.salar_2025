import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import vapeplot
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import gaussian_kde
from matplotlib import ticker
from scipy.stats import spearmanr, mannwhitneyu
from scipy.interpolate import UnivariateSpline
import matplotlib.font_manager as fm
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

class ThermoregulationDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.result_df = None
        self.result_df_cm = None
        self.chamber_df = None
        self.colors = {}
        self.fills = {}

    def read_data(self):
        self.data = pd.read_excel(self.file_path)
        print(self.data.head()) #muestra los datos de entrada
        """"Crear columana 'Fecha' y 'Hora' desde la columna 'Datetime'"""
        self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])      
        self.data['Fecha'] = self.data['Datetime'].dt.strftime('%d.%m.%Y')
        self.data['Hora'] = self.data['Datetime'].dt.hour
        cols = ['ID', 'Point', 'X', 'Y', 'Slice', 'Datetime', 'Fecha', 'Hora', 'Treatment', 'Element', 'Number', 'Element_Number']
        self.data = self.data[cols]
        
        
    def adjust_coordinates(self):
        """Ajusta las coordenadas restando las coordenadas de Corner_1."""
        self.result_df = pd.DataFrame()
        for (treatment, slice_), group in self.data.groupby(['Treatment', 'Slice']):
            reference_point = group[group['Element_Number'] == 'Corner_1'][['X', 'Y']].iloc[0]
            group['X_0'] = group['X'] - reference_point['X']
            group['Y_0'] = reference_point['Y'] - group['Y']
            self.result_df = pd.concat([self.result_df, group])

    def convert_to_cm(self):
        """Convierte las coordenadas de píxeles a centímetros usando la escala calculada."""
        self.result_df_cm = pd.DataFrame()
        for (treatment, slice_), group in self.result_df.groupby(['Treatment', 'Slice']):
            corner_1 = group[group['Element_Number'] == 'Corner_1'][['X_0', 'Y_0']].iloc[0]
            corner_7 = group[group['Element_Number'] == 'Corner_7'][['X_0', 'Y_0']].iloc[0]
            #pixel_distance = np.sqrt((corner_7['X'] - corner_1['X'])**2 + (corner_7['Y'] - corner_1['Y'])**2)
            pixel_distance = corner_7['X_0'] - corner_1['X_0']
            scale = 180 / pixel_distance
            group['X_cm'] = group['X_0'] * scale
            group['Y_cm'] = group['Y_0'] * scale
            self.result_df_cm = pd.concat([self.result_df_cm, group])

    def assign_chambers(self):
        """Asigna cada punto a su respectiva cámara basándose en las coordenadas."""
        results = []
        for (treatment, slice_), group in self.result_df_cm.groupby(['Treatment', 'Slice']):
            corners = {f'Corner_{i}': group[group['Element_Number'] == f'Corner_{i}']['X_cm'].iloc[0] for i in range(1, 8)}
            
            def assign_chamber(x):
                for i in range(1, 7):
                    if corners[f'Corner_{i}'] < x <= corners[f'Corner_{i+1}']:
                        return f'Chamber_{i}'
                return 'Out_of_bounds'
            
            group['Chamber'] = group['X_cm'].apply(assign_chamber)
            results.append(group)
        
        self.chamber_df = pd.concat(results)

    def categorize_zones(self):
        """Categoriza las filas en Top, Middle o Bottom basándose en Y_cm."""
        max_height = 30  # cm
        self.chamber_df['Zone'] = pd.cut(self.chamber_df['Y_cm'], 
                                         bins=[-float('inf'), max_height / 3, 2 * max_height / 3, float('inf')],
                                         labels=['Bottom', 'Middle', 'Top'],
                                         right=False)
        
    def assign_temperature(self):
        """Asigna una temperatura a cada cámara en función del tratamiento."""
        def get_temperature(row):
            if row['Treatment'] == 'RTR':
                return 12.04
            elif row['Treatment'] == 'WTR':
                temperatures = {
                    'Chamber_1': 9.6,
                    'Chamber_2': 11.1,
                    'Chamber_3': 12.3,
                    'Chamber_4': 13.9,
                    'Chamber_5': 15.5,
                    'Chamber_6': 16.4
                }
                return temperatures.get(row['Chamber'], np.nan)
            return np.nan
        
        self.chamber_df['Temperature'] = self.chamber_df.apply(get_temperature, axis=1)
        
    def assign_temperature_std(self):
        """Asigna la desviación estándar de la temperatura a cada cámara en función del tratamiento."""
        def get_temperature_std(row):
            if row['Treatment'] == 'RTR':
                return 0.7  # Ejemplo de desviación estándar para RTR (ajusta según tus datos)
            elif row['Treatment'] == 'WTR':
                std_temperatures = {
                    'Chamber_1': 0.5,
                    'Chamber_2': 0.7,
                    'Chamber_3': 0.4,
                    'Chamber_4': 0.6,
                    'Chamber_5': 0.4,
                    'Chamber_6': 0.5
                }
                return std_temperatures.get(row['Chamber'], np.nan)
            return np.nan
    
        # Asigna la desviación estándar de la temperatura
        self.chamber_df['Temperature_Std'] = self.chamber_df.apply(get_temperature_std, axis=1)
        

    def calculate_jacobs_index(self):
        """Calcula el índice de preferencia de Jacobs solo para las filas donde 'Element' es 'Fish'."""
        # Filtrar solo las filas donde el elemento es un Fish
        fish_df = self.chamber_df[self.chamber_df['Element'] == 'Fish'].copy()
    
        # Proporción del área ocupada por cada cámara (1/6 del área total)
        a_i = 1 / 6
    
        # Crear un nuevo DataFrame para almacenar los resultados
        jacobs_df = pd.DataFrame()
    
        for (treatment, slice_), group in fish_df.groupby(['Treatment', 'Slice']):
            # Contar el número de peces en cada cámara
            chamber_counts = group['Chamber'].value_counts(normalize=True)
            chamber_counts = chamber_counts.reindex([f'Chamber_{i}' for i in range(1, 7)], fill_value=0)
    
            # Calcular el índice de Jacobs para cada cámara
            for chamber, u_i in chamber_counts.items():
                jacobs_index = (u_i - a_i) / (u_i + a_i - 2 * u_i * a_i)
                
                # Asignar el índice a cada Fish en esa cámara
                chamber_group = group[group['Chamber'] == chamber].copy()
                chamber_group['Jacobs_Index'] = jacobs_index
                jacobs_df = pd.concat([jacobs_df, chamber_group])
    
        self.chamber_df = pd.concat([self.chamber_df[self.chamber_df['Element'] != 'Fish'], jacobs_df])
        
    def calculate_thermal_preference_index(self):
        """Calcula el índice de preferencia térmica para cada Fish del tratamiento WTR por slice."""
        # Filtrar solo los datos correspondientes a peces del tratamiento WTR
        fish_df = self.chamber_df[(self.chamber_df['Element'] == 'Fish') & (self.chamber_df['Treatment'] == 'WTR')].copy()
    
        # Crear una columna 'Thermal_Preference_Index' en self.chamber_df con valores NaN
        self.chamber_df['Thermal_Preference_Index'] = np.nan
    
        # Iterar sobre cada Slice
        for slice_, group in fish_df.groupby('Slice'):
            # Obtener el conteo de peces por recámara
            chamber_counts = group['Chamber'].value_counts().reindex([f'Chamber_{i}' for i in range(1, 7)], fill_value=0)
    
            # Verificar si el conteo de peces es mayor a 0 para evitar división por cero
            if chamber_counts.sum() > 0:
                # Inicializar el índice de preferencia térmica
                thermal_pref_index = 0
    
                # Calcular el índice de preferencia térmica usando la columna 'Temperature'
                for chamber in chamber_counts.index:
                    # Asegurarse de que estamos trabajando solo con WTR
                    chamber_temp = group[group['Chamber'] == chamber]['Temperature'].iloc[0] if not group[group['Chamber'] == chamber].empty else 0
                    thermal_pref_index += (chamber_counts[chamber] / chamber_counts.sum()) * chamber_temp
    
                # Asignar el índice solo a las filas correspondientes del grupo en el DataFrame original
                self.chamber_df.loc[group.index, 'Thermal_Preference_Index'] = thermal_pref_index

    

    def process_data(self):
        """Método principal para procesar los datos a través de los pasos definidos."""
        self.read_data()
        self.adjust_coordinates()
        self.convert_to_cm()
        self.assign_chambers()
        self.categorize_zones()
        self.assign_temperature()
        self.assign_temperature_std()
        self.calculate_jacobs_index()
        self.calculate_thermal_preference_index()

class Data_pre_analysis:
    def __init__(self, processor):
        self.processor = processor
        self.df = self.processor.chamber_df.copy() 
        self.df = self.df[self.df['Element'] == 'Fish']   
        #self.convert_hora_to_numeric()
        self.df_clean = None
        self.plots = []
        self.colors = self.processor.colors
        self.fills = self.processor.fills
                    
    def data_description(self, df=None):
        if df is None:
            df = self.df  

        print("\nFirst DataFrame rows:")
        print(df.head())
        
        print("\nDataFrame Information:")
        print(df.info())

        print("\nStatistical summary:")
        print(df.describe(include='all'))
        
    
    
    def outliers_detection(self, outliers_variables, z_score=3):
        # Copiar el DataFrame original
        df_outliers = self.df.copy()
    
        # Inicializar una lista para almacenar cada fila del resumen de outliers
        outliers_list = []
    
        for num_var in outliers_variables:
            # Filtrar solo los valores no nulos para esta variable
            valid_data = df_outliers[num_var].dropna()
            
            if len(valid_data) == 0:
                outliers_list.append({
                    'Variable': num_var,
                    'Num_Outliers': 0,
                    'Outliers_Values': []
                })
                continue
            
            # Calcular el z-score para la variable numérica
            z_scores = np.abs(stats.zscore(valid_data))
            
            # Determinar si cada dato es un outlier
            is_outlier = z_scores > z_score
            
            # Contar el número de outliers
            num_outliers = is_outlier.sum()
            
            # Obtener los valores de los outliers
            outliers_values = valid_data[is_outlier].tolist()
    
            # Añadir los resultados a la lista
            outliers_list.append({
                'Variable': num_var,
                'Num_Outliers': num_outliers,
                'Outliers_Values': outliers_values
            })
        
        outliers_summary = pd.DataFrame(outliers_list)  
        return outliers_summary
    
    def distribution_analysis(self, distribution_variables, categorical_variable):
        results_tests = []
        for var in distribution_variables:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histograma y densidad
            sns.histplot(data=self.df, x=var, kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {var}')
            
            # Q-Q plot
            stats.probplot(self.df[var], dist="norm", plot=ax2)
            ax2.set_title(f'Q-Q Plot of {var}')
            
            plt.tight_layout()
            if self.output_dir:
                fig.savefig(self.output_dir / f'distribution_{var}.png', dpi=300, bbox_inches='tight')
            self.plots.append(plt.gcf())
            print(plt.gcf())
            plt.close()

            # Test de normalidad
            _, p_val_norm = stats.normaltest(self.df[var])
            is_normal = "True" if p_val_norm > 0.05 else "False"

            # Test de homocedasticidad
            _, p_val_homo = stats.levene(*[group[var].values for name, group in self.df.groupby(categorical_variable)])
            is_homogeneous = "True" if p_val_homo > 0.05 else "False"

            results_tests.append({
                'Variable': var,
                'p_val_normality': p_val_norm,
                'Is_normal': is_normal,
                'p_val_homocedasticity': p_val_homo,
                'Is_homogeneous': is_homogeneous
            })

        # Crear tabla de resultados
        df_results = pd.DataFrame(results_tests)
        print("\nD’Agostino and Pearson test & Levene test summary")
        print(df_results)
        self.df_results_tests = df_results
    
    def variable_relationship_analysis(self, relationship_numeric_variables, relationship_categorical_variable):
        
        # Asegurarse de que todas las variables existen y son numéricas
        numeric_vars = [var for var in relationship_numeric_variables if var in self.df.columns and pd.api.types.is_numeric_dtype(self.df[var])]
        
        #Corr_matrix
        corr_matrix = self.df[relationship_numeric_variables].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Matrix')
        self.plots.append(plt.gcf())
        print(plt.gcf())
        plt.close()

        # Pairplot
        
        pairplot = sns.pairplot(self.df, vars=relationship_numeric_variables, hue=relationship_categorical_variable, palette=colors)
        self.plots.append(pairplot.fig)
        plt.show()
        plt.close()

class termoregulation_analysis:
    def __init__(self, processor, pre_analysis):
        self.processor = processor
        self.pre_analysis = pre_analysis
        self.df = self.pre_analysis.df.copy()
        self.colors = self.processor.colors
        self.fills = self.processor.fills
        self.df_Chamber = self.processor.chamber_df.copy()
        self.output_dir = None
        self.results = {}
        self.mannwhitney_stat = None
        self.mannwhitney_p_value = None
        
    def estimate_kernel_density(self):
        """
        Estima la función de densidad de Kernel para cada tratamiento
        utilizando las variables X_cm e Y_cm, etiquetando los ejes con Zone y Chamber+Temperature,
        y con una barra de color común para ambos tratamientos.
        """
        # Calcular los valores centrales de cada recámara
        fish_data = self.df[self.df['Element'] == 'Fish']
        Chamber_data = self.df_Chamber[self.df_Chamber['Element'] == 'Corner']
        grouped_data = Chamber_data.groupby(['Treatment', 'Element_Number'])['X_cm'].mean().reset_index()
        
        centers = {'RTR': [], 'WTR': []}
        for treatment in ['RTR', 'WTR']:
            treatment_data = grouped_data[grouped_data['Treatment'] == treatment]
            treatment_data = treatment_data.sort_values('Element_Number').reset_index(drop=True)
            for i in range(1, 7):
                center = (treatment_data.loc[treatment_data['Element_Number'] == f'Corner_{i+1}', 'X_cm'].values[0]-(treatment_data.loc[treatment_data['Element_Number'] == f'Corner_{i+1}', 'X_cm'].values[0] -
                          treatment_data.loc[treatment_data['Element_Number'] == f'Corner_{i}', 'X_cm'].values[0]) / 2)
                centers[treatment].append(center)
         
        # Figs
        fig, axs = plt.subplots(1, 2, figsize=(8.8, 2))
        axs[0].text(-0.001, 1.1, 'A)', transform=axs[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        axs[1].text(-0.001, 1.1, 'B)', transform=axs[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        
        max_densities = []
    
        for i, treatment in enumerate(['RTR', 'WTR']):
            treatment_data = fish_data[fish_data['Treatment'] == treatment]
            x = treatment_data['X_cm']
            y = treatment_data['Y_cm']
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            xmin, xmax = 0, 180
            ymin, ymax = 0, 30
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            z = np.reshape(kde(positions).T, xx.shape)
            max_densities.append(z.max())
            
            # plot
            cmap = plt.cm.rainbow
            im = axs[i].imshow(np.rot90(z), cmap=cmap,
                          extent=[xmin, xmax, ymin, ymax], aspect='auto')
            
            # Configurar etiquetas del eje X (Chamber + Temperature)
            chamber_temp = treatment_data.groupby('Chamber').agg({
                'X_cm': 'mean',
                'Temperature': 'first',
                'Temperature_Std': 'first'  # Asegúrate de tener esta columna con las desviaciones estándar
            }).sort_values('X_cm')
            
            # Crear etiquetas con el formato deseado
            xtick_labels = [
                f"Ch {chamber.split('_')[-1]}\n{temp:.1f} ± {temp_std:.1f}"
                for chamber, temp, temp_std in zip(chamber_temp.index, chamber_temp['Temperature'], chamber_temp['Temperature_Std'])
            ]
            
            # Asignar las etiquetas al eje X
            axs[i].set_xticks(centers[treatment])
            axs[i].set_xticklabels(xtick_labels, font='Arial', fontsize=7, fontweight='bold')
                          
            # Configurar etiquetas del eje Y (Zone)
            if i == 0:
                zone_ticks = {'Tp': 25, 'Md': 15, 'Bt': 5}
                axs[i].set_yticks(list(zone_ticks.values()))
                axs[i].set_yticklabels(list(zone_ticks.keys()), font='Arial', fontsize=7, fontweight='bold')
                axs[0].set_ylabel('Tank Zone', font='Arial', fontsize=7, fontweight='bold')
            else:
                axs[i].set_yticklabels([])  # Desactivar las etiquetas del eje Y en el segundo subplot

            
            # Ajustar grosor de ejes y ticks
        
            axs[i].tick_params(axis='both', which='major', width=2, length=4, pad=2)
            axs[i].spines['top'].set_linewidth(2)
            axs[i].spines['right'].set_linewidth(2)
            axs[i].spines['bottom'].set_linewidth(2)
            axs[i].spines['left'].set_linewidth(2)
    
        # Configurar una escala de color común
        vmax = max(max_densities)
        for i in range(2):
            axs[i].images[0].set_clim(0, vmax)
        
        # Añadir una barra de color común vertical en el lado derecho
        fig.subplots_adjust(right=0.865)
        cbar_ax = fig.add_axes([0.875, 0.11, 0.01, 0.77])
        cbar = fig.colorbar(axs[0].images[0], cax=cbar_ax, format=ticker.ScalarFormatter(useMathText=True))
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
        cbar.set_label('Density', fontsize=7, fontname='Arial', fontweight='bold')  # Ajustando el tamaño y la fuente de la etiqueta
        cbar.ax.tick_params(labelsize=6, labelright=True, width=2, length=3)  # Ajustando el tamaño de los números en la barra de color
        cbar.ax.yaxis.set_tick_params(pad=2)  # Añadiendo un poco de espacio entre los números y la barra
        cbar.outline.set_linewidth(2)

        # Cambiando la fuente de los números en la barra de color
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Arial')
            
        offset_text = cbar.ax.yaxis.get_offset_text()
        offset_text.set_fontsize(6)  # Cambia el tamaño de la fuente
        offset_text.set_fontname('Arial')  # Cambia la fuente

        # Cambiar la ubicación del texto (ajustando la posición manualmente)
        offset_text.set_position((1.8, 0))  # Ajusta los valores según la posición deseada
        
        # Ajustar el espacio entre subplots manualmente
        plt.subplots_adjust(wspace=0.04)
        fig.supxlabel('Chamber and Temperature [mean ± SD (°C)]', fontsize=8, fontweight='bold', fontname='Arial', y=-0.11)      
        return fig, axs 
        

    def preprocess_jacobs_index(self):
        """
        Preprocesa los datos para el análisis del Jacobs Index, asegurando que todas las combinaciones de Slice,
        Treatment y Chamber están representadas y que los valores faltantes son reemplazados adecuadamente.
        """
        # Primero, eliminamos los duplicados en el DataFrame original
        self.df = self.df.drop_duplicates(subset=['Slice', 'Treatment', 'Chamber'])
    
        # Obtener todas las combinaciones posibles de Slice, Treatment y Chamber
        all_combinations = pd.MultiIndex.from_product(
            [self.df['Slice'].unique(),
             self.df['Treatment'].unique(),
             ['Chamber_1', 'Chamber_2', 'Chamber_3', 'Chamber_4', 'Chamber_5', 'Chamber_6']],
            names=['Slice', 'Treatment', 'Chamber']
        )
    
        # Crear un nuevo DataFrame con todas las combinaciones
        self.preprocessed_df = pd.DataFrame(index=all_combinations).reset_index()
    
        # Fusionar con los datos originales
        self.preprocessed_df = self.preprocessed_df.merge(self.df, on=['Slice', 'Treatment', 'Chamber'], how='left')
    
        # Llenar los valores faltantes con -1
        self.preprocessed_df['Jacobs_Index'] = self.preprocessed_df['Jacobs_Index'].fillna(-1)
    
        return self.preprocessed_df
    
    def mwu_test_jacobs_index(self):
        """
        Realiza el análisis estadístico del Jacobs Index mediante el test de Mann-Whitney U.
        Devuelve los resultados del análisis.
        """
        # Preprocesar los datos
        preprocessed_df = self.preprocess_jacobs_index()
        self.results = {}
        chambers = ['Chamber_1', 'Chamber_2', 'Chamber_3', 'Chamber_4', 'Chamber_5', 'Chamber_6']
        
        for chamber in chambers:
            # Obtener los valores de Jacobs_Index para cada tratamiento
            rtr_values = preprocessed_df[(preprocessed_df['Chamber'] == chamber) & (preprocessed_df['Treatment'] == 'RTR')]['Jacobs_Index']
            wtr_values = preprocessed_df[(preprocessed_df['Chamber'] == chamber) & (preprocessed_df['Treatment'] == 'WTR')]['Jacobs_Index']
            
            # Análisis estadístico (Mann-Whitney U Test)
            stat, p_value = mannwhitneyu(rtr_values, wtr_values, alternative='two-sided')
            
            # Guardar resultados estadísticos
            self.results[chamber] = {'stat': stat, 'p_value': p_value}
            
        print("\nMann-Whitney U Test Chamber preference RTR vs WTR")
        for chamber, stats in self.results.items():
            print(f"{chamber}:")
            print(f"  Statistic: {stats['stat']:.3e}")
            print(f"  P-value: {stats['p_value']:.3e}")
    
        return self.results, preprocessed_df
    
    
    def analyze_jacobs_index(self):
        """
        Realiza el análisis estadístico del Jacobs Index y genera un gráfico unificado
        con las cámaras en el eje x y el Jacobs Index en el eje y.
        """
        # Configurar la fuente Arial
        plt.rcParams['font.family'] = 'Arial'
    
        # Preprocesar los datos
        preprocessed_df = self.preprocess_jacobs_index()
        results = self.results
    
        # Crear una figura para el gráfico unificado
        fig, ax = plt.subplots(figsize=(8.8, 3))
        ax.text(-0.01, 1.1, 'C)', transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        
        chambers = ['Chamber_1', 'Chamber_2', 'Chamber_3', 'Chamber_4', 'Chamber_5', 'Chamber_6']
    
        # Listas para almacenar posiciones en el eje X para los tratamientos
        chamber_positions = np.arange(len(chambers))
        bar_width = 0.4  # Ancho de las barras de los tratamientos
    
        # Valores para agregar asteriscos según el p-valor
        def significance_indicator(p_value):
            if p_value < 0.001:
                return '***'
            elif p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return 'ns'  # No significativo
    
        for j, treatment in enumerate(['RTR', 'WTR']):
            treatment_means = []
            treatment_stds = []
    
            for i, chamber in enumerate(chambers):
                # Filtrar datos por Chamber y Tratamiento
                chamber_data = preprocessed_df[(preprocessed_df['Chamber'] == chamber) & 
                                               (preprocessed_df['Treatment'] == treatment)]
    
                # Obtener los valores de Jacobs_Index para este tratamiento y cámara
                data = chamber_data['Jacobs_Index']
    
                # Calcular la media y la desviación estándar para las barras de error
                mean_val = np.mean(data)
                std_val = np.std(data)
    
                treatment_means.append(mean_val)
                treatment_stds.append(std_val)             
                    
            # Crear gráfico de barras para este tratamiento con barras de error del mismo color
            ax.bar(chamber_positions + j * bar_width, treatment_means, 
                   yerr=treatment_stds, width=bar_width, label=treatment,
                   color=self.fills[treatment], capsize=5, edgecolor=self.colors[treatment],
                   linewidth=2,
                   error_kw={'ecolor': self.colors[treatment], 'elinewidth': 2})
    
        # Ajustes de la gráfica
        
        ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2)  # Línea en y=0
        
        # Label axis
        ax.set_xlabel('Chamber', fontsize=10, fontweight='bold')
        ax.set_ylabel("Jacobs' Preference Index", fontsize=10, fontweight='bold', labelpad=5)
        
        # Ajustar grosor de ejes y ticks
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='y', which='major', labelsize=10, width=2, length=5)
        ax.set_xticks(chamber_positions + bar_width / 2)
        ax.set_xticklabels([f"Ch {i+1}" for i in range(len(chambers))], fontsize=10, fontweight='bold')
        
        # Definir y_ticks antes de usarlo
        y_ticks = np.linspace(-1, 1, 9)  # Aquí defines y_ticks
        
        # Modificar solo esta línea para que las etiquetas de Y estén en negrita
        ax.set_yticks(y_ticks)  # Asegúrate de establecer los ticks antes de etiquetarlos
        ax.set_yticklabels([f"{tick:.2f}" for tick in y_ticks], fontsize=10, fontweight='bold')
        
        ax.yaxis.set_tick_params(width=2, length=5)
        ax.xaxis.set_tick_params(width=2, length=5)
        ax.set_ylim(-1.25, 1.25)

       # Añadir asteriscos y líneas que conectan las barras comparadas
        line_y_position = 0.9  # Altura fija de las líneas de unión y los asteriscos
        for i, chamber in enumerate(chambers):
            p_value = results[chamber]['p_value']
            indicator = significance_indicator(p_value)
    
            if indicator != 'ns':  # Solo agregar líneas si hay significancia
                # Dibujar una línea que conecte las dos barras a una altura fija
                x1, x2 = chamber_positions[i], chamber_positions[i] + bar_width
                ax.plot([x1, x2], [line_y_position, line_y_position], color='black', linewidth=1.5)
    
                # Añadir asteriscos sobre la línea
                ax.text((x1 + x2) / 2, line_y_position + 0.05, indicator, ha='center', fontsize=14, color='black', fontweight='bold')
    

        # Mostrar gráfico
        #plt.tight_layout()
    
        
        return fig, ax
          
    def preprocess_IPT(self):
        """
        Preprocesa el DataFrame para filtrar los datos no nulos en 'Thermal_Preference_Index',
        eliminar duplicados considerando la columna 'Slice', y crear un nuevo DataFrame con los registros seleccionados.
        """
        filtered_df = self.df[self.df['Thermal_Preference_Index'].notnull()]
        filtered_df = filtered_df.drop_duplicates(subset='Slice')
        
        
       # Crear la columna 'Time'
        def calculate_time(hour):
            if hour >= 16:
                return hour - 16  # Desde las 16 hasta las 23 (Time de 0 a 7)
            else:
                return hour + 8  # Desde las 0 hasta las 12 (Time de 8 a 20)
        
        # Aplicar la función para generar la nueva columna 'Time'
        filtered_df['Time'] = filtered_df['Hora'].apply(calculate_time)
        
        # Guardar el nuevo DataFrame con la columna 'Time'
        self.IPT_df = filtered_df.copy()
        self.IPT_df['D/N'] = np.where(self.IPT_df['Time'] <= 8, 'Dark', 'Light')
        
        return self.IPT_df
    

    def fit_IPT(self):
        """
        Realiza el análisis de 'Thermal_Preference_Index' y genera gráficos de líneas y violín.
        """
       
        # Calcular el promedio de IPT por hora
        avg_IPT_per_hour = self.IPT_df.groupby('Time')['Thermal_Preference_Index'].mean().reset_index()
    
        # Ajuste polinómico de grado 3
        z = np.polyfit(avg_IPT_per_hour['Time'], avg_IPT_per_hour['Thermal_Preference_Index'], 3)
        p = np.poly1d(z)
        avg_IPT_per_hour['Polynomial_Fit'] = p(avg_IPT_per_hour['Time'])
        
        # Aplicar suavizado con media móvil
        avg_IPT_per_hour['Smoothed_IPT'] = avg_IPT_per_hour['Thermal_Preference_Index'].rolling(window=3, min_periods=1).mean()
    
        # Ajuste spline
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(avg_IPT_per_hour['Time'], avg_IPT_per_hour['Thermal_Preference_Index'], s=0.5)
        avg_IPT_per_hour['Spline_Fit'] = spline(avg_IPT_per_hour['Time'])
        
        # Calcular la correlación de Spearman
        rho_original, p_value_original = spearmanr(avg_IPT_per_hour['Time'], avg_IPT_per_hour['Thermal_Preference_Index'])
        rho_smoothed, p_value_smoothed = spearmanr(avg_IPT_per_hour['Time'], avg_IPT_per_hour['Smoothed_IPT'])
        rho_poly, p_value_poly = spearmanr(avg_IPT_per_hour['Time'], avg_IPT_per_hour['Polynomial_Fit'])
        rho_spline, p_value_spline = spearmanr(avg_IPT_per_hour['Time'], avg_IPT_per_hour['Spline_Fit'])
        
        # Calcular el MSE
        mse_original = mean_squared_error(avg_IPT_per_hour['Thermal_Preference_Index'], avg_IPT_per_hour['Thermal_Preference_Index'])
        mse_smoothed = mean_squared_error(avg_IPT_per_hour['Thermal_Preference_Index'], avg_IPT_per_hour['Smoothed_IPT'])
        mse_poly = mean_squared_error(avg_IPT_per_hour['Thermal_Preference_Index'], avg_IPT_per_hour['Polynomial_Fit'])
        mse_spline = mean_squared_error(avg_IPT_per_hour['Thermal_Preference_Index'], avg_IPT_per_hour['Spline_Fit'])
        
        # Crear un DataFrame con los resultados de la correlación y MSE
        results_df = pd.DataFrame({
            'Método': ['Datos Originales', 'Media Móvil', 'Ajuste Polinómico', 'Ajuste Spline'],
            'Spearman\'s rho': [rho_original, rho_smoothed, rho_poly, rho_spline],
            'p-value': [p_value_original, p_value_smoothed, p_value_poly, p_value_spline],
            'MSE': [mse_original, mse_smoothed, mse_poly, mse_spline]
        })
        
        # Mostrar el DataFrame
        print("\nResultados de la correlación de Spearman y MSE:")
        print(results_df)
            
        # Crear gráficos
        fig, axs = plt.subplots(2, 2, figsize=(7, 7))
        
        # Datos Originales
        sns.lineplot(data=avg_IPT_per_hour, x='Time', y='Thermal_Preference_Index', marker='o', ax=axs[0, 0], color = '#FFDAB8')
        axs[0, 0].set_title(f'Datos Originales\nSpearman\'s rho = {rho_original:.2f}, p = {p_value_original:.3e}')
        axs[0, 0].set_xlabel('Time')
        axs[0, 0].set_ylabel('IPT Promedio')
    
        # Suavizado con Media Móvil
        sns.lineplot(data=avg_IPT_per_hour, x='Time', y='Smoothed_IPT', marker='o', ax=axs[0, 1])
        axs[0, 1].set_title(f'Suavizado con Media Móvil\nSpearman\'s rho = {rho_smoothed:.2f}, p = {p_value_smoothed:.3e}')
        axs[0, 1].set_xlabel('Time')
        axs[0, 1].set_ylabel('IPT Promedio (Suavizado)')
    
        # Ajuste Polinómico
        sns.lineplot(data=avg_IPT_per_hour, x='Time', y='Polynomial_Fit', color='red', ax=axs[1, 0])
        axs[1, 0].set_title(f'Ajuste Polinómico\nSpearman\'s rho = {rho_poly:.2f}, p = {p_value_poly:.3e}')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].set_ylabel('IPT Promedio (Polinómico)')
    
        # Ajuste Spline
        sns.lineplot(data=avg_IPT_per_hour, x='Time', y='Spline_Fit', color='green', ax=axs[1, 1])
        axs[1, 1].set_title(f'Ajuste Spline\nSpearman\'s rho = {rho_spline:.2f}, p = {p_value_spline:.3e}')
        axs[1, 1].set_xlabel('Time')
        axs[1, 1].set_ylabel('IPT Promedio (Spline)')
        
        plt.tight_layout()
        if self.output_dir:
            fig.savefig(self.output_dir / 'IPT_correlation_fits.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        
        # Evaluación de residuos 
        fig, axs = plt.subplots(1, 2, figsize=(7, 3))

        # Residuales para el ajuste Spline
        residuals_spline = avg_IPT_per_hour['Thermal_Preference_Index'] - avg_IPT_per_hour['Spline_Fit']
        sns.scatterplot(x=avg_IPT_per_hour['Time'], y=residuals_spline, ax=axs[0])
        axs[0].set_title('Residuos del Ajuste Spline')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Residuos')
        axs[0].axhline(y=0, color='r', linestyle='--')
        
        # Evaluación de residuos para el ajuste polinómico
        residuals_polynomial = avg_IPT_per_hour['Thermal_Preference_Index'] - avg_IPT_per_hour['Polynomial_Fit']
        sns.scatterplot(x=avg_IPT_per_hour['Time'], y=residuals_polynomial, ax=axs[1])
        axs[1].set_title('Residuos del Ajuste Polinómico')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Residuos')
        axs[1].axhline(y=0, color='r', linestyle='--')
        
        plt.tight_layout()
        if self.output_dir:
            fig.savefig(self.output_dir / 'IPT_residuals.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    def mwu_test_IPT(self):
        """
        Realiza el análisis estadístico del IPT mediante el test de Mann-Whitney U.
        Devuelve los resultados del análisis.
        """
        # Realizar el análisis estadístico (Mann-Whitney U test)
        self.light_data = self.IPT_df[self.IPT_df['D/N'] == 'Light']['Thermal_Preference_Index']
        self.dark_data = self.IPT_df[self.IPT_df['D/N'] == 'Dark']['Thermal_Preference_Index']
        self.mannwhitney_stat, self.mannwhitney_p_value = mannwhitneyu(self.light_data, self.dark_data, alternative='two-sided')
        
        print(f"Mann-Whitney U estadístico: {self.mannwhitney_stat:.3e}")
        print(f"P-valor: {self.mannwhitney_p_value:.3e}")
        
        #return self.light_data, self.dark_data, self.mannwhitney_stat, self.mannwhitney_p_value
    
        
    
    def analyze_IPT_with_best_fit(self):
        """
        Realiza el análisis del 'Thermal_Preference_Index' y genera una figura con dos gráficos:
        1. Gráfico de la media de IPT por hora con barras de error (desviación estándar), ajuste Spline y ajuste Polinómico.
        2. Gráfico de violín comparando los valores de IPT entre el día y la noche.
        """
        
        # Calcular la media y la desviación estándar del IPT por hora
        stats_per_hour = self.IPT_df.groupby('Time')['Thermal_Preference_Index'].agg(['mean', 'std']).reset_index()
        
        # Ajuste Spline
        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(stats_per_hour['Time'], stats_per_hour['mean'], s=0.5)
        stats_per_hour['Spline_Fit'] = spline(stats_per_hour['Time'])
        
        # Ajuste Polinómico (por ejemplo, de grado 3)
        polynomial_coeff = np.polyfit(stats_per_hour['Time'], stats_per_hour['mean'], deg=3)
        polynomial_fit = np.polyval(polynomial_coeff, stats_per_hour['Time'])
        stats_per_hour['Polynomial_Fit'] = polynomial_fit
        
        # Configuración global de la fuente (Arial)
        plt.rcParams['font.family'] = 'Arial'
        
        # Crear figura y subgráficos
        fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.5))
        axs[0].text(-0.1, 1.1, 'D)', transform=axs[0].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        axs[1].text(-0.1, 1.1, 'E)', transform=axs[1].transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
        
        # Gráfico 1: Datos originales con barras de error, ajuste Spline y ajuste Polinómico
        axs[0].errorbar(stats_per_hour['Time'], stats_per_hour['mean'], yerr=stats_per_hour['std'], 
                        fmt='o', capsize=2, label='Mean ± SD', color='#FFDAB8', markersize=4)
        axs[0].plot(stats_per_hour['Time'], stats_per_hour['Spline_Fit'], label='Spline Fit', color='#FFA040', linestyle='solid', lw=2)
        axs[0].plot(stats_per_hour['Time'], stats_per_hour['Polynomial_Fit'], label='3rd Degree Polynomial Fit', color='#FFA040', linestyle='dotted', lw=2)
        
        # Configurar etiquetas 
        
        axs[0].set_xlabel('Time (h)', fontsize=10, fontweight='bold')
        axs[0].set_ylabel('Thermal Preference Index (°C)', fontsize=10, fontweight='bold')
        axs[0].legend(fontsize=8)
        
        axs[0].axvline(x=8.2, color='gray', linestyle='dotted', linewidth=1.5)
        
        rect = patches.Rectangle((-1, 10), 9.3, 14, 
                                 linewidth=1, edgecolor='none', facecolor='lightgrey', alpha=0.1)
        axs[0].add_patch(rect)
        
        rect_2 = patches.Rectangle((8.31, 10), 12.8, 14, 
                                 linewidth=1, edgecolor='none', facecolor='lightblue', alpha=0.1)
        axs[0].add_patch(rect_2)
        
        center_x_rect = -1 + (9.2 / 2)  # Centro del rectángulo rect en el eje X
        axs[0].text(center_x_rect, 11, 'Dark', fontsize=10, color='black', fontweight='bold', 
                    ha='center', va='center')
        
        # Para el rectángulo "Light"
        center_x_rect_2 = 8.31 + ((12.8 + 0) / 2)  # Centro del rectángulo rect_2 en el eje X
        axs[0].text(center_x_rect_2, 11, 'Light', fontsize=10, color='black', fontweight='bold', 
                    ha='center', va='center')
        
        # Gráfico 2: Gráfico de violín para comparar IPT entre Día y Noche
        # Agrupar los datos para violines
        
        light_data = self.IPT_df[self.IPT_df['D/N'] == 'Light']['Thermal_Preference_Index']
        dark_data = self.IPT_df[self.IPT_df['D/N'] == 'Dark']['Thermal_Preference_Index']
        
        # Crear el gráfico de violín
        parts = axs[1].violinplot([dark_data, light_data], showmeans=False, showmedians=False, showextrema=False)
        
        # Colores para los violines
        colors = ['#FFA040', '#FFA040']
        fill_colors = ['#FFDAB8', 'white'] 
        
         # Colores para los boxplot
        colors2 = ['#FFA040', '#FFA040']
        fill_colors2 = ['white', '#FFDAB8']
        
        # Configurar los violines
        for i, pc in enumerate(parts['bodies']):
            pc.set_edgecolor(colors[i])
            pc.set_linewidth(2)
            pc.set_facecolor(fill_colors[i])
            pc.set_alpha(1)
        
        # Añadir boxplot dentro del gráfico de violín
        for i, data in enumerate([dark_data, light_data]):
            axs[1].boxplot(data, positions=[i + 1], widths=0.2, showcaps=False,
                           boxprops=dict(facecolor=fill_colors2[i], color=colors2[i], linewidth=2),
                           whiskerprops=dict(color=colors2[i], linewidth=2),
                           medianprops=dict(color=colors2[i], linewidth=2),
                           patch_artist=True)
        
        
        # Añadir los promedios de cada grupo con puntos negros
        dark_mean = np.mean(dark_data)
        light_mean = np.mean(light_data)
        
        axs[1].scatter([1], [dark_mean], color='black', s=10, marker='o', zorder=10)
        axs[1].scatter([2], [light_mean], color='black', s=10, marker='o', zorder=10)
        
        axs[1].set_xlabel('Photoperiod phase', fontsize=10, fontweight='bold')
        #axs[1].set_ylabel('Thermal Preference Index (°C)', fontsize=10, fontweight='bold')
        axs[1].set_xticks([1, 2])
        axs[1].set_xticklabels(['Dark', 'Light'])
        
        # Personalización de ejes y ticks
        for ax in axs:
            ax.spines['top'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            ax.set_ylim(10.75, 13.5)
            y_ticks = np.linspace(10, 14, 6)
            ax.set_yticks(y_ticks)
            ax.tick_params(width=2, length=4, labelsize=10)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(10)
                label.set_fontname('Arial')
                label.set_fontweight('bold')
                        
        # Realizar prueba Mann-Whitney U
        mannwhitney_stat, mannwhitney_p_value = mannwhitneyu(light_data, dark_data, alternative='two-sided')
        
        # Determinar si hay diferencias significativas y añadir los asteriscos correspondientes
        significance = ''
        if mannwhitney_p_value < 0.05:
            significance = '*'
        if mannwhitney_p_value < 0.01:
            significance = '**'
        if mannwhitney_p_value < 0.001:
            significance = '***'
        
        
        # Añadir los asteriscos dentro del gráfico, entre los dos violines
        axs[1].text(1.5, 13.25, significance, ha='center', va='center', fontsize=20, color='black', fontweight='bold')
        
        # Ajustar el diseño
        plt.subplots_adjust(wspace=0.2)
        print(dark_mean)
        print(light_mean)
        #plt.tight_layout()
        return fig, axs
    
    def save_statistical_results(self):
        """
        Guarda los resultados estadísticos en archivos Excel.
        """
        if not self.output_dir:
            return
        
        # Guardar resultados de Mann-Whitney U para Jacobs Index
        jacobs_results = []
        for chamber, stats in self.results.items():
            jacobs_results.append({
                'Chamber': chamber,
                'Statistic': stats['stat'],
                'P-value': stats['p_value']
            })
        jacobs_df = pd.DataFrame(jacobs_results)
        jacobs_df.to_excel(self.output_dir / 'MWU_Jacobs_Index_results.xlsx', index=False)
        
        # Guardar resultados de Mann-Whitney U para IPT
        ipt_results = pd.DataFrame({
            'Comparison': ['Light vs Dark'],
            'Statistic': [self.mannwhitney_stat],
            'P-value': [self.mannwhitney_p_value]
        })
        ipt_results.to_excel(self.output_dir / 'MWU_IPT_results.xlsx', index=False)


