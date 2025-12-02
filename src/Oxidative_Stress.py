import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import kruskal, rankdata
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import warnings
warnings.filterwarnings("ignore")
import itertools
import string

class Oxidative_stress_Data:
    def __init__(self, file_path):
        self.file_path = file_path
        self.radical_data = None
        self.enzime_data = None
        self.Intensity_df = None
        self.Ox_Stress_df = None
        self.colors = {}
        self.fills = {}
        
    def load_data(self):
        try:
            self.radical_data = pd.read_excel(os.path.join(self.file_path, "Total_Radicals.xlsx"))
            self.enzime_data = pd.read_excel(os.path.join(self.file_path, "Enzime_Expression.xlsx"))
            print("Datos cargados exitosamente.")
        except Exception as e:
            print(f"Se produjo un error inesperado: {e}")
            
        column_names = self.enzime_data.columns.tolist()
        print(column_names)
        return self.radical_data, self.enzime_data
    
    
    def Estimate_Intensity_AU_g(self):
        radical_data_copy = self.radical_data.copy()
        blank_data = radical_data_copy[radical_data_copy['Group'] == 'DMPO-DMSO']

        radical_data_copy['Intensity-blank_[A.U]'] = None

        for index, row in radical_data_copy.iterrows():
            if row['Group'] in ['Control', 'RTR', 'WTR']:
                blank_intensity = blank_data[blank_data['Field_[G]'] == row['Field_[G]']]['Intensity_[A.U]'].mean()

                radical_data_copy.at[index, 'Intensity-blank_[A.U]'] = row['Intensity_[A.U]'] - blank_intensity
        print("La intensidad normalizada por el blanco ha sido calculada en la columna 'Intensity-blank_[A.U]'")

        radical_data_copy['Intensity_[A.U/g]'] = radical_data_copy['Intensity-blank_[A.U]'] / radical_data_copy['Sample_mass_[g]']

        self.Intensity_df = radical_data_copy
        
        print("La intensidad por gramo de muestra ha sido calculada")
        print("Nuevo dataframe 'Intensity_df' ha sido creado")
        print("La intensidad por gramo de muestra ha incorporada a Intensity_df['Intensity_[A.U/g]']")
        
        return self.Intensity_df
    
    def X_DMPO_AU_g(self):
        results = []

        self.Intensity_df = self.Intensity_df[self.Intensity_df['Group']!='DMPO-DMSO']
        unique_combinations = self.Intensity_df[['Time', 'Tissue', 'Group', 'Individual']].drop_duplicates()
    
        for _, combo in unique_combinations.iterrows():
            filtered_data = self.Intensity_df[
                (self.Intensity_df['Time'] == combo['Time']) &
                (self.Intensity_df['Tissue'] == combo['Tissue']) &
                (self.Intensity_df['Group'] == combo['Group']) &
                (self.Intensity_df['Individual'] == combo['Individual'])
            ]

            field_values = filtered_data['Field_[G]'].values
            intensity_values = filtered_data['Intensity_[A.U/g]'].values
 
            area = np.trapz(intensity_values, field_values)

            results.append({
                'Time': combo['Time'],
                'Tissue': combo['Tissue'],
                'Group': combo['Group'],
                'Individual': combo['Individual'],
                'X_DMPO_(A.U/g)': area
            })
            
        print("La concentración relativa de rádicales libre totales ha sido calculada mdiante el área bajo la curva")

        Ox_Stress_df = pd.DataFrame(results)
        
        print("Nuevo dataframe 'Ox_Stress_df' ha sido creado")
        print("La concentración relativa de rádicales libre totales ha sido incorporada al Ox_Stress_df['X_DMPO_(A.U/g)']")

        control_means = Ox_Stress_df[Ox_Stress_df['Group'] == 'Control'].groupby('Tissue')['X_DMPO_(A.U/g)'].mean().reset_index()
        control_means.columns = ['Tissue', 'Control_X_DMPO_(A.U/g)_mean']
        Ox_Stress_df = pd.merge(Ox_Stress_df, control_means, on='Tissue', how='left')
        Ox_Stress_df['X_DMPO_AU_g_FC'] = (Ox_Stress_df['X_DMPO_(A.U/g)'] - Ox_Stress_df['Control_X_DMPO_(A.U/g)_mean']) / Ox_Stress_df['Control_X_DMPO_(A.U/g)_mean']
        self.Ox_Stress_df = Ox_Stress_df
        print("El fold change Ox_Stress_df['X_DMPO_(A.U/g)'] ha sido calculada e incorporada al Ox_Stress_df['X_DMPO_(A.U/g)_FC']")
        return Ox_Stress_df
    
    def Relative_gene_expression(self):
        grouped = self.enzime_data.groupby(['Time', 'Tissue', 'Group', 'Individual'])
        agg_df = grouped.agg({
            'Cq_ef1a': ['mean', 'std'],
            'Cq_sod1': ['mean', 'std'],
            'Cq_gpx1': ['mean', 'std']
        }).reset_index()

        agg_df.columns = ['Time', 'Tissue', 'Group', 'Individual',
                          'Cq_ef1a_mean', 'Cq_ef1a_sd',
                          'Cq_sod1_mean', 'Cq_sod1_sd',
                          'Cq_gpx1_mean', 'Cq_gpx1_sd']

        self.Ox_Stress_df = pd.merge(self.Ox_Stress_df, agg_df,
                                     on=['Time', 'Tissue', 'Group', 'Individual'], how='left')

        control_means = self.enzime_data[self.enzime_data['Group'] == 'Control'].groupby('Tissue').agg({
            'Cq_ef1a': 'mean',
            'Cq_sod1': 'mean',
            'Cq_gpx1': 'mean'
        }).reset_index()
    
        control_means.columns = ['Tissue', 'Control_Cq_ef1a_mean', 'Control_Cq_sod1_mean', 'Control_Cq_gpx1_mean']

        self.Ox_Stress_df = pd.merge(self.Ox_Stress_df, control_means, on='Tissue', how='left')
        
        print("Los promedios y SD de cq de cada gen por réplica técnica fueron calculados e insertados en Ox_Stress_df")

        self.Ox_Stress_df['Delta_Cq_sod1'] = self.Ox_Stress_df['Cq_sod1_mean'] - self.Ox_Stress_df['Cq_ef1a_mean']
        self.Ox_Stress_df['Delta_Cq_gpx1'] = self.Ox_Stress_df['Cq_gpx1_mean'] - self.Ox_Stress_df['Cq_ef1a_mean']

        self.Ox_Stress_df['DD_Cq_sod1'] = self.Ox_Stress_df['Delta_Cq_sod1'] - (self.Ox_Stress_df['Control_Cq_sod1_mean'] - self.Ox_Stress_df['Control_Cq_ef1a_mean'])
        self.Ox_Stress_df['DD_Cq_gpx1'] = self.Ox_Stress_df['Delta_Cq_gpx1'] - (self.Ox_Stress_df['Control_Cq_gpx1_mean'] - self.Ox_Stress_df['Control_Cq_ef1a_mean'])

        self.Ox_Stress_df['Rel_Expression_sod1'] = 2 ** (-self.Ox_Stress_df['DD_Cq_sod1'])
        self.Ox_Stress_df['Rel_Expression_gpx1'] = 2 ** (-self.Ox_Stress_df['DD_Cq_gpx1'])
    
        print("La expresión relativa de SOD1 y GPX1 fueron calculadas e insertadas en el Ox_Stress_df")
        print(self.Ox_Stress_df.head())
    
        return self.Ox_Stress_df

    def process_Oxidative_stress_Data(self):
        """Método principal para procesar los datos a través de los pasos definidos."""
        self.load_data()
        self.Estimate_Intensity_AU_g()
        self.X_DMPO_AU_g()
        self.Relative_gene_expression()

class Data_pre_analysis:
    def __init__(self, ox_stress_data):
        self.ox_stress_data = ox_stress_data
        self.Intensity_df = self.ox_stress_data.Intensity_df
        self.df = self.ox_stress_data.Ox_Stress_df.copy()
        self.df = self.df[self.df['Group'] != 'Control']
        self.plots = []
        self.colors = self.ox_stress_data.colors
        self.fills = self.ox_stress_data.fills
        self.output_dir = None  
        
        
    def data_description(self, df=None):
        if df is None:
            df = self.df 
        
        print("\nFirst DataFrame rows:")
        print(df.head())
        
        print("\nDataFrame Information:")
        print(df.info())

        print("\nStatistical summary:")
        print(df.describe(include='all'))
        
        column_names = df.columns.tolist()
        print(column_names)
    
    def outliers_detection(self, outliers_variables, z_score=3):
        df_outliers = self.df.copy()
        outliers_list = []
    
        for num_var in outliers_variables:
            num_var_zscore = num_var + '_zscore'
            df_outliers[num_var_zscore] = np.abs(stats.zscore(df_outliers[num_var].dropna()))

            df_outliers[num_var + '_is_outlier'] = df_outliers[num_var_zscore] > z_score

            num_outliers = df_outliers[num_var + '_is_outlier'].sum()

            outliers_values = df_outliers.loc[df_outliers[num_var + '_is_outlier'], num_var].tolist()

            outliers_list.append({
                'Variable': num_var,
                'Num_Outliers': num_outliers,
                'Outliers_Values': outliers_values
            })
        
        outliers_summary = pd.DataFrame(outliers_list)  
        return outliers_summary
    
    def distribution_analysis(self, distribution_variables, categorical_variable):
        results_tests = []
        
        tissues = self.df['Tissue'].unique()
        
        for tissue in tissues:
            tissue_df = self.df[self.df['Tissue'] == tissue]
            
            for var in distribution_variables:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Histograma y densidad
                sns.histplot(data=tissue_df, x=var, kde=True, ax=ax1)
                ax1.set_title(f'Distribution of {var} for Tissue: {tissue}')
                
                # Q-Q plot
                stats.probplot(tissue_df[var], dist="norm", plot=ax2)
                ax2.set_title(f'Q-Q Plot of {var} for Tissue: {tissue}')
                
                plt.tight_layout()
                if self.output_dir:
                    fig.savefig(self.output_dir / f'distribution_{tissue}_{var}.png', dpi=300, bbox_inches='tight')
                self.plots.append(plt.gcf())
                print(plt.gcf())
                plt.close()
    
                # Test de normalidad usando Shapiro-Wilk
                _, p_val_norm = stats.shapiro(tissue_df[var])
                is_normal = "True" if p_val_norm > 0.05 else "False"
    
                # Test de homocedasticidad
                _, p_val_homo = stats.levene(*[group[var].values for name, group in tissue_df.groupby(categorical_variable)])
                is_homogeneous = "True" if p_val_homo > 0.05 else "False"
    
                results_tests.append({
                    'Tissue': tissue,
                    'Variable': var,
                    'p_val_normality': p_val_norm,
                    'Is_normal': is_normal,
                    'p_val_homocedasticity': p_val_homo,
                    'Is_homogeneous': is_homogeneous
                })
    
        # Crear tabla de resultados
        df_results = pd.DataFrame(results_tests)
        print(df_results)
        if self.output_dir:
            df_results.to_excel(self.output_dir / 'distribution_analysis_results.xlsx', index=False)
        self.df_results_tests = df_results

    
    def variable_relationship_analysis(self, relationship_numeric_variables, relationship_categorical_variable):

        tissues = self.df['Tissue'].unique()
        
        for tissue in tissues:
            tissue_df = self.df[self.df['Tissue'] == tissue]

            unique_categories = tissue_df[relationship_categorical_variable].unique()
            missing_colors = [category for category in unique_categories if category not in self.colors]
            
            if missing_colors:
                raise ValueError(f"Colors dictionary does not have entries for the following categories: {', '.join(missing_colors)}")
            
            # Corr_matrix
            numeric_vars = [var for var in relationship_numeric_variables if var in tissue_df.columns and pd.api.types.is_numeric_dtype(tissue_df[var])]
            corr_matrix = tissue_df[numeric_vars].corr(numeric_only=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
            plt.title(f'Correlation Matrix for Tissue: {tissue}')
            if self.output_dir:
                plt.savefig(self.output_dir / f'corr_matrix_{tissue}.png', dpi=300, bbox_inches='tight')
            self.plots.append(plt.gcf())
            print(plt.gcf())
            plt.close()
    
            # Pairplot
            pairplot = sns.pairplot(tissue_df, vars=numeric_vars, hue=relationship_categorical_variable, palette=self.colors)
            if self.output_dir:
                pairplot.savefig(self.output_dir / f'pairplot_{tissue}.png', dpi=300, bbox_inches='tight')
            self.plots.append(pairplot.fig)
            plt.close()

class Oxidative_Stress_analysis:
    def __init__(self, ox_stress_data, pre_analysis):
        self.ox_stress_data = ox_stress_data
        self.Intensity_df = self.ox_stress_data.Intensity_df.copy()
        self.df = self.ox_stress_data.Ox_Stress_df.copy()
        self.df = self.df[self.df['Group'] != 'Control']
        self.plots = []
        self.colors = self.ox_stress_data.colors
        self.fills = self.ox_stress_data.fills
        self.ci_df= None
        self.output_dir = None
        
    def plot_spectra(self):
        plt.rcParams['font.family'] = 'Arial'
        tissues = self.Intensity_df['Tissue'].unique()
    
        for tissue in tissues:
            fig, axs = plt.subplots(2, 2, figsize=(6, 4))
            
            times = ['Control', 50, 100, 150]
            labels = ['A)', 'B)', 'C)', 'D)']  

            for i, time in enumerate(times):
                ax = axs[i // 2, i % 2]

                if time == 'Control':
                    control_data = self.Intensity_df[(self.Intensity_df['Tissue'] == tissue) & 
                                                     (self.Intensity_df['Group'] == 'Control')]

                    control_mean = control_data.groupby('Field_[G]')['Intensity_[A.U/g]'].mean()
                    for individual in control_data['Individual'].unique():
                        individual_data = control_data[control_data['Individual'] == individual]
                        ax.plot(individual_data['Field_[G]'], individual_data['Intensity_[A.U/g]'], 'o', color='grey',
                                alpha=0.5, markersize=0.5)

                    ax.plot(control_mean.index, control_mean.values, '-', color='black', label='Promedio Control', linewidth=1)

                else:
                    treatment_data = self.Intensity_df[(self.Intensity_df['Tissue'] == tissue) & (self.Intensity_df['Time'] == time)]

                    for group in ['RTR', 'WTR']:
                        group_data = treatment_data[treatment_data['Group'] == group]
                        group_mean = group_data.groupby('Field_[G]')['Intensity_[A.U/g]'].mean()

                        for individual in group_data['Individual'].unique():
                            individual_data = group_data[group_data['Individual'] == individual]
                            ax.plot(individual_data['Field_[G]'], individual_data['Intensity_[A.U/g]'], 'o',
                                    color=self.fills[group], markersize=0.5)
                        ax.plot(group_mean.index, group_mean.values, '-', color=self.colors[group], label=f'Promedio {group}', linewidth=1)

                ax.text(-0.15, 1.1, labels[i], transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')
                
                ax.set_xlabel('Field (G)', fontsize=10, fontweight='bold')
                ax.set_ylabel('Intensity [AU/g]', fontsize=10, fontweight='bold')
                ax.xaxis.set_tick_params(width=2, length=5, labelsize=8)
                ax.yaxis.set_tick_params(width=2, length=5, labelsize=8)
                ax.set_xticks(range(3480, 3560, 20))
                ax.axhline(y=0, color='black', linestyle='dotted', linewidth=1.5)

                if tissue == 'brain':
                    ax.set_ylim(-20, 20)
                else:
                    ax.set_ylim(-2, 2)
                    
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
                    

            plt.tight_layout()

            if self.output_dir:
                fig.savefig(self.output_dir / f'{tissue}_spectra.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
        
    def kruskal_wallis_test(self):
        variables = ['X_DMPO_AU_g_FC','Rel_Expression_sod1', 'Rel_Expression_gpx1']
        results = []
        for tissue in self.df['Tissue'].unique():
            df_tissue = self.df[self.df['Tissue'] == tissue]
            for var in variables:
                df_kruskal = df_tissue[['Group', 'Time', var]].copy()
                df_kruskal['Group_Time'] = df_kruskal['Group'] + ' ' + df_kruskal['Time'].astype(str)
                groups = df_kruskal['Group_Time'].unique()
                data = [df_kruskal[df_kruskal['Group_Time'] == group][var] for group in groups]
                stat, p_value = kruskal(*data)
                results.append({
                    'Tissue': tissue,
                    'Variable': var,
                    'Statistic': stat,
                    'p-value': p_value
                })
        kw_df = pd.DataFrame(results)
        print("\nResultados de la Prueba Kruskal-Wallis:")
        print(kw_df)
        if self.output_dir:
            kw_df.to_excel(self.output_dir / 'Oxidative_stress_Kruskal-Wallis.xlsx', index=False)
        return kw_df
    
    def conover_iman_posthoc(self):
        variables = ['X_DMPO_AU_g_FC', 'Rel_Expression_sod1', 'Rel_Expression_gpx1']
        results = []
        
        for tissue in self.df['Tissue'].unique():
            df_tissue = self.df[self.df['Tissue'] == tissue]
            
            for var in variables:
                df_posthoc = df_tissue[['Group', 'Time', var]].copy()
                df_posthoc['Group_Time'] = df_posthoc['Group'] + ' ' + df_posthoc['Time'].astype(str)

                posthoc_results = sp.posthoc_conover(df_posthoc, val_col=var, group_col='Group_Time', p_adjust='holm')
                
                posthoc_df = pd.DataFrame(posthoc_results)
                
                groups = posthoc_df.index.tolist()
                
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        group1 = groups[i]
                        group2 = groups[j]
                        p_value = posthoc_df.loc[group1, group2]
                        
                        results.append({
                            'Tissue': tissue,
                            'Variable': var,
                            'group1': group1,
                            'group2': group2,
                            'p-adj': p_value
                        })

        ci_df = pd.DataFrame(results)
        self.ci_df = ci_df
        if self.output_dir:
            ci_df.to_excel(self.output_dir / 'Oxidative_stress_posthoc.xlsx', index=False)
        print(ci_df)
        return ci_df
  
    def assign_significance_letters(self):
        original_df = self.ci_df
        CI = 95
        alpha = 1 - CI/100

        variables = original_df["Variable"].unique()
        tissues = original_df["Tissue"].unique()

        all_results = []

        for variable in variables:
            for tissue in tissues:
                df = original_df[(original_df['Variable'] == variable) & (original_df['Tissue'] == tissue)].copy()

                if len(df.index) < 2:
                    df = df.rename(columns={"p-unc": "pval"})
                else:
                    df = df.rename(columns={"p-corr": "pval"})

                df.rename({'A': 'group1', 'B': 'group2', "pval": "p-adj"}, axis="columns", inplace=True)

                df["p-adj"] = df["p-adj"].astype(float)

                group1 = set(df.group1.tolist())
                group2 = set(df.group2.tolist())
                groupSet = group1 | group2
                groups = list(groupSet)

                letters = list(string.ascii_lowercase + string.digits)[:len(groups)]
                cldgroups = letters

                cld = pd.DataFrame(list(zip(groups, letters, cldgroups)))
                cld[3] = ""

                for row in df.itertuples():
                    if df["p-adj"][row[0]] > (alpha):
                        cld.iat[groups.index(df["group1"][row[0]]), 2] += cld.iat[groups.index(df["group2"][row[0]]), 1]
                        cld.iat[groups.index(df["group2"][row[0]]), 2] += cld.iat[groups.index(df["group1"][row[0]]), 1]

                    if df["p-adj"][row[0]] < (alpha):
                        cld.iat[groups.index(df["group1"][row[0]]), 3] += cld.iat[groups.index(df["group2"][row[0]]), 1]
                        cld.iat[groups.index(df["group2"][row[0]]), 3] += cld.iat[groups.index(df["group1"][row[0]]), 1]

                cld[2] = cld[2].apply(lambda x: "".join(sorted(x)))
                cld[3] = cld[3].apply(lambda x: "".join(sorted(x)))
                cld.rename(columns={0: "groups"}, inplace=True)

                cld["labels"] = ""
                letters = list(string.ascii_lowercase)
                unique = []
                for item in cld[2]:
                    for fitem in cld["labels"].unique():
                        for c in range(0, len(fitem)):
                            if not set(unique).issuperset(set(fitem[c])):
                                unique.append(fitem[c])
                    g = len(unique)

                    for kitem in cld[1]:
                        if kitem in item:
                            forbidden = set()
                            for row in cld.itertuples():
                                if letters[g] in row[5]:
                                    forbidden |= set(row[4])
                            if kitem in forbidden:
                                g = len(unique) + 1

                            if cld["labels"].loc[cld[1] == kitem].iloc[0] == "":
                                cld["labels"].loc[cld[1] == kitem] += letters[g]

                            if len(set(cld["labels"].loc[cld[1] == kitem].iloc[0]).intersection(cld.loc[cld[2] == item, "labels"].iloc[0])) <= 0:
                                if letters[g] not in list(cld["labels"].loc[cld[1] == kitem].iloc[0]):
                                    cld["labels"].loc[cld[1] == kitem] += letters[g]
                                if letters[g] not in list(cld["labels"].loc[cld[2] == item].iloc[0]):
                                    cld["labels"].loc[cld[2] == item] += letters[g]

                            if kitem in forbidden:
                                g -= 1

                cld = cld.sort_values("labels")
                cld.drop(columns=[1, 2, 3], inplace=True)
                cld['Tissue'] = tissue
                cld['Variable'] = variable

                all_results.append(cld)

        cld_df = pd.concat(all_results, ignore_index=True)
        cld_df = cld_df[['Tissue', 'Variable', 'groups', 'labels']]
        cld_df = cld_df.rename(columns={'groups': 'groupXtime', 'labels': 'Label'})
        def split_group_time(groupXtime):
            parts = groupXtime.split()
            group = parts[0]
            time = parts[1] if len(parts) > 1 else ''
            return pd.Series({'Group': group, 'Time': time})

        cld_df[['Group', 'Time']] = cld_df['groupXtime'].apply(split_group_time)
        cld_df = cld_df[['Tissue', 'Variable', 'groupXtime', 'Group', 'Time', 'Label']]
        cld_df = cld_df.sort_values(['Tissue', 'Variable', 'groupXtime'])

        if self.output_dir:
            cld_df.to_excel(self.output_dir / 'Oxidative_stress_cld.xlsx', index=False)
        return cld_df
   
    def bar_plots_with_error(self):
        plt.rcParams['font.family'] = 'Arial'
        variables = ['X_DMPO_AU_g_FC', 'Rel_Expression_sod1', 'Rel_Expression_gpx1']
        tissues = ['brain', 'muscle']
        bar_width = 0.3
        bar_spacing = 0.32

        significance_letters = self.assign_significance_letters()
        
        fig, axes = plt.subplots(len(variables), len(tissues), figsize=(7, 9))
 
        subplot_labels = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)']
        label_index = 0
        
        for i, var in enumerate(variables):
            for j, tissue in enumerate(tissues):
                ax = axes[i, j]

                ax.text(-0.15, 1.1, subplot_labels[label_index], transform=ax.transAxes, 
                        fontsize=10, fontweight='bold', va='top', ha='right')
                label_index += 1
                
                df_tissue = self.df[self.df['Tissue'] == tissue]

                letters_subset = significance_letters[
                    (significance_letters['Tissue'] == tissue) & 
                    (significance_letters['Variable'] == var)
                ]
                
                for k, group in enumerate(['RTR', 'WTR']):
                    df_group = df_tissue[df_tissue['Group'] == group]
                    means = df_group.groupby('Time')[var].mean()
                    stds = df_group.groupby('Time')[var].std()
                    index = np.arange(len(means))
                    
                    offset = bar_width / 2 + bar_spacing * k
                    
                    bars = ax.bar(index + offset, means, bar_width,
                                  color=self.fills[group], edgecolor=self.colors[group],
                                  yerr=stds, capsize=5, label=group, error_kw={'ecolor': self.colors[group], 'elinewidth': 2})
                    
                    for l, (bar, time) in enumerate(zip(bars, means.index)):
                        height = bar.get_height()
                        height_with_error = height + stds.iloc[l] + 0.05
                        height_with_error_negative = height - stds.iloc[l] - 0.05

                        letter = letters_subset[
                            (letters_subset['Group'] == group) & 
                            (letters_subset['Time'] == str(time))
                        ]['Label'].values[0] if not letters_subset.empty else ""
                        
                        if height < 0:
                            ax.text(bar.get_x() + bar.get_width() / 2., height_with_error_negative,
                                letter, ha='center', va='top', fontsize=10, fontweight='bold')
                        else:
                            ax.text(bar.get_x() + bar.get_width() / 2., height_with_error,
                                    letter, ha='center', va='bottom', fontsize=10, fontweight='bold') 
                 
                for label in ax.get_yticklabels():
                    label.set_fontsize(10)
                    label.set_fontweight('bold')
                
                if var == 'X_DMPO_AU_g_FC':
                    ax.set_ylim(-1.2, 1.2)
                    ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2)
                    ax.set_ylabel('Relative [X-DMPO] (AU/g)\n(Fold Change related to control)', fontsize=10, fontweight='bold')
                elif var == 'Rel_Expression_sod1':
                    ax.set_ylim(0, 12)
                    ax.set_ylabel('sod1 mRNA levels\n(Relative to control/ef1a)', fontsize=10, fontweight='bold')
                elif var == 'Rel_Expression_gpx1':
                    ax.set_ylim(0, 6)
                    ax.set_ylabel('gpx1a mRNA levels\n(Relative to control/ef1a)', fontsize=10, fontweight='bold')
                ax.set_xticks(index + (bar_spacing + bar_width) / 2)
                ax.xaxis.set_tick_params(width=2, length=5, labelsize=10)
                ax.yaxis.set_tick_params(width=2, length=5, labelsize=10)
                ax.set_xticklabels(['50', '100', '150'], fontsize=10, fontweight='bold')
                ax.set_xlabel('Time (days)', fontsize=10, fontweight='bold')
                
                for spine in ax.spines.values():
                    spine.set_linewidth(2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if self.output_dir:
            fig.savefig(self.output_dir / 'Oxidative_stress_barplots.pdf', format='pdf', dpi=300, bbox_inches='tight')
        self.plots.append(fig)
        plt.close()
    
    def process_Oxidative_Stress_analysis(self):
        self.plot_spectra()
        self.kruskal_wallis_test()
        self.conover_iman_posthoc()
        self.assign_significance_letters()
        self.bar_plots_with_error()

