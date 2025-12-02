from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import kruskal, rankdata
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import warnings
warnings.filterwarnings("ignore")
import string
SHOW_PLOTS = True

class DNA_Data:
    def __init__(self, file_path):
        self.file_path = file_path
        self.rtl_data = None
        self.mc_data = None
        self.ohdg_data = None
        self.Std_Curves_df = pd.DataFrame(columns=["Slope", "Intercept", "R2"], index=[])
        self.Variables_df = pd.DataFrame()
        self.colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
        self.fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}
        
    def load_data(self):
        try:
            base = Path(self.file_path)
            self.rtl_data = pd.read_csv(base / "RTL.csv")
            self.mc_data = pd.read_csv(base / "5-mC.csv")
            self.ohdg_data = pd.read_csv(base / "8-OHdG.csv")
            print("Datos cargados exitosamente.")
        except Exception as e:
            print(f"Se produjo un error inesperado: {e}")
        return self.rtl_data, self.mc_data, self.ohdg_data
          
    
    def fit_and_identify_outliers(self, X, y, threshold):
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = np.abs(y - y_pred)
        median_residual = np.median(residuals)
        mad = np.median(np.abs(residuals - median_residual))
        mad = mad if mad != 0 else 1
        outliers = np.abs(residuals - median_residual) / mad > threshold
        return X, y, y_pred, outliers, model

    def plot_regression(self, X, y, y_pred, outliers, model, ax, title, xlabel, ylabel):
        X_filtered = X[~outliers]
        y_filtered = y[~outliers]
        y_pred_filtered = model.predict(X_filtered)
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y_filtered, y_pred_filtered)

        ax.scatter(X, y, color='black', label='Datos originales')
        ax.scatter(X[outliers], y[outliers], color='red', label='Outliers')
        ax.plot(X_filtered, y_pred_filtered, color='black', label=f'Fit: Slope={slope:.2f}, Intercept={intercept:.2f}, R2={r2:.2f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        return slope, intercept, r2

    def std_curve(self):
        std_concentrations = {"Std-01": np.log10(50), "Std-02": np.log10(10), 
                              "Std-03": np.log10(2), "Std-04": np.log10(0.4), 
                              "Std-05": np.log10(0.08)}

        fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
        axs = axs.flatten()

        if "Replicate" in self.rtl_data.columns:
            self.rtl_data = self.rtl_data.groupby(["Tissue", "Treatment", "Cq_Gap8", "Cq_Tel1b"], as_index=False).mean()

        plot_idx = 0

        for tissue in ["brain_pool", "muscle_pool"]:
            tissue_data = self.rtl_data[self.rtl_data["Tissue"] == tissue]
            tissue_data = tissue_data[tissue_data["Treatment"].isin(std_concentrations.keys())]
            tissue_data["Log_Conc"] = tissue_data["Treatment"].map(std_concentrations)

            if "Cq_Gap8" in tissue_data.columns:
                X_gap8 = tissue_data["Log_Conc"].values.reshape(-1, 1)
                y_gap8 = tissue_data["Cq_Gap8"].values
                if len(y_gap8) > 1:
                    X_filtered_gap8, y_filtered_gap8, y_pred_gap8, outliers_gap8, gap8_model = self.fit_and_identify_outliers(X_gap8, y_gap8, threshold=1.4)
                    slope_gap8, intercept_gap8, r2_gap8 = self.plot_regression(X_gap8, y_gap8, y_pred_gap8, outliers_gap8, gap8_model, axs[plot_idx], f'Regresión Lineal para Cq_Gap8 ({tissue})', 'Log(Concentration)', 'Cq_Gap8')
                    self.Std_Curves_df.loc[f"Gap8_{tissue}"] = [slope_gap8, intercept_gap8, r2_gap8]
                    plot_idx += 1
                else:
                    print(f"No hay suficientes datos para realizar la regresión de Gap8 en {tissue}.")

            if "Cq_Tel1b" in tissue_data.columns:
                X_tel1b = tissue_data["Log_Conc"].values.reshape(-1, 1)
                y_tel1b = tissue_data["Cq_Tel1b"].values
                if len(y_tel1b) > 1:
                    X_filtered_tel1b, y_filtered_tel1b, y_pred_tel1b, outliers_tel1b, tel1b_model = self.fit_and_identify_outliers(X_tel1b, y_tel1b, threshold=1.4)
                    slope_tel1b, intercept_tel1b, r2_tel1b = self.plot_regression(X_tel1b, y_tel1b, y_pred_tel1b, outliers_tel1b, tel1b_model, axs[plot_idx], f'Regresión Lineal para Cq_Tel1b ({tissue})', 'Log(Concentration)', 'Cq_Tel1b')
                    self.Std_Curves_df.loc[f"Tel1b_{tissue}"] = [slope_tel1b, intercept_tel1b, r2_tel1b]
                    plot_idx += 1
                else:
                    print(f"No hay suficientes datos para realizar la regresión de Tel1b en {tissue}.")

        def process_data(data, label):
            filtered_data = data[data["Tissue"].isna()]
            filtered_data = filtered_data[filtered_data["Group"] != "NC"]
            X = filtered_data["Conc_ng/ul"].values.reshape(-1, 1)
            y = filtered_data["OD_450_nm"].values
            return X, y

        if self.mc_data is not None:
            X_mc, y_mc = process_data(self.mc_data, "MC")
            if len(y_mc) > 1:
                X_filtered_mc, y_filtered_mc, y_pred_mc, outliers_mc, mc_model = self.fit_and_identify_outliers(X_mc, y_mc, threshold=2)
                slope_mc, intercept_mc, r2_mc = self.plot_regression(X_mc, y_mc, y_pred_mc, outliers_mc, mc_model, axs[plot_idx], 'Regresión Lineal para MC Data', 'Conc_ng/ul', 'OD_450_nm')
                self.Std_Curves_df.loc["MC"] = [slope_mc, intercept_mc, r2_mc]
                plot_idx += 1
            else:
                print("No hay suficientes datos para realizar la regresión lineal en mc_data.")
                    
        if self.ohdg_data is not None:
            X_ohdg, y_ohdg = process_data(self.ohdg_data, "OHDG")
            if len(y_ohdg) > 1:
                X_filtered_ohdg, y_filtered_ohdg, y_pred_ohdg, outliers_ohdg, ohdg_model = self.fit_and_identify_outliers(X_ohdg, y_ohdg, threshold=1.5)
                slope_ohdg, intercept_ohdg, r2_ohdg = self.plot_regression(X_ohdg, y_ohdg, y_pred_ohdg, outliers_ohdg, ohdg_model, axs[plot_idx], 'Regresión Lineal para OHDG Data', 'Conc_ng/ul', 'OD_450_nm')
                self.Std_Curves_df.loc["OHDG"] = [slope_ohdg, intercept_ohdg, r2_ohdg]
                plot_idx += 1
            else:
                print("No hay suficientes datos para realizar la regresión lineal en ohdg_data.")    

        print("Regresiones RTL realizadas exitosamente.")

        output_dir = Path(__file__).resolve().parent.parent / "Output" / "Results_DNA_damage"
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output_dir / 'Std_Curves_regressions.pdf', format='pdf', dpi=300, bbox_inches='tight')
            print(f"Saved standard curves figure to {output_dir / 'Std_Curves_regressions.pdf'}")
        except Exception as e:
            print(f"Failed to save standard curves figure: {e}")
        plt.close(fig)
        
    def Amplification_Efficiency(self):
        if "Slope" in self.Std_Curves_df.columns:
            self.Std_Curves_df["Amplification_Efficiency"] = np.nan

            for gene_tissue in ["Gap8_brain_pool", "Tel1b_brain_pool", "Gap8_muscle_pool", "Tel1b_muscle_pool"]:
                if gene_tissue in self.Std_Curves_df.index:
                    slope = self.Std_Curves_df.loc[gene_tissue, "Slope"]
                    if slope != 0:
                        efficiency = 10 ** (-1 / slope)
                        self.Std_Curves_df.loc[gene_tissue, "Amplification_Efficiency"] = efficiency
                    else:
                        print(f"El slope para {gene_tissue} es 0, no se puede calcular la eficiencia.")
                else:
                    print(f"{gene_tissue} no está presente en el DataFrame.")

            print("Eficiencia de amplificación calculada exitosamente.")
        else:
            print("La columna 'Slope' no está presente en el DataFrame. Asegúrate de haber calculado las regresiones antes de intentar calcular la eficiencia.")

    def calibrator_sample(self):
        required_columns = ["Tissue", "Treatment", "Cq_Gap8", "Cq_Tel1b"]
        if all(col in self.rtl_data.columns for col in required_columns):
            filtered_data = self.rtl_data[(self.rtl_data["Tissue"].isin(["brain_pool", "muscle_pool"])) &
                                          (self.rtl_data["Treatment"] == "Std-01")]

            if not filtered_data.empty:
                calibrator_means = filtered_data.groupby("Tissue")[["Cq_Gap8", "Cq_Tel1b"]].mean()

                for tissue in ["brain_pool", "muscle_pool"]:
                    for gene in ["Gap8", "Tel1b"]:
                        sample_key = f"{gene}_{tissue}"
                        if sample_key in self.Std_Curves_df.index:
                            if "Cq_calibrator_sample" not in self.Std_Curves_df.columns:
                                self.Std_Curves_df["Cq_calibrator_sample"] = np.nan
                            self.Std_Curves_df.loc[sample_key, "Cq_calibrator_sample"] = calibrator_means.loc[tissue, f"Cq_{gene}"]

                print("Valores calibradores calculados exitosamente y añadidos al DataFrame.")
            else:
                print("No se encontraron registros con Tissue == 'brain_pool' o 'muscle_pool' y Treatment == 'Std-01'.")
        else:
            print("El DataFrame rtl_data no contiene las columnas necesarias para realizar el cálculo.")
        
        print(self.Std_Curves_df)

        
    def Variables_data(self):
        filtered_rtl_data = self.rtl_data.dropna(subset=["Age"])
        if not filtered_rtl_data.empty:
            rtl_grouped_data = (filtered_rtl_data
                                .groupby(["Age", "Treatment", "Tissue", "Individual"])
                                .agg(Cq_Gap8_Mean=("Cq_Gap8", "mean"),
                                     Cq_Tel1b_Mean=("Cq_Tel1b", "mean"))
                                .reset_index()
                                .rename(columns={"Age": "Time", "Treatment": "Group"}))
            self.Variables_df = rtl_grouped_data.copy()
        else:
            self.Variables_df = pd.DataFrame(columns=["Time", "Group", "Tissue", "Individual", 
                                                      "Cq_Gap8_Mean", "Cq_Tel1b_Mean"])

        self.mc_data["Group"].replace({"Td": "WTR", "Ti": "RTR"}, inplace=True)
        self.ohdg_data["Group"].replace({"Td": "WTR", "Ti": "RTR"}, inplace=True)

        filtered_mc_data = self.mc_data.dropna(subset=["Sample_dph"])
        if not filtered_mc_data.empty:
            mc_grouped_data = (filtered_mc_data
                               .groupby(["Sample_dph", "Group", "Tissue", "Individual"])
                               .agg(OD_450_nm_mc_Mean=("OD_450_nm", "mean"))
                               .reset_index()
                               .rename(columns={"Sample_dph": "Time"}))
            self.Variables_df = pd.merge(self.Variables_df, mc_grouped_data, how="left",
                                         on=["Time", "Group", "Tissue", "Individual"])

        filtered_ohdg_data = self.ohdg_data.dropna(subset=["Sample_dph"])
        if not filtered_ohdg_data.empty:
            ohdg_grouped_data = (filtered_ohdg_data
                                 .groupby(["Sample_dph", "Group", "Tissue", "Individual"])
                                 .agg(OD_450_nm_ohdg_Mean=("OD_450_nm", "mean"))
                                 .reset_index()
                                 .rename(columns={"Sample_dph": "Time"}))
            self.Variables_df = pd.merge(self.Variables_df, ohdg_grouped_data, how="left",
                                         on=["Time", "Group", "Tissue", "Individual"])

    def update_Variables_df(self):
        if self.Variables_df.empty:
            print("Error: Variables_df está vacío.")
            return
 
        OD_450_NC_mc = self.mc_data.loc[self.mc_data["Group"] == "NC", "OD_450_nm"].mean()
        OD_450_NC_ohdg = self.ohdg_data.loc[self.ohdg_data["Group"] == "NC", "OD_450_nm"].mean()

        self.Variables_df = self.Variables_df.assign(E_Gap8=pd.NA, Cq_cal_Gap8=pd.NA,
                                                     E_Tel1b=pd.NA, Cq_cal_Tel1b=pd.NA,
                                                     S_mc_ng=pd.NA, Slope_mc=pd.NA,
                                                     S_ohdg_ng=pd.NA, Slope_ohdg=pd.NA,
                                                     OD_NC_mc=pd.NA, OD_NC_ohdg=pd.NA,
                                                     Time_Treatment=pd.NA
                                                     )

        for tissue, prefix in [("brain", "brain_pool"), ("muscle", "muscle_pool")]:
            mask = self.Variables_df["Tissue"] == tissue
            self.Variables_df.loc[mask, ["Cq_cal_Gap8", "E_Gap8"]] = [
                self.Std_Curves_df.loc[f"Gap8_{prefix}", "Cq_calibrator_sample"],
                self.Std_Curves_df.loc[f"Gap8_{prefix}", "Amplification_Efficiency"]
            ]
            self.Variables_df.loc[mask, ["Cq_cal_Tel1b", "E_Tel1b"]] = [
                self.Std_Curves_df.loc[f"Tel1b_{prefix}", "Cq_calibrator_sample"],
                self.Std_Curves_df.loc[f"Tel1b_{prefix}", "Amplification_Efficiency"]
            ]

        self.Variables_df.loc[self.Variables_df["OD_450_nm_mc_Mean"].notna(), ["S_mc_ng", "Slope_mc"]] = [100, self.Std_Curves_df.loc["MC", "Slope"]]
        self.Variables_df.loc[self.Variables_df["OD_450_nm_ohdg_Mean"].notna(), ["S_ohdg_ng", "Slope_ohdg"]] = [300, self.Std_Curves_df.loc["OHDG", "Slope"]]

        self.Variables_df.loc[self.Variables_df["OD_450_nm_mc_Mean"].notna(), "OD_NC_mc"] = OD_450_NC_mc
        self.Variables_df.loc[self.Variables_df["OD_450_nm_ohdg_Mean"].notna(), "OD_NC_ohdg"] = OD_450_NC_ohdg

        conditions = [
            (self.Variables_df["Time"] == 40),
            (self.Variables_df["Time"] == 86),
            (self.Variables_df["Time"] == 140),
            (self.Variables_df["Time"] == 188)
        ]
        values = [0, 50, 100, 150]
        self.Variables_df["Time_Treatment"] = np.select(conditions, values, default=pd.NA)

        def reorder_columns(df, insert_after, new_columns):
            for col in new_columns:
                insert_pos = df.columns.get_loc(insert_after) + 1
                cols = list(df.columns)
                cols.insert(insert_pos, cols.pop(cols.index(col)))
                df = df[cols]
            return df
        self.Variables_df = reorder_columns(self.Variables_df, "Time", ["Time_Treatment"])
        self.Variables_df = reorder_columns(self.Variables_df, "Cq_Gap8_Mean", ["E_Gap8", "Cq_cal_Gap8"])
        self.Variables_df = reorder_columns(self.Variables_df, "Cq_Tel1b_Mean", ["E_Tel1b", "Cq_cal_Tel1b"])
        self.Variables_df = reorder_columns(self.Variables_df, "OD_450_nm_mc_Mean", ["S_mc_ng", "OD_NC_mc", "Slope_mc"])
        self.Variables_df = reorder_columns(self.Variables_df, "OD_450_nm_ohdg_Mean", ["S_ohdg_ng", "OD_NC_ohdg", "Slope_ohdg"])
    
        print("Variables_df ha sido creado, actualizado y reordenado con éxito.")
        
        print(self.Variables_df.head())
        
    def process_DNA_data(self):
        """Método principal para procesar los datos a través de los pasos definidos."""
        self.load_data()       
        self.std_curve()
        self.Amplification_Efficiency()
        self.calibrator_sample()
        self.Variables_data()
        self.update_Variables_df()
           
        
class Long_welfare_measures:
    def __init__(self, dna_data):
        self.dna_data = dna_data
        self.df = self.dna_data.Variables_df.copy() 
        self.plots = []
        self.colors = self.dna_data.colors
        self.fills = self.dna_data.fills
        
    
    def RTL_quantification(self):

        required_columns = ["Individual", "E_Tel1b", "E_Gap8", "Cq_cal_Tel1b", "Cq_cal_Gap8", 
                            "Cq_Tel1b_Mean", "Cq_Gap8_Mean"]
        
        if not all(col in self.df.columns for col in required_columns):
            print("Error: Falta una o más columnas necesarias para el cálculo de RTL.")
            return
        
        def calculate_rtl(row):
            try:
                delta_cq_target = row["Cq_cal_Tel1b"] - row["Cq_Tel1b_Mean"]
                delta_cq_reference = row["Cq_cal_Gap8"] - row["Cq_Gap8_Mean"]
                target_ratio = row["E_Tel1b"] ** delta_cq_target
                reference_ratio = row["E_Gap8"] ** delta_cq_reference
                rtl_value = target_ratio / reference_ratio
                
                return rtl_value
            except (TypeError, ValueError, ZeroDivisionError) as e:
                print(f"Error calculando RTL para Individual {row['Individual']}: {e}")
                return np.nan

        self.df["RTL"] = self.df.apply(calculate_rtl, axis=1)
        
        print("RTL ha sido calculado y añadido a self.df.")
        
    def mC_quantification(self):
        if self.df.empty:
            print("Error: self.df está vacío.")
            return

        required_columns = ["OD_450_nm_mc_Mean", "Slope_mc", "OD_NC_mc", "S_mc_ng"]
        for col in required_columns:
            if col not in self.df.columns:
                print(f"Error: La columna '{col}' no está en self.df.")
                return

        self.df["5_mC_ng"] = (self.df["OD_450_nm_mc_Mean"] - self.df["OD_NC_mc"]) / (self.df["Slope_mc"] * 2)

        self.df["5_mC_%"] = (self.df["5_mC_ng"] / self.df["S_mc_ng"]) * 100
    
        print("5_mC_ng y 5_mC_% han sido calculados y añadidos a self.df.")
        
    def OHdG_quantification(self):
        if self.df.empty:
            print("Error: self.df está vacío.")
            return

        required_columns = ["OD_450_nm_ohdg_Mean", "Slope_ohdg", "OD_NC_ohdg", "S_ohdg_ng"]
        for col in required_columns:
            if col not in self.df.columns:
                print(f"Error: La columna '{col}' no está en self.df.")
                return

        self.df["8_OHdG_ng"] = (self.df["OD_450_nm_ohdg_Mean"] - self.df["OD_NC_ohdg"]) / self.df["Slope_ohdg"]

        self.df["8_OHdG_%"] = (self.df["8_OHdG_ng"] / self.df["S_ohdg_ng"]) * 100
    
        print("8_OHdG_ng y 8_OHdG_% han sido calculados y añadidos a self.df.")
        print(self.df)
        return self.df
    
    def convert_columns_to_numeric(self):
        columns_to_convert = ['Time','Individual', 'Cq_Gap8_Mean', 'Cq_cal_Gap8', 'E_Gap8', 'Cq_Tel1b_Mean', 'Cq_cal_Tel1b',
                              'E_Tel1b', 'OD_450_nm_mc_Mean', 'Slope_mc', 'OD_NC_mc', 'S_mc_ng', 'OD_450_nm_ohdg_Mean',
                              'Slope_ohdg', 'OD_NC_ohdg', 'S_ohdg_ng', 'RTL', '5_mC_ng', '5_mC_%', '8_OHdG_ng', '8_OHdG_%']

        for column in columns_to_convert:
            if column in self.df.columns:
                self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
            else:
                print(f"Columna {column} no encontrada en el DataFrame.")
        return self.df
    
    def fold_change(self):
        columns_of_interest = ['RTL', '5_mC_ng', '5_mC_%', '8_OHdG_ng', '8_OHdG_%']

        for column in columns_of_interest:
            self.df[f'{column}_FC'] = np.nan

        tissues = self.df['Tissue'].unique()
        
        for tissue in tissues:
            tissue_df = self.df[self.df['Tissue'] == tissue]
            control_means = tissue_df[tissue_df['Group'] == 'Control'][columns_of_interest].mean()

            for group in ['RTR', 'WTR']:
                group_df = tissue_df[tissue_df['Group'] == group]
                
                for column in columns_of_interest:
                    fold_change = (group_df[column]- control_means[column]) / control_means[column]

                    self.df.loc[group_df.index, f'{column}_FC'] = fold_change
        
        print("El fold change ha sido calculado e incorporado al dataframe")
        print(self.df.head())
        return self.df
    
    def process_Long_welfare_measures(self):
        """Método principal para procesar los datos a través de los pasos definidos."""
        self.RTL_quantification()
        self.mC_quantification()
        self.OHdG_quantification()
        self.convert_columns_to_numeric()
        self.fold_change()

class Data_pre_analysis:
    def __init__(self, dna_data, wf_measures):
        self.dna_data = dna_data
        self.wf_measures = wf_measures
        self.df = self.wf_measures.df.copy()
        self.df = self.df[self.df['Group'] != 'Control']
        self.plots = []
        self.colors = self.dna_data.colors
        self.fills = self.dna_data.fills  
        
        
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

                output_dir = Path(__file__).resolve().parent.parent / "Output" / "Results_DNA_damage"
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = output_dir / f'distribution_{tissue}_{var}.png'
                try:
                    plt.gcf().savefig(fname, dpi=300, bbox_inches='tight')
                    print(f"Saved distribution figure to {fname}")
                except Exception as e:
                    print(f"Failed to save distribution figure for {tissue} {var}: {e}")
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

        df_results = pd.DataFrame(results_tests)
        print(df_results)
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
            if len(numeric_vars) > 0:
                corr_matrix = tissue_df[numeric_vars].corr(numeric_only=True)
                fig_corr = plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
                plt.title(f'Correlation Matrix for Tissue: {tissue}')

                output_dir = Path(__file__).resolve().parent.parent / "Output" / "Results_DNA_damage"
                output_dir.mkdir(parents=True, exist_ok=True)
                fname_corr = output_dir / f'corr_matrix_{tissue}.png'
                try:
                    fig_corr.savefig(fname_corr, dpi=300, bbox_inches='tight')
                    print(f"Saved correlation matrix to {fname_corr}")
                except Exception as e:
                    print(f"Failed to save correlation matrix for {tissue}: {e}")
                plt.close(fig_corr)

                # Pairplot
                try:
                    pairplot = sns.pairplot(tissue_df, vars=numeric_vars, hue=relationship_categorical_variable, palette=self.colors)
                    fname_pair = output_dir / f'pairplot_{tissue}.png'
                    pairplot.fig.savefig(fname_pair, dpi=300, bbox_inches='tight')
                    print(f"Saved pairplot to {fname_pair}")
                    plt.close(pairplot.fig)
                except Exception as e:
                    print(f"Failed to create/save pairplot for {tissue}: {e}")

        print("Saved all generated pre-analysis figures to Output/Results_DNA_damage.")
    
class Long_welfare_analysis:
    def __init__(self, dna_data, wf_measures, pre_analysis):
        self.dna_data = dna_data
        self.wf_measures = wf_measures
        self.pre_analysis = pre_analysis
        self.df = self.wf_measures.df.copy()
        self.df = self.df[self.df['Group'].isin(['RTR', 'WTR'])]
        self.plots = []
        self.colors = self.dna_data.colors
        self.fills = self.dna_data.fills
        self.ci_df= None
        
    def two_way_anova(self):
        self.df = self.df.rename(columns={"5_mC_ng_FC": "mC_ng_FC"})
        self.df = self.df.rename(columns={"8_OHdG_ng_FC": "OHdG_ng_FC"})
        variables = ['RTL_FC', 'mC_ng_FC', 'OHdG_ng_FC']
        results = []
    
        for tissue in self.df['Tissue'].unique():
            df_tissue = self.df[self.df['Tissue'] == tissue]
            
            for var in variables:
                df_anova = df_tissue[['Group', 'Time', var]].copy()
                model = ols(f'{var} ~ C(Group) + C(Time) + C(Group):C(Time)', data=df_anova).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                results.append({
                    'Tissue': tissue,
                    'Variable': var,
                    'ANOVA': anova_table
                })

                tukey = pairwise_tukeyhsd(endog=df_anova[var], groups=df_anova['Group'] + ' ' + df_anova['Time'].astype(str), alpha=0.05)
                tukey_summary = tukey.summary()
                
                print(f"\nResultados de la ANOVA de dos vías para el tejido: {tissue} y variable: {var}")
                print(anova_table)
                print(f"\nResultados del test de Tukey para el tejido: {tissue} y variable: {var}")
                print(tukey_summary)
        return results
        
        
    def kruskal_wallis_test(self):
        variables = ['RTL_FC', '5_mC_ng_FC', '8_OHdG_ng_FC']
        results = []
        for tissue in self.df['Tissue'].unique():
            df_tissue = self.df[self.df['Tissue'] == tissue]
            for var in variables:
                df_kruskal = df_tissue[['Group', 'Time_Treatment', var]].copy()
                df_kruskal['Group_Time'] = df_kruskal['Group'] + ' ' + df_kruskal['Time_Treatment'].astype(str)
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
        output_dir = Path(__file__).resolve().parent.parent / "Output" / "Results_DNA_damage"
        output_dir.mkdir(parents=True, exist_ok=True)
        kw_df.to_excel(output_dir / 'Kruskal-Wallis_results.xlsx', index=False)
        return kw_df
    
    def conover_iman_posthoc(self):
        variables = ['RTL_FC', '5_mC_ng_FC', '8_OHdG_ng_FC']
        results = []
        
        for tissue in self.df['Tissue'].unique():
            df_tissue = self.df[self.df['Tissue'] == tissue]
            
            for var in variables:
                df_posthoc = df_tissue[['Group', 'Time_Treatment', var]].copy()
                df_posthoc['Group_Time'] = df_posthoc['Group'] + ' ' + df_posthoc['Time_Treatment'].astype(str)

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

        output_dir = Path(__file__).resolve().parent.parent / "Output" / "Results_DNA_damage"
        output_dir.mkdir(parents=True, exist_ok=True)
        ci_df.to_excel(output_dir / 'DNA_Damage_posthoc_results.xlsx', index=False)

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
            return pd.Series({'Group': group, 'Time_Treatment': time})

        cld_df[['Group', 'Time_Treatment']] = cld_df['groupXtime'].apply(split_group_time)

        cld_df = cld_df[['Tissue', 'Variable', 'groupXtime', 'Group', 'Time_Treatment', 'Label']]

        cld_df = cld_df.sort_values(['Tissue', 'Variable', 'groupXtime'])

        return cld_df
    
    def bar_plots_with_error(self):
        plt.rcParams['font.family'] = 'Arial'
        variables = ['8_OHdG_ng_FC', 'RTL_FC', '5_mC_ng_FC']
        tissues = ['brain', 'muscle']
        bar_width = 0.3  
        bar_spacing = 0.32  
        
        significance_letters = self.assign_significance_letters()
        
        fig, axes = plt.subplots(len(variables), len(tissues), figsize=(7, 10))

        subplot_labels = ['A)', 'D)', 'B)', 'E)', 'C)', 'F)']
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
                    means = df_group.groupby('Time_Treatment')[var].mean()
                    stds = df_group.groupby('Time_Treatment')[var].std()
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
                            (letters_subset['Time_Treatment'] == str(time))
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
                    
                if var =='RTL_FC':
                    ax.set_ylim(-1.5, 1.5)
                    ax.set_ylabel('T/S ratio\n(Fold Change related to control)', fontsize=10, fontweight='bold')
                if var =='5_mC_ng_FC':
                    ax.set_ylim(-0.8, 0.8)
                    ax.set_ylabel('5-methylcytosine (%)\n(Fold Change related to control)', fontsize=10, fontweight='bold')
                if var =='8_OHdG_ng_FC':
                    ax.set_ylim(-0.4, 0.4)
                    ax.set_ylabel('8-hydroxy-2-deoxyguanosine (%)\n(Fold Change related to control)', fontsize=10, fontweight='bold')
                
                ax.set_xticks(index + (bar_spacing + bar_width) / 2)
                ax.xaxis.set_tick_params(width=2, length=5, labelsize=10)
                ax.yaxis.set_tick_params(width=2, length=5, labelsize=10)
                ax.set_xticklabels(['50', '100', '150'], fontsize=10, fontweight='bold')
                ax.set_xlabel('Time (days)', fontsize=10, fontweight='bold')
                
                ax.axhline(y=0, color='black', linestyle='dotted', linewidth=2) 

                for spine in ax.spines.values():
                    spine.set_linewidth(2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_dir = Path(__file__).resolve().parent.parent / "Output" / "Results_DNA_damage"
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = output_dir / 'DNA_damage_barplots.pdf'
        try:
            fig.savefig(fname, format='pdf', dpi=300, bbox_inches='tight')
            print(f"Saved bar plots to {fname}")
        except Exception as e:
            print(f"Failed to save bar plots figure: {e}")
        plt.close(fig)
      
    def process_Long_welfare(self):
        """Método principal para procesar los datos a través de los pasos definidos."""
        self.kruskal_wallis_test()
        self.conover_iman_posthoc()
        self.bar_plots_with_error()



