import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from scipy.optimize import differential_evolution, curve_fit
from scipy.stats import probplot
from scipy.stats import norm
from scipy.stats import normaltest, levene
import statsmodels.api as sm
from statsmodels.api import OLS
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.formula.api as smf
from io import StringIO
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.utils import resample
from sklearn.model_selection import KFold
import inspect
import warnings
warnings.filterwarnings("ignore")

# Module-level color definitions
colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}

class data_processing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_clean = None
        self.preprocessing_summary = {}
        self.plots = []
        self.colors = colors
        self.fills = fills
        
    def load_data(self, sheet_name=0):
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading file: {e}")
            
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
        
    def outliers_duplicate_null_detection(self, variables, categorical_variable, z_score=3):
        df = self.df.copy()
        analysis_list = []

        for group in df[categorical_variable].unique():
            group_df = df[df[categorical_variable] == group]
            
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
        
                analysis_list.append({
                    'Group': group,
                    'Variable': var,
                    'Num_Outliers': num_outliers,
                    'Outliers_Values': outliers_values,
                    'Num_Duplicates': num_duplicates,
                    'Duplicate_Values': duplicate_values,
                    'Num_Nulls': num_nulls,
                    'Null_Values': null_values
                })
        
        analysis_summary = pd.DataFrame(analysis_list)
        print("\nOutliers, Duplicates, and Nulls Summary by Group:")
        print(analysis_summary)
        return analysis_summary

    def data_clean(self, variables, categorical_variable, handle_outliers=True, outlier_method='remove', 
                   z_score=3, remove_na=True, remove_duplicates=True):
        self.df_clean = self.df.copy()
        
        for group in self.df_clean[categorical_variable].unique():
            group_mask = self.df_clean[categorical_variable] == group
            group_df = self.df_clean[group_mask]
            
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
        
            if remove_na:
                group_df = group_df.dropna()
                print(f"Missing values (NaN) were removed for group {group}.")
            else:
                print(f"Skipping removal of missing values for group {group}.")
        
            if remove_duplicates:
                group_df = group_df.drop_duplicates()
                print(f"Duplicate rows were removed for group {group}.")
            else:
                print(f"Skipping removal of duplicate rows for group {group}.")
            
            self.df_clean.loc[group_mask] = group_df
            
        print("\nStatistical summary data clean:")
        print(self.df_clean.describe(include='all'))
        return self.df_clean

    def scatter_plots(self, x, y_list, hue, title, colors=None, theme='white', font='Arial', font_size=12, 
                  name_axis_x=None, name_axis_y=None, y_um_var=None, scale_x='linear', scale_y='linear',
                  nrows=1, ncols=1, fig_size=(12, 8)):
        """
        Crea subplots de gráficos de dispersión para una lista de variables Y.
        
        Parámetros:
        - x: Variable en el eje X (string).
        - y_list: Lista de variables en el eje Y (lista de strings).
        - hue: Variable que distingue los grupos con colores (string).
        - title: Título de la figura (string).
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

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size[0], fig_size[1]))

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
        for i, y in enumerate(y_list):
            ax = axes[i]
            if colors:
                sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=colors, ax=ax)
            else:
                sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax)

            ax.set_xlabel(name_axis_x if name_axis_x else x, fontsize=font_size, fontname=font)
            ax.set_ylabel(f"{y} {y_um_var if y_um_var else ''}", fontsize=font_size, fontname=font)

            ax.set_xscale(scale_x)
            ax.set_yscale(scale_y)

            ax.tick_params(axis='both', which='both', labelsize=font_size, colors='black')
            ax.set_axisbelow(True)
            ax.set_title(f"{title} - {y}", fontsize=font_size + 2, fontname=font)
    
        plt.tight_layout()
        self.plots.append(fig)
        try:
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = output_dir / f"scatter_{title.replace(' ', '_')}.png"
            fig.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved scatter plot to {fname}")
        except Exception as e:
            print(f"Warning: could not save scatter plot: {e}")
        plt.close()
        
    def norm_homo_analysis(self, dependent_variables, categorical_variable, time_variable):
        data=self.df_clean if self.df_clean is not None else self.df
        results_tests = []
        for var in dependent_variables:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            # Histograma y densidad
            sns.histplot(data=data, x=var, kde=True, ax=ax1)
            ax1.set_title(f'Distribution of {var}')
            
            # Q-Q plot
            stats.probplot(data[var], dist="norm", plot=ax2)
            ax2.set_title(f'Q-Q Plot of {var}')
            
            # Modelo OLS
            formula = f'{var} ~ C({categorical_variable}) + {time_variable}'
            model = ols(formula, data=data).fit()
            fitted_values = model.fittedvalues
            residuals = model.resid
            
            # Residual plot (agregado como tercer subplot)
            sns.residplot(x=fitted_values, y=residuals, lowess=True, line_kws={'color': 'red'}, ax=ax3)
            ax3.axhline(0, linestyle='--', color='black', linewidth=2)
            ax3.set_title(f'Residual Plot of {var}')
            ax3.set_xlabel('Fitted Values')
            ax3.set_ylabel('Residuals')

            plt.tight_layout()

            try:
                output_dir = Path("Output") / "Results_Growth"
                output_dir.mkdir(parents=True, exist_ok=True)
                fname = output_dir / f"norm_homo_{var}.png"
                plt.gcf().savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Saved norm/homo figure to {fname}")
            except Exception as e:
                print(f"Warning: could not save norm/homo figure: {e}")
            plt.close()
            
            # Test de normalidad
            _, p_val_norm = stats.normaltest(data[var])
            is_normal = "True" if p_val_norm > 0.05 else "False"

            # Test de homocedasticidad
            _, p_val_homo = stats.levene(*[group[var].values for name, group in data.groupby(categorical_variable)])
            is_homogeneous = "True" if p_val_homo > 0.05 else "False"

            results_tests.append({
                'Variable': var,
                'p_val_normality': p_val_norm,
                'Is_normal': is_normal,
                'p_val_homocedasticity': p_val_homo,
                'Is_homogeneous': is_homogeneous
            })

        df_results = pd.DataFrame(results_tests)
        print("\nD’Agostino and Pearson test & Levene test summary")
        print(df_results)
        self.df_results_tests = df_results
        
    def variable_relationship_analysis(self, relationship_numeric_variables, relationship_categorical_variable):
        data = self.df_clean if self.df_clean is not None else self.df
        numeric_vars = [var for var in relationship_numeric_variables if var in data.columns and 
                        pd.api.types.is_numeric_dtype(data[var])]
        
        #Corr_matrix
        corr_matrix = data[relationship_numeric_variables].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlation Matrix')

        try:
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = output_dir / "corr_matrix.png"
            plt.gcf().savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {fname}")
        except Exception as e:
            print(f"Warning: could not save correlation matrix: {e}")
        plt.close()
        
        # Pairplot
        try:
            pairplot = sns.pairplot(data, vars=relationship_numeric_variables, hue=relationship_categorical_variable,
                                    palette=self.colors)
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = output_dir / "pairplot.png"
            pairplot.fig.savefig(fname, dpi=300, bbox_inches='tight')
            print(f"Saved pairplot to {fname}")
            plt.close(pairplot.fig)
        except Exception as e:
            print(f"Warning: could not save pairplot: {e}")
        
    def ancova_analysis(self, numeric_variables, categorical_variable, independent_variable):
        data = self.df_clean if self.df_clean is not None else self.df
        results_ancova = {}
        
        for var in numeric_variables:
            model = ols(f'{var} ~ C({categorical_variable}) + {independent_variable}', data=data).fit()
            anova_table = anova_lm(model)

            anova_table['Source'] = anova_table.index

            results_ancova[var] = anova_table[['Source', 'df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]

        self.results_ancova_dict = results_ancova

        for var, df in self.results_ancova_dict.items():
            print(f"\nANCOVA for {var}")
            print(df)

    
class NLM_Analysis:
    def __init__(self, pre_analysis):
        self.pre_analysis = pre_analysis
        self.df = self.pre_analysis.df_clean if self.pre_analysis.df_clean is not None else self.pre_analysis.df
        self.colors = self.pre_analysis.colors
        self.fills = self.pre_analysis.fills
        self.models = {
            'Linear': (self.lineal_model, self.lineal_initial_guess),
            'Exponential': (self.exponential_model, self.exponential_initial_guess),
            'Power': (self.power_model, self.power_model_initial_guess),
            'Simplified Logistic': (self.simplified_logistic_model, self.simplified_logistic_initial_guess),
            'Logistic': (self.logistic_model, self.logistic_initial_guess),
            'Gompertz': (self.gompertz_model, self.gompertz_initial_guess),
            'von Bertalanffy': (self.von_bertalanffy_model, self.von_bertalanffy_initial_guess),
            'Brody': (self.brody_model, self.brody_model_initial_guess)
        }
        self.fitted_params = {}
        self.model_metrics = {}
        self.normalization_methods = {
            'none': self.no_normalization,
            'max': self.max_normalization,
            'minmax': self.minmax_normalization,
            'standard': self.standard_normalization,
            'robust': self.robust_normalization
        }
        self.fitting_methods = {'curve_fit': self.fit_curve_fit,}
        self.current_normalization = None
        

    def normalize_data(self, dependent_var, method='max'):
        if method not in self.normalization_methods:
            raise ValueError(f"Método de normalización '{method}' no reconocido.")
        
        self.current_normalization = method
        return self.normalization_methods[method](dependent_var)

    def no_normalization(self, dependent_var):
        self.df[f'{dependent_var}_normalized'] = self.df[dependent_var]
        return None
        
    def max_normalization(self, dependent_var):
        max_value = self.df[dependent_var].max()
        self.df[f'{dependent_var}_normalized'] = self.df[dependent_var] / max_value
        return max_value

    def minmax_normalization(self, dependent_var):
        scaler = MinMaxScaler()
        self.df[f'{dependent_var}_normalized'] = scaler.fit_transform(self.df[[dependent_var]])
        return scaler

    def standard_normalization(self, dependent_var):
        scaler = StandardScaler()
        self.df[f'{dependent_var}_normalized'] = scaler.fit_transform(self.df[[dependent_var]])
        return scaler

    def robust_normalization(self, dependent_var):
        scaler = RobustScaler()
        self.df[f'{dependent_var}_normalized'] = scaler.fit_transform(self.df[[dependent_var]])
        return scaler
    
    
    # Model definitions
    
    def lineal_model(self, t, a, b):
        return a + b * t
        
    def simplified_logistic_model(self, t, L, k):
        return L / (1 + np.exp(-k * t))

    def power_model(self, t, a, k):
        return a*(t**k)
         
    def exponential_model(self, t, a, b):
        return a * np.exp(b * t)

    def logistic_model(self, t, L, k, t0):
        return L / (1 + np.exp(-k * (t - t0)))

    def gompertz_model(self, t, a, b, c):
        return a * np.exp(-np.exp(-c * (t - b)))

    def von_bertalanffy_model(self, t, Linf, K, t0):
        return Linf * (1 - np.exp(-K * (t - t0)))

    def brody_model(self, t, a, c, b):
        return a * (1 - c * np.exp(-b * t))

    # Initial guess

    def lineal_initial_guess(self, x, y):
        b = (y[-1] - y[0]) / (x[-1] - x[0])
        a = y[0] - b * x[0]
        return [a, b]

    def simplified_logistic_initial_guess(self, x, y):
        L = y.max()
        k = 0.1
        return [L, k]

    def power_model_initial_guess(self, x, y):
        a = y.max()
        k= 0.01
        return[a, k]

    def exponential_initial_guess(self, x, y):
        a = y.min()
        b = 0.01
        return [a, b]

    def logistic_initial_guess(self, x, y):
        L = y.max()
        k = 0.1
        t0 = x.mean()
        return [L, k, t0]

    def gompertz_initial_guess(self, x, y):
        a = y.max()
        b = x.mean()
        c = 0.1
        return [a, b, c]

    def von_bertalanffy_initial_guess(self, x, y):
        Linf = y.max() * 10
        K = 0.01
        t0 = x.min()
        return [Linf, K, t0]

    def brody_model_initial_guess(self, x, y):
        a = y.max()
        c = 1
        b = 0.01
        return[a, c, b]

    def fit_curve_fit(self, model_func, x, y, initial_guess):
        try:
            params, _ = curve_fit(model_func, x, y, p0=initial_guess, maxfev=100000, method='trf')
            return params
        except (RuntimeError, ValueError):
            return None


    def optimize_initial_guess(self, model_func, x, y, bounds):
        """
        Optimiza la conjetura inicial usando un método de optimización global (differential_evolution).
        """
        result = differential_evolution(lambda p: np.sum((model_func(x, *p) - y) ** 2), bounds, maxiter=1000)
        return result.x

    def fit_model(self, model_name, x, y, fitting_method='curve_fit', use_optimization=False):
        """
        Ajusta el modelo seleccionado a los datos usando el método de ajuste especificado.
        """
        model_func, initial_guess_func = self.models[model_name]
        initial_guess = initial_guess_func(x, y)

        if use_optimization:
            bounds = [(0, 10)] * len(initial_guess)
            initial_guess = self.optimize_initial_guess(model_func, x, y, bounds)

        fit_func = self.fitting_methods[fitting_method]
        params = fit_func(model_func, x, y, initial_guess)

        if params is None:
            print(f"Warning: Could not fit {model_name} model using {fitting_method}. Trying alternative method.")
            alternative_method = 'differential_evolution' if fitting_method == 'curve_fit' else 'curve_fit'
            params = self.fitting_methods[alternative_method](model_func, x, y, initial_guess)
        
        if params is None:
            print(f"Error: Could not fit {model_name} model using any method.")
        
        return params
    
    def calculate_aic(self, n, mse, num_params):
        return 2 * num_params + n * np.log(mse)

    def calculate_bic(self, n, mse, num_params):
        return np.log(n) * num_params + n * np.log(mse)

    def calculate_log_likelihood(self, y, y_pred):
        n = len(y)
        mse = np.mean((y - y_pred)**2)
        return -n/2 * np.log(2 * np.pi * mse) - 1/(2 * mse) * np.sum((y - y_pred)**2)

    def evaluate_model(self, model_name, x, y, params):
        model_func, _ = self.models[model_name]
        y_pred = model_func(x, *params)
        
        n = len(y)
        num_params = len(params)
        
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        rss = np.sum((y - y_pred)**2)
        rse = np.sqrt(rss / (n - num_params))
        
        aic = self.calculate_aic(n, mse, num_params)
        bic = self.calculate_bic(n, mse, num_params)
        log_likelihood = self.calculate_log_likelihood(y, y_pred)
        
        return {
            'R2': r2,
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'RSS': rss,
            'RSE': rse,
            'AIC': aic,
            'BIC': bic,
            'Log-likelihood': log_likelihood
        }
    def fit_all_models(self, dependent_var, categorical_var, time_var, normalization_method='none', fitting_method='curve_fit'): 
        scaler = self.normalize_data(dependent_var, method=normalization_method)
        
        for treatment in self.df[categorical_var].unique():
            treatment_data = self.df[self.df[categorical_var] == treatment]
            x = treatment_data[time_var].values
            y = treatment_data[f'{dependent_var}_normalized'].values
            
            for model_name in self.models.keys():
                params = self.fit_model(model_name, x, y, fitting_method)
                if params is not None:
                    metrics = self.evaluate_model(model_name, x, y, params)
                    
                    if treatment not in self.fitted_params:
                        self.fitted_params[treatment] = {}
                    self.fitted_params[treatment][model_name] = params
                    
                    if treatment not in self.model_metrics:
                        self.model_metrics[treatment] = {}
                    self.model_metrics[treatment][model_name] = metrics
        
        return scaler

    def plot_fitted_models(self, dependent_var, categorical_var, time_var, scaler, name_xaxis, name_yaxis):
        plt.rcParams['font.family'] = 'Arial'
        fig, axs = plt.subplots(4, 2, figsize=(6, 8))
        axs = axs.ravel()
        
        for i, (model_name, (model_func, _)) in enumerate(self.models.items()):
            ax = axs[i]
            for treatment in self.df[categorical_var].unique():
                treatment_data = self.df[self.df[categorical_var] == treatment]
                x = treatment_data[time_var].values
                y = treatment_data[dependent_var].values
                
                ax.scatter(x, y, color=self.colors[treatment], alpha=0.5, label=f'{treatment} (Data)', s=2)
                
                if treatment in self.fitted_params and model_name in self.fitted_params[treatment]:
                    params = self.fitted_params[treatment][model_name]
                    x_smooth = np.linspace(x.min(), x.max(), 300)
                    y_smooth = model_func(x_smooth, *params)
                    
                    # Desnormalizar los datos si es necesario
                    if self.current_normalization == 'max':
                        y_smooth *= scaler
                    elif self.current_normalization in ['minmax', 'standard', 'robust']:
                        y_smooth = scaler.inverse_transform(y_smooth.reshape(-1, 1)).flatten()
                    
                    ax.plot(x_smooth, y_smooth, color=self.colors[treatment], label=f'{treatment} (Fitted)')
                
            ax.set_title(f'{model_name} Model', fontsize=8, fontweight='bold')
            ax.set_xlabel(name_xaxis, fontsize=8, fontweight='bold')
            ax.set_ylabel(name_yaxis, fontsize=8, fontweight='bold')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(8)
                label.set_fontweight('bold')
        
        plt.tight_layout()

        try:
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            fname = output_dir / "Model_candidatos.pdf"
            fig.savefig(fname, format='pdf', dpi=300, bbox_inches='tight')
            print(f"Saved fitted models to {fname}")
        except Exception as e:
            print(f"Warning: could not save fitted models figure: {e}")
        plt.close(fig)
        
    def compare_models(self):
        comparison_data = []

        for treatment in self.model_metrics:
            for model in self.model_metrics[treatment]:
                metrics = self.model_metrics[treatment][model]
                comparison_data.append({
                    'Treatment': treatment,
                    'Model': model,
                    **metrics
                })

        comparison_df = pd.DataFrame(comparison_data)
        try:
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            comparison_df.to_excel(output_dir / 'Bondad_ajustes.xlsx', index=False)
            print(f"Saved model comparison to {output_dir / 'Bondad_ajustes.xlsx'}")
        except Exception as e:
            print(f"Warning: could not save model comparison: {e}")
        
        print("Model Comparison:")
        print(comparison_df)

        ordered_models_data = []

        metrics_to_compare = ['R2', 'MSE', 'RMSE', 'MAE', 'RSS', 'RSE', 'AIC', 'BIC', 'Log-likelihood']

        for treatment in comparison_df['Treatment'].unique():
            treatment_df = comparison_df[comparison_df['Treatment'] == treatment]

            for metric in metrics_to_compare:
                if metric in ['R2', 'Log-likelihood']:
                    ordered_models = treatment_df.sort_values(by=metric, ascending=False)['Model'].tolist()
                else:
                    ordered_models = treatment_df.sort_values(by=metric, ascending=True)['Model'].tolist()

                ordered_dict = {'Treatment': treatment, 'Metric': metric}

                for i, model in enumerate(ordered_models):
                    ordered_dict[f'{i+1}th Best Model'] = model
                
                ordered_models_data.append(ordered_dict)

        ordered_models_df = pd.DataFrame(ordered_models_data)

        print("\nBest Models Ordered by Metric:")
        print(ordered_models_df)

        model_counts = ordered_models_df['1th Best Model'].value_counts()
        
        print("\nModel Performance Summary (1st Best Model Count):")
        print(model_counts)

        best_overall_model = model_counts.index[0]
        print(f"\nBest Overall Model: {best_overall_model}")
        
    def parameter_significance_test(self, dependent_var, categorical_var, time_var, selected_models):
        results = []
        covariance_matrices = {}
        correlation_matrices = {}
    
        pd.options.display.float_format = '{:.4f}'.format
    
        for treatment in self.df[categorical_var].unique():
            treatment_data = self.df[self.df[categorical_var] == treatment]
            x = treatment_data[time_var].values
            y = treatment_data[f'{dependent_var}_normalized'].values
    
            for model_name in selected_models:
                if treatment in self.fitted_params and model_name in self.fitted_params[treatment]:
                    params = self.fitted_params[treatment][model_name]
    
                    model_func, _ = self.models[model_name]
                    param_names = list(inspect.signature(model_func).parameters.keys())[1:len(params) + 1]
    
                    residuals = y - model_func(x, *params)
                    ss_residuals = np.sum(residuals ** 2)
                    n = len(y)
                    p = len(params)
                    dof = max(0, n - p)
                    mse = ss_residuals / dof
    
                    epsilon = np.sqrt(np.finfo(float).eps)
                    jacobian = np.zeros((n, p))
                    for i in range(p):
                        params_plus = params.copy()
                        params_plus[i] += epsilon
                        jacobian[:, i] = (model_func(x, *params_plus) - model_func(x, *params)) / epsilon
    
                    cov_matrix = np.linalg.inv(jacobian.T.dot(jacobian)) * mse
                    covariance_matrices[(treatment, model_name)] = (cov_matrix, param_names)
    
                    se = np.sqrt(np.diag(cov_matrix))
    
                    t_stats = params / se
                    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    
                    ci_lower = params - se * stats.t.ppf(0.975, dof)
                    ci_upper = params + se * stats.t.ppf(0.975, dof)
    
                    for i, param_name in enumerate(param_names):
                        results.append({
                            'Treatment': treatment,
                            'Model': model_name,
                            'Parameter': param_name,
                            'Estimate': params[i],
                            'Std. Error': se[i],
                            't-value': t_stats[i],
                            'p-value': p_values[i],
                            'CI Lower': ci_lower[i],
                            'CI Upper': ci_upper[i],
                            'Degrees of Freedom': dof
                        })

                    diag_inv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
                    corr_matrix = diag_inv.dot(cov_matrix).dot(diag_inv)
    
                    correlation_matrices[(treatment, model_name)] = (corr_matrix, param_names)
    
        results_df = pd.DataFrame(results)
        try:
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_excel(output_dir / 'Parameter_significance.xlsx', index=False)
            print(f"Saved parameter significance to {output_dir / 'Parameter_significance.xlsx'}")
        except Exception as e:
            print(f"Warning: could not save parameter significance: {e}")
        print("\nParameter Significance Test Results:")
        print(results_df)
    
        print("\nCovariance Matrices for each Treatment and Model:")
        for (treatment, model_name), (cov_matrix, param_names) in covariance_matrices.items():
            print(f"\nTreatment: {treatment}, Model: {model_name}")
            print(pd.DataFrame(cov_matrix, columns=param_names, index=param_names))
    
        print("\nCorrelation Matrices for each Treatment and Model:")
        for (treatment, model_name), (corr_matrix, param_names) in correlation_matrices.items():
            print(f"\nTreatment: {treatment}, Model: {model_name}")
            print(pd.DataFrame(corr_matrix, columns=param_names, index=param_names))
    
        pd.reset_option('display.float_format')
    
        return results_df, covariance_matrices, correlation_matrices
    
    
    def check_model_assumptions(self, dependent_var, categorical_var, time_var, selected_models):
        results = []

        for treatment in self.df[categorical_var].unique():
            treatment_data = self.df[self.df[categorical_var] == treatment]
            x = treatment_data[time_var].values
            y = treatment_data[f'{dependent_var}_normalized'].values
            
            for model_name, (model_func, _) in self.models.items():
                if model_name not in selected_models:
                    continue
                
                params = self.fitted_params[treatment][model_name]
                y_pred = model_func(x, *params)
                residuals = y - y_pred
                std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
                plt.rcdefaults()
                sns.reset_defaults()
                plt.rcParams['font.family'] = 'Arial'
                plt.rcParams['font.size'] = 8
                fig = plt.figure(figsize=(8, 8))
                
                # 1. Normalidad de los residuos
                ax1 = fig.add_subplot(331)
                ax2 = fig.add_subplot(332)
                
                # Histograma de los residuos
                sns.histplot(residuals, kde=True, ax=ax1)
                ax1.set_title("Histogram of Residuals")
                
                # QQ-plot de los residuos
                qqplot(residuals, line='s', ax=ax2)
                ax2.set_title("Q-Q Plot of Residuals")
                
                # 2. Homocedasticidad
                ax3 = fig.add_subplot(333)
                ax4 = fig.add_subplot(334)
                
                # Residuos vs valores predichos
                ax3.scatter(y_pred, residuals)
                ax3.set_xlabel("Predicted Values")
                ax3.set_ylabel("Residuals")
                ax3.set_title("Residuals vs Predicted")
                ax3.axhline(y=0, color='r', linestyle='--')
                
                # Residuos estandarizados vs valores predichos
                ax4.scatter(y_pred, std_residuals)
                ax4.set_xlabel("Predicted Values")
                ax4.set_ylabel("Standardized Residuals")
                ax4.set_title("Standardized Residuals vs Predicted")
                ax4.axhline(y=0, color='r', linestyle='--')
                
                # 3. Independencia de los errores
                ax5 = fig.add_subplot(335)
                ax6 = fig.add_subplot(336)
                
                # Residuos vs Orden cronológico
                ax5.scatter(range(len(residuals)), residuals)
                ax5.set_xlabel("Order")
                ax5.set_ylabel("Residuals")
                ax5.set_title("Residuals vs Order")
                ax5.axhline(y=0, color='r', linestyle='--')
                
                # Correlograma (ACF plot)
                plot_acf(residuals, ax=ax6)
                ax6.set_title("Autocorrelation Function (ACF)")
                
                # 4. Linealidad de la transformación
                ax7 = fig.add_subplot(337)
                
                # Gráfico de residuos parciales
                smoothed = lowess(residuals, x)
                ax7.scatter(x, residuals)
                ax7.plot(smoothed[:, 0], smoothed[:, 1], color='r')
                ax7.set_xlabel("Time")
                ax7.set_ylabel("Partial Residuals")
                ax7.set_title("Partial Residuals Plot")
                
                # 5. Puntos influyentes o valores atípicos
                ax8 = fig.add_subplot(338)
                ax9 = fig.add_subplot(339)
                
                # Calcular leverage y distancia de Cook manualmente
                X = np.column_stack((np.ones_like(x), x))  # Agregar columna de unos para el término constante
                hat_matrix = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
                leverage = np.diag(hat_matrix)
                
                p = len(params)
                mse = np.sum(residuals**2) / (len(x) - p)
                cook_distance = (residuals**2 / (p * mse)) * (leverage / (1 - leverage)**2)
        
                # Leverage vs residuos estandarizados
                ax8.scatter(leverage, std_residuals)
                ax8.set_xlabel("Leverage")
                ax8.set_ylabel("Standardized Residuals")
                ax8.set_title("Leverage vs Standardized Residuals")
                
                # Distancia de Cook
                ax9.stem(cook_distance)
                ax9.set_xlabel("Observation")
                ax9.set_ylabel("Cook's Distance")
                ax9.set_title("Cook's Distance")
                
                plt.tight_layout()

                try:
                    output_dir = Path("Output") / "Results_Growth"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    fname = output_dir / f"model_assumptions_{treatment}_{model_name.replace(' ', '_')}.png"
                    fig.savefig(fname, dpi=300, bbox_inches='tight')
                    print(f"Saved model assumptions to {fname}")
                except Exception as e:
                    print(f"Warning: could not save model assumptions figure: {e}")
                plt.close(fig)

                # Prueba de Breusch-Pagan para homocedasticidad
                exog = sm.add_constant(y_pred)
                bp_test = het_breuschpagan(residuals, exog)

                bp_result = {
                    'Modelo': model_name,
                    'Tratamiento': treatment,
                    'Prueba': 'Breusch-Pagan',
                    'Estadístico': bp_test[0],
                    'p-valor': bp_test[1],
                    'Resultado': 'Rechaza H0 (No homocedástico)' if bp_test[1] < 0.05 else 'No rechaza H0 (Homocedástico)'
                }

                results.append(bp_result)

                # Prueba de White para homocedasticidad
                exog = np.column_stack((y_pred, y_pred**2))
                exog = sm.add_constant(exog)
                white_test = het_white(residuals**2, exog)
                white_result = {
                    'Modelo': model_name,
                    'Tratamiento': treatment,
                    'Prueba': 'White',
                    'Estadístico': white_test[0],
                    'p-valor': white_test[1],
                    'Resultado': 'Rechaza H0 (No homocedástico)' if white_test[1] < 0.05 else 'No rechaza H0 (Homocedástico)'
                }

                results.append(white_result)
                
                # Prueba de normalidad de D'Agostino-Pearson
                dagostino_test = normaltest(residuals)
                normality_result = {
                    'Modelo': model_name,
                    'Tratamiento': treatment,
                    'Prueba': 'D\'Agostino-Pearson',
                    'Estadístico': dagostino_test.statistic,
                    'p-valor': dagostino_test.pvalue,
                    'Resultado': 'Rechaza H0 (No normal)' if dagostino_test.pvalue < 0.05 else 'No rechaza H0 (Normal)'
                }
                
                # Prueba de homocedasticidad de Levene
                levene_test = levene(residuals, y_pred)
                homoscedasticity_result = {
                    'Modelo': model_name,
                    'Tratamiento': treatment,
                    'Prueba': 'Levene',
                    'Estadístico': levene_test.statistic,
                    'p-valor': levene_test.pvalue,
                    'Resultado': 'Rechaza H0 (No homocedástico)' if levene_test.pvalue < 0.05 else 'No rechaza H0 (Homocedástico)'
                }

                results.append(normality_result)
                results.append(homoscedasticity_result)

                # DataFrame de puntos influyentes según distancia de Cook
                cook_df = pd.DataFrame({
                    'Observación': range(len(cook_distance)),
                    'Cook\'s Distance': cook_distance
                })
                cook_df_sorted = cook_df.sort_values(by="Cook's Distance", ascending=False).reset_index(drop=True)

                print(cook_df_sorted.head(7))

        results_df = pd.DataFrame(results)
        print(results_df)
        return results_df

    def jackknife_analysis(self, dependent_var, categorical_var, time_var, selected_models):
        results = []
        
        for treatment in self.df[categorical_var].unique():
            treatment_data = self.df[self.df[categorical_var] == treatment]
            n = len(treatment_data)
            
            for model_name in selected_models:
                model_func, _ = self.models[model_name]

                x_full = treatment_data[time_var].values
                y_full = treatment_data[f'{dependent_var}_normalized'].values
                full_params = self.fit_model(model_name, x_full, y_full)
                
                if full_params is None:
                    continue
                
                param_names = list(inspect.signature(model_func).parameters.keys())[1:len(full_params) + 1]
                jackknife_params = []
                
                # Jackknife resampling
                for i in range(n):
                    jackknife_sample = treatment_data.drop(treatment_data.index[i])
                    x = jackknife_sample[time_var].values
                    y = jackknife_sample[f'{dependent_var}_normalized'].values
                    
                    params = self.fit_model(model_name, x, y)
                    if params is not None:
                        jackknife_params.append(params)
                
                if not jackknife_params:
                    continue
                
                jackknife_params = np.array(jackknife_params)

                jackknife_mean = np.mean(jackknife_params, axis=0)
                jackknife_bias = (n - 1) * (jackknife_mean - full_params)
                jackknife_var = ((n - 1) / n) * np.sum((jackknife_params - jackknife_mean)**2, axis=0)
                jackknife_se = np.sqrt(jackknife_var)

                ci_lower = full_params - 1.96 * jackknife_se
                ci_upper = full_params + 1.96 * jackknife_se
                
                for i, param_name in enumerate(param_names):
                    results.append({
                        'Model': model_name,
                        'Treatment': treatment,
                        'Parameter': param_name,
                        'Estimate': full_params[i],
                        'Jackknife Mean': jackknife_mean[i],
                        'Bias': jackknife_bias[i],
                        'Variance': jackknife_var[i],
                        'Std Error': jackknife_se[i],
                        'CI Lower': ci_lower[i],
                        'CI Upper': ci_upper[i]
                    })
        results_df = pd.DataFrame(results)
        try:
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_excel(output_dir / 'Jackknife.xlsx', index=False)
            print(f"Saved jackknife results to {output_dir / 'Jackknife.xlsx'}")
        except Exception as e:
            print(f"Warning: could not save jackknife results: {e}")
        print("\nJackknife Analysis Results:")
        print(results_df.to_string(index=False))
        return results_df

    def bootstrap_analysis(
        self, dependent_var, categorical_var, time_var, selected_models,
        n_iterations=1000, confidence_level=0.95, parametric=True,
        outlier_filter=True, outlier_threshold=5, n_timepoints=100
    ):
        """
        Bootstrap analysis extendido:
          - ICs de parámetros.
          - ICs de curvas predichas por tratamiento.
          - Comparaciones de curvas completas entre tratamientos.
        """
        results = []
        bootstrap_by_group = {}
        curves_by_group = {}
    
        x_min = self.df[time_var].min()
        x_max = self.df[time_var].max()
        x_grid = np.sort(self.df[time_var].unique())
    
        for treatment in self.df[categorical_var].unique():
            treatment_data = self.df[self.df[categorical_var] == treatment]
    
            for model_name in selected_models:
                model_func, _ = self.models[model_name]
                bootstrap_params = []
                bootstrap_curves = []
    
                for _ in range(n_iterations):
                    if parametric:
                        # --- Parametric bootstrap ---
                        x = treatment_data[time_var].values
                        y = treatment_data[f'{dependent_var}_normalized'].values
                        base_params = self.fit_model(model_name, x, y)
                        if base_params is None:
                            continue
                        y_pred = model_func(x, *base_params)
                        residuals = y - y_pred
                        y_sim = y_pred + np.random.choice(residuals, size=len(residuals), replace=True)
                        params = self.fit_model(model_name, x, y_sim)
                    else:
                        # --- Nonparametric bootstrap ---
                        bootstrap_sample = treatment_data.sample(n=len(treatment_data), replace=True)
                        x = bootstrap_sample[time_var].values
                        y = bootstrap_sample[f'{dependent_var}_normalized'].values
                        params = self.fit_model(model_name, x, y)
    
                    if params is not None:
                        bootstrap_params.append(params)
                        bootstrap_curves.append(model_func(x_grid, *params))
    
                if bootstrap_params:
                    param_names = list(inspect.signature(model_func).parameters.keys())[1:len(params) + 1]
                    bootstrap_array = np.array(bootstrap_params)

                    if outlier_filter:
                        clean_bootstrap = []
                        clean_curves = []
                        for row, curve in zip(bootstrap_array, bootstrap_curves):
                            if all(np.abs(row) < outlier_threshold * np.nanmax(bootstrap_array, axis=0)):
                                clean_bootstrap.append(row)
                                clean_curves.append(curve)
                        bootstrap_array = np.array(clean_bootstrap)
                        bootstrap_curves = np.array(clean_curves)
    
                    key = (model_name, treatment)
                    bootstrap_by_group[key] = {
                        param_names[i]: bootstrap_array[:, i] for i in range(len(param_names))
                    }

                    curves_by_group[key] = np.array(bootstrap_curves)

                    for i, param_name in enumerate(param_names):
                        values = bootstrap_array[:, i]
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        ci_lower, ci_upper = np.percentile(
                            values, [(1-confidence_level)*100/2, 100 - (1-confidence_level)*100/2]
                        )
                        results.append({
                            'Model': model_name,
                            'Treatment': treatment,
                            'Parameter': param_name,
                            'Bootstrap Mean': mean_val,
                            'Std. Error': std_val,
                            'CI Lower': ci_lower,
                            'CI Upper': ci_upper
                        })
    
        results_df = pd.DataFrame(results)
        try:
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_excel(output_dir / 'Bootstrap.xlsx', index=False)
            print(f"Saved bootstrap results to {output_dir / 'Bootstrap.xlsx'}")
        except Exception as e:
            print(f"Warning: could not save bootstrap results: {e}")
        print("\n=== Bootstrap Parameter Estimates ===")
        print(results_df.to_string(index=False))

        # --- Comparaciones entre grupos ---
        print("\n=== Bootstrap Differences Between Groups ===")
        group_names = list(self.df[categorical_var].unique())
        
        for model_name in selected_models:
            param_names = list(inspect.signature(self.models[model_name][0]).parameters.keys())[1:]
            if len(group_names) != 2:
                print(f"[!] Se requieren exactamente dos grupos para '{model_name}'.")
                continue
            
            group1, group2 = group_names
            key1, key2 = (model_name, group1), (model_name, group2)
            
            if key1 not in bootstrap_by_group or key2 not in bootstrap_by_group:
                print(f"[!] Faltan datos bootstrap para comparar '{group1}' y '{group2}' en '{model_name}'.")
                continue
            
            for param_name in param_names:
                values1 = bootstrap_by_group[key1].get(param_name)
                values2 = bootstrap_by_group[key2].get(param_name)
                
                if values1 is None or values2 is None:
                    continue
                
                n = min(len(values1), len(values2))
                delta = np.array(values1[:n]) - np.array(values2[:n])
                delta_mean = np.mean(delta)
                delta_std = np.std(delta)
                ci_lower, ci_upper = np.percentile(delta, [2.5, 97.5])
                
                print(f"Model: {model_name} | Parameter: {param_name}")
                print(f"  Mean Δ: {delta_mean:.6f}")
                print(f"  Std. Error: {delta_std:.6f}")
                print(f"  95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                print(f"  Significant: {'YES' if ci_lower > 0 or ci_upper < 0 else 'NO'}\n")
    
        # --- Comparaciones entre curvas ---
        print("\n=== Bootstrap Curve Comparisons Between Groups ===")
        curve_comparison_results = []
        group_names = list(self.df[categorical_var].unique())
        if len(group_names) == 2:
            group1, group2 = group_names
            for model_name in selected_models:
                key1, key2 = (model_name, group1), (model_name, group2)
                if key1 not in curves_by_group or key2 not in curves_by_group:
                    continue
    
                curves1 = curves_by_group[key1]
                curves2 = curves_by_group[key2]
                n = min(len(curves1), len(curves2))
                deltas = curves2[:n] - curves1[:n]
                mean_delta = np.mean(deltas, axis=0)
                ci_lower = np.percentile(deltas, 2.5, axis=0)
                ci_upper = np.percentile(deltas, 97.5, axis=0)
    
                # Área bajo la curva (AUC)
                auc1 = np.trapz(curves1, x_grid, axis=1)
                auc2 = np.trapz(curves2, x_grid, axis=1)
                delta_auc = auc2 - auc1
                auc_mean = np.mean(delta_auc)
                auc_ci = np.percentile(delta_auc, [2.5, 97.5])
                auc_significant = 'YES' if auc_ci[0] > 0 or auc_ci[1] < 0 else 'NO'
    
                print(f"\nModel: {model_name}")
                print(f"  ΔAUC (mean): {auc_mean:.6f}")
                print(f"  95% CI for ΔAUC: [{auc_ci[0]:.6f}, {auc_ci[1]:.6f}]")
                print(f"  Significant (AUC): {auc_significant}")
    
                # Chequeo de significancia a lo largo de la curva
                sig_points = (ci_lower > 0) | (ci_upper < 0)
                perc_sig = np.mean(sig_points) * 100
                print(f"  % of timepoints with significant difference: {perc_sig:.1f}%")

                curve_comparison_results.append({
                    'Model': model_name,
                    'Group_1': group1,
                    'Group_2': group2,
                    'Delta_AUC_Mean': auc_mean,
                    'Delta_AUC_CI_Lower': auc_ci[0],
                    'Delta_AUC_CI_Upper': auc_ci[1],
                    'AUC_Significant': auc_significant,
                    'Perc_Significant_Timepoints': perc_sig
                })

        if curve_comparison_results:
            curve_comparison_df = pd.DataFrame(curve_comparison_results)
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            curve_comparison_df.to_excel(output_dir / 'Bootstrap_Curve_Comparisons.xlsx', index=False)

        sig_df = pd.DataFrame({
            'Time': x_grid,
            'Significant': sig_points
        })

        output_dir = Path("Output") / "Results_Growth"
        output_dir.mkdir(parents=True, exist_ok=True)
        sig_df.to_excel(output_dir / 'sig_points.xlsx', index=False)
        return results_df, curves_by_group, x_grid, sig_df

    def cross_validation(self, dependent_var, categorical_var, time_var, selected_models, n_splits=5):
        results = []
        for treatment in self.df[categorical_var].unique():
            treatment_data = self.df[self.df[categorical_var] == treatment]
            
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for model_name in selected_models:
                model_func, _ = self.models[model_name]
                cv_scores = []
                
                for train_index, test_index in kf.split(treatment_data):
                    train_data = treatment_data.iloc[train_index]
                    test_data = treatment_data.iloc[test_index]
                    
                    x_train = train_data[time_var].values
                    y_train = train_data[f'{dependent_var}_normalized'].values
                    x_test = test_data[time_var].values
                    y_test = test_data[f'{dependent_var}_normalized'].values
                    
                    params = self.fit_model(model_name, x_train, y_train)
                    if params is not None:
                        y_pred = model_func(x_test, *params)
                        mse = np.mean((y_test - y_pred)**2)
                        cv_scores.append(mse)
                
                if cv_scores:
                    results.append({
                        'Model': model_name,
                        'Treatment': treatment,
                        'Mean MSE': np.mean(cv_scores),
                        'Std MSE': np.std(cv_scores)
                    })
        
        results_df = pd.DataFrame(results)
        try:
            output_dir = Path("Output") / "Results_Growth"
            output_dir.mkdir(parents=True, exist_ok=True)
            results_df.to_excel(output_dir / 'Cross_validaton.xlsx', index=False)
            print(f"Saved cross-validation results to {output_dir / 'Cross_validaton.xlsx'}")
        except Exception as e:
            print(f"Warning: could not save cross-validation results: {e}")
        print("\nCross-Validation Results:")
        print(results_df.to_string(index=False))
        return results_df
        
    
        
    def process_analysis(self, dependent_var, categorical_var, y_sd, time_var, name_xaxis, name_yaxis, selected_models, 
                         normalization_method='none', fitting_method='curve_fit'):
        
        
        """
        Método principal que ejecuta el análisis completo en el orden adecuado.
        :param dependent_var: Nombre de la variable dependiente (e.g. 'peso', 'longitud').
        :param categorical_var: Nombre de la variable categórica (e.g. 'tratamiento').
        :param time_var: Nombre de la variable de tiempo (e.g. 'tiempo').
        :param name_xaxis: Etiqueta para el eje X en los gráficos.
        :param name_yaxis: Etiqueta para el eje Y en los gráficos.
        """
        
        scaler = self.fit_all_models(dependent_var, categorical_var, time_var, normalization_method, fitting_method)
        
        # Graficar los modelos ajustados
        self.plot_fitted_models(dependent_var, categorical_var, time_var, scaler, name_xaxis, name_yaxis)
        
        # Comparar los modelos ajustados
        self.compare_models()
        
        self.parameter_significance_test(dependent_var, categorical_var, time_var, selected_models)
        self.check_model_assumptions(dependent_var, categorical_var, time_var, selected_models)
        self.jackknife_analysis(dependent_var, categorical_var, time_var, selected_models)
        self.bootstrap_analysis(dependent_var, categorical_var, time_var, selected_models)
        self.cross_validation(dependent_var, categorical_var, time_var, selected_models)
        
        # Generar figura final de crecimiento
        self.plot_growth_final_figure(dependent_var, categorical_var, y_sd, time_var, name_xaxis, name_yaxis, selected_models)
    
    def plot_growth_final_figure(self, dependent_var, categorical_var, y_sd, time_var, name_xaxis, name_yaxis, selected_models):
        """
        Crea una figura final del ajuste del modelo de crecimiento con anotaciones de significancia.
        Basado en el método plot_Mortality_Growth del notebook, solo el subplot de crecimiento.
        """
        # Leer datos de significancia
        output_dir = Path("Output") / "Results_Growth"
        sig_file = output_dir / 'sig_points.xlsx'
        
        if not sig_file.exists():
            print("Warning: sig_points.xlsx not found. Cannot create final figure.")
            return
        
        sig_df = pd.read_excel(sig_file)
        
        # Leer datos de bootstrap comparisons para obtener Delta AUC
        bootstrap_file = output_dir / 'Bootstrap_Curve_Comparisons.xlsx'
        if not bootstrap_file.exists():
            print("Warning: Bootstrap_Curve_Comparisons.xlsx not found. Using default values.")
            delta_auc = 0.0
            ci_lower, ci_upper = 0.0, 0.0
        else:
            bootstrap_df = pd.read_excel(bootstrap_file)
            # Buscar el modelo seleccionado
            model_name = selected_models[0]
            model_row = bootstrap_df[bootstrap_df['Model'] == model_name]
            if not model_row.empty:
                delta_auc = model_row['Delta_AUC_Mean'].values[0]
                ci_lower = model_row['Delta_AUC_CI_Lower'].values[0]
                ci_upper = model_row['Delta_AUC_CI_Upper'].values[0]
            else:
                delta_auc = 0.0
                ci_lower, ci_upper = 0.0, 0.0
        
        # Configurar la figura
        sns.reset_defaults()
        plt.rcdefaults()
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
        plt.rcParams['font.family'] = 'Arial'
        
        model_name = selected_models[0]
        model_func, _ = self.models[model_name]
        
        # Graficar datos y ajustes para cada tratamiento
        for treatment in self.df[categorical_var].unique():
            treatment_data = self.df[self.df[categorical_var] == treatment]
            x = treatment_data[time_var].values
            y = treatment_data[dependent_var].values
            y_sd_vals = treatment_data[y_sd].values

            ax.scatter(x, y, color=self.colors[treatment], alpha=0.5, label=f'{treatment} (Data)', s=2, zorder=3)
            ax.errorbar(x, y, yerr=y_sd_vals, fmt='none', color=self.colors[treatment], alpha=0.2,
                       capsize=2, capthick=0.5, elinewidth=0.5, zorder=2)

            if treatment in self.fitted_params and model_name in self.fitted_params[treatment]:
                params = self.fitted_params[treatment][model_name]
                x_smooth = np.linspace(x.min(), x.max(), 300)
                y_smooth = model_func(x_smooth, *params)

                if self.current_normalization == 'max':
                    y_smooth *= self.df[dependent_var].max()
                elif self.current_normalization in ['minmax', 'standard', 'robust']:
                    scaler = self.normalization_methods[self.current_normalization](dependent_var)
                    y_smooth = scaler.inverse_transform(y_smooth.reshape(-1, 1)).flatten()
                
                ax.plot(x_smooth, y_smooth, color=self.colors[treatment], label=f'{treatment} (Fitted)', linewidth=1.5)

        sig_times = sig_df[sig_df['Significant'] == True]['Time'].values
        if len(sig_times) > 0:
            sig_start = sig_times.min()
            sig_end = sig_times.max()

            y_max = max([max(treatment_data[dependent_var] + treatment_data[y_sd])
                        for treatment_data in [self.df[self.df[categorical_var] == t] 
                                              for t in self.df[categorical_var].unique()]])
            y_line = y_max + (y_max * 0.05) 

            ax.plot([sig_start, sig_end], [y_line, y_line], color='black', linewidth=1)

            cap_height = y_max * 0.02
            ax.plot([sig_start, sig_start], [y_line - cap_height/2, y_line + cap_height/2], 
                   color='black', linewidth=1)
            ax.plot([sig_end, sig_end], [y_line - cap_height/2, y_line + cap_height/2], 
                   color='black', linewidth=1)
            
            x_mid = (sig_start + sig_end) / 2
            ax.text(x_mid, y_line + cap_height, '*', ha='center', va='bottom', 
                   fontsize=8, color='black', fontweight='bold')
        

        ax.text(0.05, 0.95, f'ΔAUC = {delta_auc:.2f} g·day\n95% CI [{ci_lower:.2f}, {ci_upper:.2f}]',
               transform=ax.transAxes, ha='left', va='top', fontsize=6, 
               fontweight='bold', color='black')

        ax.set_xlabel(name_xaxis, fontsize=8, fontweight='bold')
        ax.set_ylabel(name_yaxis, fontsize=8, fontweight='bold')
        ax.set_xlim(-5, max(self.df[time_var]) + 5)
        ax.set_ylim(0, 5)
        ax.tick_params(axis='both', which='major', labelsize=8, width=2, length=5)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(8)
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        plt.tight_layout()
        fig.savefig(output_dir / 'Growth_Final_Figure.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved final growth figure to {output_dir / 'Growth_Final_Figure.pdf'}")


