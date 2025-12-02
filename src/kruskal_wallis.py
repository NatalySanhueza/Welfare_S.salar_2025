import pandas as pd
from scipy.stats import kruskal
from itertools import combinations
import scikit_posthocs as sp
import string

class StatisticalAnalysis:
    def __init__(self, data):
        self.data = data.copy()
        self.result_df = None  

    def kruskal_wallis_analysis(self, dependent_vars, independent_vars):
        """
        Realiza el análisis de Kruskal-Wallis.

        Parámetros:
        - dependent_vars (list): Lista de nombres de las columnas de las variables dependientes.
        - independent_vars (list): Lista de nombres de las columnas de las variables independientes.
        
        Retorna:
        - pd.DataFrame: Tabla con resultados del test Kruskal-Wallis.
        """

        self.data['var_independiente_combinada'] = self.data[independent_vars].astype(str).agg(' '.join, axis=1)
        results = []

        for dep_var in dependent_vars:
            unique_combinations = self.data['var_independiente_combinada'].unique()
            grouped_data = {comb: self.data[self.data['var_independiente_combinada'] == comb][dep_var].dropna() 
                            for comb in unique_combinations}
            group_data = [group for group in grouped_data.values() if len(group) > 0]
            if len(group_data) > 1:
                stat, p_value = kruskal(*group_data)
                for comb, group in grouped_data.items():
                    row = {var: comb.split()[i] for i, var in enumerate(independent_vars)}
                    row['var_independiente_combinada'] = comb
                    row['variable_dependiente'] = dep_var
                    row['estadistico'] = stat
                    row['p_valor'] = p_value
                    results.append(row)

        result_df = pd.DataFrame(results)
        filtered_df = result_df[['variable_dependiente', 'estadistico', 'p_valor']]
        unique_df = filtered_df.drop_duplicates(subset=['variable_dependiente'])
        print('\n Kruskal Wallis Test:')
        print(unique_df)
        return result_df

    def post_hoc_analysis(self, dependent_vars, independent_vars, method='conover', correction='bonferroni'):
        """
        Realiza análisis post hoc para el test de Kruskal-Wallis.
    
        Parámetros:
        - dependent_vars (list): Lista de nombres de las columnas de las variables dependientes.
        - independent_vars (list): Lista de nombres de las columnas de las variables independientes.
        - method (str): Método de comparación post hoc ('conover', 'dunn', o 'mann_whitney').
        - correction (str): Método de corrección para comparaciones múltiples.
    
        Retorna:
        - pd.DataFrame: Tabla con resultados del análisis post hoc.
        """

        self.data['var_independiente_combinada'] = self.data[independent_vars].astype(str).agg(' '.join, axis=1)
        
        results = []
    
        for dep_var in dependent_vars:
            if method == 'conover':
                post_hoc_result = sp.posthoc_conover(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=None)
            elif method == 'dunn':
                post_hoc_result = sp.posthoc_dunn(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=None)
            elif method == 'mann_whitney':
                post_hoc_result = sp.posthoc_mannwhitney(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=None)
            else:
                raise ValueError("Método no reconocido. Use 'conover', 'dunn', o 'mann_whitney'.")
            
            if method == 'conover':
                post_hoc_result_adj = sp.posthoc_conover(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=correction)
            elif method == 'dunn':
                post_hoc_result_adj = sp.posthoc_dunn(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=correction)
            elif method == 'mann_whitney':
                post_hoc_result_adj = sp.posthoc_mannwhitney(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=correction)

            groups = post_hoc_result.index.tolist()

            for group1, group2 in combinations(groups, 2):
                p_value = post_hoc_result.loc[group1, group2]
                p_value_adj = post_hoc_result_adj.loc[group1, group2]
                reject = p_value_adj < 0.05
                
                results.append(self._create_result_row(dep_var, group1, group2, p_value, p_value_adj, reject, independent_vars))
    
        result_df = pd.DataFrame(results)
        sort_columns = ['dependent_variable'] + [f'{var}_grupo1' for var in independent_vars] + [f'{var}_grupo2' for var in independent_vars]
        result_df = result_df.sort_values(by=sort_columns)
    
        print("\n Post-hoc Test:")
        print(f"\nTest of Multiple Comparisons: {method.capitalize()} | p adjustment method: {correction}")
        self.result_df = result_df
        print(result_df)
    
        return result_df
    
    def _create_result_row(self, dep_var, group1, group2, p_value, p_value_adj, reject, independent_vars):
        """
        Crea una fila de resultados para el DataFrame.
        """
        row = {
            'dependent_variable': dep_var,
            'group1': str(group1),
            'group2': str(group2),
            'p-value': p_value,
            'p-adj': p_value_adj,
            'reject': reject
        }

        group1_values = str(group1).split()
        group2_values = str(group2).split()
        for i, var in enumerate(independent_vars):
            row[f'{var}_grupo1'] = group1_values[i] if i < len(group1_values) else ''
            row[f'{var}_grupo2'] = group2_values[i] if i < len(group2_values) else ''
        
        return row
    


    def compact_letter_display(self, dependent_vars, independent_vars):
        '''
        Creates a compact letter display. This creates a dataframe consisting of
        2 columns, a column containing the treatment groups and a column containing
        the letters that have been assigned to the treatment groups. These letters
        are part of what's called the compact letter display. Treatment groups that
        share at least 1 letter are similar to each other, while treatment groups
        that don't share any letters are significantly different from each other.

        Parameters
        ----------
        df : Pandas dataframe
            Pandas dataframe containing raw Tukey test results from statsmodels.
        alpha : float
            The alpha value. The default is 0.05.

        Returns
        -------
        A dataframe representing the compact letter display, created from the Tukey
        test results.

        '''
        CI = 95
        df = self.result_df
        alpha = 1-CI/100

        all_results = []

        for dep_var in dependent_vars:

            if len(df.index)<2: df = df.rename(columns = {"p-unc" : "pval"})
            else: df = df.rename(columns = {"p-corr" : "pval"})

            df.rename({'A': 'group1', 'B': 'group2', "pval":"p-adj"}, axis="columns", inplace=True)


            df["p-adj"] = df["p-adj"].astype(float)

            group1 = set(df.group1.tolist())
            group2 = set(df.group2.tolist())
            groupSet = group1 | group2
            groups = list(groupSet)

            letters = list(string.ascii_lowercase+string.digits)[:len(groups)]
            cldgroups = letters

            cld = pd.DataFrame(list(zip(groups, letters, cldgroups)))
            cld[3]=""

            for row in df.itertuples():
                if df["p-adj"][row[0]] > (alpha):
                    cld.iat[groups.index(df["group1"][row[0]]), 2] += cld.iat[groups.index(df["group2"][row[0]]), 1]
                    cld.iat[groups.index(df["group2"][row[0]]), 2] += cld.iat[groups.index(df["group1"][row[0]]), 1]

                if df["p-adj"][row[0]] < (alpha):
                        cld.iat[groups.index(df["group1"][row[0]]), 3] +=  cld.iat[groups.index(df["group2"][row[0]]), 1]
                        cld.iat[groups.index(df["group2"][row[0]]), 3] +=  cld.iat[groups.index(df["group1"][row[0]]), 1]

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
                            g=len(unique)+1

                        if cld["labels"].loc[cld[1] == kitem].iloc[0] == "":
                           cld["labels"].loc[cld[1] == kitem] += letters[g] 

                        if len(set(cld["labels"].loc[cld[1] == kitem].iloc[0]).intersection(cld.loc[cld[2] == item, "labels"].iloc[0])) <= 0:
                            if letters[g] not in list(cld["labels"].loc[cld[1] == kitem].iloc[0]):
                                cld["labels"].loc[cld[1] == kitem] += letters[g]
                            if letters[g] not in list(cld["labels"].loc[cld[2] == item].iloc[0]):
                                cld["labels"].loc[cld[2] == item] += letters[g]

                        if kitem in forbidden:
                            g-=1

            cld = cld.sort_values("labels")
            cld.drop(columns=[1, 2, 3], inplace=True)
            cld['dependent_variable'] = dep_var
            all_results.append(cld)
            cld_df = pd.concat(all_results, ignore_index=True)
            cld_df = cld_df[['dependent_variable', 'groups', 'labels']]
            cld_df[independent_vars] = cld_df['groups'].str.split(' ', expand=True)

            print(f"\nCompact Letter Display (CLD) for Test of Multiple Comparisons:")
            print(cld_df)
            print('\n')
            return(cld_df)

    def run_full_analysis(self, dependent_vars, independent_vars, post_hoc_method='conover', post_hoc_correction='bonferroni'):
        """
        Ejecuta el análisis completo: Kruskal-Wallis seguido del análisis post hoc.

        Parámetros:
        - dependent_vars (list): Lista de nombres de las columnas de las variables dependientes.
        - independent_vars (list): Lista de nombres de las columnas de las variables independientes.
        - post_hoc_method (str): Método de comparación post hoc ('conover' o 'wilcoxon').
        - post_hoc_correction (str): Método de corrección para comparaciones múltiples ('bonferroni' o 'holm').

        Retorna:
        - tuple: (DataFrame de Kruskal-Wallis, DataFrame de análisis post hoc)
        """
        kw_results = self.kruskal_wallis_analysis(dependent_vars, independent_vars)
        post_hoc_results = self.post_hoc_analysis(dependent_vars, independent_vars, method=post_hoc_method, correction=post_hoc_correction)
        cld_results = self.compact_letter_display(dependent_vars, independent_vars)
        return kw_results, post_hoc_results, cld_results
