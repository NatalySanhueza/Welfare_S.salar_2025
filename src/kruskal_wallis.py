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
        Realiza el análisis de Kruskal-Wallis para cada variable dependiente y combina las variables independientes.

        Parámetros:
        - dependent_vars (list): Lista de nombres de las columnas de las variables dependientes.
        - independent_vars (list): Lista de nombres de las columnas de las variables independientes.
        
        Retorna:
        - pd.DataFrame: Tabla con resultados del test Kruskal-Wallis.
        """
        
        # Crear combinaciones únicas de las variables independientes
        self.data['var_independiente_combinada'] = self.data[independent_vars].astype(str).agg(' '.join, axis=1)
        
        # Lista para almacenar resultados
        results = []

        # Iterar sobre cada variable dependiente
        for dep_var in dependent_vars:
            
            # Obtener todas las combinaciones únicas de la variable combinada
            unique_combinations = self.data['var_independiente_combinada'].unique()
            
            # Crear diccionario para agrupar datos por cada combinación de la variable independiente combinada
            grouped_data = {comb: self.data[self.data['var_independiente_combinada'] == comb][dep_var].dropna() 
                            for comb in unique_combinations}
            
            # Preparar datos para el test de Kruskal-Wallis, usando solo combinaciones con datos
            group_data = [group for group in grouped_data.values() if len(group) > 0]
            
            # Verificar que haya suficientes grupos para el test
            if len(group_data) > 1:
                # Realizar el test Kruskal-Wallis
                stat, p_value = kruskal(*group_data)
                
                # Añadir resultados a la lista para cada combinación única
                for comb, group in grouped_data.items():
                    row = {var: comb.split()[i] for i, var in enumerate(independent_vars)}
                    row['var_independiente_combinada'] = comb
                    row['variable_dependiente'] = dep_var
                    row['estadistico'] = stat
                    row['p_valor'] = p_value
                    results.append(row)

        # Crear DataFrame con resultados
        result_df = pd.DataFrame(results)
        # Filtrar el DataFrame y mostrar resultados únicos
        filtered_df = result_df[['variable_dependiente', 'estadistico', 'p_valor']]
        unique_df = filtered_df.drop_duplicates(subset=['variable_dependiente'])
        print('\n Kruskal Wallis Test:')
        print(unique_df) # type: ignore
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
        
        # Crear combinaciones únicas de las variables independientes
        self.data['var_independiente_combinada'] = self.data[independent_vars].astype(str).agg(' '.join, axis=1)
        
        results = []
    
        for dep_var in dependent_vars:
            # Realizar el test post-hoc sin ajuste de p-valores
            if method == 'conover':
                post_hoc_result = sp.posthoc_conover(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=None)
            elif method == 'dunn':
                post_hoc_result = sp.posthoc_dunn(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=None)
            elif method == 'mann_whitney':
                post_hoc_result = sp.posthoc_mannwhitney(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=None)
            else:
                raise ValueError("Método no reconocido. Use 'conover', 'dunn', o 'mann_whitney'.")
            
            # Realizar el test post-hoc con ajuste de p-valores
            if method == 'conover':
                post_hoc_result_adj = sp.posthoc_conover(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=correction)
            elif method == 'dunn':
                post_hoc_result_adj = sp.posthoc_dunn(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=correction)
            elif method == 'mann_whitney':
                post_hoc_result_adj = sp.posthoc_mannwhitney(self.data, val_col=dep_var, group_col='var_independiente_combinada', p_adjust=correction)
            
            # Obtener todas las combinaciones únicas de grupos
            groups = post_hoc_result.index.tolist()
            
            # Crear resultados para cada par de grupos
            for group1, group2 in combinations(groups, 2):
                p_value = post_hoc_result.loc[group1, group2]
                p_value_adj = post_hoc_result_adj.loc[group1, group2]
                reject = p_value_adj < 0.05
                
                results.append(self._create_result_row(dep_var, group1, group2, p_value, p_value_adj, reject, independent_vars))
    
        result_df = pd.DataFrame(results)
        
        # Ordenar el DataFrame
        sort_columns = ['dependent_variable'] + [f'{var}_grupo1' for var in independent_vars] + [f'{var}_grupo2' for var in independent_vars]
        result_df = result_df.sort_values(by=sort_columns)
    
        print("\n Post-hoc Test:")
        print(f"\nTest of Multiple Comparisons: {method.capitalize()} | p adjustment method: {correction}")
        self.result_df = result_df
        print(result_df) # type: ignore
    
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
        
        # Agregar valores de las variables independientes
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

            if len(df.index)<2: df = df.rename(columns = {"p-unc" : "pval"})    #the pval column  has different names based on test and numerosity
            else: df = df.rename(columns = {"p-corr" : "pval"})

            df.rename({'A': 'group1', 'B': 'group2', "pval":"p-adj"}, axis="columns", inplace=True)


            df["p-adj"] = df["p-adj"].astype(float)

            # Creating a list of the different treatment groups from Tukey's
            group1 = set(df.group1.tolist())  # Dropping duplicates by creating a set
            group2 = set(df.group2.tolist())  # Dropping duplicates by creating a set
            groupSet = group1 | group2  # Set operation that creates a union of 2 sets
            groups = list(groupSet)

            # Creating lists of letters that will be assigned to treatment groups
            letters = list(string.ascii_lowercase+string.digits)[:len(groups)]
            cldgroups = letters

            # the following algoritm is a simplification of the classical cld,

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

            # this part will reassign the final name to the group
            # for sure there are more elegant ways of doing this
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
                        #Checking if there are forbidden pairing (proposition of solution to the imperfect script)                
                        forbidden = set()
                        for row in cld.itertuples():
                            if letters[g] in row[5]:
                                forbidden |= set(row[4])
                        if kitem in forbidden:
                            g=len(unique)+1

                        if cld["labels"].loc[cld[1] == kitem].iloc[0] == "":
                           cld["labels"].loc[cld[1] == kitem] += letters[g] 

                        # Checking if columns 1 & 2 of cld share at least 1 letter
                        if len(set(cld["labels"].loc[cld[1] == kitem].iloc[0]).intersection(cld.loc[cld[2] == item, "labels"].iloc[0])) <= 0:
                            if letters[g] not in list(cld["labels"].loc[cld[1] == kitem].iloc[0]):
                                cld["labels"].loc[cld[1] == kitem] += letters[g]
                            if letters[g] not in list(cld["labels"].loc[cld[2] == item].iloc[0]):
                                cld["labels"].loc[cld[2] == item] += letters[g]

                        if kitem in forbidden: #back to previous letter
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

    


# Crear un conjunto de datos de ejemplo
#np.random.seed(42)
#data = pd.DataFrame({
#    'peso': np.random.normal(100, 15, 200),
#    'longitud': np.random.normal(50, 5, 200),
#    'tiempo': np.random.choice([50, 100], 200),
#    'tratamiento': np.random.choice(['A', 'B', 'C'], 200),
#    'ubicacion': np.random.choice(['Norte', 'Sur'], 200)
#})
#
## Inicializar la clase StatisticalAnalysis
#analysis = StatisticalAnalysis(data)
#
## Definir variables dependientes e independientes
#dependent_vars = ['peso', 'longitud']
#independent_vars = ['tiempo', 'tratamiento']
#
#
## Ejecutar el análisis completo
#kw_results, post_hoc_results, cld_results = analysis.run_full_analysis(
#    dependent_vars, 
#    independent_vars, 
#    post_hoc_method='dunn', 
#    post_hoc_correction='bonferroni'
# )

# Imprimir resultados del análisis de Kruskal-Wallis
#print("Resultados del análisis de Kruskal-Wallis:")
#print(kw_results)
#print("\n")
#
## Imprimir resultados del análisis post hoc
#print("Resultados del análisis post hoc:")
#print(post_hoc_results)
#
## Ejemplo de cómo filtrar los resultados post hoc para una variable dependiente específica
#peso_post_hoc = post_hoc_results[post_hoc_results['dependent_vars'] == 'peso']
#print("\nResultados post hoc para la variable 'peso':")
#print(peso_post_hoc)
#
## Ejemplo de cómo encontrar diferencias significativas
#significativas = post_hoc_results[post_hoc_results['rechazar_h0'] == True]
#print("\nComparaciones significativas:")
#print(significativas)

# Guardar resultados en archivos CSV
#kw_results.to_csv('kruskal_wallis_results.csv', index=False)
#post_hoc_results.to_csv('post_hoc_results.csv', index=False)