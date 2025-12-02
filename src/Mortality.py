from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pandas as pd

class MortalityAnalysis:
    def __init__(self, file_path_mortality, initial_population):
        self.file_path = file_path_mortality
        self.data = pd.read_excel(self.file_path)
        self.initial_population = initial_population
        self.preprocess_data()
        self.colors = {'RTR': '#00AFBB', 'WTR': '#FFA040'}
        self.fills = {'RTR': '#CFFFFF', 'WTR': '#FFDAB8'}

    def preprocess_data(self):
        km_data = pd.DataFrame(columns=['Time', 'Group', 'Event'])

        for _, row in self.data.iterrows():
            time = row['Time']
            group = row['Group']
            n_dead = row['N_Dead']

            km_data = pd.concat([km_data, pd.DataFrame({
                'Time': [time] * n_dead,
                'Group': [group] * n_dead,
                'Event': [1] * n_dead
            })], ignore_index=True)

        for group in self.initial_population.keys():
            group_data = km_data[km_data['Group'] == group]
            n_alive = self.initial_population[group] - len(group_data)

            km_data = pd.concat([km_data, pd.DataFrame({
                'Time': [self.data['Time'].max()] * n_alive,
                'Group': [group] * n_alive,
                'Event': [0] * n_alive
            })], ignore_index=True)

        km_data = km_data.infer_objects()

        self.data_grouped = km_data
        
        print("Datos agrupados y procesados:")
        print(self.data_grouped.head())

    def plot_kaplan_meier(self, ax=None):
        """
        Genera las curvas de probabilidad de mortalidad acumulada para cada grupo.
        """
        from matplotlib import pyplot as plt
        kmf = KaplanMeierFitter()
        plt.rcParams['font.family'] = 'Arial'

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 2.5))
            created_fig = True
        else:
            fig = ax.figure
            created_fig = False
        
        for group in self.data_grouped['Group'].unique():
            group_data = self.data_grouped[self.data_grouped['Group'] == group]
            kmf.fit(durations=group_data['Time'], event_observed=group_data['Event'], label=group)
            kmf.plot_cumulative_density(ax=ax, color=self.colors[group], linewidth=2, ci_alpha=0.1, ci_show=False)
        
        result, groups = self.compare_mortality()

        ax.text(80, 0.05, f"Log-rank p-value = {result.p_value:.2e}",
                ha='center', va='bottom', fontsize=6, fontweight='bold', color='black')
        
        ax.set_xlim(-5, 165)
        ax.set_xticks(range(0, 161, 20))

        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        ax.tick_params(axis='both', which='major', length=5, labelsize=8, width=2)
        for label in ax.get_yticklabels():
            label.set_fontsize(8)
            label.set_fontweight('bold')
        
        for label in ax.get_xticklabels():
            label.set_fontsize(8)
            label.set_fontweight('bold')
        
        ax.get_legend().remove()
        ax.set_xlabel('Time (days)', fontsize=8, fontweight='bold')
        ax.set_ylabel('Cumulative Mortality Probability', fontsize=8, fontweight='bold')
    
        return fig, ax

    def compare_mortality(self):
        """
        Compara las curvas de supervivencia entre grupos utilizando la prueba log-rank.
        """
        groups = self.data_grouped['Group'].unique()
        
        if len(groups) != 2:
            raise ValueError("Se requiere exactamente dos grupos para la comparación.")
    
        mask_group1 = self.data_grouped['Group'] == groups[0]
        mask_group2 = self.data_grouped['Group'] == groups[1]
    
        result = logrank_test(
            self.data_grouped[mask_group1]['Time'], self.data_grouped[mask_group2]['Time'], 
            event_observed_A=self.data_grouped[mask_group1]['Event'], 
            event_observed_B=self.data_grouped[mask_group2]['Event']
        )
        
        print(f"Prueba log-rank entre {groups[0]} y {groups[1]}:")
        print(f"Test statistic: {result.test_statistic}")
        print(f"p-valor: {result.p_value}")
        
        return result, groups
    


