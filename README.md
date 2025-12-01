# Cumulative Welfare in Atlantic Salmon (*Salmo salar*) Under Different Thermal Regimes

[![Article Status](https://img.shields.io/badge/Article-Under%20Review-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)]()
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

## About

This repository contains the complete data analysis pipeline for a comprehensive study on long-term welfare assessment in Atlantic salmon subjected to different thermal regimes. The analyses evaluate multiple welfare indicators to understand the impacts of thermal conditions on fish health and wellbeing.

**Analyses included:**
- **DNA Damage**: Quantification of telomere length (RTL), DNA methylation (5-mC), and oxidative DNA damage (8-OHdG)
- **Oxidative Stress**: Free radical quantification and antioxidant gene expression (SOD1, GPX1)
- **Growth**: Non-linear growth modeling using multiple candidate models
- **Mortality**: Kaplan-Meier survival analysis
- **Thermoregulatory Behavior**: Spatial distribution analysis and thermal preference index
- **Cumulative Welfare**: Principal component analysis integrating multiple welfare indicators

**Note**: This research is currently under review. The repository is provided for transparency and reproducibility purposes.

## Publication

**Status**: Manuscript under review

**Authors**: Nataly Sanhueza et al.

**Correspondence**: natalysanhueza@udec.cl

### How to Cite

When citing this work before publication, please use:

```
Sanhueza, N. et al. (2025). Behavioral Thermoregulation Influences Cumulative Welfare in Atlantic Salmon (Salmo salar). GitHub repository: https://github.com/NatalySanhueza/Welfare_S.salar_2025
```

*This citation will be updated once the manuscript is published.*

## Project Structure

```
Welfare_S.salar_2025/
├── Data/                              # Raw data files
│   ├── Cumulative_welfare/
│   ├── DNA_Damage/
│   │   ├── 5-mC.csv
│   │   ├── 8-OHdG.csv
│   │   └── RTL.csv
│   ├── Growth/
│   ├── Mortality/
│   ├── Oxidative_Stress/
│   └── Thermoregulatory_Behavior/
├── scripts/                           # Execution scripts for each analysis
│   ├── run_cumulative_welfare.py
│   ├── run_DNA_damage.py
│   ├── run_Growth.py
│   ├── run_Mortality.py
│   ├── run_Oxidative_Stress.py
│   └── run_Thermoregulatory_Behavior.py
├── src/                               # Core analysis modules
│   ├── __init__.py
│   ├── Cumulative_welfare.py
│   ├── DNA_Damage.py
│   ├── Growth.py
│   ├── kruskal_wallis.py
│   ├── Mortality.py
│   ├── Oxidative_Stress.py
│   └── Thermoregulatory_Behavior.py
├── Output/                            # Generated results (figures and tables)
│   ├── Results_Cumulative_welfare/
│   ├── Results_DNA_damage/
│   ├── Results_Growth/
│   ├── Results_Mortality/
│   ├── Results_Oxidative_Stress/
│   └── Results_Thermoregulatory_Behavior/
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Requirements

### System Requirements
- **Operating System**: Windows 10/11, macOS, or Linux
- **Python**: 3.8 or higher
- **Memory**: Minimum 4 GB RAM (8 GB recommended for Growth analysis)
- **Disk Space**: ~500 MB for installation + outputs

### Python Dependencies

All required packages are listed in `requirements.txt`:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- scikit-learn
- scikit-posthocs
- openpyxl
- lifelines
- vapeplot

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/NatalySanhueza/Welfare_S.salar_2025.git
cd Welfare_S.salar_2025
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Each analysis can be executed independently. All outputs (figures and statistical results) are automatically saved to the corresponding `Output/Results_[analysis]/` directory.

### Setting Matplotlib Backend (Important for headless execution)

**Windows (PowerShell):**
```powershell
$env:MPLBACKEND='Agg'
```

**macOS/Linux:**
```bash
export MPLBACKEND=Agg
```

### Running Individual Analyses

**DNA Damage Analysis:**
```bash
python scripts/run_DNA_damage.py
```
*Outputs*: 18 files including regression curves, distribution plots, correlation matrices, and statistical test results.

**Cumulative Welfare Analysis:**
```bash
python scripts/run_cumulative_welfare.py
```
*Outputs*: 23 files including PCA biplots, scatter plots, and statistical comparisons per tissue group.

**Growth Analysis:**
```bash
python scripts/run_Growth.py
```
*Outputs*: 10+ files including model comparison tables, fitted curves, bootstrap analyses, and cross-validation results.

**Mortality Analysis:**
```bash
python scripts/run_Mortality.py
```
*Outputs*: Kaplan-Meier survival curves with log-rank test results.

**Oxidative Stress Analysis:**
```bash
python scripts/run_Oxidative_Stress.py
```
*Outputs*: EPR spectra, free radical quantification, gene expression bar plots, and statistical comparisons.

**Thermoregulatory Behavior Analysis:**
```bash
python scripts/run_Thermoregulatory_Behavior.py
```
*Outputs*: Kernel density maps, Jacobs preference index, thermal preference analysis, and Mann-Whitney U test results.

### All-in-One Execution (Optional)

To run all analyses sequentially:

```bash
python scripts/run_DNA_damage.py
python scripts/run_cumulative_welfare.py
python scripts/run_Growth.py
python scripts/run_Mortality.py
python scripts/run_Oxidative_Stress.py
python scripts/run_Thermoregulatory_Behavior.py
```

## Results

All analyses generate:
- **Figures**: High-resolution PDFs and PNGs for publication
- **Statistical Tables**: Excel files with test results, p-values, and effect sizes
- **Model Outputs**: Parameter estimates, goodness-of-fit metrics, and validation results

### Key Outputs by Analysis

| Analysis | Key Figures | Statistical Outputs |
|----------|-------------|---------------------|
| DNA Damage | Standard curves, distribution plots, correlation matrices, bar plots | Kruskal-Wallis, post-hoc tests |
| Cumulative Welfare | PCA biplots, scatter plots, correlation matrices | Kruskal-Wallis per group, CLD labels |
| Growth | Model candidates, assumptions checks, bootstrap curves | Model comparison, parameter significance, jackknife, cross-validation |
| Mortality | Kaplan-Meier curves | Log-rank test |
| Oxidative Stress | EPR spectra, bar plots with error bars | Mann-Whitney U, Kruskal-Wallis |
| Thermoregulatory | Kernel density heatmaps, preference indices | Mann-Whitney U for chambers and photoperiod |

## Data Availability

All raw data and analysis scripts are provided in this repository to ensure full reproducibility. Data files are located in the `Data/` directory, organized by analysis type.

### Experimental Design

- **Treatment groups**: 
  - RTR: Restricted Thermal Range (constant 12°C)
  - WTR: Wide Thermal Range (9.6-16.4°C gradient)
- **Duration**: 160 days
- **Sample size**: Varies by analysis (see individual data files and publication for details)
- **Tissues analyzed**: Brain and muscle (where applicable)

## Reproducibility

All analyses follow these principles:
- **No hardcoded paths**: All file paths are dynamically resolved
- **Consistent execution**: Uniform command structure across all analyses
- **Automatic output organization**: Results saved to designated directories
- **Version controlled**: Full commit history available
- **Documented dependencies**: Exact package versions in requirements.txt

### Verification

To verify the installation and execution:

1. Check that the virtual environment is activated
2. Run one analysis (e.g., `python scripts/run_Mortality.py`)
3. Verify outputs in `Output/Results_Mortality/`

## Troubleshooting

**Import Errors:**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

**Matplotlib Display Issues:**
- Set backend: `$env:MPLBACKEND='Agg'` (Windows) or `export MPLBACKEND=Agg` (Linux/Mac)

**Memory Issues (Growth Analysis):**
- Close other applications
- Reduce bootstrap iterations if needed (modify in source code)

**Path Errors:**
- Verify you're running commands from the repository root directory
- Check that `Data/` directory contains all required files
- Check that `Data/` directory contains all required files

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) - see the LICENSE file for details.

**Important**: While the code and analysis pipeline are openly shared, the scientific findings and interpretations are part of ongoing research. Please contact the authors before using or citing this work in publications.

## Acknowledgments

See acknowledgments section in the associated publication.

## Contact

For questions or collaboration inquiries, please contact:

**Nataly Sanhueza**  
Email: natalysanhueza@udec.cl 
GitHub: [@NatalySanhueza](https://github.com/NatalySanhueza)

---

*Last updated: November 2025*
