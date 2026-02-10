# Election Prediction Analysis Toolkit

En enkel, men omfattende analyse av valgdata - for moro skyld! 🚀

This toolkit provides comprehensive analysis of election prediction data, including error analysis, bias detection, clustering of prediction patterns, and beautiful visualizations.

## Features

- 📊 **Error Analysis**: Calculate total and party-specific prediction errors
- 🎯 **Bias Detection**: Identify systematic over/under-estimation patterns
- 🎭 **Clustering Analysis**: Group similar prediction patterns using machine learning
- 📈 **Rich Visualizations**: Create publication-quality plots and charts
- 🧪 **Well Tested**: Comprehensive unit tests for reliability
- 📦 **Easy Installation**: Simple pip install with all dependencies

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd roars-valgdata

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Basic Usage

```python
from src import main

# Run complete analysis
results = main.main()

# Or run individual components
from src.data_loader import load_and_prepare_data
from src.error_analysis import calculate_total_errors

predictions_df, results_df = load_and_prepare_data()
error_df = calculate_total_errors(predictions_df, results_df)
```

### Command Line Usage

```bash
# Run analysis with default settings
python -m src.main

# Specify custom data file
python -m src.main --data-file "my_predictions.csv"

# Adjust number of clusters
python -m src.main --n-clusters 4

# Skip plot generation
python -m src.main --no-plots
```

## Project Structure

```
roars-valgdata/
├── src/                          # Main package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── error_analysis.py        # Error calculation functions
│   ├── bias_analysis.py         # Bias detection functions
│   ├── clustering.py            # Clustering analysis
│   ├── visualization.py         # Plotting functions
│   └── main.py                  # Main orchestration
├── tests/                       # Unit tests
│   ├── __init__.py
│   └── test_analysis.py
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── README.md                    # This file
├── LICENSE                      # MIT License
├── analyse_data.ipynb           # Original Jupyter notebook
└── *.csv                        # Data files
```

## Analysis Components

### 1. Error Analysis
- Calculate absolute errors for each prediction
- Identify best and worst performers
- Analyze party-specific prediction difficulty
- Generate ranking and distribution plots

### 2. Bias Detection
- Detect systematic over/under-estimation
- Identify optimistic vs pessimistic predictors
- Analyze party-level bias patterns
- Create bias heatmaps and scatter plots

### 3. Clustering Analysis
- Group similar prediction patterns using K-means and hierarchical clustering
- Visualize relationships with PCA and t-SNE
- Create similarity networks
- Identify political archetypes

### 4. Visualization
- Error distribution histograms and box plots
- Party-wise error heatmaps and violin plots
- Bias detection scatter plots and heatmaps
- Clustering dendrograms and network graphs

## Data Format

The toolkit expects prediction data in CSV format with the following structure:

```csv
Tidsmerke,E-postadresse,Eget navn,Arbeiderpartiet,Høyre,FrP,SV,SP,Rødt,MDG,KrF,Venstre,...
06.09.2025 kl. 15.37.20,email@example.com,Person Name,"29,6","14,5","19,3","5,8",6,"6,5","5,5",5,"3,9",...
```

- `Tidsmerke`: Timestamp (DD.MM.YYYY kl. HH.MM.SS format)
- `Eget navn`: Respondent name
- Party columns: Prediction percentages (comma as decimal separator)

## Configuration

Edit `src/config.py` to customize:

- Party names and order
- Synthetic results (for testing)
- Plotting parameters
- Clustering settings
- File paths

## Development

### Running Tests

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. Create new module in `src/`
2. Add functions to `__init__.py`
3. Write comprehensive tests
4. Update documentation
5. Add to `setup.py` if needed

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Plotting
- seaborn: Statistical visualizations
- scikit-learn: Machine learning
- scipy: Scientific computing
- networkx: Network analysis

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Acknowledgments

- Original analysis by Roar and team
- Built with love for Norwegian election nerds 🇳🇴
