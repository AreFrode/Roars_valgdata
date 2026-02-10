# Configuration for election prediction analysis

# Party names (Norwegian political parties)
PARTIES = [
    'Arbeiderpartiet',  # Labour Party
    'Høyre',           # Conservative Party
    'FrP',             # Progress Party
    'SV',              # Socialist Left Party
    'SP',              # Centre Party
    'Rødt',            # Red Party
    'MDG',             # Green Party
    'KrF',             # Christian Democratic Party
    'Venstre'          # Liberal Party
]

# Synthetic results for 2025 election (replace with real results when available)
SYNTHETIC_RESULTS = {
    'Arbeiderpartiet': 28.2,
    'Høyre': 14.6,
    'FrP': 23.9,
    'SV': 5.5,
    'SP': 5.6,
    'Rødt': 5.3,
    'MDG': 4.7,
    'KrF': 4.2,
    'Venstre': 3.7
}

# Clustering parameters
DEFAULT_N_CLUSTERS = 3
PCA_COMPONENTS = 2
TSNE_PERPLEXITY = 30

# Plotting parameters
FIGURE_SIZE = (20, 16)
PLOT_STYLE = 'default'

# File paths
DATA_FILE = "Valgresultat 2025👍 (Svar) - Fasit.csv"
RESULTS_FILE = "results.csv"
ERROR_RESULTS_FILE = "Resultater.csv"

# Data processing
SKIP_ROWS = 1
TIMESTAMP_FORMAT = '%d.%m.%Y kl. %H.%M.%S'
