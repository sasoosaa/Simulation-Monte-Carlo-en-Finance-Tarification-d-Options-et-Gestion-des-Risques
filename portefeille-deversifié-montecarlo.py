import numpy as np
import pandas as pd
import yfinance as yf
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc

# Télécharger les données d'un actif
def download_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data['Adj Close']
    except Exception as e:
        print(f"Erreur lors du téléchargement de {ticker}: {e}")
        return None

# Initialisation de l'application Dash avec Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Mise en page de l'application
app.layout = dbc.Container([
    # Barre d'information en haut
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Simulation Monte Carlo - Portefeuille Diversifié", className="ms-2"),
            dbc.Button("Info", id="info-button", color="info", className="ms-auto"),
        ]),
        color="primary",
        dark=True,
    ),

    # Modal pour afficher les informations
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Méthode Monte Carlo et Portefeuille Diversifié")),
        dbc.ModalBody([
            html.P("""
                La méthode Monte Carlo est une technique de simulation utilisée pour modéliser la probabilité de différents résultats dans un processus qui ne peut pas être facilement prédit en raison de l'intervention de variables aléatoires.
            """),
            html.P("""
                Dans le contexte d'un portefeuille diversifié, la méthode Monte Carlo permet de simuler l'évolution future de la valeur du portefeuille en tenant compte de la volatilité des actifs et de leurs corrélations.
            """),
            html.P("""
                Cette application utilise les données historiques des actifs pour calculer les rendements et la volatilité, puis simule plusieurs scénarios possibles pour prédire la valeur future du portefeuille.
            """),
            html.P("""
                Les résultats incluent une simulation graphique, une distribution des prix simulés, ainsi que des métriques de risque comme la Value at Risk (VaR) et la Conditional Value at Risk (CVaR).
            """),
        ]),
        dbc.ModalFooter(
            dbc.Button("Fermer", id="close-modal", className="ms-auto")
        ),
    ], id="info-modal", is_open=False),

    # Contenu principal de l'application
    dbc.Card([
        dbc.CardBody([
            html.H5("Paramètres du portefeuille", className="card-title text-info"),
            dbc.Row([
                dbc.Col([
                    html.Label("Tickers des actifs (séparés par des virgules)"),
                    dcc.Input(id="tickers", type="text", placeholder="Ex: AAPL, MSFT, TSLA", className="form-control"),
                ], width=6),
                dbc.Col([
                    html.Label("Pourcentages d'allocation (séparés par des virgules, total = 100%)"),
                    dcc.Input(id="allocations", type="text", placeholder="Ex: 40, 30, 30", className="form-control"),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Date de début"),
                    dcc.Input(id="start-date", type="text", placeholder="AAAA-MM-JJ", className="form-control"),
                ], width=6),
                dbc.Col([
                    html.Label("Date de fin"),
                    dcc.Input(id="end-date", type="text", placeholder="AAAA-MM-JJ", className="form-control"),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Valeur initiale du portefeuille (€)"),
                    dcc.Input(id="initial-value", type="number", value=1000000, className="form-control"),
                ], width=6),
                dbc.Col([
                    html.Label("Horizon temporel (jours)"),
                    dcc.Input(id="time-horizon", type="number", value=365, className="form-control"),
                ], width=6),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Nombre d'itérations"),
                    dcc.Input(id="iterations", type="number", value=1000, className="form-control"),
                ], width=6),
                dbc.Col([
                    html.Label("Niveau de confiance (%)"),
                    dcc.Input(id="confidence-level", type="number", value=95, className="form-control"),
                ], width=6),
            ]),
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("Lancer l'analyse", id="run-analysis", color="primary", className="me-2"),
        ], width="auto"),
    ], className="mb-4 text-center"),
    
    dbc.Card([
        dbc.CardBody([
            html.H5("Simulation Monte Carlo - Portefeuille", className="card-title text-info"),
            dcc.Graph(id="monte-carlo-plot"),
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H5("Distribution des prix simulés", className="card-title text-info"),
            dcc.Graph(id="histogram-plot"),
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H5("Valeurs de Risque", className="card-title text-info"),
            html.Div(id="risk-metrics", className="mt-3")
        ])
    ], className="mb-4"),

    # Nouveaux graphiques supplémentaires
    dbc.Card([
        dbc.CardBody([
            html.H5("Heatmap de corrélation des actifs", className="card-title text-info"),
            dcc.Graph(id="correlation-heatmap"),
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H5("Rendements cumulés des actifs", className="card-title text-info"),
            dcc.Graph(id="cumulative-returns-plot"),
        ])
    ], className="mb-4"),

    dbc.Card([
        dbc.CardBody([
            html.H5("Distribution des rendements du portefeuille", className="card-title text-info"),
            dcc.Graph(id="returns-distribution-plot"),
        ])
    ], className="mb-4"),
], fluid=True)

# Callback pour gérer l'ouverture et la fermeture du modal
@app.callback(
    Output("info-modal", "is_open"),
    [Input("info-button", "n_clicks"), Input("close-modal", "n_clicks")],
    [State("info-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Callback principal pour l'analyse
@app.callback(
    [Output("monte-carlo-plot", "figure"),
     Output("histogram-plot", "figure"),
     Output("risk-metrics", "children"),
     Output("correlation-heatmap", "figure"),
     Output("cumulative-returns-plot", "figure"),
     Output("returns-distribution-plot", "figure")],
    [Input("run-analysis", "n_clicks")],
    [State("tickers", "value"),
     State("allocations", "value"),
     State("start-date", "value"),
     State("end-date", "value"),
     State("initial-value", "value"),
     State("time-horizon", "value"),
     State("iterations", "value"),
     State("confidence-level", "value")]
)
def run_analysis(n_clicks, tickers, allocations, start_date, end_date, initial_value, time_horizon, iterations, confidence_level):
    if not n_clicks:
        return go.Figure(), go.Figure(), "", go.Figure(), go.Figure(), go.Figure()

    try:
        tickers_list = [ticker.strip() for ticker in tickers.split(",")]
        allocations_list = [float(alloc) for alloc in allocations.split(",")]
    except:
        return go.Figure(), go.Figure(), "Erreur : Vérifiez les valeurs entrées.", go.Figure(), go.Figure(), go.Figure()

    if len(tickers_list) != len(allocations_list) or sum(allocations_list) != 100:
        return go.Figure(), go.Figure(), "Erreur : Les allocations doivent totaliser 100%.", go.Figure(), go.Figure(), go.Figure()

    prices = pd.DataFrame()
    for ticker in tickers_list:
        data = download_data(ticker, start_date, end_date)
        if data is not None:
            prices[ticker] = data

    if prices.empty:
        return go.Figure(), go.Figure(), "Erreur : Impossible de télécharger les données.", go.Figure(), go.Figure(), go.Figure()

    returns = prices.pct_change().dropna()
    portfolio_returns = returns.dot(np.array(allocations_list) / 100)

    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    portfolio_values = []

    for _ in range(iterations):
        simulated_prices = [initial_value]
        for _ in range(time_horizon):
            next_price = simulated_prices[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal())
            simulated_prices.append(next_price)
        portfolio_values.append(simulated_prices)

    mean_portfolio_values = np.mean(portfolio_values, axis=0)

    # Graphique Monte Carlo
    monte_carlo_fig = go.Figure()
    for simulation in portfolio_values[:100]:
        monte_carlo_fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=simulation, mode='lines', line=dict(color='rgba(0,0,255,0.1)'), showlegend=False))
    monte_carlo_fig.add_trace(go.Scatter(x=list(range(time_horizon + 1)), y=mean_portfolio_values, mode='lines', line=dict(color='red', width=2), name='Moyenne'))
    monte_carlo_fig.update_layout(xaxis_title="Jours", yaxis_title="Valeur du Portefeuille (€)", template="plotly_white")

    # Histogramme des prix simulés
    simulated_prices = initial_value * np.exp(time_horizon * (mu - 0.5 * sigma**2) + sigma * np.sqrt(time_horizon) * np.random.normal(0, 1, iterations))
    var_percentile = np.percentile(simulated_prices, (1 - confidence_level / 100) * 100)
    cvar = initial_value - np.mean(simulated_prices[simulated_prices <= var_percentile])

    histogram_fig = go.Figure()
    histogram_fig.add_trace(go.Histogram(x=simulated_prices, nbinsx=50, name="Prix simulés"))
    histogram_fig.add_vline(x=var_percentile, line_dash="dash", line_color="red", name="VaR")
    histogram_fig.update_layout(xaxis_title="Valeur (€)", yaxis_title="Fréquence", template="plotly_white")

    # Heatmap de corrélation
    correlation_matrix = returns.corr()
    heatmap_fig = px.imshow(correlation_matrix, text_auto=True, title="Matrice de corrélation des actifs", color_continuous_scale="Viridis")
    heatmap_fig.update_layout(xaxis_title="Actifs", yaxis_title="Actifs")

    # Rendements cumulés
    cumulative_returns = (1 + returns).cumprod()
    cumulative_returns_fig = go.Figure()
    for ticker in cumulative_returns.columns:
        cumulative_returns_fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns[ticker], mode='lines', name=ticker))
    cumulative_returns_fig.update_layout(xaxis_title="Date", yaxis_title="Rendement cumulé", template="plotly_white")

    # Distribution des rendements
    returns_distribution_fig = go.Figure()
    returns_distribution_fig.add_trace(go.Histogram(x=portfolio_returns, nbinsx=50, name="Rendements du portefeuille"))
    returns_distribution_fig.update_layout(xaxis_title="Rendements", yaxis_title="Fréquence", template="plotly_white")

    # Métriques de risque
    risk_metrics = f" Value at Risk (VaR) à : {var_percentile:.2f}€,  Conditional Value at Risk (CVaR): {cvar:.2f}€"

    return monte_carlo_fig, histogram_fig, risk_metrics, heatmap_fig, cumulative_returns_fig, returns_distribution_fig

if __name__ == "__main__":
    app.run_server(debug=True)