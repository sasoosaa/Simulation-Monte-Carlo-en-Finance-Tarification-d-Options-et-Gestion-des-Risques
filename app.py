import numpy as np
import pandas as pd
import yfinance as yf
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

class MonteCarloPricing:
    def __init__(self, ticker, K=None, T_days=None, r=None, sigma=None, simulations=1000, option_type='european', avg_type='arithmetic'):
        # Valider et garantir des paramètres positifs.
        self.ticker = ticker
        self.simulations = max(100, int(simulations))  # Minimum 100 simulations
        
        # Récupérer les données boursières avec gestion des erreurs
        try:
            self.data = self._fetch_stock_data()
        except Exception as e:
            print(f"Data fetch error: {e}")
            self.data = self._default_stock_data()
        
        # Définir les paramètres avec des valeurs de repli
        self.S0 = float(self.data.get('price', 100))
        self.K = float(K if K is not None else self.data['options']['strike'])
        
        # Garantir un nombre de jours jusqu'à l'échéance positif
        self.steps = max(10, int(T_days if T_days is not None else self.data['options']['days_to_expiry']))
        self.T = self.steps / 365  # Convertir en années
        
        # Valider les autres paramètres numériques
        self.r = float(r if r is not None else self.data.get('risk_free_rate', 0.05))
        self.sigma = float(sigma if sigma is not None else self.data.get('volatility', 0.2))
        
        self.option_type = option_type
        self.avg_type = avg_type
        self.dt = self.T / self.steps
    
    def _default_stock_data(self):
        """Fournir des données par défaut en cas d'échec de la récupération"""
        return {
            'price': 100,
            'volatility': 0.2,
            'risk_free_rate': 0.05,
            'options': {
                'strike': 100,
                'days_to_expiry': 30
            }
        }
    
    def _fetch_stock_data(self):
        """Récupérer les données des actions et options depuis Yahoo Finance"""
        try:
            # Obtenir les informations sur le ticker de l'action
            stock = yf.Ticker(self.ticker)
            
            # Récupérer le prix actuel de l'action
            price = float(stock.history(period="1d")['Close'].iloc[0])
            
            # Obtenir les données d'options (prochaine échéance)
            options = stock.options
            if not options:
                raise ValueError(f"No options data available for {self.ticker}")
            
            # Extraire la chaîne d'options de la prochaine échéance
            next_expiry = options[0]
            option_chain = stock.option_chain(next_expiry)
            
            # Calculer le prix d'exercice moyen et la volatilité implicite des options d'achat
            calls = option_chain.calls
            avg_strike = float(calls['strike'].mean())
            implied_vol = float(calls['impliedVolatility'].mean())
            
            # Déterminer le nombre de jours jusqu'à l'échéance
            expiry_date = pd.to_datetime(next_expiry)
            days_to_expiry = max(10, (expiry_date - pd.Timestamp.now()).days)
            
            # Estimer le taux sans risque (approximé à partir du rendement des bons du Trésor à 10 ans)
            risk_free_rate = float(yf.Ticker('^TNX').history(period='1d')['Close'].iloc[0] / 100)
            
            return {
                'price': price,
                'volatility': implied_vol,
                'risk_free_rate': risk_free_rate,
                'options': {
                    'expiry_date': next_expiry,
                    'days_to_expiry': days_to_expiry,
                    'strike': avg_strike
                }
            }
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return self._default_stock_data()
    
    def simulate(self):
        """Simuler les trajectoires de prix avec une gestion robuste des erreurs"""
        try:
            np.random.seed(42)
            
            # Vérifier que les dimensions des matrices sont positives
            steps = max(10, int(self.steps))
            sims = max(100, int(self.simulations))
            
            S = np.zeros((steps, sims))
            S[0] = self.S0
            
            drift = (self.r - 0.5*self.sigma**2)*self.dt
            vol = self.sigma*np.sqrt(self.dt)
            
            for t in range(1, steps):
                Z = np.random.normal(0, 1, sims)
                S[t] = S[t-1] * np.exp(drift + vol*Z)
            
            return S
        except Exception as e:
            print(f"Simulation error: {e}")
            # Fournir une simulation de secours avec des paramètres minimaux en cas d’erreur
            return np.full((10, 100), self.S0)
    
    def calculate_prices(self, S):
        discount = np.exp(-self.r*self.T)
        
        if self.option_type == 'asian':
            if self.avg_type == 'arithmetic':
                avg_price = np.mean(S, axis=0)
            else:  # geometric
                avg_price = np.exp(np.mean(np.log(S), axis=0))
            
            call = discount * np.mean(np.maximum(avg_price - self.K, 0))
            put = discount * np.mean(np.maximum(self.K - avg_price, 0))
        else:  # européen
            call = discount * np.mean(np.maximum(S[-1] - self.K, 0))
            put = discount * np.mean(np.maximum(self.K - S[-1], 0))
        
        return call, put


app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.LUX],
                suppress_callback_exceptions=True)

app.title = "Financial Simulations "

app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Analyse Financière ", className="ms-2"),
            dbc.Nav([
                dbc.NavLink("Options", href="/options", active="exact"),
                dbc.NavLink("Portfeuille", href="/portfolio", active="exact"),
                dbc.NavLink("Infos", href="/infos", active="exact"),
            ], className="ms-auto", navbar=True)
        ]),
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    dcc.Location(id='url'),
    html.Div(id='page-content')
], fluid=True)


options_layout = dbc.Container([
    dbc.Card([
        dbc.CardBody([
            html.H4("Calculateur de Tarification d'Options ", className="card-title text-primary"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Stock Ticker"),
                    dbc.Input(id="opt-ticker", value="AAPL", type="text")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("prix d'exercice (€)"),
                    dbc.Input(id="opt-strike", type="number", placeholder="Auto-detect")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("Jours jusqu'à l'échéance"),
                    dbc.Input(id="opt-days", type="number", placeholder="Auto-detect")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("Type d'option"),
                    dcc.Dropdown(
                        id='opt-type',
                        options=[
                            {'label': 'Européenne', 'value': 'european'},
                            {'label': 'Asiatique', 'value': 'asian'}
                        ],
                        value='european'
                    )
                ], md=3),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Taux Sans Risque (%)"),
                    dbc.Input(id="opt-rate", type="number", placeholder="Auto-detect")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("Volatilité (%)"),
                    dbc.Input(id="opt-vol", type="number", placeholder="Auto-detect")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("Simulations"),
                    dbc.Input(id="opt-sims", value=1000, type="number")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("Type de moyenne"),
                    dcc.Dropdown(
                        id='opt-avg-type',
                        options=[
                            {'label': 'Géométrique', 'value': 'geometric'}
                            ],
                            value='geometric',  # Valeur par défaut forcée
                            disabled=True,      # Empêche la modification
                            style={'backgroundColor': '#f8f9fa'}  # Style visuel pour indiquer le statut désactivé
                            )
                            ], md=3),
            ]),
            
            dbc.Button("Calculate", 
                      id="opt-calculate", 
                      color="primary", 
                      className="mt-3 w-100")
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id="opt-paths"), md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Results", className="text-success"),
                    html.Div(id="opt-results", className="mt-3 fs-5")
                ])
            ])
        ], md=4)
    ])
])


portfolio_layout = dbc.Container([
    dbc.Card([
        dbc.CardBody([
            html.H4("Analyse de Portefeuille", className="card-title text-primary"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Actifs (séparés par des virgules)"),
                    dbc.Input(id="pf-tickers", placeholder="AAPL, MSFT, ...")
                ], md=4),
                
                dbc.Col([
                    dbc.Label("Allocations (%)"),
                    dbc.Input(id="pf-allocations", placeholder="40, 30, 30")
                ], md=4),
                
                dbc.Col([
                    dbc.Label("Niveau de confiance (%)"),
                    dbc.Input(id="pf-confidence", value=95, type="number")
                ], md=4),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Date de début"),
                    dbc.Input(id="pf-start", placeholder="YYYY-MM-DD")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("Date de fin"),
                    dbc.Input(id="pf-end", placeholder="YYYY-MM-DD")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("Investissement initial (€)"),
                    dbc.Input(id="pf-initial", value=100000, type="number")
                ], md=3),
                
                dbc.Col([
                    dbc.Label("Taux sans risque (%)"),
                    dbc.Input(id="pf-risk-free", value=2.5, type="number")
                ], md=3),
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Horizon temporel (jours)"),
                    dbc.Input(id="pf-horizon", value=365, type="number")
                ], md=6),
                
                dbc.Col([
                    dbc.Label("Simulations"),
                    dbc.Input(id="pf-sims", value=1000, type="number")
                ], md=6),
            ]),
            
            dbc.Button("Analyser", 
                      id="pf-analyze", 
                      color="primary", 
                      className="mt-3 w-100")
        ])
    ], className="mb-4"),
    
    dbc.Card([
        dbc.CardBody([
            html.H4("Métriques de risque", className="text-danger"),
            html.Div(id="pf-risk-metrics", className="mt-3 fs-5")
        ])
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id="pf-simulations"), md=8),
        dbc.Col(dcc.Graph(id="pf-correlation"), md=4)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Graph(id="pf-distribution"), md=6),
        dbc.Col(dcc.Graph(id="pf-returns"), md=6)
    ])
])


info_layout = dbc.Container([
    dbc.Card([
        dbc.CardBody([
            # Titre principal
            html.H2("Méthode Monte Carlo", className="text-primary mb-4"),
            
            # Section : Principe Général
            html.H4("Principe Général", className="mt-4"),
            dcc.Markdown(
                '''
                La méthode Monte Carlo est une technique numérique qui utilise l'échantillonnage aléatoire 
                pour résoudre des problèmes mathématiques ou physiques complexes. Son nom vient du quartier 
                de Monte Carlo à Monaco, réputé pour ses casinos et les jeux de hasard.
                ''', 
                className="mb-3"
            ),
            
            # Section : Application en Finance
            html.H4("Application en Finance", className="mt-4"),
            dcc.Markdown(
                '''
                **Utilisations typiques :**  
                - Valorisation d'options exotiques  
                - Analyse de risque de portefeuille  
                - Prévisions de marchés complexes  
                - Calculs de Value at Risk (VaR)  

                **Avantages :**  
                - Flexibilité dans la modélisation  
                - Gestion de dimensions multiples  
                - Précision améliorée avec plus de simulations  
                ''', 
                className="mb-3"
            ),
            
            # Section : Dans cette Application
            html.H4("Dans cette Application", className="mt-4"),
            dcc.Markdown(
                '''
                - **Options :** Simulation de trajectoires browniennes géométriques  
                - **Portefeuille :** Modélisation des corrélations entre actifs  
                - **Paramètres clés :**  
                    - Nombre de simulations (précision vs performance)  
                    - Volatilité (mesure du risque)  
                    - Horizon temporel (période de projection)  
                ''', 
                className="mb-3"
            ),
            
            # Section : Équations Clés (corrigée)
            html.H4("Équations Clés", className="mt-4"),
            dcc.Markdown(
                r"""
                **Mouvement Brownien Géométrique :**  
                $$
                dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
                $$   

                **Discounted Payoff (Options) :**  
                $$
                \text{Prix} = \mathbb{E}\left[ e^{-rT} \cdot \text{Payoff}(S_T) \right]
                $$  

                **Value at Risk (Portefeuille) :**  
                $$
                \text{VaR}_\alpha = \inf \left\{ l \in \mathbb{R} : \mathbb{P}(L > l) \leq 1 - \alpha \right\}
                $$  
                """, 
                className="mb-3",
                mathjax=True  # Activation de MathJax si nécessaire
            )
        ])
    ], className="mt-4")
], fluid=True)

@callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page(pathname):
    if pathname == "/portfolio":
        return portfolio_layout
    elif pathname == "/infos":
        return info_layout
    return options_layout


@callback(
    [Output("opt-paths", "figure"),
     Output("opt-results", "children")],
    Input("opt-calculate", "n_clicks"),
    [State("opt-ticker", "value"),
     State("opt-strike", "value"),
     State("opt-days", "value"),
     State("opt-rate", "value"),
     State("opt-vol", "value"),
     State("opt-sims", "value"),
     State("opt-type", "value"),
     State("opt-avg-type", "value")]
)
def update_options(_, ticker, K, T, r, vol, sims, option_type, avg_type):
    if not ticker:
        raise PreventUpdate
    
    # Créer un modèle avec des paramètres détectés automatiquement
    model = MonteCarloPricing(
        ticker, 
        K=K,  # Sera utilisé si aucun paramètre n'est spécifié
        T_days=T,  # Sera utilisé si aucun paramètre n'est spécifié
        r=r/100 if r else None,  # Convertir en décimal si fourni
        sigma=vol/100 if vol else None,  # Convertir en décimal si fourni
        simulations=sims, 
        option_type=option_type, 
        avg_type=avg_type
    )
    
    paths = model.simulate()
    call, put = model.calculate_prices(paths)
    
    fig = go.Figure()
    for i in range(min(20, sims)):
        fig.add_trace(go.Scatter(
            y=paths[:, i],
            mode='lines',
            line=dict(width=1),
            opacity=0.7
        ))
    fig.update_layout(
        title=f"Price Path Simulations for {ticker}",
        xaxis_title="Days",
        yaxis_title="Price (€)",
        template="plotly_white"
    )
    
    results = f"""
    Current Price: €{model.S0:.2f}
    Call Price: €{call:.2f}
    Put Price: €{put:.2f}
    
    Details:
    Strike: €{model.K:.2f}
    Volatility: {model.sigma*100:.2f}%
    Risk-Free Rate: {model.r*100:.2f}%
    Days to Expiry: {int(model.steps)}
    Simulations: {sims:,}
    Type: {option_type.capitalize()}{' ('+avg_type+')' if option_type == 'asian' else ''}
    """
    
    return fig, results

def calculate_risk_metrics(returns, confidence, risk_free_rate):
    var = np.percentile(returns, 100 - confidence)
    cvar = returns[returns <= var].mean()
    sharpe = (returns.mean() * 252 - risk_free_rate/100) / (returns.std() * np.sqrt(252))
    return {'var': var, 'cvar': cvar, 'sharpe': sharpe}

def download_asset_data(tickers, start, end):
    data = pd.DataFrame()
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end)['Adj Close']
            data[t] = df
        except:
            continue
    return data.dropna()

@callback(
    [Output("pf-simulations", "figure"),
     Output("pf-correlation", "figure"),
     Output("pf-distribution", "figure"),
     Output("pf-returns", "figure"),
     Output("pf-risk-metrics", "children")],
    Input("pf-analyze", "n_clicks"),
    [State("pf-tickers", "value"),
     State("pf-allocations", "value"),
     State("pf-start", "value"),
     State("pf-end", "value"),
     State("pf-initial", "value"),
     State("pf-horizon", "value"),
     State("pf-sims", "value"),
     State("pf-confidence", "value"),
     State("pf-risk-free", "value")]
)
def update_portfolio(_, tickers, allocs, start, end, initial, horizon, sims, confidence, risk_free):
    if not tickers or not allocs:
        raise PreventUpdate
    
    try:
        tickers = [t.strip() for t in tickers.split(",")]
        allocs = [float(a.strip())/100 for a in allocs.split(",")]
        
        if len(tickers) != len(allocs) or abs(sum(allocs)-1) > 0.01:
            raise ValueError("Invalid allocations")
            
        data = download_asset_data(tickers, start, end)
        if data.empty:
            raise ValueError("No data downloaded")
            
        returns = data.pct_change().dropna()
        portfolio_returns = returns.dot(allocs)
        
        # Calcul des métriques de risque
        risk_metrics = calculate_risk_metrics(portfolio_returns, confidence, risk_free)
        
        # Simulation Monte Carlo
        mu = portfolio_returns.mean() * 252
        sigma = portfolio_returns.std() * np.sqrt(252)
        simulations = []
        for _ in range(sims):
            prices = [initial]
            for _ in range(horizon):
                ret = np.random.normal(mu/252, sigma/np.sqrt(252))
                prices.append(prices[-1] * np.exp(ret))
            simulations.append(prices)
        
        # Génération des graphiques
        sim_fig = go.Figure()
        for s in simulations[:100]:
            sim_fig.add_trace(go.Scatter(y=s, line=dict(width=1, color='blue'), opacity=0.1))
        sim_fig.update_layout(title="Portfolio Value Simulations", template="plotly_white")
        
        corr_fig = go.Figure(go.Heatmap(
            z=returns.corr().values,
            x=returns.columns,
            y=returns.columns,
            colorscale='Blues'
        )).update_layout(title="Asset Correlation Matrix")
        
        dist_fig = go.Figure(go.Histogram(x=portfolio_returns, nbinsx=50))
        dist_fig.add_vline(x=risk_metrics['var'], line_dash="dash", line_color="red")
        dist_fig.update_layout(title="Returns Distribution", template="plotly_white")
        
        cum_returns = (1 + returns).cumprod()
        ret_fig = go.Figure()
        for col in cum_returns:
            ret_fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[col], name=col))
        ret_fig.update_layout(title="Historical Cumulative Returns", template="plotly_white")
        
        metrics_text = f"""
        VaR ({confidence}%): {risk_metrics['var']:.2%}
        CVaR ({confidence}%): {risk_metrics['cvar']:.2%}
        Sharpe Ratio: {risk_metrics['sharpe']:.2f}
        """
        
        return sim_fig, corr_fig, dist_fig, ret_fig, metrics_text
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return [go.Figure()]*4 + [f"Error: {str(e)}"]


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)