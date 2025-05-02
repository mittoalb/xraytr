import os
import re
import xraylib
import numpy as np
import pandas as pd
import requests
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Initialize xraylib
xraylib.XRayInit()

# Load offline cache of densities and names (CSV with columns formula,density_g_cm3,name)
df_off = pd.DataFrame()
if os.path.exists('densities.csv'):
    df_off = pd.read_csv('densities.csv', index_col='formula')
density_table = df_off['density_g_cm3'].to_dict()
name_table    = df_off['name'].to_dict()

# Helper: parse energy input
def parse_energies(s):
    try:
        if ':' in s:
            a, b, c = map(float, s.split(':'))
            return np.arange(a, b + 1e-9, c)
        return np.array([float(s)])
    except:
        return np.array([])

# Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H3(
        "X‑ray Transmissivity & Refractive Indices",
        style={'color':'yellow','fontWeight':'bold','fontSize':'32px','margin-bottom':'20px'}
    ),
    html.Div([
        html.Label("Formula:", style={'color':'white'}),
        dcc.Input(id='formula', type='text', value='SiO2', style={'margin-right':'10px'}),
        html.Div(id='name-display', style={'color':'white','fontStyle':'italic'})
    ], style={'display':'flex','alignItems':'center','margin-bottom':'10px'}),
    html.Div([
        html.Label("Density (g/cm³):", style={'color':'white'}),
        dcc.Input(id='density-input', type='text', placeholder='auto', style={'margin-right':'10px'}),
        html.Div(id='density-note', style={'color':'white','fontStyle':'italic'})
    ], style={'display':'flex','alignItems':'center','margin-bottom':'10px'}),
    html.Div([
        html.Label("Thickness (mm):", style={'color':'white'}),
        dcc.Input(id='thickness', type='number', value=1.0, style={'margin-right':'20px'}),
        html.Label("Energy (keV start:stop:step):", style={'color':'white'}),
        dcc.Input(id='energy', type='text', value='8.0:20.0:0.5')
    ], style={'display':'flex','alignItems':'center','margin-bottom':'20px'}),
    html.Button('Compute', id='compute-btn', n_clicks=0, style={'margin-bottom':'20px'}),
    dcc.Graph(id='trans-plot', style={'height':'45vh'}),
    dcc.Graph(id='delta-beta-plot', style={'height':'45vh'})
], style={'backgroundColor':'black','padding':'20px','height':'100vh'})

@app.callback(
    [Output('name-display','children'),
     Output('density-note','children'),
     Output('trans-plot','figure'),
     Output('delta-beta-plot','figure'),
     Output('density-input','value')],
    [Input('compute-btn','n_clicks')],
    [State('formula','value'), State('density-input','value'),
     State('thickness','value'), State('energy','value')]
)
def update_graph(n_clicks, formula, den_in, t_mm, e_str):
    E = parse_energies(e_str)
    if not formula or E.size == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, den_in

    raw = formula.strip()
    if re.fullmatch(r'[A-Za-z]{1,2}', raw):
        f = raw[0].upper() + raw[1:].lower()
    else:
        f = raw

    try:
        Z = xraylib.SymbolToAtomicNumber(f)
        is_elem = True
    except:
        is_elem = False

    # Resolve name & density
    if is_elem:
        name = name_table.get(f, f)
        if den_in:
            try:
                rho = float(den_in)
                note = 'Using user density'
            except:
                rho = xraylib.ElementDensity(Z)
                note = f'Invalid input; element density {rho:.4g}'
        else:
            rho = xraylib.ElementDensity(Z)
            note = f'Element density {rho:.4g}'
    else:
        rho = None; name = None
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/{f}/property/Density,IUPACName/JSON"
            resp = requests.get(url, timeout=5)
            props = resp.json()['PropertyTable']['Properties'][0]
            rho = props.get('Density')
            name = props.get('IUPACName')
        except:
            pass
        name = name or name_table.get(f, f)
        if den_in:
            try:
                rho = float(den_in)
                note = 'Using user-provided density'
            except:
                rho = rho or density_table.get(f, 1.0)
                note = f'Invalid input; using available density {rho:.4g}'
        elif rho is not None:
            rho = float(rho)
            note = f'PubChem density {rho:.4g}'
        elif f in density_table:
            rho = density_table[f]
            note = f'Table density {rho:.4g}'
        else:
            rho = 1.0
            note = 'Defaulting to 1.0'

    if f not in density_table:
        with open('densities.csv','a') as cf:
            cf.write(f"{f},{rho:.6g},{name}\n")
        density_table[f] = rho
        name_table[f] = name

    t_cm = float(t_mm) / 10.0
    try:
        mu_rho = np.array([xraylib.CS_Total(Z, e) if is_elem else xraylib.CS_Total_CP(f, e) for e in E])
    except:
        mu_rho = np.zeros_like(E)
    T = np.exp(-mu_rho * rho * t_cm)

    n_real = np.array([xraylib.Refractive_Index_Re(f, e, rho) for e in E])
    n_imag = np.array([xraylib.Refractive_Index_Im(f, e, rho) for e in E])
    delta = 1.0 - n_real
    beta = n_imag

    y_max_T = np.nanmax(T) if T.size else 1.0
    y_max_db = max(np.nanmax(delta), np.nanmax(beta)) if delta.size else 1.0
    grid = dict(showgrid=True, gridcolor='gray', gridwidth=1, griddash='dash')
    base = {'paper_bgcolor':'black','plot_bgcolor':'black','font':{'color':'white','size':14},'margin':{'l':60,'r':20,'t':50,'b':60}}

    trans_fig = {'data':[{'x':E,'y':T,'mode':'lines+markers','line':{'color':'cyan'}}],
                 'layout':{**base,'xaxis':{**grid,'title':'Energy (keV)','tickformat':'.3g'},
                           'yaxis':{**grid,'title':'T','range':[0,y_max_T*1.05],'tickformat':'.2e'}}}
    db_fig = {'data':[{'x':E,'y':delta,'name':'δ'},{'x':E,'y':beta,'name':'β'}],
              'layout':{**base,'xaxis':{**grid,'title':'Energy (keV)','tickformat':'.3g'},
                        'yaxis':{**grid,'title':'δ & β','range':[0,y_max_db*1.05],'tickformat':'.2e'}}}

    return name, note, trans_fig, db_fig, f"{rho:.4g}"

if __name__ == '__main__':
    app.run(debug=True)
