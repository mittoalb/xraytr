import os
import re
import xraylib
import numpy as np
import pandas as pd
import requests
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

# Import X-ray edge data
from xray_edges import get_edge_data, search_elements, filter_by_energy_range

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
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    html.H1(
        "X‑ray server Tools",
        style={'color':'yellow','fontWeight':'bold','fontSize':'32px','margin-bottom':'20px','textAlign':'center'}
    ),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Transmissivity & Refractive Index', value='tab-1', 
                style={'backgroundColor':'#333','color':'white'}, 
                selected_style={'backgroundColor':'#666','color':'yellow'}),
        dcc.Tab(label='X-ray Absorption Edges', value='tab-2', 
                style={'backgroundColor':'#333','color':'white'}, 
                selected_style={'backgroundColor':'#666','color':'yellow'})
    ], style={'height':'50px'}),
    
    html.Div(id='tab-content')
    
], style={'backgroundColor':'black','padding':'20px','minHeight':'100vh'})

# Tab 1 content (original functionality)
tab1_content = html.Div([
    html.H3(
        "X‑ray Transmissivity & Refractive Indices",
        style={'color':'yellow','fontWeight':'bold','fontSize':'24px','margin-bottom':'20px'}
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
    ], style={'display':'flex','alignItems':'center','margin-bottom':'10px'}),
    html.Div([
        html.Label("Plot mode:", style={'color':'white','margin-right':'10px'}),
        dcc.RadioItems(
            id='plot-mode',
            options=[
                {'label': 'δ & β separate', 'value': 'separate'},
                {'label': 'δ/β ratio', 'value': 'ratio'}
            ],
            value='separate',
            labelStyle={'display': 'inline-block', 'margin-right': '20px', 'color': 'white'}
        )
    ], style={'display':'flex','alignItems':'center','margin-bottom':'20px'}),
    html.Button('Compute', id='compute-btn', n_clicks=0, style={'margin-bottom':'20px'}),
    dcc.Graph(id='trans-plot', style={'height':'45vh'}),
    dcc.Graph(id='delta-beta-plot', style={'height':'45vh'})
])

# Tab 2 content (X-ray edge table)
tab2_content = html.Div([
    html.H3(
        "X-ray Absorption Edge Energies",
        style={'color':'yellow','fontWeight':'bold','fontSize':'24px','margin-bottom':'20px'}
    ),
    html.Div([
        html.P(
            "All energies are in eV (electron volts). Use these values for X-ray absorption spectroscopy analysis.",
            style={'color':'white', 'fontSize':'16px', 'marginBottom':'20px'}
        ),
        html.Div([
            html.Div([
                html.Label("Search element:", style={'color':'white', 'marginRight':'10px'}),
                dcc.Input(id='element-search', type='text', placeholder='Enter element name or symbol', 
                         style={'width':'200px', 'marginRight':'10px'}),
                html.Button('Clear', id='clear-search', n_clicks=0, style={'marginRight':'20px'})
            ], style={'display':'flex', 'alignItems':'center', 'marginBottom':'10px'}),
            
            html.Div([
                html.Label("Edge type:", style={'color':'white', 'marginRight':'10px'}),
                dcc.Dropdown(
                    id='edge-type-dropdown',
                    options=[
                        {'label': 'All edges', 'value': 'all'},
                        {'label': 'K-edge only', 'value': 'K'},
                        {'label': 'L₁-edge only', 'value': 'L1'},
                        {'label': 'L₂-edge only', 'value': 'L2'},
                        {'label': 'L₃-edge only', 'value': 'L3'}
                    ],
                    value='all',
                    style={'width':'150px', 'marginRight':'20px', 'color':'black'}
                ),
                html.Label("Energy range (eV):", style={'color':'white', 'marginRight':'10px'}),
                dcc.Input(id='energy-min', type='number', placeholder='Min', 
                         style={'width':'80px', 'marginRight':'5px'}),
                html.Span(" - ", style={'color':'white'}),
                dcc.Input(id='energy-max', type='number', placeholder='Max', 
                         style={'width':'80px'})
            ], style={'display':'flex', 'alignItems':'center', 'marginBottom':'20px'})
        ])
    ]),
    
    html.Div(
        id='edge-table-container',
        style={'maxHeight':'70vh', 'overflowY':'auto', 'border':'1px solid #444'}
    )
])

@app.callback(Output('tab-content', 'children'),
              Input('tabs', 'value'))
def render_content(active_tab):
    if active_tab == 'tab-1':
        return tab1_content
    elif active_tab == 'tab-2':
        return tab2_content

@app.callback(
    Output('edge-table-container', 'children'),
    [Input('element-search', 'value'),
     Input('edge-type-dropdown', 'value'),
     Input('energy-min', 'value'),
     Input('energy-max', 'value'),
     Input('clear-search', 'n_clicks')]
)
def update_edge_table(search_term, edge_type, energy_min, energy_max, clear_clicks):
    # Handle None values
    if edge_type is None:
        edge_type = 'all'
    
    # Get all edge data
    edge_data = get_edge_data()
    
    # Filter by search term
    if search_term:
        edge_data = search_elements(search_term)
    
    # Filter by energy range if edge type is specified
    if edge_type != 'all' and (energy_min is not None or energy_max is not None):
        edge_data = filter_by_energy_range(edge_data, energy_min, energy_max, edge_type)
    
    # Create table based on edge type selection
    if edge_type == 'all':
        # Show all edges
        columns = [
            {'name': 'Z', 'id': 'z', 'type': 'numeric'},
            {'name': 'Element', 'id': 'element', 'type': 'text'},
            {'name': 'Symbol', 'id': 'symbol', 'type': 'text'},
            {'name': 'K-edge (eV)', 'id': 'k_edge', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
            {'name': 'L₁-edge (eV)', 'id': 'l1_edge', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
            {'name': 'L₂-edge (eV)', 'id': 'l2_edge', 'type': 'numeric', 'format': {'specifier': ',.1f'}},
            {'name': 'L₃-edge (eV)', 'id': 'l3_edge', 'type': 'numeric', 'format': {'specifier': ',.1f'}}
        ]
        
        data = []
        for row in edge_data:
            data.append({
                'z': row[0],
                'element': row[1],
                'symbol': row[2],
                'k_edge': row[3] if row[3] is not None else 'N/A',
                'l1_edge': row[4] if row[4] is not None else 'N/A',
                'l2_edge': row[5] if row[5] is not None else 'N/A',
                'l3_edge': row[6] if row[6] is not None else 'N/A'
            })
    else:
        # Show only specific edge
        edge_names = {'K': 'K-edge', 'L1': 'L₁-edge', 'L2': 'L₂-edge', 'L3': 'L₃-edge'}
        edge_indices = {'K': 3, 'L1': 4, 'L2': 5, 'L3': 6}
        
        columns = [
            {'name': 'Z', 'id': 'z', 'type': 'numeric'},
            {'name': 'Element', 'id': 'element', 'type': 'text'},
            {'name': 'Symbol', 'id': 'symbol', 'type': 'text'},
            {'name': f'{edge_names[edge_type]} (eV)', 'id': 'edge_energy', 'type': 'numeric', 'format': {'specifier': ',.1f'}}
        ]
        
        data = []
        for row in edge_data:
            edge_value = row[edge_indices[edge_type]]
            if edge_value is not None:  # Only include elements with this edge
                data.append({
                    'z': row[0],
                    'element': row[1],
                    'symbol': row[2],
                    'edge_energy': edge_value
                })
    
    # Create DataTable
    table = dash_table.DataTable(
        data=data,
        columns=columns,
        style_cell={
            'backgroundColor': '#222',
            'color': 'white',
            'border': '1px solid #444',
            'textAlign': 'left',
            'padding': '8px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': '#3498db',
            'color': 'white',
            'fontWeight': 'bold',
            'border': '1px solid #444'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'symbol'},
                'textAlign': 'center',
                'fontWeight': 'bold'
            },
            {
                'if': {'column_id': 'z'},
                'textAlign': 'center'
            },
            {
                'if': {'column_id': ['k_edge', 'l1_edge', 'l2_edge', 'l3_edge', 'edge_energy']},
                'textAlign': 'right',
                'color': '#00FFFF'  # Cyan for energy values
            }
        ],
        sort_action='native',
        filter_action='native',
        page_size=20,
        style_table={'overflowX': 'auto'}
    )
    
    return table

@app.callback(
    [Output('name-display','children'),
     Output('density-note','children'),
     Output('trans-plot','figure'),
     Output('delta-beta-plot','figure'),
     Output('density-input','value')],
    [Input('compute-btn','n_clicks')],
    [State('formula','value'), State('density-input','value'),
     State('thickness','value'), State('energy','value'),
     State('plot-mode','value')]
)
def update_graph(n_clicks, formula, den_in, t_mm, e_str, plot_mode):
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
    grid = dict(showgrid=True, gridcolor='gray', gridwidth=1, griddash='dash')
    base = {'paper_bgcolor':'black','plot_bgcolor':'black','font':{'color':'white','size':14},'margin':{'l':60,'r':20,'t':50,'b':60}}

    trans_fig = {'data':[{'x':E,'y':T,'mode':'lines+markers','line':{'color':'cyan'}}],
                 'layout':{**base,'xaxis':{**grid,'title':'Energy (keV)','tickformat':'.3g'},
                           'yaxis':{**grid,'title':'T', 'range':[0,y_max_T*1.05],'tickformat':'.2e'}}}

    # Choose plot based on mode
    if plot_mode == 'ratio':
        ratio = np.divide(delta, beta, out=np.zeros_like(delta), where=beta!=0)
        y_max_ratio = np.nanmax(ratio) if ratio.size else 1.0
        db_fig = {'data':[{'x':E,'y':ratio,'mode':'lines+markers','line':{'color':'orange'},'name':'δ/β'}],
                  'layout':{**base,'xaxis':{**grid,'title':'Energy (keV)','tickformat':'.3g'},
                            'yaxis':{**grid,'title':'δ/β ratio','range':[0,y_max_ratio*1.05],'tickformat':'.2e'}}}
    else:  # separate
        y_max_db = max(np.nanmax(delta), np.nanmax(beta)) if delta.size else 1.0
        db_fig = {'data':[{'x':E,'y':delta,'name':'δ'},{'x':E,'y':beta,'name':'β'}],
                  'layout':{**base,'xaxis':{**grid,'title':'Energy (keV)','tickformat':'.3g'},
                            'yaxis':{**grid,'title':'δ & β','range':[0,y_max_db*1.05],'tickformat':'.2e'}}}

    return name, note, trans_fig, db_fig, f"{rho:.4g}"

# Clear search functionality
@app.callback(
    [Output('element-search', 'value'),
     Output('energy-min', 'value'),
     Output('energy-max', 'value')],
    [Input('clear-search', 'n_clicks')]
)
def clear_search(n_clicks):
    if n_clicks > 0:
        return '', None, None
    return dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8009)
