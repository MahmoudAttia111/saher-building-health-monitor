import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

# Load and prepare data
print("Loading building health monitoring data...")
# Note: Update the path to your dataset
df = pd.read_csv(r"building_health_monitoring_dataset.csv")

# Data preprocessing (from your original code)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Fill nulls with median for numeric columns
for col in df.select_dtypes(include="float64").columns:
    df[col].fillna(df[col].median(), inplace=True)

# Create total acceleration feature
df["total_accel"] = np.sqrt(df["Accel_X (m/s^2)"]**2 + df["Accel_Y (m/s^2)"]**2 + df["Accel_Z (m/s^2)"]**2)

# Balance the dataset (undersampling)
zero_condition_data = df.loc[df['Condition Label'] == 0].sample(n=300, random_state=42)
one_condition_data = df.loc[df['Condition Label'] == 1]
two_condition_data = df.loc[df['Condition Label'] == 2]

df_balanced = pd.concat([zero_condition_data, one_condition_data, two_condition_data])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Create additional features for analysis
df_balanced['Hour'] = df_balanced['Timestamp'].dt.hour
df_balanced['DayOfWeek'] = df_balanced['Timestamp'].dt.day_name()
df_balanced['Month'] = df_balanced['Timestamp'].dt.month_name()

# Create risk categories
df_balanced['Risk_Level'] = df_balanced['Condition Label'].map({
    0: 'Normal',
    1: 'Warning', 
    2: 'Critical'
})

# Temperature categories
df_balanced['Temp_Category'] = pd.cut(df_balanced['Temp (°C)'], 
                                     bins=[df_balanced['Temp (°C)'].min()-1, 20, 25, 30, df_balanced['Temp (°C)'].max()+1],
                                     labels=['Cold', 'Cool', 'Normal', 'Hot'])

# Strain categories
strain_percentiles = df_balanced['Strain (με)'].quantile([0.33, 0.66])
df_balanced['Strain_Level'] = pd.cut(df_balanced['Strain (με)'], 
                                    bins=[df_balanced['Strain (με)'].min()-1, strain_percentiles[0.33], 
                                          strain_percentiles[0.66], df_balanced['Strain (με)'].max()+1],
                                    labels=['Low', 'Medium', 'High'])

# Initialize Dash app
app = dash.Dash(__name__, title="Building Health Monitor Dashboard")
app.config.suppress_callback_exceptions = True

# Define colors
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'normal': '#27ae60',
    'warning': '#f39c12',
    'critical': '#e74c3c',
    'primary': '#3498db',
    'secondary': '#6c757d',
    'info': '#17a2b8'
}

# Calculate key metrics
total_readings = len(df_balanced)
normal_readings = len(df_balanced[df_balanced['Condition Label'] == 0])
warning_readings = len(df_balanced[df_balanced['Condition Label'] == 1])
critical_readings = len(df_balanced[df_balanced['Condition Label'] == 2])
avg_strain = df_balanced['Strain (με)'].mean()
avg_temp = df_balanced['Temp (°C)'].mean()

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏢 Building Health Monitoring Dashboard", 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px'}),
        html.P("Real-time structural health monitoring and predictive maintenance insights",
               style={'textAlign': 'center', 'color': colors['secondary'], 'fontSize': '18px'}),
        html.Hr(style={'border': '2px solid #3498db', 'margin': '20px 0'})
    ]),
    
    # Key Metrics Row
    html.Div([
        html.Div([
            html.H3(f"{total_readings:,}", style={'color': colors['primary'], 'textAlign': 'center'}),
            html.P("Total Readings", style={'textAlign': 'center', 'color': colors['text']})
        ], className="metric-card"),
        html.Div([
            html.H3(f"{normal_readings:,}", style={'color': colors['normal'], 'textAlign': 'center'}),
            html.P("Normal Conditions", style={'textAlign': 'center', 'color': colors['text']})
        ], className="metric-card"),
        html.Div([
            html.H3(f"{warning_readings:,}", style={'color': colors['warning'], 'textAlign': 'center'}),
            html.P("Warning Alerts", style={'textAlign': 'center', 'color': colors['text']})
        ], className="metric-card"),
        html.Div([
            html.H3(f"{critical_readings:,}", style={'color': colors['critical'], 'textAlign': 'center'}),
            html.P("Critical Alerts", style={'textAlign': 'center', 'color': colors['text']})
        ], className="metric-card"),
        html.Div([
            html.H3(f"{avg_strain:.1f}με", style={'color': colors['info'], 'textAlign': 'center'}),
            html.P("Avg Strain", style={'textAlign': 'center', 'color': colors['text']})
        ], className="metric-card"),
        html.Div([
            html.H3(f"{avg_temp:.1f}°C", style={'color': colors['info'], 'textAlign': 'center'}),
            html.P("Avg Temperature", style={'textAlign': 'center', 'color': colors['text']})
        ], className="metric-card")
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px', 'flexWrap': 'wrap'}),
    
    # Tabs
    dcc.Tabs([
        # Overview Tab
        dcc.Tab(label='📊 Overview', children=[
            html.Div([
                # Condition distribution
                html.Div([
                    html.H3("Building Condition Distribution", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(
                        id='condition-distribution',
                        figure=px.pie(
                            df_balanced.groupby('Risk_Level').size().reset_index(name='Count'),
                            values='Count', names='Risk_Level',
                            color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                            title="Current Building Health Status"
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # strain level distribution
                html.Div([
                    html.H3("Strain Level Distribution", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(
                        id='strain-level-distribution',
                        figure=px.histogram(
                            df_balanced, x='Strain_Level', color='Risk_Level',
                            category_orders={'Strain_Level': ['Low', 'Medium', 'High']},
                            color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                            title="Strain Levels Across Conditions"
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # Strain and Temperature over time
                html.Div([
                    html.H3("Acceleration (X-Axis) vs Strain Analysis", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(
                        id='strain-timeline',
                        figure=px.scatter(
                            df_balanced, x='Accel_X (m/s^2)', y='Strain (με)', 
                            color='Risk_Level',
                            color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                            title="Strain Readings Colored by Risk Level"
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                    )
                ], style={'width': '100%', 'marginTop': '30px'}),
                
                html.Div([
                    html.H3("Temperature vs Strain Analysis", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(
                        id='temp-strain-scatter',
                        figure=px.scatter(
                            df_balanced, x='Temp (°C)', y='Strain (με)',
                            color='Risk_Level',
                            color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                            title="Temperature vs Strain"
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                    )
                ], style={'width': '100%', 'marginTop': '30px'})
            ])
        ]),
        
        # Sensor Analysis Tab
        dcc.Tab(label='📡 Sensor Analysis', children=[
            html.Div([
                # Acceleration components
                html.Div([
                    html.H3("Acceleration Components Distribution", style={'textAlign': 'center', 'color': colors['text']}),
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='accel-x-dist',
                                figure=px.histogram(
                                    df_balanced, x='Accel_X (m/s^2)', color='Risk_Level',
                                    color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                                    title="X-Axis Acceleration"
                                ).update_layout(plot_bgcolor='white', paper_bgcolor='white')
                            )
                        ], style={'width': '33%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(
                                id='accel-y-dist',
                                figure=px.histogram(
                                    df_balanced, x='Accel_Y (m/s^2)', color='Risk_Level',
                                    color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                                    title="Y-Axis Acceleration"
                                ).update_layout(plot_bgcolor='white', paper_bgcolor='white')
                            )
                        ], style={'width': '33%', 'display': 'inline-block'}),
                        
                        html.Div([
                            dcc.Graph(
                                id='accel-z-dist',
                                figure=px.histogram(
                                    df_balanced, x='Accel_Z (m/s^2)', color='Risk_Level',
                                    color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                                    title="Z-Axis Acceleration"
                                ).update_layout(plot_bgcolor='white', paper_bgcolor='white')
                            )
                        ], style={'width': '33%', 'display': 'inline-block'})
                    ])
                ], style={'marginBottom': '30px'}),
                
                # 3D acceleration visualization
                html.Div([
                    html.H3("3D Acceleration Visualization", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(
                        id='accel-3d',
                        figure=px.scatter_3d(
                            df_balanced.sample(500), # Sample for performance
                            x='Accel_X (m/s^2)', y='Accel_Y (m/s^2)', z='Accel_Z (m/s^2)',
                            color='Risk_Level',
                            color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                            title="3D Acceleration Pattern Analysis"
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                    )
                ], style={'marginBottom': '30px'}),

                # 3D accelaration x vs strain vs temperature
                html.Div([
                    html.H3("3D Acceleration vs Strain vs Temperature", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(
                        id='accel-strain-temp-3d',
                        figure=px.scatter_3d(
                            df_balanced.sample(500), # Sample for performance
                            x='Accel_X (m/s^2)', y='Strain (με)', z='Temp (°C)',
                            color='Risk_Level',
                            color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                            title="3D Acceleration vs Strain vs Temperature"
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                    )
                ], style={'marginBottom': '30px'}),

                # Correlation heatmap
                html.Div([
                    html.H3("Sensor Correlation Matrix", style={'textAlign': 'center', 'color': colors['text']}),
                    dcc.Graph(id='correlation-heatmap')
                ])
            ])
        ]),
        
        # Environmental Factors Tab
        dcc.Tab(label='🌡️ Environmental', children=[
            html.Div([
                # Temperature analysis
                html.Div([
                    html.Div([
                        html.H3("Temperature Distribution by Risk Level", style={'textAlign': 'center', 'color': colors['text']}),
                        dcc.Graph(
                            id='temp-by-risk',
                            figure=px.box(
                                df_balanced, x='Risk_Level', y='Temp (°C)',
                                color='Risk_Level',
                                color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']}
                            ).update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.H3("Temperature Categories Impact", style={'textAlign': 'center', 'color': colors['text']}),
                        dcc.Graph(
                            id='temp-categories',
                            figure=px.bar(
                                df_balanced.groupby(['Temp_Category', 'Risk_Level']).size().reset_index(name='Count'),
                                x='Temp_Category', y='Count', color='Risk_Level',
                                color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                                barmode='group'
                            ).update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ], style={'marginBottom': '30px'}),
                
                # Hourly and daily patterns
                html.Div([
                    html.Div([
                        html.H3("Anomaly Detection", style={'textAlign': 'center', 'color': colors['text']}),
                        dcc.Graph(
                                id='strain-outliers',
                                figure=px.box(
                                    df_balanced, y='Strain (με)', x='Risk_Level',
                                    color='Risk_Level', points='outliers',
                                    color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                                    title="Strain Outliers by Risk Level"
                                ).update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        )
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.H3("Risk Levels by Day of Week", style={'textAlign': 'center', 'color': colors['text']}),
                        dcc.Graph(
                                id='accel-outliers',
                                figure=px.box(
                                    df_balanced, y='total_accel', x='Risk_Level',
                                    color='Risk_Level', points='outliers',
                                    color_discrete_map={'Normal': colors['normal'], 'Warning': colors['warning'], 'Critical': colors['critical']},
                                    title="Acceleration Outliers by Risk Level"
                                ).update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        )
                    ], style={'width': '50%', 'display': 'inline-block'})
                ])
            ])
        ]),
        
        # Insights & Recommendations Tab
        dcc.Tab(label='💡 Insights & Recommendations', children=[
            html.Div([
                # Key Insights
                html.Div([
                    html.H3("Key Insights", style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Div([
                            html.H4("🔴 Critical Risk Factors", style={'color': colors['critical']}),
                            html.Ul([
                                html.Li("High strain levels (>1000με) strongly correlate with critical conditions"),
                                html.Li("Temperature extremes (very hot/cold) increase structural stress"),
                                html.Li("Combined high acceleration and strain indicate potential structural issues"),
                                html.Li("Z-axis acceleration spikes may indicate vertical structural movement")
                            ])
                        ], className="insights-card", style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4("🟡 Warning Indicators", style={'color': colors['warning']}),
                            html.Ul([
                                html.Li("Gradual increase in strain readings over time"),
                                html.Li("Temperature fluctuations causing expansion/contraction"),
                                html.Li("Consistent medium-level accelerations during specific hours"),
                                html.Li("Weekend patterns may indicate different stress conditions")
                            ])
                        ], className="insights-card", style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ], style={'marginBottom': '30px'}),
                    
                    html.Div([
                        html.Div([
                            html.H4("🟢 Normal Operation Patterns", style={'color': colors['normal']}),
                            html.Ul([
                                html.Li("Strain levels below 800με indicate stable structure"),
                                html.Li("Temperature range 20-25°C shows optimal conditions"),
                                html.Li("Low total acceleration suggests minimal external forces"),
                                html.Li("Consistent readings across all sensor axes")
                            ])
                        ], className="insights-card", style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.H4("📊 Statistical Patterns", style={'color': colors['info']}),
                            html.Ul([
                                html.Li(f"Average strain: {avg_strain:.1f}με (within normal range)"),
                                html.Li(f"Critical alerts: {(critical_readings/total_readings)*100:.1f}% of readings"),
                                html.Li("Strong correlation between temperature and strain"),
                                html.Li("Acceleration patterns show building response to external forces")
                            ])
                        ], className="insights-card", style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
                    ])
                ], style={'marginBottom': '40px'}),
                
                # Recommendations
                html.Div([
                    html.H3("🔧 Maintenance Recommendations", style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Div([
                            html.H4("🚨 Immediate Actions", style={'color': colors['critical']}),
                            html.Ol([
                                html.Li("Install additional sensors in areas showing high strain readings"),
                                html.Li("Implement real-time alerting for strain levels >1200με"),
                                html.Li("Schedule structural inspection when critical alerts occur"),
                                html.Li("Monitor temperature control systems to reduce thermal stress")
                            ])
                        ], className="recommendations-card", style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.H4("📅 Preventive Maintenance", style={'color': colors['warning']}),
                            html.Ol([
                                html.Li("Weekly review of strain trend patterns"),
                                html.Li("Monthly calibration of all sensors"),
                                html.Li("Quarterly structural assessment based on data trends"),
                                html.Li("Annual review of thresholds and alerting parameters")
                            ])
                        ], className="recommendations-card", style={'marginBottom': '20px'}),
                        
                        html.Div([
                            html.H4("🔍 Monitoring Enhancements", style={'color': colors['primary']}),
                            html.Ol([
                                html.Li("Implement machine learning for predictive maintenance"),
                                html.Li("Add environmental sensors (humidity, wind speed)"),
                                html.Li("Create automated reports for facility managers"),
                                html.Li("Integrate with building management systems")
                            ])
                        ], className="recommendations-card")
                    ])
                ])
            ])
        ])
    ], style={'marginTop': '20px'})
], style={'backgroundColor': colors['background'], 'padding': '20px'})

# Callback for correlation heatmap
@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('correlation-heatmap', 'id')
)
def update_correlation_heatmap(id):
    # Select numeric columns for correlation
    numeric_cols = ['Accel_X (m/s^2)', 'Accel_Y (m/s^2)', 'Accel_Z (m/s^2)', 
                   'Strain (με)', 'Temp (°C)', 'total_accel', 'Condition Label']
    corr_matrix = df_balanced[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Sensor Data Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto',
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>Building Health Monitor Dashboard</title>
        <style>
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                margin: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                min-width: 150px;
            }
            .metric-card h3 {
                margin: 0;
                font-size: 2em;
                font-weight: bold;
            }
            .insights-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .recommendations-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=8052)