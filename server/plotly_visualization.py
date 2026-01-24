import plotly.graph_objects as go
import pandas as pd

# Function to plot predicted RUL and sensor trends for last 50 cycles
# df: DataFrame with columns including 'time_cycles', 'predicted_rul', and sensor columns
# sensor_cols: list of sensor column names to plot
# unit_nr: engine unit number to filter

def plot_predicted_rul_and_sensors(df, sensor_cols, unit_nr):
    # Use the provided dataframe directly (assumed to be already filtered for the unit)
    df_unit = df.copy()

    # Select last 50 cycles
    df_last = df_unit[df_unit['time_cycles'] > df_unit['time_cycles'].max() - 50]

    # Calculate dynamic range for sensors for secondary axis
    # Flatten the values of all sensor columns to find global min/max
    sensor_values = df_last[sensor_cols].values.flatten()
    y2_min = float(min(sensor_values) - 5)
    y2_max = float(max(sensor_values) + 5)

    fig = go.Figure()

    # Plot predicted RUL
    fig.add_trace(go.Scatter(
        x=df_last['time_cycles'].tolist(),
        y=df_last['predicted_rul'].tolist(),
        mode='lines+markers',
        name='Predicted RUL',
        line=dict(color='red', width=3)
    ))

    # Plot sensor trends (average of selected sensors)
    for sensor in sensor_cols:
        fig.add_trace(go.Scatter(
            x=df_last['time_cycles'].tolist(),
            y=df_last[sensor].tolist(),
            mode='lines',
            name=f'Sensor {sensor}',
            yaxis='y2',
            line=dict(width=1)
        ))

    fig.update_layout(
        title=f'Predicted RUL and Sensor Trends for Engine {unit_nr}',
        xaxis_title='Time Cycles',
        yaxis_title='Predicted RUL (Cycles)',
        template='plotly_white',
        autosize=True,
        margin=dict(l=40, r=60, t=60, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.25,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="white",
            borderwidth=1,
            font=dict(color="white")
        ),
        hovermode="x unified",
        # Add vertical line for 'Current State'
        shapes=[dict(
            type="line",
            xref="x",
            yref="paper",
            x0=df_last['time_cycles'].max(),
            y0=0,
            x1=df_last['time_cycles'].max(),
            y1=1,
            line=dict(color="white", width=1, dash="dash")
        )],
        annotations=[dict(
            x=df_last['time_cycles'].max(),
            y=1.05,
            xref="x",
            yref="paper",
            text="Current State",
            showarrow=False,
            font=dict(color="white", size=12),
            bgcolor="rgba(0,0,0,0.5)"
        )],
        yaxis=dict(
            title=dict(text="Predicted RUL", font=dict(color="red")),
            tickfont=dict(color="red"),
            autorange=True
        ),
        yaxis2=dict(
            title=dict(text="Sensor Value", font=dict(color="rgb(148, 163, 184)")),
            tickfont=dict(color="rgb(148, 163, 184)"),
            anchor="x",
            overlaying="y",
            side="right",
            range=[y2_min, y2_max]
        )
    )

    return fig

