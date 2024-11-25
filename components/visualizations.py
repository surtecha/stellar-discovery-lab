import plotly.graph_objects as go
import matplotlib.pyplot as plt

def create_animated_area_chart(x_values, y_values, user_value, title, x_title, y_title, log_scale=False):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            fill='tozeroy',
            name='Dataset',
            line=dict(color='#4FADFF', width=2),
            fillcolor='rgba(79, 173, 255, 0.3)',
            mode='lines',
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[user_value],
            y=[max(y_values)],  # Always at max height
            mode='markers',
            name='Your Input',
            marker=dict(
                color='#E24A33',
                size=12,
                symbol='diamond'
            ),
        )
    )
    
    frames = []
    n_points = 50
    
    for i in range(n_points + 1):
        frame_y = [y * (i / n_points) for y in y_values]
        
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_values,
                        y=frame_y,
                        fill='tozeroy',
                        name='Dataset',
                        line=dict(color='#4FADFF', width=2),
                        fillcolor='rgba(79, 173, 255, 0.3)',
                        mode='lines',
                    ),
                    go.Scatter(
                        x=[user_value],
                        y=[max(y_values)],  # Keep diamond at final position
                        mode='markers',
                        name='Your Input',
                        marker=dict(
                            color='#E24A33',
                            size=12,  # Constant size
                            symbol='diamond'
                        ),
                    )
                ],
                name=f'frame{i}'
            )
        )
    
    fig.frames = frames
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template='plotly_dark',
        showlegend=True,
        height=400,
        xaxis=dict(
            type='log' if log_scale else 'linear',
            gridwidth=1,
            showgrid=True,
        ),
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1.15,
                x=0.15,
                xanchor='right',
                yanchor='top',
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=40, redraw=True),
                            fromcurrent=True,
                            mode='immediate',
                            transition=dict(duration=10)
                        )]
                    )
                ]
            )
        ],
        sliders=[],
        margin=dict(t=100)
    )
    
    if log_scale:
        fig.update_xaxes(
            ticktext=['0.1', '0.2', '0.5', '1', '2', '5', '10', '20', '50', '100'],
            tickvals=[0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100],
        )
    
    return fig

def create_probability_chart(prediction_prob):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    probabilities = [prediction_prob[0][0], prediction_prob[0][1]]
    labels = ['CONFIRMED', 'FALSE POSITIVE']
    colors = ['#4FADFF', '#E24A33']
    
    bars = ax.bar(labels, probabilities, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title('Prediction Probabilities', color='white', pad=10, fontsize=16)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height/2.,
                f'{height:.1%}',
                ha='center', va='center', color='white', fontsize=12, weight='bold')
    
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('white')
    
    plt.xticks(color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    
    return fig

def create_donut_chart(values, labels, title, colors=['#4FADFF', '#E24A33']):
    fig = go.Figure(data=[go.Pie(
        values=values,
        labels=labels,
        hole=.3,
        textinfo='percent',
        marker_colors=colors,
        textfont=dict(size=16, color='white'),
        textposition='inside',
        hoverinfo='label+percent+value',
        hoverlabel=dict(font=dict(size=14, color='white')),
        pull=[0, 0.1]
    )])
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        height=400
    )
    
    return fig