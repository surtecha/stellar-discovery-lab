import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
def create_transit_simulation(koi_count, koi_prad):
    # Planet colors
    planet_colors = [
        'rgb(255, 69, 0)',   # Red-Orange
        'rgb(0, 128, 255)',  # Bright Blue
        'rgb(50, 205, 50)',  # Lime Green
        'rgb(147, 112, 219)', # Purple
        'rgb(255, 20, 147)',  # Deep Pink
        'rgb(255, 140, 0)',   # Dark Orange
        'rgb(0, 255, 255)'    # Cyan
    ]
    # Initialize orbits based on koi_count and koi_prad
    orbits = []
    for i in range(int(koi_count)):
        # Scale orbit radius based on planetary radius
        radius = (i + 1) * 2.0 * (koi_prad / 2.26)  # Normalized to your default value
        color = planet_colors[i % len(planet_colors)]
        
        orbits.append({
            "radius": radius,
            "mass": koi_prad,  # Use planetary radius as proxy for mass
            "color": color,
            "start_angle": np.random.uniform(0, 2 * np.pi),
            "speed_factor": 1,
            "planet_size": 6 + (koi_prad * 2),
            "transit_width": 0.8
        })
    # Create figure with subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]], 
                       horizontal_spacing=0.02, column_widths=[0.5, 0.5])
    # Time setup
    frames_per_revolution = 120
    total_revolutions = 10
    time_steps = np.linspace(0, 2 * np.pi * total_revolutions, 
                           frames_per_revolution * total_revolutions)
    # Add initial empty trace for combined transits
    fig.add_trace(
        go.Scatter(x=[], y=[], mode='lines', name='Combined Transits',
                  line=dict(color='rgb(147, 112, 219)', width=2)),
        row=1, col=1
    )
    # Setup 3D visualization
    max_radius = max(orbit["radius"] for orbit in orbits)
    observer_x = max_radius * 1.2
    rectangle_width = max_radius / 10
    sun_radius = rectangle_width
    # Create sun
    phi = np.linspace(0, 2*np.pi, 50)
    theta = np.linspace(0, np.pi, 25)
    phi, theta = np.meshgrid(phi, theta)
    x_sun = sun_radius * np.sin(theta) * np.cos(phi)
    y_sun = sun_radius * np.sin(theta) * np.sin(phi)
    z_sun = sun_radius * np.cos(theta)
    fig.add_trace(
        go.Surface(
            x=x_sun, y=y_sun, z=z_sun,
            colorscale=[[0, '#ffff00'], [1, '#ffd700']],
            showscale=False,
            lighting=dict(ambient=0.8, diffuse=1, fresnel=2, specular=1, roughness=0.5),
            name="Sun"
        ),
        row=1, col=2
    )
    # Add orbit circles
    for i, orbit in enumerate(orbits):
        circle_points = np.linspace(0, 2 * np.pi, 100)
        x_orbit = orbit["radius"] * np.cos(circle_points)
        y_orbit = orbit["radius"] * np.sin(circle_points)
        z_orbit = np.zeros_like(circle_points)
        
        fig.add_trace(
            go.Scatter3d(
                x=x_orbit, y=y_orbit, z=z_orbit,
                mode="lines", line=dict(color=orbit["color"], dash="dot", width=2),
                name=f"Planet {i + 1} Orbit",
                showlegend=False
            ),
            row=1, col=2
        )
    # Add observer rectangle
    x_coords = [0, 0, observer_x, observer_x, 0]
    y_coords = [-rectangle_width, rectangle_width, rectangle_width, -rectangle_width, -rectangle_width]
    z_coords = [0, 0, 0, 0, 0]
    fig.add_trace(
        go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode="lines", line=dict(color="white", width=2),
            name="Observer",
        ),
        row=1, col=2
    )
    def calculate_transit_depth(x, y, radius, transit_width, observer_x):
        if x < 0 or x > observer_x:
            return 0
        scaled_y = y / rectangle_width
        if abs(scaled_y) >= 1:
            return 0
        return (1.0 - scaled_y**2) * 0.3
    # Calculate transit depths
    all_transit_depths = np.zeros(len(time_steps))
    for i, t in enumerate(time_steps):
        total_depth = 0
        for orbit in orbits:
            angle = orbit["start_angle"] + t * orbit["speed_factor"]
            x = orbit["radius"] * np.cos(angle)
            y = orbit["radius"] * np.sin(angle)
            transit_contribution = calculate_transit_depth(
                x, y, orbit["radius"], orbit["transit_width"], observer_x
            ) * orbit["mass"]
            total_depth += transit_contribution
        all_transit_depths[i] = -total_depth if total_depth > 0 else 0
    # Create animation frames
    frames = []
    for i, t in enumerate(time_steps):
        frame_data = []
        
        frame_data.append(
            go.Scatter(
                x=time_steps[:i+1],
                y=all_transit_depths[:i+1],
                mode='lines',
                line=dict(color='rgb(147, 112, 219)', width=2),
                name='Transit Detection'
            )
        )
        
        frame_data.append(
            go.Surface(
                x=x_sun, y=y_sun, z=z_sun,
                colorscale=[[0, '#ffff00'], [1, '#ffd700']],
                showscale=False,
                lighting=dict(
                    ambient=0.8, diffuse=1, fresnel=2, specular=1, roughness=0.5
                ),
                name="Sun"
            )
        )
        
        for j, orbit in enumerate(orbits):
            angle = orbit["start_angle"] + t * orbit["speed_factor"]
            x_planet = orbit["radius"] * np.cos(angle)
            y_planet = orbit["radius"] * np.sin(angle)
            z_planet = 0
            
            frame_data.append(
                go.Scatter3d(
                    x=[x_planet], y=[y_planet], z=[z_planet],
                    mode="markers",
                    marker=dict(
                        size=orbit["planet_size"], 
                        color=orbit["color"],
                        symbol='circle',
                        line=dict(color='white', width=1)
                    ),
                    name=f"Planet {j + 1}",
                    showlegend=True
                )
            )
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    # Update layout
    max_depth = abs(min(all_transit_depths))
    fig.update_layout(
        height=600,
        title="Planet Transit Simulation",
        xaxis_title="Time",
        yaxis_title="Transit Detection (Combined Mass)",
        yaxis=dict(
            range=[-max_depth * 1.2, max_depth * 0.2],
            zeroline=True,
            zerolinecolor='gray',
            gridcolor='lightgray'
        ),
        xaxis=dict(
            range=[0, max(time_steps)],
            zeroline=True,
            zerolinecolor='gray',
            gridcolor='lightgray'
        ),
        scene=dict(
            xaxis=dict(range=[-1.2 * max_radius, 1.2 * max_radius],
                      showticklabels=False, title='X'),
            yaxis=dict(range=[-1.2 * max_radius, 1.2 * max_radius],
                      showticklabels=False, title='Y'),
            zaxis=dict(range=[-1.2 * max_radius, 1.2 * max_radius],
                      showticklabels=False, title='Z'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": True},
                                  "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        showlegend=True,
        template="plotly_dark"
    )
    fig.frames = frames
    return fig