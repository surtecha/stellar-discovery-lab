import streamlit as st
import streamlit.components.v1 as components

def load_threejs_simulation(koi_count, koi_prad):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Exoplanet Transit Simulation</title>
        <style>
            #simulation-container {{
                width: 100%;
                height: 800px;
                position: relative;
                display: flex;
                background: #000;
            }}
            #threejs-container {{
                width: 50%;
                height: 100%;
                position: relative;
            }}
            #graph-container {{
                width: 50%;
                height: 100%;
            }}
            #controls {{
                position: absolute;
                top: 10px;
                left: 10px;
                z-index: 100;
                background: rgba(0,0,0,0.7);
                padding: 10px;
                border-radius: 5px;
                color: white;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script async src="https://unpkg.com/es-module-shims@1.3.6/dist/es-module-shims.js"></script>
        <script type="importmap">
            {{
                "imports": {{
                    "three": "https://unpkg.com/three@0.159.0/build/three.module.js",
                    "three/addons/": "https://unpkg.com/three@0.159.0/examples/jsm/"
                }}
            }}
        </script>
        <script type="module">
            import * as THREE from 'three';
            import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

            let scene, camera, renderer, controls;
            let planets = [];
            let isPlaying = true;
            let observerPlane;
            let transitData = {{ x: [], y: [] }};
            let currentTime = 0;
            
            const planetColors = [
                0xff4500, 0x0080ff, 0x32cd32,
                0x9370db, 0xff1493, 0xff8c00, 0x00ffff
            ];

            const koi_count = {koi_count};
            const koi_prad = {koi_prad};

            function initScene() {{
                scene = new THREE.Scene();
                scene.background = new THREE.Color(0x000000);

                const container = document.getElementById('threejs-container');
                camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
                camera.position.set(20, 20, 20);

                renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                container.appendChild(renderer.domElement);

                controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;

                // Lights
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                scene.add(ambientLight);
                
                const pointLight = new THREE.PointLight(0xffffff, 2);
                scene.add(pointLight);

                // Star
                const starGeometry = new THREE.SphereGeometry(2, 32, 32);
                const starMaterial = new THREE.MeshPhongMaterial({{
                    emissive: 0xffff00,
                    emissiveIntensity: 1
                }});
                const star = new THREE.Mesh(starGeometry, starMaterial);
                scene.add(star);

                // Observer line
                const maxRadius = (koi_count + 1) * 4;
                const lineGeometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, -maxRadius/2, 0),
                    new THREE.Vector3(0, maxRadius/2, 0)
                ]);
                const lineMaterial = new THREE.LineBasicMaterial({{
                    color: 0xffffff,
                    transparent: true,
                    opacity: 0.5
                }});
                observerPlane = new THREE.Line(lineGeometry, lineMaterial);
                observerPlane.position.x = maxRadius * 0.8;
                scene.add(observerPlane);

                // Create planets with tilted orbits
                for (let i = 0; i < koi_count; i++) {{
                    const radius = (i + 1) * 4;
                    const planetSize = 0.5 + (koi_prad * 0.2);
                    
                    // Planet
                    const planetGeometry = new THREE.SphereGeometry(planetSize, 32, 32);
                    const planetMaterial = new THREE.MeshPhongMaterial({{
                        color: planetColors[i % planetColors.length]
                    }});
                    const planet = new THREE.Mesh(planetGeometry, planetMaterial);
                    
                    // Create orbit points
                    const orbitPoints = [];
                    const segments = 64;
                    for (let j = 0; j <= segments; j++) {{
                        const theta = (j / segments) * Math.PI * 2;
                        const x = radius * Math.cos(theta);
                        const y = radius * Math.sin(theta);
                        orbitPoints.push(new THREE.Vector3(x, y, 0));
                    }}
                    
                    const orbitGeometry = new THREE.BufferGeometry().setFromPoints(orbitPoints);
                    const orbitMaterial = new THREE.LineBasicMaterial({{
                        color: 0x666666,
                        opacity: 0.5,
                        transparent: true
                    }});
                    const orbit = new THREE.Line(orbitGeometry, orbitMaterial);
                    
                    // Add random tilt to orbit
                    const tiltAngle = (Math.random() - 0.5) * Math.PI / 6;
                    orbit.rotation.x = tiltAngle;
                    
                    scene.add(orbit);
                    scene.add(planet);
                    
                    planets.push({{
                        mesh: planet,
                        orbit: orbit,
                        radius: radius,
                        speed: 0.01 / Math.sqrt(radius),
                        angle: Math.random() * Math.PI * 2,
                        tilt: tiltAngle,
                        size: planetSize
                    }});
                }}
            }}

            function initGraph() {{
                const layout = {{
                    title: {{
                        text: 'Transit Light Curve',
                        font: {{ color: '#fff' }}
                    }},
                    xaxis: {{ 
                        title: 'Time',
                        gridcolor: '#333',
                        color: '#fff'
                    }},
                    yaxis: {{ 
                        title: 'Relative Brightness',
                        range: [0.95, 1.01],
                        gridcolor: '#333',
                        color: '#fff'
                    }},
                    plot_bgcolor: '#111',
                    paper_bgcolor: '#111',
                    font: {{ color: '#fff' }},
                    showlegend: false,
                    margin: {{ t: 50, r: 20, b: 50, l: 60 }}
                }};

                Plotly.newPlot('graph-container', [{{
                    x: [],
                    y: [],
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#9370db', width: 2 }}
                }}], layout);
            }}

            function calculateTransitDepth() {{
                let totalBrightness = 1.0;
                const observerX = observerPlane.position.x;
                
                planets.forEach(planet => {{
                    // Get planet position in world coordinates
                    const worldPos = new THREE.Vector3();
                    planet.mesh.getWorldPosition(worldPos);
                    
                    // Check if planet is near the observer line
                    if (Math.abs(worldPos.x - observerX) < 0.5) {{
                        // Calculate normalized position along observer line
                        const normalizedY = worldPos.y / (observerPlane.geometry.attributes.position.array[4]);
                        if (Math.abs(normalizedY) < 1) {{
                            // Calculate transit depth based on planet size and position
                            const transitDepth = (planet.size * planet.size * 0.02) * 
                                               (1 - Math.pow(normalizedY, 2));
                            totalBrightness -= transitDepth;
                        }}
                    }}
                }});
                
                return Math.max(totalBrightness, 0.95);
            }}

            function updateGraph() {{
                const brightness = calculateTransitDepth();
                transitData.x.push(currentTime);
                transitData.y.push(brightness);

                if (transitData.x.length > 500) {{
                    transitData.x.shift();
                    transitData.y.shift();
                }}

                Plotly.update('graph-container', {{
                    x: [transitData.x],
                    y: [transitData.y]
                }});
            }}

            function animate() {{
                requestAnimationFrame(animate);
                
                if (isPlaying) {{
                    currentTime += 0.1;
                    
                    planets.forEach(planet => {{
                        planet.angle += planet.speed;
                        
                        // Calculate position considering orbit tilt
                        const x = planet.radius * Math.cos(planet.angle);
                        const y = planet.radius * Math.sin(planet.angle);
                        
                        // Apply orbit tilt transformation
                        planet.mesh.position.set(
                            x,
                            y * Math.cos(planet.tilt),
                            y * Math.sin(planet.tilt)
                        );
                    }});
                    
                    updateGraph();
                }}
                
                controls.update();
                renderer.render(scene, camera);
            }}

            function onWindowResize() {{
                const container = document.getElementById('threejs-container');
                camera.aspect = container.clientWidth / container.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(container.clientWidth, container.clientHeight);
            }}

            function init() {{
                initScene();
                initGraph();
                
                window.addEventListener('resize', onWindowResize, false);
                document.getElementById('playPause').addEventListener('click', () => {{
                    isPlaying = !isPlaying;
                    document.getElementById('playPause').textContent = isPlaying ? 'Pause' : 'Play';
                }});
                
                animate();
            }}

            init();
        </script>
    </head>
    <body>
        <div id="simulation-container">
            <div id="threejs-container"></div>
            <div id="graph-container"></div>
            <div id="controls">
                <button id="playPause">Pause</button>
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(html_content, height=800)