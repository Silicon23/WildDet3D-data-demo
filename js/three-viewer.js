/**
 * Three.js Viewer for Pointcloud and 3D Bounding Box Visualization
 * 
 * Renders PLY pointclouds with RGB colors and 3D bounding boxes as wireframes.
 * 
 * Controls:
 * - Left drag: Rotate
 * - Right drag / Middle drag: Pan
 * - Scroll: Zoom
 * - Double-click: Reset view
 */

class ThreeViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.pointCloud = null;
        this.bboxGroup = null;
        
        // Store initial camera state for reset
        this.initialCameraState = null;
        
        // Model colors
        this.modelColors = {
            'la3d': 0xa855f7,      // Purple - highest priority
            'sam3d': 0x22c55e,
            'algorithm_regression': 0xf97316,
            'algorithm': 0xf97316,
            '3d_mood': 0xef4444,
            'detany3d': 0x3b82f6,
            'default': 0x666666
        };
        
        this.initialized = false;
    }
    
    init() {
        if (this.initialized) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        // Clear container
        this.container.innerHTML = '';
        
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);
        
        // Camera - use a reasonable FOV
        this.camera = new THREE.PerspectiveCamera(50, width / height, 0.001, 1000);
        this.camera.position.set(0, 0, 5);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;
        this.controls.screenSpacePanning = true;
        this.controls.rotateSpeed = 0.8;
        this.controls.panSpeed = 0.8;
        this.controls.zoomSpeed = 1.2;
        
        // Very permissive zoom limits - will be adjusted based on scene
        this.controls.maxDistance = 500;
        this.controls.minDistance = 0.001;
        
        // Mouse controls:
        // Left drag = rotate
        // Right drag / middle drag = pan (translate)
        // Scroll = zoom
        this.controls.mouseButtons = {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.PAN
        };
        
        // Touch controls for trackpad
        this.controls.touches = {
            ONE: THREE.TOUCH.ROTATE,
            TWO: THREE.TOUCH.DOLLY_PAN
        };
        
        // Double-click to reset view
        this.renderer.domElement.addEventListener('dblclick', () => this.resetView());
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Bounding box group
        this.bboxGroup = new THREE.Group();
        this.scene.add(this.bboxGroup);
        
        // Add reset button
        this.addResetButton();
        
        // Handle resize
        this.resizeObserver = new ResizeObserver(() => this.handleResize());
        this.resizeObserver.observe(this.container);
        
        // Animation loop
        this.animate();
        
        this.initialized = true;
    }
    
    addResetButton() {
        const btn = document.createElement('button');
        btn.className = 'viewer-reset-btn';
        btn.innerHTML = '⟲ Reset View';
        btn.title = 'Reset camera view (or double-click)';
        btn.addEventListener('click', () => this.resetView());
        this.container.appendChild(btn);
    }
    
    handleResize() {
        if (!this.renderer || !this.camera) return;
        
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        if (this.controls) {
            this.controls.update();
        }
        
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    async loadPointcloud(plyPath) {
        return new Promise((resolve, reject) => {
            const loader = new THREE.PLYLoader();
            
            loader.load(
                plyPath,
                (geometry) => {
                    // Remove old pointcloud
                    if (this.pointCloud) {
                        this.scene.remove(this.pointCloud);
                        this.pointCloud.geometry.dispose();
                        this.pointCloud.material.dispose();
                    }
                    
                    // Create material with vertex colors
                    const material = new THREE.PointsMaterial({
                        size: 0.012,  // Slightly larger points for visibility
                        vertexColors: geometry.hasAttribute('color'),
                        sizeAttenuation: true
                    });
                    
                    if (!geometry.hasAttribute('color')) {
                        material.color = new THREE.Color(0x6b7280);
                    }
                    
                    // Create points
                    this.pointCloud = new THREE.Points(geometry, material);
                    
                    // Transform from OpenCV camera coords (Y-down, Z-forward) 
                    // to Three.js coords (Y-up, Z-backward)
                    this.pointCloud.rotation.x = Math.PI;  // Rotate 180° around X
                    
                    this.scene.add(this.pointCloud);
                    
                    // Center camera on pointcloud
                    this.centerOnPointcloud();
                    
                    // Save initial state for reset
                    this.saveInitialState();
                    
                    resolve();
                },
                (progress) => {
                    // Progress callback
                },
                (error) => {
                    console.error('Error loading PLY:', error);
                    reject(error);
                }
            );
        });
    }
    
    centerOnPointcloud() {
        if (!this.pointCloud) return;
        
        const geometry = this.pointCloud.geometry;
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();
        
        // Get center in local coordinates
        const localCenter = new THREE.Vector3();
        geometry.boundingBox.getCenter(localCenter);
        
        // Transform center to world coordinates (apply the rotation)
        // Since we rotate 180° around X, y and z are negated
        const worldCenter = new THREE.Vector3(
            localCenter.x,
            -localCenter.y,
            -localCenter.z
        );
        
        const radius = geometry.boundingSphere.radius;
        
        // Position camera to view the entire pointcloud
        // Camera looks along -Z in Three.js, so position it at +Z from center
        const distance = radius * 2.5;  // A bit further than the radius for full view
        
        this.camera.position.set(
            worldCenter.x,
            worldCenter.y,
            worldCenter.z + distance
        );
        
        // Look at the center
        this.controls.target.copy(worldCenter);
        this.controls.update();
        
        // Adjust zoom limits based on scene size
        this.controls.minDistance = radius * 0.1;
        this.controls.maxDistance = radius * 20;
    }
    
    saveInitialState() {
        this.initialCameraState = {
            position: this.camera.position.clone(),
            target: this.controls.target.clone()
        };
    }
    
    resetView() {
        if (this.initialCameraState) {
            // Animate to initial position
            this.camera.position.copy(this.initialCameraState.position);
            this.controls.target.copy(this.initialCameraState.target);
            this.controls.update();
        } else if (this.pointCloud) {
            // Fallback: recenter on pointcloud
            this.centerOnPointcloud();
        }
    }
    
    clearBboxes() {
        while (this.bboxGroup.children.length > 0) {
            const child = this.bboxGroup.children[0];
            this.bboxGroup.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        }
    }
    
    addBbox(box3d, model = 'default', label = null) {
        const corners = this.box3dToCorners(box3d);
        const color = this.modelColors[model] || this.modelColors.default;
        
        // Transform corners from OpenCV coords to Three.js coords
        // (negate Y and Z to rotate 180° around X axis)
        const transformedCorners = corners.map(([x, y, z]) => [x, -y, -z]);
        
        // Create edges
        const edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], // bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], // top face
            [0, 4], [1, 5], [2, 6], [3, 7]  // vertical edges
        ];
        
        const material = new THREE.LineBasicMaterial({ 
            color: color,
            linewidth: 2
        });
        
        for (const [i, j] of edges) {
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(transformedCorners[i][0], transformedCorners[i][1], transformedCorners[i][2]),
                new THREE.Vector3(transformedCorners[j][0], transformedCorners[j][1], transformedCorners[j][2])
            ]);
            const line = new THREE.Line(geometry, material);
            this.bboxGroup.add(line);
        }
    }
    
    box3dToCorners(box3d) {
        /**
         * Convert 10D box format [x, y, z, w, h, l, qw, qx, qy, qz] to 8 corners.
         * Box is in OpenCV camera coordinates (X-right, Y-down, Z-forward).
         */
        const [cx, cy, cz, w, h, l, qw, qx, qy, qz] = box3d;
        
        // Rotation matrix from quaternion
        const R = this.quaternionToRotationMatrix(qw, qx, qy, qz);
        
        // Half dimensions
        const hw = w / 2;
        const hh = h / 2;
        const hl = l / 2;
        
        // Local corners (before rotation)
        const localCorners = [
            [-hw, -hh, -hl],
            [ hw, -hh, -hl],
            [ hw,  hh, -hl],
            [-hw,  hh, -hl],
            [-hw, -hh,  hl],
            [ hw, -hh,  hl],
            [ hw,  hh,  hl],
            [-hw,  hh,  hl]
        ];
        
        // Transform to world coordinates
        const corners = localCorners.map(([lx, ly, lz]) => {
            // Rotate
            const rx = R[0][0]*lx + R[0][1]*ly + R[0][2]*lz;
            const ry = R[1][0]*lx + R[1][1]*ly + R[1][2]*lz;
            const rz = R[2][0]*lx + R[2][1]*ly + R[2][2]*lz;
            
            // Translate
            return [rx + cx, ry + cy, rz + cz];
        });
        
        return corners;
    }
    
    quaternionToRotationMatrix(qw, qx, qy, qz) {
        /**
         * Convert quaternion to 3x3 rotation matrix.
         */
        const R = [
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ];
        return R;
    }
    
    setBoxes(boxes) {
        /**
         * Set multiple bounding boxes at once.
         * 
         * @param {Array} boxes - Array of {box3d, model, label} objects
         */
        this.clearBboxes();
        
        for (const boxData of boxes) {
            this.addBbox(
                boxData.box3d,
                boxData.model || 'default',
                boxData.label || null
            );
        }
    }
    
    dispose() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        
        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }
        
        this.clearBboxes();
        
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        this.container.innerHTML = '';
        this.initialized = false;
    }
}

// Export for use in other modules
window.ThreeViewer = ThreeViewer;
