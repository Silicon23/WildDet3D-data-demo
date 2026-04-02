/**
 * Three.js Viewer for Pointcloud and 3D Bounding Box Visualization
 *
 * Renders PLY pointclouds with RGB colors and 3D bounding boxes as wireframes.
 * Supports image plane display and turntable auto-rotation.
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

        // Oscillating tilt state
        this._tiltActive = false;
        this._tiltTime = 0;

        // Camera intrinsics for matching image perspective
        this._intrinsics = null;
        this._imageSize = null;

        // Default box color
        this.defaultColor = 0x666666;

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
        this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
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

        // Lighting — full ambient so textured mesh shows original image colors
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
        this.scene.add(ambientLight);

        // Bounding box group
        this.bboxGroup = new THREE.Group();
        this.scene.add(this.bboxGroup);

        // Add buttons
        this.addResetButton();
        this.addAutoRotateButton();

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
        btn.innerHTML = '&#x27F2; Reset View';
        btn.title = 'Reset camera view (or double-click)';
        btn.addEventListener('click', () => this.resetView());
        this.container.appendChild(btn);
    }

    addAutoRotateButton() {
        const btn = document.createElement('button');
        btn.className = 'viewer-reset-btn';
        btn.style.right = 'auto';
        btn.style.left = '8px';
        btn.innerHTML = 'Auto View';
        btn.title = 'Toggle gentle camera sway';
        this._tiltBtn = btn;
        btn.addEventListener('click', () => {
            this._tiltActive = !this._tiltActive;
            if (this._tiltActive) {
                this._tiltTime = 0;
                this._tiltOrigin = {
                    position: this.camera.position.clone(),
                    target: this.controls.target.clone()
                };
            } else {
                // Return to original position
                if (this._tiltOrigin) {
                    this.camera.position.copy(this._tiltOrigin.position);
                    this.controls.target.copy(this._tiltOrigin.target);
                    this.controls.update();
                }
            }
            btn.style.background = this._tiltActive
                ? 'rgba(59, 130, 246, 0.85)' : 'rgba(0, 0, 0, 0.7)';
        });
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

        // Oscillating tilt: gentle sway to reveal depth
        if (this._tiltActive && this._tiltOrigin) {
            this._tiltTime += 0.004;
            const angle = Math.sin(this._tiltTime) * 0.25; // ~15° max swing

            const origin = this._tiltOrigin;
            const offset = new THREE.Vector3().subVectors(origin.position, origin.target);
            const radius = offset.length();

            // Rotate offset around Y axis (horizontal tilt)
            const baseAngle = Math.atan2(offset.x, offset.z);
            const newAngle = baseAngle + angle;

            this.camera.position.set(
                origin.target.x + radius * Math.sin(newAngle),
                origin.position.y + radius * 0.04 * Math.sin(this._tiltTime * 0.7),
                origin.target.z + radius * Math.cos(newAngle)
            );
            this.controls.target.copy(origin.target);
        }

        if (this.controls) {
            this.controls.update();
        }

        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }

    // ========================================================================
    // Pointcloud
    // ========================================================================

    async loadPointcloud(plyPath) {
        // Support both .glb (mesh) and .ply (point cloud) formats
        const isGLB = plyPath.endsWith('.glb');

        if (isGLB) {
            return this._loadGLB(plyPath);
        } else {
            return this._loadPLY(plyPath);
        }
    }

    async _loadGLB(glbPath) {
        return new Promise((resolve, reject) => {
            const loader = new THREE.GLTFLoader();
            loader.load(
                glbPath,
                (gltf) => {
                    // Remove old scene mesh
                    if (this.sceneMesh) {
                        this.scene.remove(this.sceneMesh);
                    }

                    this.sceneMesh = gltf.scene;

                    // Convert to unlit material with vertex colors
                    this.sceneMesh.traverse((child) => {
                        if (child.isMesh && child.material) {
                            const oldMat = child.material;
                            child.material = new THREE.MeshBasicMaterial({
                                vertexColors: true,
                                side: THREE.DoubleSide
                            });
                            oldMat.dispose();
                        }
                    });

                    this.scene.add(this.sceneMesh);

                    // Compute bounding box for camera placement
                    const box = new THREE.Box3().setFromObject(this.sceneMesh);
                    this._sceneBounds = { box };

                    this.centerOnScene();
                    this.saveInitialState();
                    this._startAutoView();
                    resolve();
                },
                undefined,
                (error) => {
                    console.error('Error loading GLB:', error);
                    // Fallback: try loading .ply with same name
                    const plyFallback = glbPath.replace('.glb', '.ply');
                    this._loadPLY(plyFallback).then(resolve).catch(reject);
                }
            );
        });
    }

    async _loadPLY(plyPath) {
        return new Promise((resolve, reject) => {
            const loader = new THREE.PLYLoader();

            loader.load(
                plyPath,
                (geometry) => {
                    if (this.pointCloud) {
                        this.scene.remove(this.pointCloud);
                        this.pointCloud.geometry.dispose();
                        this.pointCloud.material.dispose();
                    }

                    const material = new THREE.PointsMaterial({
                        size: 0.012,
                        vertexColors: geometry.hasAttribute('color'),
                        sizeAttenuation: true
                    });

                    if (!geometry.hasAttribute('color')) {
                        material.color = new THREE.Color(0x6b7280);
                    }

                    this.pointCloud = new THREE.Points(geometry, material);
                    this.pointCloud.rotation.x = Math.PI;
                    this.scene.add(this.pointCloud);

                    this.centerOnPointcloud();
                    this.saveInitialState();
                    this._startAutoView();
                    resolve();
                },
                undefined,
                (error) => {
                    console.error('Error loading PLY:', error);
                    reject(error);
                }
            );
        });
    }

    /**
     * Set camera intrinsics so the 3D view matches the original image perspective.
     * @param {Array} intrinsics - 3x3 matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
     * @param {Array} imageSize - [width, height]
     */
    setCameraIntrinsics(intrinsics, imageSize) {
        this._intrinsics = intrinsics;
        this._imageSize = imageSize;
    }

    /**
     * Position camera at the original camera location using intrinsics.
     * This makes the 3D view match the 2D image framing.
     */
    centerOnScene() {
        if (!this._sceneBounds) return;

        const box = this._sceneBounds.box;
        const center = new THREE.Vector3();
        box.getCenter(center);
        const size = new THREE.Vector3();
        box.getSize(size);
        const radius = size.length() / 2;

        if (this._intrinsics && this._imageSize) {
            // Use intrinsics: place camera at origin, set FOV to match image
            const fy = this._intrinsics[1][1];
            const imgH = this._imageSize[1];
            const vfov = 2 * Math.atan(imgH / (2 * fy)) * (180 / Math.PI);
            this.camera.fov = vfov;
            this.camera.updateProjectionMatrix();

            // Camera at origin (where the real camera was), looking at scene center
            // Mesh is already in Three.js coords (Y flipped, Z flipped in Python)
            this.camera.position.set(0, 0, 0);
            this.controls.target.copy(center);
        } else {
            // Fallback: fit to bounding box
            const distance = radius * 2.5;
            this.camera.position.set(center.x, center.y, center.z + distance);
            this.controls.target.copy(center);
        }

        this.controls.update();
        this.controls.minDistance = 0.01;
        this.controls.maxDistance = radius * 20;
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

    _startAutoView() {
        this._tiltActive = true;
        this._tiltTime = 0;
        this._tiltOrigin = {
            position: this.camera.position.clone(),
            target: this.controls.target.clone()
        };
        if (this._tiltBtn) {
            this._tiltBtn.style.background = 'rgba(59, 130, 246, 0.85)';
        }
    }

    saveInitialState() {
        this.initialCameraState = {
            position: this.camera.position.clone(),
            target: this.controls.target.clone()
        };
    }

    resetView() {
        // Stop tilt on reset
        this._tiltActive = false;
        if (this._tiltBtn) this._tiltBtn.style.background = 'rgba(0, 0, 0, 0.7)';

        if (this.initialCameraState) {
            this.camera.position.copy(this.initialCameraState.position);
            this.controls.target.copy(this.initialCameraState.target);
            this.controls.update();
        } else if (this.pointCloud) {
            this.centerOnPointcloud();
        }
    }

    // ========================================================================
    // Bounding Boxes
    // ========================================================================

    clearBboxes() {
        while (this.bboxGroup.children.length > 0) {
            const child = this.bboxGroup.children[0];
            this.bboxGroup.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        }
    }

    /**
     * Add a 3D bounding box wireframe.
     * @param {Array} box3d - 10D box format
     * @param {number} color - Numeric hex color (e.g. 0xff0000)
     * @param {number|null} annotationIdx - annotation index for click identification
     */
    addBbox(box3d, color = null, annotationIdx = null) {
        const corners = this.box3dToCorners(box3d);
        color = color || this.defaultColor;

        // Transform corners from OpenCV coords to Three.js coords
        // (negate Y and Z to rotate 180° around X axis)
        const transformedCorners = corners.map(([x, y, z]) => [x, -y, -z]);
        const verts = transformedCorners.map(c => new THREE.Vector3(c[0], c[1], c[2]));

        // Create edges as cylinders for thick lines (WebGL Line is always 1px)
        const edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], // bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], // top face
            [0, 4], [1, 5], [2, 6], [3, 7]  // vertical edges
        ];

        const edgeMaterial = new THREE.MeshBasicMaterial({ color: color });
        // Scale tube radius by box depth so edges look consistent on screen
        const boxCenter = new THREE.Vector3(
            (verts[0].x + verts[6].x) / 2,
            (verts[0].y + verts[6].y) / 2,
            (verts[0].z + verts[6].z) / 2
        );
        const depth = boxCenter.distanceTo(this.camera.position);
        const tubeRadius = depth * 0.001;

        for (const [i, j] of edges) {
            const start = verts[i];
            const end = verts[j];
            const dir = new THREE.Vector3().subVectors(end, start);
            const len = dir.length();
            if (len < 1e-6) continue;

            const cylGeom = new THREE.CylinderBufferGeometry(tubeRadius, tubeRadius, len, 6, 1);
            cylGeom.translate(0, len / 2, 0);
            cylGeom.rotateX(Math.PI / 2);

            const mesh = new THREE.Mesh(cylGeom, edgeMaterial);
            mesh.position.copy(start);
            mesh.lookAt(end);
            if (annotationIdx !== null) mesh.userData.annotationIdx = annotationIdx;
            this.bboxGroup.add(mesh);
        }

        // Add semi-transparent faces for visibility
        // 6 faces: bottom(0123), top(4567), front(0154), back(2376), left(0374), right(1265)
        const faceIndices = [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 5, 4], [2, 3, 7, 6],
            [0, 3, 7, 4], [1, 2, 6, 5]
        ];

        const positions = [];
        for (const [a, b, c, d] of faceIndices) {
            // Two triangles per quad: (a,b,c) and (a,c,d)
            positions.push(
                verts[a].x, verts[a].y, verts[a].z,
                verts[b].x, verts[b].y, verts[b].z,
                verts[c].x, verts[c].y, verts[c].z,
                verts[a].x, verts[a].y, verts[a].z,
                verts[c].x, verts[c].y, verts[c].z,
                verts[d].x, verts[d].y, verts[d].z
            );
        }

        const faceGeometry = new THREE.BufferGeometry();
        faceGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        const faceMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide,
            depthWrite: false
        });
        const faceMesh = new THREE.Mesh(faceGeometry, faceMaterial);
        if (annotationIdx !== null) faceMesh.userData.annotationIdx = annotationIdx;
        this.bboxGroup.add(faceMesh);
    }

    box3dToCorners(box3d) {
        const [cx, cy, cz, w, h, l, qw, qx, qy, qz] = box3d;
        const R = this.quaternionToRotationMatrix(qw, qx, qy, qz);
        const hw = w / 2;
        const hh = h / 2;
        const hl = l / 2;

        const localCorners = [
            [-hw, -hh, -hl], [ hw, -hh, -hl],
            [ hw,  hh, -hl], [-hw,  hh, -hl],
            [-hw, -hh,  hl], [ hw, -hh,  hl],
            [ hw,  hh,  hl], [-hw,  hh,  hl]
        ];

        return localCorners.map(([lx, ly, lz]) => {
            const rx = R[0][0]*lx + R[0][1]*ly + R[0][2]*lz;
            const ry = R[1][0]*lx + R[1][1]*ly + R[1][2]*lz;
            const rz = R[2][0]*lx + R[2][1]*ly + R[2][2]*lz;
            return [rx + cx, ry + cy, rz + cz];
        });
    }

    quaternionToRotationMatrix(qw, qx, qy, qz) {
        return [
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ];
    }

    /**
     * Set multiple bounding boxes at once.
     * Automatically zooms camera to frame the boxes rather than the full pointcloud.
     * @param {Array} boxes - Array of {box3d, color} objects
     */
    setBoxes(boxes, skipAutoZoom = false) {
        this.clearBboxes();
        for (const boxData of boxes) {
            this.addBbox(boxData.box3d, boxData.color || null,
                         boxData.annotationIdx !== undefined ? boxData.annotationIdx : null);
        }

        // Auto zoom: only on first call, not on re-renders (hide/show toggling)
        if (!skipAutoZoom && boxes.length > 0 && !this._intrinsics && !this._hasSetBoxes) {
            this._zoomToBoxes(boxes);
            this._hasSetBoxes = true;
        }
    }

    /**
     * Zoom camera to frame the 3D bounding boxes with some padding.
     */
    _zoomToBoxes(boxes) {
        // Compute bounding box of all 3D boxes (in Three.js coords)
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (const boxData of boxes) {
            const [cx, cy, cz, w, h, l] = boxData.box3d;
            // Transform to Three.js coords: negate Y and Z
            const tx = cx, ty = -cy, tz = -cz;
            const r = Math.max(w, h, l) / 2;
            minX = Math.min(minX, tx - r);
            minY = Math.min(minY, ty - r);
            minZ = Math.min(minZ, tz - r);
            maxX = Math.max(maxX, tx + r);
            maxY = Math.max(maxY, ty + r);
            maxZ = Math.max(maxZ, tz + r);
        }

        const center = new THREE.Vector3(
            (minX + maxX) / 2,
            (minY + maxY) / 2,
            (minZ + maxZ) / 2
        );

        const size = new THREE.Vector3(maxX - minX, maxY - minY, maxZ - minZ);
        const radius = size.length() / 2;

        // Place camera in FRONT of all boxes (maxZ = closest to camera) + padding
        const camZ = maxZ + radius * 1.5;

        this.camera.position.set(center.x, center.y, camZ);
        this.controls.target.copy(center);
        this.controls.update();

        this.controls.minDistance = radius * 0.3;
        this.controls.maxDistance = radius * 15;

        this.saveInitialState();
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    dispose() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }

        if (this.pointCloud) {
            this.scene.remove(this.pointCloud);
            this.pointCloud.geometry.dispose();
            this.pointCloud.material.dispose();
        }

        if (this.sceneMesh) {
            this.scene.remove(this.sceneMesh);
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
