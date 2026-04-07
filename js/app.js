/**
 * Main Application Logic for WildDet3D-Bench Visualization
 *
 * Handles data loading, state management, and UI coordination.
 * Simplified from tinyval version: single box per annotation, no model comparison.
 */

// ============================================================================
// Configuration
// ============================================================================

const DATA_BASE = 'https://huggingface.co/datasets/Silicon23/WildDet3D-demo/resolve/main/data';
const IMAGES_PER_PAGE = 24;

// ============================================================================
// Category Color Utilities
// ============================================================================

function categoryToHue(categoryName) {
    let hash = 0;
    for (let i = 0; i < categoryName.length; i++) {
        hash = categoryName.charCodeAt(i) + ((hash << 5) - hash);
        hash = hash & hash; // Convert to 32-bit integer
    }
    return ((hash % 360) + 360) % 360;
}

function categoryToColor(categoryName) {
    const hue = categoryToHue(categoryName);
    return `hsl(${hue}, 70%, 60%)`;
}

function hslToHex(h, s, l) {
    s /= 100;
    l /= 100;
    const a = s * Math.min(l, 1 - l);
    const f = n => {
        const k = (n + h / 30) % 12;
        const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
        return Math.round(255 * color).toString(16).padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}`;
}

function categoryToHexColor(categoryName) {
    return hslToHex(categoryToHue(categoryName), 70, 60);
}

function categoryToThreeColor(categoryName) {
    return parseInt(categoryToHexColor(categoryName).slice(1), 16);
}

// ============================================================================
// State
// ============================================================================

let appState = {
    // Index page state
    index: null,
    images: [],
    filteredImages: [],
    currentPage: 1,
    selectedScene: null,
    datasetFilter: 'all',
    searchQuery: '',
    shuffled: false,

    // Detail page state
    currentImage: null,
    boxData: null,
    cameraParams: null,
    showLabels: true,
    hiddenAnnotations: new Set(),  // annotation indices hidden by user
    categoryOverrides: {},         // annotationIdx -> custom category name

    // Renderers (Section 1 only)
    filteredViewer: null,
    filtered2DRenderer: null,
    filtered3DRenderer: null
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    const isDetailPage = window.location.pathname.endsWith('image.html');

    if (isDetailPage) {
        initDetailPage();
    } else {
        initIndexPage();
    }
});

// ============================================================================
// Index Page
// ============================================================================

async function initIndexPage() {
    try {
        // Load index data
        const response = await fetch(`${DATA_BASE}/index.json`);
        appState.index = await response.json();
        appState.images = appState.index.images;
        appState.filteredImages = [...appState.images];

        // Update stats
        updateStats();

        // Initialize scene tree
        const sceneTree = new SceneTree('tree-container', onSceneSelect);
        sceneTree.render(appState.index.scene_tree);

        // Sidebar open/close
        const sidebarOpenBtn = document.getElementById('sidebar-open');
        const sidebarCloseBtn = document.getElementById('sidebar-close');
        const sidebar = document.querySelector('.sidebar');

        sidebarOpenBtn?.addEventListener('click', () => {
            sidebar?.classList.remove('collapsed');
            sidebarOpenBtn.classList.add('hidden');
        });
        sidebarCloseBtn?.addEventListener('click', () => {
            sidebar?.classList.add('collapsed');
            sidebarOpenBtn?.classList.remove('hidden');
        });

        // Search input
        document.getElementById('search-input')?.addEventListener('input', (e) => {
            appState.searchQuery = e.target.value.trim();
            filterImages();
        });

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                appState.datasetFilter = btn.dataset.filter;
                filterImages();
            });
        });

        // Shuffle toggle
        document.getElementById('shuffle-toggle')?.addEventListener('click', (e) => {
            appState.shuffled = !appState.shuffled;
            e.currentTarget.classList.toggle('active', appState.shuffled);
            filterImages();
        });

        // Initial render
        renderImageGrid();

    } catch (error) {
        console.error('Error loading index:', error);
        document.getElementById('image-grid').innerHTML =
            '<div class="loading">Error loading data. Make sure to run prepare_data.py first.</div>';
    }
}

function updateStats() {
    const totalInDataset = appState.index.total_images_in_dataset || appState.index.total_images;
    const withBoxes = appState.index.total_images;
    const categories = Object.keys(appState.index.images_by_scene || {}).length;

    document.getElementById('stat-total').textContent = totalInDataset.toLocaleString();
    document.getElementById('stat-scored').textContent = withBoxes.toLocaleString();
    document.getElementById('stat-categories').textContent = categories.toLocaleString();
}

function onSceneSelect(scenePath) {
    appState.selectedScene = scenePath;
    appState.currentPage = 1;

    // Update title
    const title = scenePath
        ? scenePath.split('/').pop().replace(/_/g, ' ')
        : 'All Images';
    document.getElementById('content-title').textContent = title;

    filterImages();
}

function filterImages() {
    let filtered = [...appState.images];

    // Filter by scene
    if (appState.selectedScene) {
        const sceneImages = new Set(
            appState.index.images_by_scene[appState.selectedScene] || []
        );
        // Also include images from child scenes
        Object.entries(appState.index.images_by_scene).forEach(([path, ids]) => {
            if (path.startsWith(appState.selectedScene + '/')) {
                ids.forEach(id => sceneImages.add(id));
            }
        });
        filtered = filtered.filter(img => sceneImages.has(img.image_id));
    }

    // Filter by dataset+split (format: "source_split" e.g. "coco_val", "lvis_train")
    if (appState.datasetFilter !== 'all') {
        filtered = filtered.filter(img => `${img.source}_${img.split}` === appState.datasetFilter);
    }

    // Filter by search query
    if (appState.searchQuery) {
        const query = appState.searchQuery.toLowerCase();
        filtered = filtered.filter(img =>
            String(img.image_id).includes(query) ||
            String(img.original_id).includes(query) ||
            img.formatted_id.includes(query)
        );
    }

    // Shuffle if toggled on (fresh shuffle each time)
    if (appState.shuffled) {
        for (let i = filtered.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [filtered[i], filtered[j]] = [filtered[j], filtered[i]];
        }
    }

    appState.filteredImages = filtered;
    appState.currentPage = 1;
    renderImageGrid();
}

function renderImageGrid() {
    const grid = document.getElementById('image-grid');
    const pagination = document.getElementById('pagination');

    if (!grid) return;

    const images = appState.filteredImages;
    const totalPages = Math.ceil(images.length / IMAGES_PER_PAGE);
    const start = (appState.currentPage - 1) * IMAGES_PER_PAGE;
    const end = start + IMAGES_PER_PAGE;
    const pageImages = images.slice(start, end);

    if (pageImages.length === 0) {
        grid.innerHTML = '<div class="loading">No images found</div>';
        pagination.innerHTML = '';
        return;
    }

    // Render images
    grid.innerHTML = pageImages.map(img => {
        const splitLabel = img.split.charAt(0).toUpperCase() + img.split.slice(1);
        const sourceLabel = img.source.toUpperCase();
        return `
        <div class="image-card" data-id="${img.image_id}">
            <div class="image-card-thumb">
                <img src="${DATA_BASE}/images_annotated/${img.file_name}"
                     alt="Image ${img.image_id}"
                     loading="lazy"
                     onerror="this.parentElement.innerHTML='<div class=\\'loading\\'>Image not found</div>'">
            </div>
            <div class="image-card-info">
                <div class="image-card-id">#${img.formatted_id || img.image_id}</div>
                <div class="image-card-meta">
                    <span class="badge badge-${img.source}">${sourceLabel} ${splitLabel}</span>
                    <span class="annotation-count">${img.num_valid_boxes} boxes</span>
                </div>
            </div>
        </div>
    `}).join('');

    // Add click handlers
    grid.querySelectorAll('.image-card').forEach(card => {
        card.addEventListener('click', () => {
            const imageId = card.dataset.id;
            window.location.href = `image.html?id=${imageId}`;
        });
    });

    // Render pagination
    renderPagination(totalPages);
}

function renderPagination(totalPages) {
    const pagination = document.getElementById('pagination');
    if (!pagination || totalPages <= 1) {
        if (pagination) pagination.innerHTML = '';
        return;
    }

    const currentPage = appState.currentPage;
    let pages = [];

    // Always show first page
    pages.push(1);

    // Show pages around current
    for (let i = Math.max(2, currentPage - 2); i <= Math.min(totalPages - 1, currentPage + 2); i++) {
        if (pages[pages.length - 1] !== i - 1) {
            pages.push('...');
        }
        pages.push(i);
    }

    // Always show last page
    if (totalPages > 1) {
        if (pages[pages.length - 1] !== totalPages - 1) {
            pages.push('...');
        }
        pages.push(totalPages);
    }

    pagination.innerHTML = `
        <button class="pagination-btn" ${currentPage === 1 ? 'disabled' : ''} data-page="prev">
            &larr;
        </button>
        ${pages.map(p =>
            p === '...'
                ? '<span class="pagination-btn" style="cursor:default">...</span>'
                : `<button class="pagination-btn ${p === currentPage ? 'active' : ''}" data-page="${p}">${p}</button>`
        ).join('')}
        <button class="pagination-btn" ${currentPage === totalPages ? 'disabled' : ''} data-page="next">
            &rarr;
        </button>
    `;

    pagination.querySelectorAll('.pagination-btn[data-page]').forEach(btn => {
        btn.addEventListener('click', () => {
            const page = btn.dataset.page;
            if (page === 'prev') {
                appState.currentPage = Math.max(1, currentPage - 1);
            } else if (page === 'next') {
                appState.currentPage = Math.min(totalPages, currentPage + 1);
            } else if (!isNaN(parseInt(page))) {
                appState.currentPage = parseInt(page);
            }
            renderImageGrid();
            // Scroll to top of grid
            document.getElementById('image-grid')?.scrollTo(0, 0);
        });
    });
}

// ============================================================================
// Detail Page
// ============================================================================

async function initDetailPage() {
    const urlParams = new URLSearchParams(window.location.search);
    const imageId = urlParams.get('id');

    if (!imageId) {
        showError('No image ID specified');
        return;
    }

    try {
        // Load index to get image info
        const indexResponse = await fetch(`${DATA_BASE}/index.json`);
        appState.index = await indexResponse.json();

        // Find image in index
        const image = appState.index.images.find(
            img => String(img.image_id) === imageId || String(img.original_id) === imageId
        );

        if (!image) {
            showError(`Image ${imageId} not found`);
            return;
        }

        appState.currentImage = image;

        // Update header info
        updateImageInfo(image);

        // Load data in parallel
        await Promise.all([
            loadCameraParams(image),
            loadBoxData(image)
        ]);

        // Initialize renderers (await image loading)
        await initRenderers(image);

        // Show all hidden boxes
        document.getElementById('show-all-boxes')?.addEventListener('click', () => {
            appState.hiddenAnnotations.clear();
            renderFilteredSection();
            syncAnnotationListCheckboxes();
        });

        // Label toggle button
        document.getElementById('toggle-labels')?.addEventListener('click', (e) => {
            appState.showLabels = !appState.showLabels;
            e.currentTarget.classList.toggle('active', appState.showLabels);
            renderFilteredSection();
        });

        // Initial render
        renderFilteredSection();
        renderAnnotationList();

        // Set up download buttons and click-to-hide (once)
        setupDownloadButtons();
        setupClickToHide();

    } catch (error) {
        console.error('Error initializing detail page:', error);
        showError('Error loading image data');
    }
}

function showError(message) {
    document.querySelector('.detail-content').innerHTML =
        `<div class="detail-section"><div class="loading">${message}</div></div>`;
}

function updateImageInfo(image) {
    document.getElementById('image-id').textContent = image.formatted_id || image.image_id;

    const sourceBadge = document.getElementById('source-badge');
    const splitLabel = image.split.charAt(0).toUpperCase() + image.split.slice(1);
    sourceBadge.textContent = `${image.source.toUpperCase()} ${splitLabel}`;
    sourceBadge.className = `badge badge-${image.source}`;

    document.getElementById('scene-path').textContent =
        image.scene_path ? image.scene_path.replace(/\//g, ' / ') : 'Unknown scene';

    document.getElementById('dimensions').textContent =
        image.width && image.height ? `${image.width} x ${image.height}` : '';

    document.title = `Image ${image.formatted_id || image.image_id} - WildDet3D-Bench`;
}

async function loadCameraParams(image) {
    const path = `${DATA_BASE}/camera/${image.dataset}/${image.split}/${image.formatted_id}.json`;
    try {
        const response = await fetch(path);
        appState.cameraParams = await response.json();
    } catch (error) {
        console.warn('Could not load camera params:', error);
    }
}

async function loadBoxData(image) {
    const path = `${DATA_BASE}/boxes/${image.dataset}_${image.split}_${image.formatted_id}.json`;
    try {
        const response = await fetch(path);
        if (response.ok) {
            appState.boxData = await response.json();
        }
    } catch (error) {
        console.warn('Could not load box data:', error);
    }
}

async function initRenderers(image) {
    const imagePath = `${DATA_BASE}/images/${image.file_name}`;
    const pointcloudPath = `${DATA_BASE}/pointclouds/${image.dataset}/${image.split}/${image.formatted_id}.glb`;

    // Initialize 2D renderers
    appState.filtered2DRenderer = new OverlayRenderer('canvas-2d-filtered', 'img-2d-filtered');
    appState.filtered3DRenderer = new OverlayRenderer('canvas-3d-filtered', 'img-3d-filtered');

    // Set intrinsics
    if (appState.cameraParams) {
        const intrinsics = appState.cameraParams.intrinsics;
        appState.filtered2DRenderer.setIntrinsics(intrinsics);
        appState.filtered3DRenderer.setIntrinsics(intrinsics);
    }

    // Load images - await so boxes can be drawn immediately after
    await Promise.all([
        appState.filtered2DRenderer.loadImage(imagePath),
        appState.filtered3DRenderer.loadImage(imagePath)
    ]);

    // Initialize 3D viewer (separate try/catch so WebGL failure doesn't break 2D)
    try {
        const viewer = new ThreeViewer('threejs-filtered');
        viewer.init();

        if (appState.cameraParams && appState.cameraParams.image_size) {
            viewer.setCameraIntrinsics(
                appState.cameraParams.intrinsics,
                appState.cameraParams.image_size
            );
        }

        viewer.loadPointcloud(pointcloudPath).catch(console.warn);
        appState.filteredViewer = viewer;
    } catch (e) {
        console.warn('3D viewer failed (no WebGL?):', e.message);
        appState.filteredViewer = null;
    }
}

// ============================================================================
// Visualization: Valid Boxes (ignore3D=0)
// ============================================================================

function getValidBoxes() {
    if (!appState.boxData || !appState.boxData.boxes3d) return [];

    const results = [];
    const { boxes3d, boxes2d, categories, ignore3D } = appState.boxData;

    for (let i = 0; i < boxes3d.length; i++) {
        // Skip ignored annotations
        if (ignore3D[i] !== 0) continue;

        const innerBoxes = boxes3d[i];
        if (!innerBoxes || innerBoxes.length === 0) continue;

        // Human-selected box is the single element
        const box = innerBoxes[0];
        if (!box || !box.box3d) continue;

        const originalName = categories[i] || `object_${i}`;
        const categoryName = appState.categoryOverrides[i] ?? originalName;

        results.push({
            annotationIdx: i,
            bbox2d: boxes2d[i] || null,
            box3d: box.box3d,
            categoryName: categoryName,
            originalName: originalName,
            color: categoryToColor(categoryName),
            hexColor: categoryToHexColor(categoryName),
            threeColor: categoryToThreeColor(categoryName)
        });
    }

    return results;
}

function getVisibleBoxes() {
    return getValidBoxes().filter(item => !appState.hiddenAnnotations.has(item.annotationIdx));
}

function renderFilteredSection() {
    const visibleBoxes = getVisibleBoxes();
    const showLabels = appState.showLabels;

    // Store for click hit-testing
    appState._lastVisibleBoxes = visibleBoxes;

    // Draw 2D boxes with category colors
    if (appState.filtered2DRenderer?.imageLoaded) {
        appState.filtered2DRenderer.clear();
        for (const item of visibleBoxes) {
            if (item.bbox2d) {
                appState.filtered2DRenderer.draw2DBox(
                    item.bbox2d,
                    item.color,
                    showLabels ? item.categoryName : null
                );
            }
        }
    }

    // Draw 3D boxes projected to 2D with category colors
    if (appState.filtered3DRenderer?.imageLoaded) {
        const boxes = visibleBoxes.map(item => ({
            box3d: item.box3d,
            color: item.hexColor,
            label: showLabels ? item.categoryName : null
        }));
        appState.filtered3DRenderer.draw3DBoxes(boxes);
    }

    // Update 3D viewer with category colors
    if (appState.filteredViewer) {
        const boxes = visibleBoxes.map(item => ({
            box3d: item.box3d,
            color: item.threeColor,
            annotationIdx: item.annotationIdx
        }));
        appState.filteredViewer.setBoxes(boxes);
    }
}

// ============================================================================
// Click-to-hide: toggle annotation visibility
// ============================================================================

function setupClickToHide() {
    // 2D overlay canvas: click inside a 2D bbox to hide that annotation
    const canvas2d = document.getElementById('canvas-2d-filtered');
    if (canvas2d) {
        canvas2d.addEventListener('click', (e) => {
            handleCanvasClick(e, canvas2d, 'bbox2d');
        });
    }

    // 3D projected canvas: click inside a projected 3D bbox to hide
    const canvas3d = document.getElementById('canvas-3d-filtered');
    if (canvas3d) {
        canvas3d.addEventListener('click', (e) => {
            handleCanvasClick(e, canvas3d, 'box3d');
        });
    }

    // Three.js viewer: click on a box wireframe to hide
    setupThreeJsClickToHide();
}

function handleCanvasClick(e, canvas, mode) {
    const rect = canvas.getBoundingClientRect();
    const clickX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const clickY = (e.clientY - rect.top) * (canvas.height / rect.height);

    const renderer = mode === 'bbox2d' ? appState.filtered2DRenderer : appState.filtered3DRenderer;
    if (!renderer) return;

    const boxes = appState._lastVisibleBoxes || [];

    // Check boxes in reverse order (topmost drawn last)
    for (let i = boxes.length - 1; i >= 0; i--) {
        const item = boxes[i];
        let hit = false;

        if (mode === 'bbox2d' && item.bbox2d) {
            // Hit test against 2D bounding box
            const [x1, y1, x2, y2] = item.bbox2d;
            const sx1 = x1 * renderer.scaleX, sy1 = y1 * renderer.scaleY;
            const sx2 = x2 * renderer.scaleX, sy2 = y2 * renderer.scaleY;
            hit = clickX >= sx1 && clickX <= sx2 && clickY >= sy1 && clickY <= sy2;
        } else if (mode === 'box3d' && item.box3d && renderer.intrinsics) {
            // Hit test against projected 3D bounding box (use 2D bounding rect of projected corners)
            const corners = renderer.box3dToCorners(item.box3d);
            if (corners.every(c => c[2] > 0.1)) {
                const pts = corners.map(c => renderer.project3Dto2D(c));
                const xs = pts.map(p => p[0] * renderer.scaleX);
                const ys = pts.map(p => p[1] * renderer.scaleY);
                const minX = Math.min(...xs), maxX = Math.max(...xs);
                const minY = Math.min(...ys), maxY = Math.max(...ys);
                hit = clickX >= minX && clickX <= maxX && clickY >= minY && clickY <= maxY;
            }
        }

        if (hit) {
            toggleAnnotation(item.annotationIdx);
            return;
        }
    }
}

function setupThreeJsClickToHide() {
    const viewer = appState.filteredViewer;
    if (!viewer || !viewer.renderer) return;

    const raycaster = new THREE.Raycaster();
    raycaster.params.Line = { threshold: 0.5 };
    const mouse = new THREE.Vector2();

    viewer.renderer.domElement.addEventListener('click', (e) => {
        // Skip if the user dragged (to avoid hiding on rotate/pan)
        if (viewer._didDrag) return;

        const rect = viewer.renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, viewer.camera);
        const intersects = raycaster.intersectObjects(viewer.bboxGroup.children, true);

        if (intersects.length > 0) {
            // Find which annotation this mesh belongs to
            const hitObj = intersects[0].object;
            const annIdx = hitObj.userData?.annotationIdx;
            if (annIdx !== undefined) {
                toggleAnnotation(annIdx);
            }
        }
    });

    // Track drag state to distinguish click from drag
    viewer.renderer.domElement.addEventListener('pointerdown', () => { viewer._didDrag = false; });
    viewer.renderer.domElement.addEventListener('pointermove', () => { viewer._didDrag = true; });
}

function toggleAnnotation(annotationIdx) {
    if (appState.hiddenAnnotations.has(annotationIdx)) {
        appState.hiddenAnnotations.delete(annotationIdx);
    } else {
        appState.hiddenAnnotations.add(annotationIdx);
    }
    renderFilteredSection();
    syncAnnotationListCheckboxes();
}

// ============================================================================
// Annotation List Panel
// ============================================================================

function renderAnnotationList() {
    const container = document.getElementById('annotation-list-items');
    if (!container) return;

    const allValid = getValidBoxes();
    if (allValid.length === 0) {
        container.innerHTML = '<span style="color:var(--text-muted);font-size:12px;">No valid annotations</span>';
        return;
    }

    container.innerHTML = allValid.map(item => {
        const hidden = appState.hiddenAnnotations.has(item.annotationIdx);
        return `
        <div class="annotation-item ${hidden ? 'hidden' : ''}" data-idx="${item.annotationIdx}">
            <input type="checkbox" ${hidden ? '' : 'checked'} data-idx="${item.annotationIdx}" title="Show/hide">
            <span class="annotation-item-color" style="background:${item.color}"></span>
            <input type="text" class="annotation-item-name" value="${item.categoryName}" data-idx="${item.annotationIdx}" title="Edit category name">
        </div>`;
    }).join('');

    // Checkbox: toggle visibility
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', () => {
            const idx = parseInt(cb.dataset.idx);
            if (cb.checked) {
                appState.hiddenAnnotations.delete(idx);
            } else {
                appState.hiddenAnnotations.add(idx);
            }
            cb.closest('.annotation-item').classList.toggle('hidden', !cb.checked);
            renderFilteredSection();
        });
    });

    // Text input: edit category name
    container.querySelectorAll('.annotation-item-name').forEach(input => {
        input.addEventListener('input', () => {
            const idx = parseInt(input.dataset.idx);
            const newName = input.value.trim();
            if (newName) {
                appState.categoryOverrides[idx] = newName;
            } else {
                delete appState.categoryOverrides[idx];
            }
            renderFilteredSection();
        });
    });
}

function syncAnnotationListCheckboxes() {
    const container = document.getElementById('annotation-list-items');
    if (!container) return;
    container.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        const idx = parseInt(cb.dataset.idx);
        const hidden = appState.hiddenAnnotations.has(idx);
        cb.checked = !hidden;
        cb.closest('.annotation-item').classList.toggle('hidden', hidden);
    });
}

// ============================================================================
// Download / Export
// ============================================================================

function setupDownloadButtons() {
    const imgId = appState.currentImage?.formatted_id || 'image';

    // Download 2D bbox image
    document.getElementById('dl-2d')?.addEventListener('click', () => {
        downloadCanvas('canvas-2d-filtered', `${imgId}_2d_boxes.png`);
    });

    // Download 3D projected image
    document.getElementById('dl-3d-proj')?.addEventListener('click', () => {
        downloadCanvas('canvas-3d-filtered', `${imgId}_3d_projected.png`);
    });

    // Download 3D viewer snapshot
    document.getElementById('dl-3d-snap')?.addEventListener('click', () => {
        const viewer = appState.filteredViewer;
        if (!viewer || !viewer.renderer) return;
        viewer.renderer.render(viewer.scene, viewer.camera);
        const canvas = viewer.renderer.domElement;
        canvas.toBlob((blob) => {
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = `${imgId}_3d_view.png`;
            link.href = url;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }, 'image/png');
    });

    // Export 3D rotation video
    document.getElementById('dl-3d-video')?.addEventListener('click', (e) => {
        export3DVideo(e.currentTarget, `${imgId}_3d_rotation.webm`);
    });

    // Combined download: 3D projection + point cloud side by side
    document.getElementById('dl-combined')?.addEventListener('click', () => {
        downloadCombined(`${imgId}_combined.png`);
    });
}

function downloadCanvas(canvasId, filename) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = filename;
    link.href = canvas.toDataURL('image/png');
    link.click();
}

function downloadCombined(filename) {
    const projCanvas = document.getElementById('canvas-3d-filtered');
    const viewer = appState.filteredViewer;

    if (!projCanvas) { console.error('downloadCombined: projCanvas not found'); return; }
    if (!viewer || !viewer.renderer) { console.error('downloadCombined: viewer/renderer not ready'); return; }

    const PANEL_W = 800;
    const PANEL_H = 600;

    const combined = document.createElement('canvas');
    combined.width = PANEL_W * 2;
    combined.height = PANEL_H;
    const ctx = combined.getContext('2d');

    // Black background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, combined.width, combined.height);

    // Left panel: projection fitted with symmetric padding
    drawFitted(ctx, projCanvas, 0, 0, PANEL_W, PANEL_H);

    // Right panel: re-render Three.js at exact panel size for a clean capture
    const origDpr = viewer.renderer.getPixelRatio();
    const origW = viewer.renderer.domElement.clientWidth;
    const origH = viewer.renderer.domElement.clientHeight;
    viewer.renderer.setPixelRatio(1);
    viewer.renderer.setSize(PANEL_W, PANEL_H);
    viewer.camera.aspect = PANEL_W / PANEL_H;
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.render(viewer.scene, viewer.camera);
    ctx.drawImage(viewer.renderer.domElement, PANEL_W, 0);

    // Restore original renderer size
    viewer.renderer.setPixelRatio(origDpr);
    viewer.renderer.setSize(origW, origH);
    viewer.camera.aspect = origW / origH;
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.render(viewer.scene, viewer.camera);

    // Use blob download for better browser compatibility
    combined.toBlob((blob) => {
        if (!blob) { console.error('downloadCombined: toBlob returned null'); return; }
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.download = filename;
        link.href = url;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }, 'image/png');
}

/** Draw a source canvas aspect-fitted and centered into a target rectangle.
 *  Padding is always symmetric: either top+bottom OR left+right, never both. */
function drawFitted(ctx, source, tx, ty, tw, th) {
    const srcRatio = source.width / source.height;
    const tgtRatio = tw / th;
    let dw, dh;
    if (srcRatio > tgtRatio) {
        dw = tw;
        dh = tw / srcRatio;
    } else {
        dh = th;
        dw = th * srcRatio;
    }
    const dx = tx + (tw - dw) / 2;
    const dy = ty + (th - dh) / 2;
    ctx.drawImage(source, dx, dy, dw, dh);
}

function export3DVideo(btn, filename) {
    if (btn.classList.contains('recording')) return;

    const viewer = appState.filteredViewer;
    if (!viewer || !viewer.renderer) return;

    // Pause live tilt during export
    const wasTilting = viewer._tiltActive;
    viewer._tiltActive = false;

    const canvas = viewer.renderer.domElement;
    // Manual frame mode: only captures when we call requestFrame()
    const stream = canvas.captureStream(0);
    const videoTrack = stream.getVideoTracks()[0];

    const recorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9',
        videoBitsPerSecond: 8000000
    });
    const chunks = [];

    recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const link = document.createElement('a');
        link.download = filename;
        link.href = URL.createObjectURL(blob);
        link.click();
        URL.revokeObjectURL(link.href);

        btn.classList.remove('recording');
        btn.innerHTML = '&#x23FA; Export Video';
        viewer._tiltActive = wasTilting;
    };

    btn.classList.add('recording');
    btn.innerHTML = '&#x23F3; Exporting...';

    // Get the origin camera state for the tilt
    const origin = viewer._tiltOrigin || {
        position: viewer.camera.position.clone(),
        target: viewer.controls.target.clone()
    };
    const offset = new THREE.Vector3().subVectors(origin.position, origin.target);
    const radius = offset.length();
    const baseAngle = Math.atan2(offset.x, offset.z);

    // Render one full sine cycle: 4 sec at 30fps
    const fps = 30;
    const durationMs = 4000;
    const totalFrames = Math.ceil(durationMs / 1000 * fps);
    const tStep = (2 * Math.PI) / totalFrames;

    recorder.start();

    const exportStart = performance.now();
    let frame = 0;

    function renderNextFrame() {
        if (frame >= totalFrames) {
            recorder.stop();
            viewer.camera.position.copy(origin.position);
            viewer.controls.target.copy(origin.target);
            viewer.controls.update();
            return;
        }

        const t = frame * tStep;
        const angle = Math.sin(t) * 0.25;
        const newAngle = baseAngle + angle;

        viewer.camera.position.set(
            origin.target.x + radius * Math.sin(newAngle),
            origin.position.y + radius * 0.04 * Math.sin(t * 0.7),
            origin.target.z + radius * Math.cos(newAngle)
        );
        viewer.controls.target.copy(origin.target);
        viewer.controls.update();
        viewer.renderer.render(viewer.scene, viewer.camera);

        if (videoTrack.requestFrame) {
            videoTrack.requestFrame();
        }

        frame++;

        // Wait until real wall-clock time matches target frame time
        const targetTime = exportStart + frame * (1000 / fps);
        const delay = Math.max(0, targetTime - performance.now());
        setTimeout(renderNextFrame, delay);
    }

    renderNextFrame();
}
