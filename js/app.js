/**
 * Main Application Logic for Tinyval Visualization
 * 
 * Handles data loading, state management, and UI coordination.
 */

// ============================================================================
// Configuration
// ============================================================================

const DATA_BASE = 'data';
const IMAGES_PER_PAGE = 24;

// NOTE: 'algorithm' boxes are disabled due to coordinate alignment bug in pipeline
// The algorithm pipeline uses coco.annToMask() which returns masks at original resolution,
// but depth/camera are for SR images. This causes small objects to be placed at wrong locations.
// See README.md "Known Issues" section for details on the fix.
// To re-enable: remove 'algorithm' from DISABLED_MODELS and add back to MODEL_PRIORITY
const DISABLED_MODELS = new Set(['algorithm', 'algorithm_regression']);
const MODEL_PRIORITY = ['sam3d', '3d_mood', 'detany3d'];
// const MODEL_PRIORITY = ['sam3d', 'algorithm', '3d_mood', 'detany3d'];  // Re-enable after pipeline fix

const MODEL_DISPLAY_NAMES = {
    'sam3d': 'SAM3D',
    'algorithm_regression': 'Algorithm',
    'algorithm': 'Algorithm',
    '3d_mood': '3D-MOOD',
    'detany3d': 'DetAny3D'
};

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
    
    // Detail page state
    currentImage: null,
    scoredBoxes: null,
    unscoredBoxes: null,
    cameraParams: null,
    threshold: 5,
    selectedObject: null,
    activeModels: new Set(['sam3d', '3d_mood', 'detany3d']),  // algorithm disabled - see README
    
    // Renderers
    filteredViewer: null,
    exploreViewer: null,
    filtered2DRenderer: null,
    filtered3DRenderer: null,
    explore2DRenderer: null,
    explore3DRenderer: null
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
        
        // Collapse all button
        document.getElementById('btn-collapse-all')?.addEventListener('click', () => {
            sceneTree.collapseAll();
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
        
        // Initial render
        renderImageGrid();
        
    } catch (error) {
        console.error('Error loading index:', error);
        document.getElementById('image-grid').innerHTML = 
            '<div class="loading">Error loading data. Make sure to run prepare_data.py first.</div>';
    }
}

function updateStats() {
    const total = appState.index.total_images;
    const scored = appState.images.filter(img => img.has_scored_boxes).length;
    const categories = Object.keys(appState.index.images_by_scene || {}).length;
    
    document.getElementById('stat-total').textContent = total.toLocaleString();
    document.getElementById('stat-scored').textContent = scored.toLocaleString();
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
    
    // Filter by dataset
    if (appState.datasetFilter !== 'all') {
        filtered = filtered.filter(img => img.source === appState.datasetFilter);
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
    grid.innerHTML = pageImages.map(img => `
        <div class="image-card" data-id="${img.image_id}">
            <div class="image-card-thumb">
                <img src="${DATA_BASE}/images/${img.file_name}" 
                     alt="Image ${img.image_id}"
                     loading="lazy"
                     onerror="this.parentElement.innerHTML='<div class=\\'loading\\'>Image not found</div>'">
            </div>
            <div class="image-card-info">
                <div class="image-card-id">#${img.formatted_id || img.image_id}</div>
                <div class="image-card-meta">
                    <span class="badge badge-${img.source}">${img.source.toUpperCase()}</span>
                </div>
            </div>
        </div>
    `).join('');
    
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
        const dataPromises = [
            loadCameraParams(image),
            loadScoredBoxes(image),
            loadUnscoredBoxes(image)
        ];
        
        await Promise.all(dataPromises);
        
        // Initialize renderers
        initRenderers(image);
        
        // Setup controls
        setupThresholdControl();
        setupModelToggles();
        
        // Initial render
        renderFilteredSection();
        renderExploreSection();
        
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
    sourceBadge.textContent = image.source.toUpperCase();
    sourceBadge.className = `badge badge-${image.source}`;
    
    document.getElementById('scene-path').textContent = 
        image.scene_path ? image.scene_path.replace(/\//g, ' / ') : 'Unknown scene';
    
    document.getElementById('dimensions').textContent = 
        image.width && image.height ? `${image.width} × ${image.height}` : '';
    
    document.title = `Image ${image.formatted_id || image.image_id} - Tinyval Visualization`;
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

async function loadScoredBoxes(image) {
    const path = `${DATA_BASE}/boxes_scored/${image.dataset}_${image.split}_${image.formatted_id}.json`;
    try {
        const response = await fetch(path);
        if (response.ok) {
            appState.scoredBoxes = await response.json();
        }
    } catch (error) {
        console.warn('Could not load scored boxes:', error);
    }
}

async function loadUnscoredBoxes(image) {
    const path = `${DATA_BASE}/boxes_unscored/${image.dataset}_${image.split}_${image.formatted_id}.json`;
    try {
        const response = await fetch(path);
        if (response.ok) {
            appState.unscoredBoxes = await response.json();
        }
    } catch (error) {
        console.warn('Could not load unscored boxes:', error);
    }
}

function initRenderers(image) {
    const imagePath = `${DATA_BASE}/images/${image.file_name}`;
    const pointcloudPath = `${DATA_BASE}/pointclouds/${image.dataset}/${image.split}/${image.formatted_id}.ply`;
    
    // Initialize 2D renderers
    appState.filtered2DRenderer = new OverlayRenderer('canvas-2d-filtered', 'img-2d-filtered');
    appState.filtered3DRenderer = new OverlayRenderer('canvas-3d-filtered', 'img-3d-filtered');
    appState.explore2DRenderer = new OverlayRenderer('canvas-2d-explore', 'img-2d-explore');
    appState.explore3DRenderer = new OverlayRenderer('canvas-3d-explore', 'img-3d-explore');
    
    // Set intrinsics
    if (appState.cameraParams) {
        const intrinsics = appState.cameraParams.intrinsics;
        appState.filtered2DRenderer.setIntrinsics(intrinsics);
        appState.filtered3DRenderer.setIntrinsics(intrinsics);
        appState.explore2DRenderer.setIntrinsics(intrinsics);
        appState.explore3DRenderer.setIntrinsics(intrinsics);
    }
    
    // Load images
    appState.filtered2DRenderer.loadImage(imagePath);
    appState.filtered3DRenderer.loadImage(imagePath);
    appState.explore2DRenderer.loadImage(imagePath);
    appState.explore3DRenderer.loadImage(imagePath);
    
    // Initialize 3D viewers
    appState.filteredViewer = new ThreeViewer('threejs-filtered');
    appState.filteredViewer.init();
    appState.filteredViewer.loadPointcloud(pointcloudPath).catch(console.warn);
    
    appState.exploreViewer = new ThreeViewer('threejs-explore');
    appState.exploreViewer.init();
    appState.exploreViewer.loadPointcloud(pointcloudPath).catch(console.warn);
}

// ============================================================================
// Section 1: Filtered Best Boxes
// ============================================================================

function setupThresholdControl() {
    const slider = document.getElementById('threshold-slider');
    const valueDisplay = document.getElementById('threshold-value');
    
    if (!slider) return;
    
    slider.value = appState.threshold;
    valueDisplay.textContent = appState.threshold;
    
    slider.addEventListener('input', (e) => {
        appState.threshold = parseInt(e.target.value);
        valueDisplay.textContent = appState.threshold;
        renderFilteredSection();
    });
}

function getBoxScore(box) {
    /**
     * Get total score for a box.
     */
    return box.vlm_total_score ?? (
        (box.vlm_scores?.category ?? 0) +
        (box.vlm_scores?.scale ?? 0) +
        (box.vlm_scores?.translation ?? 0) +
        (box.vlm_scores?.shape ?? 0) +
        (box.vlm_scores?.tilt ?? 0) +
        (box.vlm_scores?.rotation ?? 0)
    );
}

function getFilteredBoxes() {
    /**
     * Get the best box for each annotation based on threshold and priority.
     * Returns array of {annotation, box, model, scores}
     * 
     * Structure: boxes3d is array of arrays (one array per annotation).
     * boxes2d and categories are at top level corresponding to each annotation.
     */
    if (!appState.scoredBoxes || !appState.scoredBoxes.boxes3d) {
        return [];
    }
    
    const results = [];
    const boxes3d = appState.scoredBoxes.boxes3d;
    const boxes2d = appState.scoredBoxes.boxes2d || [];
    const categories = appState.scoredBoxes.categories || [];
    
    for (let annIdx = 0; annIdx < boxes3d.length; annIdx++) {
        const annBoxes = boxes3d[annIdx];
        if (!annBoxes || annBoxes.length === 0) continue;
        
        // Filter out disabled models first, then by threshold
        const enabledBoxes = annBoxes.filter(box => !DISABLED_MODELS.has(box.model) && !DISABLED_MODELS.has(box.source));
        const qualifying = enabledBoxes.filter(box => getBoxScore(box) >= appState.threshold);
        
        if (qualifying.length === 0) continue;
        
        // Find max score
        const maxScore = Math.max(...qualifying.map(box => getBoxScore(box)));
        
        // Get tied boxes
        const tied = qualifying.filter(box => getBoxScore(box) === maxScore);
        
        // Select by priority
        let selectedBox = null;
        for (const model of MODEL_PRIORITY) {
            const match = tied.find(box => 
                box.model === model || box.source === model
            );
            if (match) {
                selectedBox = match;
                break;
            }
        }
        if (!selectedBox) selectedBox = tied[0];
        
        // Build annotation-like object from top-level data
        const bbox = boxes2d[annIdx] || null;
        const categoryName = categories[annIdx] || `object_${annIdx}`;
        
        results.push({
            annotationIdx: annIdx,
            annotation: { bbox },
            box: selectedBox,
            model: selectedBox.model || selectedBox.source || 'unknown',
            categoryName: categoryName
        });
    }
    
    return results;
}

function getCategoryName(categoryId) {
    if (!categoryId) return 'unknown';
    if (!appState.index?.categories) return `cat_${categoryId}`;
    return appState.index.categories[categoryId] || `cat_${categoryId}`;
}

function renderFilteredSection() {
    const filteredBoxes = getFilteredBoxes();
    const totalAnnotations = appState.scoredBoxes?.boxes3d?.length || 0;
    
    // Update info
    document.getElementById('threshold-info').textContent = 
        `Showing ${filteredBoxes.length} / ${totalAnnotations} objects`;
    
    // Draw 2D boxes (ground truth boxes for filtered annotations)
    setTimeout(() => {
        if (appState.filtered2DRenderer?.imageLoaded) {
            appState.filtered2DRenderer.clear();
            for (const item of filteredBoxes) {
                if (item.annotation.bbox) {
                    appState.filtered2DRenderer.draw2DBox(
                        item.annotation.bbox,
                        null,
                        item.categoryName
                    );
                }
            }
        }
    }, 100);
    
    // Draw 3D boxes projected to 2D
    setTimeout(() => {
        if (appState.filtered3DRenderer?.imageLoaded) {
            const boxes = filteredBoxes.map(item => ({
                box3d: item.box.box3d,  // Note: box3d not bbox3d
                model: item.model,
                label: item.categoryName
            }));
            appState.filtered3DRenderer.draw3DBoxes(boxes);
        }
    }, 100);
    
    // Update 3D viewer
    if (appState.filteredViewer) {
        const boxes = filteredBoxes.map(item => ({
            box3d: item.box.box3d,  // Note: box3d not bbox3d
            model: item.model,
            label: item.categoryName
        }));
        appState.filteredViewer.setBoxes(boxes);
    }
}

// ============================================================================
// Section 2: Explore All Boxes
// ============================================================================

function setupModelToggles() {
    const toggles = document.querySelectorAll('.model-toggle');
    
    toggles.forEach(toggle => {
        toggle.addEventListener('click', () => {
            const model = toggle.dataset.model;
            
            if (appState.activeModels.has(model)) {
                appState.activeModels.delete(model);
                toggle.classList.remove('active');
            } else {
                appState.activeModels.add(model);
                toggle.classList.add('active');
            }
            
            renderExploreSection();
        });
    });
}

function getObjectsWithBoxes() {
    /**
     * Get list of annotations that have at least one 3D box.
     */
    const boxes = appState.unscoredBoxes || appState.scoredBoxes;
    if (!boxes || !boxes.boxes3d) return [];
    
    const objects = [];
    const boxes2d = boxes.boxes2d || [];
    const categories = boxes.categories || [];
    
    for (let i = 0; i < boxes.boxes3d.length; i++) {
        const annBoxes = boxes.boxes3d[i];
        if (annBoxes && annBoxes.length > 0) {
            objects.push({
                index: i,
                bbox: boxes2d[i] || null,
                categoryName: categories[i] || `object_${i}`,
                boxCount: annBoxes.length
            });
        }
    }
    
    return objects;
}

function renderObjectButtons() {
    const container = document.getElementById('object-buttons');
    if (!container) return;
    
    const objects = getObjectsWithBoxes();
    
    if (objects.length === 0) {
        container.innerHTML = '<span class="no-objects">No objects with 3D boxes</span>';
        return;
    }
    
    container.innerHTML = objects.map((obj, idx) => `
        <button class="object-btn ${idx === 0 ? 'active' : ''}" data-index="${obj.index}">
            ${obj.categoryName} (${obj.boxCount})
        </button>
    `).join('');
    
    // Select first object by default
    if (appState.selectedObject === null && objects.length > 0) {
        appState.selectedObject = objects[0].index;
    }
    
    // Add click handlers
    container.querySelectorAll('.object-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            container.querySelectorAll('.object-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            appState.selectedObject = parseInt(btn.dataset.index);
            renderExploreSection();
        });
    });
}

function getBoxesForSelectedObject() {
    /**
     * Get boxes for the selected object filtered by active models.
     * Uses unscored boxes for visualization but scored boxes for scores.
     */
    const boxes = appState.unscoredBoxes || appState.scoredBoxes;
    if (!boxes || !boxes.boxes3d || appState.selectedObject === null) return [];
    
    const annBoxes = boxes.boxes3d[appState.selectedObject] || [];
    const boxes2d = boxes.boxes2d || [];
    const categories = boxes.categories || [];
    const bbox = boxes2d[appState.selectedObject] || null;
    const categoryName = categories[appState.selectedObject] || `object_${appState.selectedObject}`;
    
    // Get scored boxes for this annotation to look up scores
    const scoredAnnBoxes = appState.scoredBoxes?.boxes3d?.[appState.selectedObject] || [];
    
    // Build a map of model -> scores from scored boxes
    const modelScores = {};
    for (const scoredBox of scoredAnnBoxes) {
        const model = scoredBox.model || scoredBox.source || 'unknown';
        if (scoredBox.vlm_scores) {
            modelScores[model] = {
                scores: scoredBox.vlm_scores,
                totalScore: scoredBox.vlm_total_score
            };
        }
    }
    
    // Filter by active models (also excludes disabled models)
    const filtered = annBoxes.filter(box => {
        const model = box.model || box.source || 'unknown';
        if (DISABLED_MODELS.has(model)) return false;  // Skip disabled models
        return appState.activeModels.has(model);
    });
    
    return filtered.map(box => {
        const model = box.model || box.source || 'unknown';
        // Try to get scores from scored boxes first, then from the box itself
        const scoreData = modelScores[model] || {};
        
        return {
            box: box,
            model: model,
            bbox: bbox,
            categoryName: categoryName,
            scores: box.vlm_scores || scoreData.scores || null,
            totalScore: box.vlm_total_score ?? scoreData.totalScore ?? null
        };
    });
}

function renderExploreSection() {
    // Render object buttons on first call
    if (document.getElementById('object-buttons')?.querySelector('.object-btn') === null) {
        renderObjectButtons();
    }
    
    const boxesData = getBoxesForSelectedObject();
    const boxes = appState.unscoredBoxes || appState.scoredBoxes;
    const boxes2d = boxes?.boxes2d || [];
    const categories = boxes?.categories || [];
    const bbox = boxes2d[appState.selectedObject] || null;
    const categoryName = categories[appState.selectedObject] || `object_${appState.selectedObject}`;
    
    // Draw 2D box for selected object
    setTimeout(() => {
        if (appState.explore2DRenderer?.imageLoaded) {
            appState.explore2DRenderer.clear();
            if (bbox) {
                appState.explore2DRenderer.draw2DBox(
                    bbox,
                    null,
                    categoryName
                );
            }
        }
    }, 100);
    
    // Draw 3D boxes projected to 2D
    setTimeout(() => {
        if (appState.explore3DRenderer?.imageLoaded) {
            const boxes3d = boxesData.map(item => ({
                box3d: item.box.box3d,  // Note: box3d not bbox3d
                model: item.model,
                label: MODEL_DISPLAY_NAMES[item.model] || item.model
            }));
            appState.explore3DRenderer.draw3DBoxes(boxes3d);
        }
    }, 100);
    
    // Update 3D viewer
    if (appState.exploreViewer) {
        const boxes3d = boxesData.map(item => ({
            box3d: item.box.box3d,  // Note: box3d not bbox3d
            model: item.model,
            label: MODEL_DISPLAY_NAMES[item.model] || item.model
        }));
        appState.exploreViewer.setBoxes(boxes3d);
    }
    
    // Update score comparison table
    renderScoreTable(boxesData);
}

function renderScoreTable(boxesData) {
    const tbody = document.getElementById('score-table-body');
    if (!tbody) return;
    
    if (boxesData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-message">Select an object and models to view scores</td></tr>';
        return;
    }
    
    tbody.innerHTML = boxesData.map(item => {
        const scores = item.scores || {};
        const modelClass = item.model.replace(/_/g, '');
        const displayName = MODEL_DISPLAY_NAMES[item.model] || item.model;
        
        // Calculate total if not provided
        let total = item.totalScore;
        if (total === null) {
            total = (scores.category ?? 0) + (scores.scale ?? 0) + (scores.translation ?? 0) +
                    (scores.shape ?? 0) + (scores.tilt ?? 0) + (scores.rotation ?? 0);
        }
        
        const scoreCell = (val) => {
            if (val === undefined || val === null) return '<td class="score-cell">-</td>';
            const scoreClass = val >= 2 ? 'score-2' : (val >= 1 ? 'score-1' : 'score-0');
            return `<td class="score-cell ${scoreClass}">${val}</td>`;
        };
        
        return `
            <tr>
                <td>
                    <span class="model-indicator ${item.model}"></span>
                    ${displayName}
                </td>
                ${scoreCell(scores.category)}
                ${scoreCell(scores.scale)}
                ${scoreCell(scores.translation)}
                ${scoreCell(scores.shape)}
                ${scoreCell(scores.tilt)}
                ${scoreCell(scores.rotation)}
                <td class="score-total">${total}</td>
            </tr>
        `;
    }).join('');
}

// Initialize on page load
