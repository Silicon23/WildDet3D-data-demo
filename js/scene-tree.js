/**
 * Scene Tree Component
 * 
 * Renders a collapsible hierarchical tree of scene categories.
 */

class SceneTree {
    constructor(containerId, onSelect) {
        this.container = document.getElementById(containerId);
        this.onSelect = onSelect;
        this.selectedPath = null;
        this.expandedPaths = new Set();
    }
    
    render(treeData) {
        if (!this.container || !treeData) return;
        
        this.container.innerHTML = '';
        
        // Add root-level "All Images" option
        const allNode = this.createAllNode(treeData.image_count || 0);
        this.container.appendChild(allNode);
        
        // Render tree children
        if (treeData.children && treeData.children.length > 0) {
            for (const child of treeData.children) {
                const nodeEl = this.renderNode(child, 0);
                this.container.appendChild(nodeEl);
            }
        }
    }
    
    createAllNode(imageCount) {
        const node = document.createElement('div');
        node.className = 'tree-node';
        
        const header = document.createElement('div');
        header.className = 'tree-node-header' + (this.selectedPath === null ? ' selected' : '');
        header.innerHTML = `
            <span class="tree-toggle empty">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="9 18 15 12 9 6"></polyline>
                </svg>
            </span>
            <span class="tree-node-name">All Images</span>
            <span class="tree-node-count">${imageCount}</span>
        `;
        
        header.addEventListener('click', () => {
            this.selectPath(null);
        });
        
        node.appendChild(header);
        return node;
    }
    
    renderNode(nodeData, depth) {
        const node = document.createElement('div');
        node.className = 'tree-node';
        node.dataset.path = nodeData.path;
        
        const hasChildren = nodeData.children && nodeData.children.length > 0;
        const isExpanded = this.expandedPaths.has(nodeData.path);
        const isSelected = this.selectedPath === nodeData.path;
        
        // Header
        const header = document.createElement('div');
        header.className = 'tree-node-header' + (isSelected ? ' selected' : '');
        
        const toggleClass = hasChildren ? (isExpanded ? 'expanded' : '') : 'empty';
        
        header.innerHTML = `
            <span class="tree-toggle ${toggleClass}">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="9 18 15 12 9 6"></polyline>
                </svg>
            </span>
            <span class="tree-node-name">${nodeData.name}</span>
            <span class="tree-node-count">${nodeData.image_count}</span>
        `;
        
        // Click handler
        header.addEventListener('click', (e) => {
            if (hasChildren && e.target.closest('.tree-toggle')) {
                this.toggleExpand(nodeData.path);
            } else {
                this.selectPath(nodeData.path);
            }
        });
        
        node.appendChild(header);
        
        // Children container
        if (hasChildren) {
            const children = document.createElement('div');
            children.className = 'tree-children' + (isExpanded ? ' expanded' : '');
            
            for (const child of nodeData.children) {
                const childNode = this.renderNode(child, depth + 1);
                children.appendChild(childNode);
            }
            
            node.appendChild(children);
        }
        
        return node;
    }
    
    toggleExpand(path) {
        if (this.expandedPaths.has(path)) {
            this.expandedPaths.delete(path);
        } else {
            this.expandedPaths.add(path);
        }
        
        // Update DOM
        const node = this.container.querySelector(`[data-path="${path}"]`);
        if (node) {
            const toggle = node.querySelector(':scope > .tree-node-header .tree-toggle');
            const children = node.querySelector(':scope > .tree-children');
            
            if (this.expandedPaths.has(path)) {
                toggle?.classList.add('expanded');
                children?.classList.add('expanded');
            } else {
                toggle?.classList.remove('expanded');
                children?.classList.remove('expanded');
            }
        }
    }
    
    selectPath(path) {
        // Update selection state
        const previousSelected = this.container.querySelector('.tree-node-header.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }
        
        this.selectedPath = path;
        
        if (path === null) {
            // Select "All Images"
            const allHeader = this.container.querySelector('.tree-node:first-child .tree-node-header');
            if (allHeader) {
                allHeader.classList.add('selected');
            }
        } else {
            const node = this.container.querySelector(`[data-path="${path}"]`);
            if (node) {
                const header = node.querySelector(':scope > .tree-node-header');
                header?.classList.add('selected');
            }
        }
        
        // Notify callback
        if (this.onSelect) {
            this.onSelect(path);
        }
    }
    
    collapseAll() {
        this.expandedPaths.clear();
        
        // Update DOM
        const toggles = this.container.querySelectorAll('.tree-toggle.expanded');
        const children = this.container.querySelectorAll('.tree-children.expanded');
        
        toggles.forEach(t => t.classList.remove('expanded'));
        children.forEach(c => c.classList.remove('expanded'));
    }
    
    expandPath(path) {
        if (!path) return;
        
        const parts = path.split('/');
        let currentPath = '';
        
        for (const part of parts) {
            currentPath = currentPath ? `${currentPath}/${part}` : part;
            this.expandedPaths.add(currentPath);
        }
        
        // Re-render to apply expansions
        // This could be optimized to just update the DOM
    }
}

// Export for use in other modules
window.SceneTree = SceneTree;
