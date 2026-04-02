/**
 * Overlay Renderer for 2D and 3D Bounding Box Visualization on Canvas
 * 
 * Draws 2D bounding boxes (COCO format) and projected 3D bounding boxes
 * onto a canvas overlay on top of RGB images.
 */

class OverlayRenderer {
    constructor(canvasId, imgId) {
        this.canvas = document.getElementById(canvasId);
        this.img = document.getElementById(imgId);
        this.ctx = this.canvas.getContext('2d');
        
        // Default box color
        this.defaultColor = '#fbbf24';
        
        this.imageLoaded = false;
        this.intrinsics = null;
    }
    
    async loadImage(imagePath) {
        return new Promise((resolve, reject) => {
            this.img.onload = () => {
                this.imageLoaded = true;
                this.resizeCanvas();
                this.drawImage();
                resolve();
            };
            this.img.onerror = reject;
            this.img.src = imagePath;
        });
    }
    
    setIntrinsics(intrinsics) {
        this.intrinsics = intrinsics;
    }
    
    resizeCanvas() {
        if (!this.imageLoaded) return;
        
        const container = this.canvas.parentElement;
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        
        const imgRatio = this.img.naturalWidth / this.img.naturalHeight;
        const containerRatio = containerWidth / containerHeight;
        
        let drawWidth, drawHeight;
        if (imgRatio > containerRatio) {
            drawWidth = containerWidth;
            drawHeight = containerWidth / imgRatio;
        } else {
            drawHeight = containerHeight;
            drawWidth = containerHeight * imgRatio;
        }
        
        this.canvas.width = drawWidth;
        this.canvas.height = drawHeight;
        
        // Store scale factors for coordinate conversion
        this.scaleX = drawWidth / this.img.naturalWidth;
        this.scaleY = drawHeight / this.img.naturalHeight;
    }
    
    drawImage() {
        if (!this.imageLoaded) return;
        this.ctx.drawImage(this.img, 0, 0, this.canvas.width, this.canvas.height);
    }
    
    clear() {
        if (!this.canvas) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawImage();
    }
    
    /**
     * Draw a 2D bounding box.
     * Format: [x1, y1, x2, y2] - top-left and bottom-right corners
     */
    draw2DBox(bbox, color = null, label = null) {
        const [x1, y1, x2, y2] = bbox;
        
        const sx1 = x1 * this.scaleX;
        const sy1 = y1 * this.scaleY;
        const sx2 = x2 * this.scaleX;
        const sy2 = y2 * this.scaleY;
        const sw = sx2 - sx1;
        const sh = sy2 - sy1;
        
        this.ctx.strokeStyle = color || this.defaultColor;
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(sx1, sy1, sw, sh);
        
        if (label) {
            this.drawLabel(label, sx1, sy1 - 5, color || this.defaultColor);
        }
    }
    
    /**
     * Draw a 3D bounding box projected to 2D
     * @param {Array} box3d - 10D box format [cx,cy,cz,w,h,l,qw,qx,qy,qz]
     * @param {string} color - CSS color string (e.g. '#ff0000' or 'hsl(120, 70%, 60%)')
     * @param {string|null} label - Optional label text
     */
    draw3DBox(box3d, color = null, label = null) {
        if (!this.intrinsics) {
            console.warn('Intrinsics not set, cannot project 3D box');
            return;
        }

        const corners = this.box3dToCorners(box3d);
        const corners2d = corners.map(c => this.project3Dto2D(c));

        // Check if any corner is behind camera
        const valid = corners.every(c => c[2] > 0.1);
        if (!valid) return;

        this.ctx.strokeStyle = color || this.defaultColor;
        this.ctx.lineWidth = 2;

        // Draw edges
        const edges = [
            [0, 1], [1, 2], [2, 3], [3, 0], // bottom face
            [4, 5], [5, 6], [6, 7], [7, 4], // top face
            [0, 4], [1, 5], [2, 6], [3, 7]  // vertical edges
        ];
        
        for (const [i, j] of edges) {
            this.ctx.beginPath();
            this.ctx.moveTo(corners2d[i][0] * this.scaleX, corners2d[i][1] * this.scaleY);
            this.ctx.lineTo(corners2d[j][0] * this.scaleX, corners2d[j][1] * this.scaleY);
            this.ctx.stroke();
        }
        
        if (label) {
            const topCenter = [
                (corners2d[4][0] + corners2d[5][0] + corners2d[6][0] + corners2d[7][0]) / 4,
                Math.min(corners2d[4][1], corners2d[5][1], corners2d[6][1], corners2d[7][1])
            ];
            this.drawLabel(label, topCenter[0] * this.scaleX, topCenter[1] * this.scaleY - 5, color);
        }
    }
    
    /**
     * Project a 3D point to 2D image coordinates
     */
    project3Dto2D(point3d) {
        const [x, y, z] = point3d;
        
        const fx = this.intrinsics[0][0];
        const fy = this.intrinsics[1][1];
        const cx = this.intrinsics[0][2];
        const cy = this.intrinsics[1][2];
        
        const u = (fx * x / z) + cx;
        const v = (fy * y / z) + cy;
        
        return [u, v];
    }
    
    /**
     * Convert 10D box format to 8 corners
     */
    box3dToCorners(box3d) {
        const [cx, cy, cz, w, h, l, qw, qx, qy, qz] = box3d;
        
        // Rotation matrix from quaternion
        const R = this.quaternionToRotationMatrix(qw, qx, qy, qz);
        
        // Half dimensions
        const hw = w / 2;
        const hh = h / 2;
        const hl = l / 2;
        
        // Local corners
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
    
    drawLabel(text, x, y, bgColor) {
        this.ctx.font = '12px Inter, sans-serif';
        const metrics = this.ctx.measureText(text);
        const padding = 4;
        
        // Background
        this.ctx.fillStyle = bgColor;
        this.ctx.fillRect(
            x - padding,
            y - 14,
            metrics.width + padding * 2,
            18
        );
        
        // Text
        this.ctx.fillStyle = '#000000';
        this.ctx.fillText(text, x, y);
    }
    
    /**
     * Draw multiple 2D bounding boxes
     */
    draw2DBoxes(boxes, colors = null, labels = null) {
        this.clear();
        
        for (let i = 0; i < boxes.length; i++) {
            const color = colors ? colors[i] : null;
            const label = labels ? labels[i] : null;
            this.draw2DBox(boxes[i], color, label);
        }
    }
    
    /**
     * Draw multiple 3D bounding boxes
     * @param {Array} boxes - Array of {box3d, color, label} objects
     */
    draw3DBoxes(boxes) {
        this.clear();

        for (const boxData of boxes) {
            this.draw3DBox(
                boxData.box3d,
                boxData.color || null,
                boxData.label || null
            );
        }
    }
}

// Export for use in other modules
window.OverlayRenderer = OverlayRenderer;
