// © 2025 Kazuki Higo
// Licensed under the PolyForm Noncommercial License 1.0.0.
// See: https://polyformproject.org/licenses/noncommercial/1.0.0/

// Lightweight sketch pad for RGB line drawing on a black canvas.
// Exposes a global SketchPad class for use from any page.
(function () {
  class SketchPad {
    constructor(canvas, options = {}) { // 初期化関数
      this.canvas = typeof canvas === 'string' ? document.getElementById(canvas) : canvas;
      if (!this.canvas) {
        throw new Error('SketchPad: canvas element is required');
      }
      this.ctx = this.canvas.getContext('2d'); // 2D描画コンテキストを取得

      const size = options.size || 800;
      this.canvas.width = options.width || size;
      this.canvas.height = options.height || size;
      this.background = options.background || '#000000';
      this.penWidth = options.penWidth || 2;
      this.historyLimit = options.historyLimit || 32;

      this.currentColor = options.initialColor || '#ff0000';
      this.isDrawing = false;
      this.lastPoint = null;
      this.history = [];

      this.reset(true); // キャンバスを黒で塗りつぶす
      this.bindEvents(); // ウスやタッチの操作を「監視」する仕組みを設定
    }

    bindEvents() {
      this.handlePointerDown = (e) => {
        e.preventDefault();
        this.isDrawing = true;
        this.lastPoint = this.getPoint(e);
        this.pushHistory();
      };

      this.handlePointerMove = (e) => {
        if (!this.isDrawing) return;
        const point = this.getPoint(e);
        this.drawLine(this.lastPoint, point);
        this.lastPoint = point;
      };

      this.handlePointerUp = () => {
        this.isDrawing = false;
        this.lastPoint = null;
      };

      const c = this.canvas;
      c.addEventListener('pointerdown', this.handlePointerDown);
      c.addEventListener('pointermove', this.handlePointerMove);
      window.addEventListener('pointerup', this.handlePointerUp);
    }

    destroy() {
      const c = this.canvas;
      c.removeEventListener('pointerdown', this.handlePointerDown);
      c.removeEventListener('pointermove', this.handlePointerMove);
      window.removeEventListener('pointerup', this.handlePointerUp);
    }

    getPoint(event) {
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
    }

    drawLine(from, to) {
      const ctx = this.ctx;
      ctx.lineWidth = this.penWidth;
      ctx.lineCap = 'round';
      ctx.strokeStyle = this.currentColor;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }

    pushHistory() {
      const snapshot = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
      this.history.push(snapshot);
      if (this.history.length > this.historyLimit) {
        this.history.shift();
      }
    }

    undo() {
      if (this.history.length === 0) return;
      this.history.pop();
      const prev = this.history[this.history.length - 1];
      if (prev) {
        this.ctx.putImageData(prev, 0, 0);
      } else {
        this.reset(true);
      }
    }

    reset(skipHistory = false) {
      this.ctx.fillStyle = this.background;
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.strokeStyle = this.currentColor;
      if (!skipHistory) {
        this.history = [];
      }
      this.pushHistory();
    }

    setColor(color) {
      this.currentColor = color;
    }

    setPenWidth(width) {
      this.penWidth = width;
    }

    save(filename = 'sketch_result.png') {
      const url = this.canvas.toDataURL('image/png');
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();
    }

    toDataURL() {
      return this.canvas.toDataURL('image/png');
    }

    toBlob(callback) {
      this.canvas.toBlob(callback, 'image/png');
    }

    getImageData() {
      return this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    }
  }

  window.SketchPad = SketchPad;
})();
