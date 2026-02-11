// © 2025 Kazuki Higo
// Licensed under the PolyForm Noncommercial License 1.0.0.
// See: https://polyformproject.org/licenses/noncommercial/1.0.0/

// デバッグモード設定
const DEBUG_MODE = true;

function debugLog(message) {
    if (DEBUG_MODE) {
        console.log(`[DEBUG] ${message}`);
    }
}

let scene, camera, renderer, controls;
let terrainMesh, terrainGroup;
let isRotating = false;
let sketchWindow = null;
let currentSketchData = null;
let overlayMesh = null;
let sketchOverlayTexture = null;
let sketchOverlaySource = null;

// データ管理
let rawHeightData = null;      // サーバーから受け取った生の高さデータ
let processedHeightData = null; // スムージング適用後の高さデータ
let currentTexture = null;     // ロードしたテクスチャ画像

let currentWidth = 0;
let currentHeight = 0;
let baseScale = 20.0; // 正規化された(0~1)標高をどれくらいの高さとして表示するか

document.addEventListener('DOMContentLoaded', async function() {
    initThreeJS();
    
    // スケッチ一覧を読み込む
    loadSketchList();

    // num_steps スライダー
    document.getElementById('num-steps').addEventListener('input', function(e) {
        const val = parseInt(e.target.value);
        document.getElementById('val-steps').textContent = val;
    });

    // Load Sketch ボタン
    document.getElementById('load-sketch-button').addEventListener('click', async function() {
        const selector = document.getElementById('sketch-selector');
        const filename = selector.value;
        if (!filename) {
            alert('Please select a sketch from the list.');
            return;
        }
        
        try {
            // 保存されたスケッチを読み込む
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = function() {
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                currentSketchData = canvas.toDataURL('image/png');
                console.log('Loaded sketch:', filename);
                document.getElementById('sketch-button').style.backgroundColor = '#FF9800';
                document.getElementById('sketch-button').textContent = `${filename} Loaded \u2713`;
                updateSketchOverlay();
            };
            img.src = `/static/data/${filename}`;
        } catch (err) {
            console.error('Error loading sketch:', err);
            alert('Failed to load sketch: ' + err.message);
        }
    });

    // Sketch ボタン
    document.getElementById('sketch-button').addEventListener('click', function() {
        if (sketchWindow && !sketchWindow.closed) {
            sketchWindow.focus();
            return;
        }
        sketchWindow = window.open('/sketch', 'SketchPad', 'width=1000,height=900');
    });

    // スケッチウィンドウからのメッセージを受け取る
    window.addEventListener('message', function(event) {
        if (event.data && event.data.type === 'sketch-complete') {
            currentSketchData = event.data.imageData;
            console.log('Sketch data received, starting generation...');
            document.getElementById('sketch-button').style.backgroundColor = '#FF9800';
            document.getElementById('sketch-button').textContent = 'Sketch Ready \u2713';

            updateSketchOverlay();

            // 自動的に生成開始
            generateTerrain();
        }
    });

    // Generate ボタン
    document.getElementById('generate-button').addEventListener('click', generateTerrain);
    
    // 背景色
    document.getElementById('bg-color-selector').addEventListener('change', function(e) {
        const isWhite = e.target.value === 'white';
        scene.background = new THREE.Color(isWhite ? 0xffffff : 0x000000);
        const loader = document.getElementById('loading');
        loader.style.color = isWhite ? 'black' : 'white';
        loader.style.textShadow = isWhite ? 'none' : '1px 1px 2px black';
    });

    // テクスチャ ON/OFF
    document.getElementById('texturing-toggle').addEventListener('change', updateMaterial);

    // スケッチ線オーバーレイ ON/OFF
    document.getElementById('sketch-overlay-toggle').addEventListener('change', updateSketchOverlay);

    // ワイヤーフレーム
    document.getElementById('wireframe-toggle').addEventListener('change', function(e) {
        if (terrainMesh) {
            terrainMesh.material.wireframe = e.target.checked;
        }
    });

    // 高さスケール
    document.getElementById('displacement-scale').addEventListener('input', function(e) {
        const val = parseFloat(e.target.value);
        document.getElementById('val-scale').textContent = val.toFixed(1);
        updateGeometryHeight(); // 再描画
    });

    // 平滑化
    document.getElementById('blur-kernel').addEventListener('input', function(e) {
        const val = parseInt(e.target.value);
        document.getElementById('val-blur').textContent = val + " px";
        
        // カーネルサイズが変わったら再計算
        if (rawHeightData) {
            applySmoothing(val);
            updateGeometryHeight();
        }
    });

    // 7. 照明
    document.getElementById('light-intensity').addEventListener('input', function(e) {
        const val = parseFloat(e.target.value);
        document.getElementById('val-light').textContent = val.toFixed(1);
        const light = scene.getObjectByName('dirLight');
        if (light) light.intensity = val;
    });

    // 8. 回転
    document.getElementById('toggle-rotation').addEventListener('click', function() {
        isRotating = !isRotating;
        this.textContent = isRotating ? 'Pause Rotation' : 'Resume Rotation';
    });
});

async function generateTerrain() {
    const numSteps = parseInt(document.getElementById('num-steps').value);
    const saveSketch = document.getElementById('save-sketch-toggle').checked;
    const seed = parseInt(document.getElementById('seed-input').value);
    
    debugLog(`Generate button clicked, num_steps: ${numSteps}, seed: ${seed}, has sketch: ${!!currentSketchData}, save: ${saveSketch}`);
    
    // プログレス表示を初期化
    document.getElementById('loading').style.display = 'block';
    document.getElementById('loading-message').textContent = 'Initializing...';
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    document.getElementById('generate-button').disabled = true;

    // 推定時間計算（1 step 約 0.08秒）
    const estimatedTimeMs = numSteps * 80;
    const updateInterval = 100; // 100msごとに更新
    let elapsedTime = 0;
    let progressPercent = 0;
    
    // プログレス更新用タイマー
    const progressTimer = setInterval(() => {
        elapsedTime += updateInterval;
        progressPercent = Math.min(95, (elapsedTime / estimatedTimeMs) * 100);
        
        document.getElementById('progress-bar').style.width = progressPercent + '%';
        document.getElementById('progress-text').textContent = Math.round(progressPercent) + '%';
        
        // メッセージ更新
        if (progressPercent < 30) {
            document.getElementById('loading-message').textContent = 'Generating...';
        } else if (progressPercent < 70) {
            document.getElementById('loading-message').textContent = 'Processing...';
        } else {
            document.getElementById('loading-message').textContent = 'Finalizing...';
        }
    }, updateInterval);

    // リセット処理
    if (terrainGroup) {
        scene.remove(terrainGroup);
        if (overlayMesh) {
            overlayMesh.material.dispose();
            overlayMesh = null;
        }
        if (sketchOverlayTexture) {
            sketchOverlayTexture.dispose();
            sketchOverlayTexture = null;
            sketchOverlaySource = null;
        }
        if (terrainMesh) { 
            terrainMesh.geometry.dispose(); 
            terrainMesh.material.dispose(); 
        }
        terrainMesh = null;
        terrainGroup = null;
    }

    try {
        debugLog("Sending fetch request to /infer...");
        const requestBody = { 
            num_inference_steps: numSteps,
            save_sketch: saveSketch,
            seed: seed
        };
        
        // スケッチデータがあれば追加
        if (currentSketchData) {
            requestBody.sketch_image = currentSketchData;
            debugLog("Including sketch image in request");
        }
        
        debugLog(`Request body: ${JSON.stringify({...requestBody, sketch_image: currentSketchData ? 'data...' : null})}`);
        
        const res = await fetch('/infer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(requestBody)
        });
        if (!res.ok) throw new Error('Server error');
        const data = await res.json();
        debugLog(`Response data: width=${data.width}, height=${data.height}, format=${data.format}`);

        currentWidth = data.width;
        currentHeight = data.height;

        // --- Texture ロード（PNG画像として） ---
        const texImage = new Image();
        texImage.onload = function() {
            currentTexture = new THREE.Texture(texImage);
            currentTexture.needsUpdate = true;
            debugLog("Texture loaded from PNG");
        };
        texImage.src = 'data:image/png;base64,' + data.texture;

        // --- Heightmap ロード（PNG画像として、uint8グレースケール） ---
        const hgtImage = new Image();
        await new Promise((resolve, reject) => {
            hgtImage.onload = function() {
                // Canvas経由でピクセルデータを取得
                const canvas = document.createElement('canvas');
                canvas.width = data.width;
                canvas.height = data.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(hgtImage, 0, 0);
                
                const imageData = ctx.getImageData(0, 0, data.width, data.height);
                const pixels = imageData.data;
                
                // uint8グレースケール（0-255）をfloat (0-1)に変換
                rawHeightData = new Float32Array(data.width * data.height);
                for (let i = 0; i < data.width * data.height; i++) {
                    const rgbaIndex = i * 4;
                    rawHeightData[i] = pixels[rgbaIndex] / 255.0; // R成分を使用（グレースケールなのでR=G=B）
                }
                
                debugLog("Heightmap loaded from PNG");
                resolve();
            };
            hgtImage.onerror = reject;
            hgtImage.src = 'data:image/png;base64,' + data.heightmap;
        });
        
        document.getElementById('info-width').textContent = data.width;
        document.getElementById('info-height').textContent = data.height;
        document.getElementById('info-res').textContent = `${data.width} x ${data.height}`;
        
        // 標高範囲表示（パーセンタイル2%-98%）
        if (data.elevation_min !== undefined && data.elevation_max !== undefined) {
            document.getElementById('info-elevation').textContent = `${data.elevation_min}m - ${data.elevation_max}m`;
        } else {
            document.getElementById('info-elevation').textContent = '-';
        }

        // 表示用データを初期化
        processedHeightData = new Float32Array(rawHeightData);

        // --- スムージングの初期適用 ---
        const kernelSize = parseInt(document.getElementById('blur-kernel').value);
        if (kernelSize > 1) {
            applySmoothing(kernelSize);
        }

        // --- メッシュ生成 ---
        createTerrainMesh();

        updateSketchOverlay();

        // --- 画像表示 ---
        const imageDisplayContainer = document.getElementById('image-display-container');
        
        // スケッチ画像の表示
        if (currentSketchData) {
            const sketchDisplay = document.getElementById('sketch-display');
            const sketchImg = document.getElementById('sketch-img');
            sketchImg.src = currentSketchData;
            sketchDisplay.style.display = 'flex';
            debugLog("Displaying sketch image");
        } else {
            document.getElementById('sketch-display').style.display = 'none';
        }
        
        // Viz画像の表示
        if (data.viz_image) {
            const vizDisplay = document.getElementById('viz-display');
            const vizImg = document.getElementById('viz-img');
            vizImg.src = 'data:image/png;base64,' + data.viz_image;
            vizDisplay.style.display = 'flex';
            debugLog("Displaying viz image");
        } else {
            document.getElementById('viz-display').style.display = 'none';
        }
        
        // 少なくとも1つの画像がある場合はコンテナを表示
        if (currentSketchData || data.viz_image) {
            imageDisplayContainer.style.display = 'block';
        }
        
        // プログレスバーを100%に
        clearInterval(progressTimer);
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-text').textContent = '100%';
        document.getElementById('loading-message').textContent = 'Complete!';
        
        // 少し待ってから非表示
        setTimeout(() => {
            document.getElementById('loading').style.display = 'none';
        }, 500);

    } catch (err) {
        console.error("Error generating terrain:", err);
        alert('Error generating terrain: ' + err.message);
        clearInterval(progressTimer);
    } finally {
        debugLog("Cleaning up UI...");
        document.getElementById('generate-button').disabled = false;
    }
}

/**
 * 高さデータの平滑化処理 (Box Blur)
 * rawHeightData を読み込み、計算結果を processedHeightData に書き込む
 */
function applySmoothing(kernelSize) {
    if (!rawHeightData || kernelSize <= 1) {
        // コピーして終了
        processedHeightData.set(rawHeightData);
        return;
    }

    const w = currentWidth;
    const h = currentHeight;
    const half = Math.floor(kernelSize / 2);

    // 計算負荷軽減のため、単純な移動平均(Box Blur)を実装
    // 注意: JSでの重いループはUIをブロックする可能性があるため、
    // 画像サイズが大きい場合は本来Workerなどが望ましいが、ここでは直接計算する。
    
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let sum = 0;
            let count = 0;

            // カーネル範囲ループ
            for (let ky = -half; ky <= half; ky++) {
                for (let kx = -half; kx <= half; kx++) {
                    const nx = x + kx;
                    const ny = y + ky;

                    // 範囲内チェック
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        const idx = ny * w + nx;
                        sum += rawHeightData[idx];
                        count++;
                    }
                }
            }
            processedHeightData[y * w + x] = sum / count;
        }
    }
}

function createTerrainMesh() {
    terrainGroup = new THREE.Group();

    // セグメント数 (最大512)
    const maxSegs = 512;
    const wSegs = Math.min(currentWidth - 1, maxSegs);
    const hSegs = Math.min(currentHeight - 1, maxSegs);
    
    const aspectRatio = currentHeight / currentWidth;
    const sizeX = 100;
    const sizeY = 100 * aspectRatio;

    const geometry = new THREE.PlaneGeometry(sizeX, sizeY, wSegs, hSegs);
    
    // マテリアル作成
    const material = new THREE.MeshStandardMaterial({ 
        side: THREE.DoubleSide,
        wireframe: false,
        roughness: 0.8
    });
    material.wireframe = document.getElementById('wireframe-toggle').checked;

    terrainMesh = new THREE.Mesh(geometry, material);
    terrainGroup.add(terrainMesh);
    terrainGroup.rotation.x = -Math.PI / 2;
    scene.add(terrainGroup);

    // テクスチャ設定を適用
    updateMaterial();
    
    // 高さ適用
    updateGeometryHeight();

    updateSketchOverlay();
}

/**
 * テクスチャのON/OFF切り替え
 */
function updateMaterial() {
    if (!terrainMesh) return;
    
    const useTexture = document.getElementById('texturing-toggle').checked;
    
    if (useTexture && currentTexture) {
        terrainMesh.material.map = currentTexture;
        terrainMesh.material.color.setHex(0xffffff); // テクスチャ本来の色
    } else {
        terrainMesh.material.map = null;
        terrainMesh.material.color.setHex(0xcccccc); // グレー
    }
    terrainMesh.material.needsUpdate = true;
}

/**
 * ジオメトリの高さを更新する (processedHeightData を使用)
 */
function updateGeometryHeight() {
    if (!terrainMesh || !processedHeightData) return;

    const scale = parseFloat(document.getElementById('displacement-scale').value);
    const geo = terrainMesh.geometry;
    const pos = geo.attributes.position;
    const vertexCount = pos.count;
    
    const gridX = geo.parameters.widthSegments + 1;
    const gridY = geo.parameters.heightSegments + 1;

    for (let i = 0; i < vertexCount; i++) {
        const gx = i % gridX;
        const gy = Math.floor(i / gridX);

        const imgX = Math.floor((gx / (gridX - 1)) * (currentWidth - 1));
        const imgY = Math.floor((gy / (gridY - 1)) * (currentHeight - 1));

        const index = imgY * currentWidth + imgX;
        
        // processedHeightData (スムージング済み) を使用
        let h = 0;
        if (processedHeightData[index] !== undefined) {
            h = processedHeightData[index] * scale * baseScale;
        }
        pos.setZ(i, h);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
}

function buildSketchOverlayTexture(dataUrl) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);

            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const pixels = imageData.data;
            const threshold = 5;

            for (let i = 0; i < pixels.length; i += 4) {
                const r = pixels[i];
                const g = pixels[i + 1];
                const b = pixels[i + 2];
                if (r <= threshold && g <= threshold && b <= threshold) {
                    pixels[i + 3] = 0;
                } else {
                    pixels[i + 3] = 255;
                }
            }

            ctx.putImageData(imageData, 0, 0);
            const texture = new THREE.Texture(canvas);
            texture.needsUpdate = true;
            resolve(texture);
        };
        img.onerror = reject;
        img.src = dataUrl;
    });
}

function updateSketchOverlay() {
    if (!terrainMesh || !terrainGroup) return;

    const enabled = document.getElementById('sketch-overlay-toggle').checked;
    if (!enabled || !currentSketchData) {
        if (overlayMesh) {
            overlayMesh.visible = false;
        }
        return;
    }

    if (overlayMesh && sketchOverlaySource === currentSketchData) {
        overlayMesh.visible = true;
        return;
    }

    buildSketchOverlayTexture(currentSketchData)
        .then((texture) => {
            if (sketchOverlayTexture) {
                sketchOverlayTexture.dispose();
            }
            sketchOverlayTexture = texture;
            sketchOverlaySource = currentSketchData;

            if (!overlayMesh) {
                const overlayMaterial = new THREE.MeshBasicMaterial({
                    map: sketchOverlayTexture,
                    transparent: true,
                    depthWrite: false,
                    polygonOffset: true,
                    polygonOffsetFactor: -1,
                    polygonOffsetUnits: -1
                });
                overlayMesh = new THREE.Mesh(terrainMesh.geometry, overlayMaterial);
                overlayMesh.renderOrder = 1;
                terrainGroup.add(overlayMesh);
            } else {
                overlayMesh.material.map = sketchOverlayTexture;
                overlayMesh.material.needsUpdate = true;
                overlayMesh.visible = true;
            }
        })
        .catch((err) => {
            console.error('Error creating sketch overlay:', err);
        });
}

function initThreeJS() {
    const container = document.body;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);

    const width = window.innerWidth;
    const height = window.innerHeight;
    const uiWidth = 320; // UIパネルの幅 + マージン

    camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
    camera.position.set(0, 100, 150);
    camera.lookAt(0, 0, 0);
    
    // 地形を右に寄せるためのオフセット設定
    // (全体幅, 全体高さ, 描画開始X, 描画開始Y, 有効幅, 有効高さ)
    camera.setViewOffset(width, height, -uiWidth / 2, 0, width, height);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(50, 100, 50);
    dirLight.name = 'dirLight';
    scene.add(dirLight);
    
    const ambLight = new THREE.AmbientLight(0x404040);
    scene.add(ambLight);

    animate();

    window.addEventListener('resize', () => {
        const w = window.innerWidth;
        const h = window.innerHeight;
        camera.aspect = w / h;
        camera.setViewOffset(w, h, -uiWidth / 2, 0, w, h);
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
}

function animate() {
    requestAnimationFrame(animate);
    if (isRotating && terrainGroup) {
        terrainGroup.rotation.z += 0.005;
    }
    controls.update();
    renderer.render(scene, camera);
}

/**
 * 保存されたスケッチ一覧を取得してドロップダウンに追加
 */
async function loadSketchList() {
    try {
        const response = await fetch('/sketches');
        const data = await response.json();
        
        const selector = document.getElementById('sketch-selector');
        // 既存のオプション（"-- Select saved sketch --"）以外をクリア
        selector.innerHTML = '<option value="">-- Select saved sketch --</option>';
        
        if (data.sketches && data.sketches.length > 0) {
            data.sketches.forEach(filename => {
                const option = document.createElement('option');
                option.value = filename;
                option.textContent = filename;
                selector.appendChild(option);
            });
            debugLog(`Loaded ${data.sketches.length} saved sketches`);
        } else {
            debugLog('No saved sketches found');
        }
    } catch (err) {
        console.error('Error loading sketch list:', err);
    }
}
