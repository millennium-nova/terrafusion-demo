from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
import base64

app = Flask(__name__)

# パスの指定
BASE_DATA_DIR = os.path.join('static', 'data')
HEIGHTMAP_DIR = os.path.join(BASE_DATA_DIR, 'heightmap')
TEX_DIR = os.path.join(BASE_DATA_DIR, 'texture')

os.makedirs(HEIGHTMAP_DIR, exist_ok=True)
os.makedirs(TEX_DIR, exist_ok=True)

# 対応拡張子
HEIGHTMAP_EXTS = ('.tif', '.tiff', '.png')
TEX_EXTS = ('.png', '.jpg', '.jpeg')

def find_image_pairs():
    """
    Heightmapフォルダを基準に走査し、Textureフォルダから同名の画像を探す
    """
    pairs = {}
    if os.path.exists(HEIGHTMAP_DIR):
        for f in os.listdir(HEIGHTMAP_DIR):
            if f.lower().endswith(HEIGHTMAP_EXTS):
                base_name = os.path.splitext(f)[0]
                pairs[base_name] = {'heightmap': f, 'texture': None}

    if os.path.exists(TEX_DIR):
        for f in os.listdir(TEX_DIR):
            if f.lower().endswith(TEX_EXTS):
                base_name = os.path.splitext(f)[0]
                if base_name in pairs:
                    pairs[base_name]['texture'] = f
    return pairs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images')
def list_images():
    pairs = find_image_pairs()
    return jsonify(sorted(list(pairs.keys())))

@app.route('/generate', methods=['POST'])
def generate():
    target_name = request.json.get('name')
    pairs = find_image_pairs()
    
    if target_name not in pairs:
        return jsonify({'error': 'Dataset not found'}), 404

    file_info = pairs[target_name]
    
    # Heightmap 読み込み
    heightmap_filename = file_info['heightmap']
    heightmap_path = os.path.join(HEIGHTMAP_DIR, heightmap_filename)
    
    # 標高データは画像のデータを保持
    heightmap_img = cv2.imread(heightmap_path, cv2.IMREAD_UNCHANGED)
    
    if heightmap_img is None:
        return jsonify({'error': 'Failed to load Heightmap file'}), 500

    # 1チャンネル(グレースケール)に強制
    if len(heightmap_img.shape) == 3:
        # BGRならグレースケールへ (通常heightmapならどれか1chでも良いが)
        heightmap_img = cv2.cvtColor(heightmap_img, cv2.COLOR_BGR2GRAY)

    # 標高シフトと正規化
    # 計算用にfloat32に変換
    height_data_float = heightmap_img.astype(np.float32)

    # 生データの値を保存
    orig_min = float(np.min(height_data_float))
    orig_max = float(np.max(height_data_float))

    # 外れ値を除外した範囲を計算 (1%〜99%)
    min_p = np.percentile(height_data_float, 1)
    max_p = np.percentile(height_data_float, 99)
    height_range = max_p - min_p

    # 全ての地形を0~1にスケール
    # ただし、起伏が極端に小さい(平坦な)場合に異常に引き伸ばされないようマージンを設定
    # これにより、標高差がマージン以下の地形は、平坦さが維持される
    scaling_margin = 200.0
    denominator = max(height_range, scaling_margin)

    # シフトと正規化
    height_data_float -= min_p
    height_data_float /= denominator
    
    # 0〜1の範囲にクリップ
    height_data_float = np.clip(height_data_float, 0, 1.0)

    heightmap_img = height_data_float
    # ========================================================

    height, width = heightmap_img.shape[:2]

    # テクスチャ読み込みと整形
    tex_filename = file_info['texture']
    
    if tex_filename:
        tex_path = os.path.join(TEX_DIR, tex_filename)
        tex_img = cv2.imread(tex_path, cv2.IMREAD_COLOR)
        
        if tex_img is not None:
            tex_img = cv2.cvtColor(tex_img, cv2.COLOR_BGR2RGB)
            # ロバスト性: サイズ不一致ならリサイズ
            if tex_img.shape[:2] != (height, width):
                tex_img = cv2.resize(tex_img, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            tex_img = np.full((height, width, 3), 128, dtype=np.uint8)
    else:
        tex_img = np.full((height, width, 3), 200, dtype=np.uint8)

    # 3. データ転送
    _, buffer_tex = cv2.imencode('.png', cv2.cvtColor(tex_img, cv2.COLOR_RGB2BGR))
    texture_base64 = base64.b64encode(buffer_tex).decode('utf-8')

    heightmap_bytes = heightmap_img.tobytes()
    heightmap_base64 = base64.b64encode(heightmap_bytes).decode('utf-8')
    
    dtype_str = str(heightmap_img.dtype)

    return jsonify({
        'width': width,
        'height': height,
        'texture': texture_base64,
        'heightmap_raw': heightmap_base64,
        'dtype': dtype_str,
        'orig_min': orig_min,
        'orig_max': orig_max
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)