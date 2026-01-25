# © 2025 Kazuki Higo
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import base64
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from src.pipeline import TerraFusionPipeline

app = Flask(__name__)

# パスの指定
BASE_DATA_DIR = os.path.join('static', 'data')
HEIGHTMAP_DIR = os.path.join(BASE_DATA_DIR, 'heightmap')
TEX_DIR = os.path.join(BASE_DATA_DIR, 'texture')

os.makedirs(HEIGHTMAP_DIR, exist_ok=True)
os.makedirs(TEX_DIR, exist_ok=True)

# ===== Initialize models on startup =====
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tokenizer and TextEncoder
tokenizer = CLIPTokenizer.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="text_encoder").to(device)

# VAE
texture_vae = AutoencoderKL.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="texture_vae").to(device)
heightmap_vae = AutoencoderKL.from_pretrained("Millennium-Nova/terrafusion-heightmap-vae").to(device)

# UNet
unet = UNet2DConditionModel.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="unet").to(device)

# Noise scheduler
scheduler = DDPMScheduler.from_pretrained("Millennium-Nova/uncond-terrain-ldm", subfolder="scheduler")

# ===== Initialize pipeline =====
pipeline = TerraFusionPipeline(
    texture_vae=texture_vae,
    heightmap_vae=heightmap_vae,
    scheduler=scheduler,
    unet=unet,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
)
pipeline.to(device)
print("Models loaded successfully.")

# デバッグ用ヘルパー関数
def debug_log(message):
    """デバッグモードの時だけログを出力"""
    if app.debug:
        print(f"[DEBUG] {message}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/infer', methods=['POST'])
def infer():
    """
    GPU推論エンドポイント
    リクエスト: { "num_inference_steps": 20 }
    レスポンス: { "heightmap": base64, "texture": base64, "width": w, "height": h }
    """
    debug_log("/infer endpoint called")
    try:
        debug_log("Parsing request data...")
        data = request.json
        debug_log(f"Received data: {data}")
        num_inference_steps = int(data.get('num_inference_steps', 20))
        
        # プロンプト（固定）
        prompt = "A satellite terrain image."
        
        print(f"Starting inference with num_steps={num_inference_steps}")
        debug_log(f"Device: {pipeline.device}")
        
        # 推論実行
        generator = torch.Generator(device=pipeline.device)
        debug_log("Generator created, starting pipeline...")
        
        with torch.no_grad():
            outputs = pipeline(
                prompt=[prompt],
                batch_size=1,
                num_inference_steps=num_inference_steps,
                make_viz=False,
                generator=generator,
            )
        
        debug_log("Pipeline completed successfully")
        
        # 出力: uint8 (0-255) と int16 を float32 (0-1) に変換
        tex_uint8 = outputs.textures[0]  # (H, W, 3) uint8
        hgt_int16 = outputs.heightmaps[0]  # (H, W) int16
        
        debug_log(f"Output shapes - Texture: {tex_uint8.shape}, Heightmap: {hgt_int16.shape}")
        
        height, width = tex_uint8.shape[:2]
        
        # Texture: uint8 -> float32 (0-1)
        tex_float32 = tex_uint8.astype(np.float32) / 255.0
        
        # Heightmap: int16 -> float32 (0-1)
        # int16は通常-32768～32767だが、pipelineの出力は0～positive range
        # 0～2000（height_scale）として出力されているため、正規化する
        # ここでは単純に0-1の範囲に正規化
        hgt_min = np.min(hgt_int16)
        hgt_max = np.max(hgt_int16)
        if hgt_max > hgt_min:
            hgt_float32 = (hgt_int16.astype(np.float32) - hgt_min) / (hgt_max - hgt_min)
        else:
            hgt_float32 = np.zeros_like(hgt_int16, dtype=np.float32)
        
        # Base64エンコード
        debug_log("Starting base64 encoding...")
        tex_bytes = tex_float32.astype(np.float32).tobytes()
        tex_base64 = base64.b64encode(tex_bytes).decode('utf-8')
        
        hgt_bytes = hgt_float32.astype(np.float32).tobytes()
        hgt_base64 = base64.b64encode(hgt_bytes).decode('utf-8')
        
        debug_log(f"Encoded data sizes - Texture: {len(tex_base64)} chars, Heightmap: {len(hgt_base64)} chars")
        debug_log("Sending response...")
        
        return jsonify({
            'width': int(width),
            'height': int(height),
            'texture': tex_base64,
            'heightmap': hgt_base64,
            'dtype': 'float32'
        })
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)