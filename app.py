# © 2025 Kazuki Higo
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See: https://polyformproject.org/licenses/noncommercial/1.0.0/

from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import base64
import torch
from PIL import Image
import io
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler, ControlNetModel
from torchvision import transforms
from src.pipeline_controlnet import TerraFusionControlNetPipeline

app = Flask(__name__)

#精度
WEIGHT_DTYPE = torch.float16

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

# Helper function to load models from local path or HuggingFace
def load_model(model_name, subfolder=None):
    local_path = os.path.join('model', subfolder)

    if os.path.exists(local_path):
        print(f"Loading {subfolder} from local path: {local_path}")
        if subfolder == "tokenizer":
            return CLIPTokenizer.from_pretrained(local_path)
        elif subfolder == "text_encoder":
            return CLIPTextModel.from_pretrained(local_path).to(device).to(WEIGHT_DTYPE)
        elif subfolder in ["texture_vae", "heightmap_vae"]:
            return AutoencoderKL.from_pretrained(local_path).to(device).to(WEIGHT_DTYPE)
        elif subfolder == "unet":
            return UNet2DConditionModel.from_pretrained(local_path).to(device).to(WEIGHT_DTYPE)
        elif subfolder == "scheduler":
            return DDPMScheduler.from_pretrained(local_path)
        else:
            return ControlNetModel.from_pretrained(local_path).to(device).to(WEIGHT_DTYPE)
    else:
        print(f"Loading {subfolder} from HuggingFace: Millennium-Nova/uncond-terrain-ldm")
        if subfolder == "tokenizer":
            return CLIPTokenizer.from_pretrained(f"Millennium-Nova/{model_name}", subfolder=subfolder)
        elif subfolder == "text_encoder":
            return CLIPTextModel.from_pretrained(f"Millennium-Nova/{model_name}", subfolder=subfolder).to(device).to(WEIGHT_DTYPE)
        elif subfolder in ["texture_vae", "heightmap_vae"]:
            return AutoencoderKL.from_pretrained(f"Millennium-Nova/{model_name}", subfolder=subfolder).to(device).to(WEIGHT_DTYPE)
        elif subfolder == "unet":
            return UNet2DConditionModel.from_pretrained(f"Millennium-Nova/{model_name}", subfolder=subfolder).to(device).to(WEIGHT_DTYPE)
        elif subfolder == "scheduler":
            return DDPMScheduler.from_pretrained(f"Millennium-Nova/{model_name}", subfolder=subfolder)
        else:
            return ControlNetModel.from_pretrained(f"Millennium-Nova/{model_name}").to(device).to(WEIGHT_DTYPE)
        
# Tokenizer and TextEncoder
tokenizer = load_model("uncond-terrain-ldm", "tokenizer")
text_encoder = load_model("uncond-terrain-ldm", "text_encoder")

# VAE
texture_vae = load_model("uncond-terrain-ldm", "texture_vae")
heightmap_vae = load_model("uncond-terrain-ldm", "heightmap_vae")

# UNet
unet = load_model("uncond-terrain-ldm", "unet")

# Noise scheduler
scheduler = load_model("uncond-terrain-ldm", "scheduler")

# ControlNet (optional, can be None)
controlnet = load_model("terra-fusion-controlnet", "controlnet")

# ===== Initialize pipeline =====
pipeline = TerraFusionControlNetPipeline(
    texture_vae=texture_vae,
    heightmap_vae=heightmap_vae,
    scheduler=scheduler,
    unet=unet,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    controlnet=controlnet,
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

@app.route('/sketch')
def sketch():
    return render_template('sketch.html')

@app.route('/sketches', methods=['GET'])
def list_sketches():
    """保存されたスケッチファイル一覧を返す"""
    try:
        sketch_files = []
        if os.path.exists(BASE_DATA_DIR):
            files = os.listdir(BASE_DATA_DIR)
            sketch_files = [f for f in files if f.startswith('sketch') and f.endswith('.png')]
            sketch_files.sort(reverse=True)  # 新しい順
        return jsonify({'sketches': sketch_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/infer', methods=['POST'])
def infer():
    """
    GPU推論エンドポイント
    リクエスト: { "num_inference_steps": 20, "sketch_image": "base64_png_data...", "save_sketch": true (optional) }
    レスポンス: { "heightmap": base64, "texture": base64, "width": w, "height": h }
    """
    debug_log("/infer endpoint called")
    try:
        debug_log("Parsing request data...")
        data = request.json
        debug_log(f"Received data keys: {data.keys()}")
        num_inference_steps = int(data.get('num_inference_steps', 20))
        sketch_base64 = data.get('sketch_image', None)
        save_sketch = data.get('save_sketch', False)
        seed = int(data.get('seed', 0))
        
        debug_log(f"sketch_image present: {sketch_base64 is not None}")
        if sketch_base64:
            debug_log(f"sketch_image length: {len(sketch_base64)}")
        debug_log(f"Seed value: {seed}")
        
        # プロンプト（固定）
        prompt = "A satellite terrain image."
        
        # スケッチ画像の前処理
        input_sketch = None
        if sketch_base64:
            debug_log("Processing sketch image...")
            try:
                # Base64デコード
                if ',' in sketch_base64:
                    sketch_base64 = sketch_base64.split(',')[1]
                sketch_bytes = base64.b64decode(sketch_base64)
                sketch_image = Image.open(io.BytesIO(sketch_bytes)).convert("RGB")
                debug_log(f"Sketch image loaded: {sketch_image.size}")
                
                # スケッチ画像を保存するか確認
                if save_sketch:
                    sketch_path = os.path.join(BASE_DATA_DIR, f"sketch_{int(torch.cuda.FloatTensor(1).uniform_() * 10000)}.png")
                    sketch_image.save(sketch_path)
                    debug_log(f"Sketch image saved to: {sketch_path}")
                
                # 前処理（inference_controlnet_higo.pyと同様）
                resolution = 512  # デフォルト解像度
                preprocess = transforms.Compose([
                    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                ])
                input_sketch = preprocess(sketch_image).unsqueeze(0).to(device)
                debug_log(f"Sketch preprocessed: {input_sketch.shape}")
            except Exception as e:
                debug_log(f"Error processing sketch: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Failed to process sketch image: {str(e)}'}), 400
        else:
            debug_log("No sketch image provided - this is an error!")
            return jsonify({'error': 'Sketch image is required but not provided'}), 400
        
        print(f"Starting inference with num_steps={num_inference_steps}, seed={seed}, sketch={input_sketch is not None}")
        debug_log(f"Device: {pipeline.device}")
        
        # 推論実行
        generator = torch.Generator(device=pipeline.device)
        if seed != 0:
            generator.manual_seed(seed)
            debug_log(f"Generator seed set to: {seed}")
        else:
            debug_log("Generator using random seed")
        debug_log("Generator created, starting pipeline...")
        
        with torch.no_grad():
            debug_log(f"Scheduler type: {type(pipeline.scheduler)}")
            debug_log(f"Scheduler config: {pipeline.scheduler.config}")
            outputs = pipeline(
                prompt=[prompt],
                batch_size=1,
                num_inference_steps=num_inference_steps,
                make_viz=True,
                generator=generator,
                height_scale=8000,
                image=input_sketch
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
        # 0～8000（height_scale）として出力されているため、正規化する
        # ここでは単純に0-1の範囲に正規化
        hgt_min = np.percentile(hgt_int16, 1)
        hgt_max = np.percentile(hgt_int16, 99)
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
        
        # パーセンタイルの標高値を計算（メートル単位のint16から）
        elevation_p2 = float(hgt_min)  # 既に計算済み
        elevation_p98 = float(hgt_max)  # 既に計算済み
        debug_log(f"Elevation range: {elevation_p2:.1f}m - {elevation_p98:.1f}m")
        
        # Viz画像の処理（あれば）
        viz_base64 = None
        if outputs.viz_images is not None and len(outputs.viz_images) > 0:
            viz_np = outputs.viz_images[0]  # numpy array (H, W, 3) uint8
            viz_img = Image.fromarray(viz_np)  # numpy -> PIL Image
            buffered = io.BytesIO()
            viz_img.save(buffered, format="PNG")
            viz_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            debug_log(f"Viz image encoded: {len(viz_base64)} chars")
        
        debug_log("Sending response...")
        
        return jsonify({
            'width': int(width),
            'height': int(height),
            'texture': tex_base64,
            'heightmap': hgt_base64,
            'dtype': 'float32',
            'viz_image': viz_base64,
            'elevation_min': round(elevation_p2, 1),
            'elevation_max': round(elevation_p98, 1)
        })
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)