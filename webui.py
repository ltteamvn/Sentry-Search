import os
import sys
import tempfile
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pysrt
from moviepy import VideoFileClip, concatenate_videoclips
import gradio as gr

# Optional imports for AI
try:
    from transformers import AutoProcessor, AutoModel
    from qwen_vl_utils import process_vision_info
    HAS_AI = True
except ImportError:
    HAS_AI = False

def process_video_srt_core(video_path, srt_path, mode, threshold, model_name, progress):
    log_messages = []
    
    def log(msg):
        log_messages.append(msg)
        print(msg)
        return "\n".join(log_messages)

    # Kiểm tra sự tồn tại của các tệp tin cục bộ
    if not os.path.exists(video_path):
        yield log(f"Lỗi: Không tìm thấy file Video tại đường dẫn: {video_path}"), None
        return
    if not os.path.exists(srt_path):
        yield log(f"Lỗi: Không tìm thấy file phụ đề SRT tại đường dẫn: {srt_path}"), None
        return

    yield log("Đang tải file SRT..."), None
    try:
        subs = pysrt.open(srt_path)
    except Exception as e:
        yield log(f"Lỗi đọc file SRT: {e}"), None
        return

    yield log("Đang tải video gốc..."), None
    try:
        video = VideoFileClip(video_path)
        total_duration = video.duration
    except Exception as e:
        yield log(f"Lỗi đọc file Video: {e}"), None
        return

    clips = []
    video_embeddings_matrix = None
    processor = None
    model = None
    device = "cpu"

    if HAS_AI:
        yield log(f"Đang tải mô hình {model_name} (có thể mất vài phút lần đầu)..."), None
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            yield log(f"Sử dụng thiết bị phần cứng: {device.upper()}"), None
            
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            # Dùng float16 nếu có CUDA để tiết kiệm bộ nhớ và chạy nhanh hơn
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype)
            model = model.to(device)
            model.eval()
            
            yield log("Đã tải mô hình AI thành công. Bắt đầu phân tích video (Video Indexing)..."), None
            
            chunk_duration = 5.0  # Mỗi đoạn 5 giây
            video_embeddings = []
            chunk_times = []
            
            num_chunks = int(total_duration / chunk_duration)
            if num_chunks == 0: 
                num_chunks = 1
            
            for c_idx in progress.tqdm(range(num_chunks), desc="Đang phân tích các phân cảnh video"):
                c_start = c_idx * chunk_duration
                c_end = min(c_start + chunk_duration, total_duration)
                chunk_times.append(c_start)
                
                mid_time = c_start + (c_end - c_start) / 2
                frame = video.get_frame(mid_time)
                
                pil_img = Image.fromarray(frame)
                # Giới hạn kích thước ảnh tối đa là 480px để tránh lỗi tràn bộ nhớ VRAM (CUDA OOM) trên card 4GB
                pil_img.thumbnail((480, 480), Image.Resampling.LANCZOS)
                
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": pil_img}, 
                        {"type": "text", "text": "Describe this image."}
                    ]}
                ]
                
                try:
                    image_inputs, video_inputs = process_vision_info(messages)
                    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(
                        text=[text_prompt], 
                        images=image_inputs, 
                        videos=video_inputs, 
                        padding=True, 
                        return_tensors="pt"
                    ).to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        embed = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs.image_embeds
                        embed = F.normalize(embed, p=2, dim=1)
                        video_embeddings.append(embed)
                except Exception as e:
                    yield log(f"Cảnh báo: Lỗi embed video chunk {c_idx}: {e}"), None
                    # Xác định số chiều vector dựa trên mô hình
                    embed_dim = 2048 if "8b" in model_name.lower() else 768
                    video_embeddings.append(torch.zeros((1, embed_dim)).to(device))
            
            if len(video_embeddings) > 0:
                video_embeddings_matrix = torch.cat(video_embeddings, dim=0)
                yield log(f"Đã hoàn thành Indexing {len(video_embeddings)} chunks."), None

            # Nhúng phụ đề hàng loạt để tăng tốc
            text_embeddings_matrix = None
            if video_embeddings_matrix is not None:
                try:
                    yield log("Đang nhúng toàn bộ phụ đề bằng AI (Batching)..."), None
                    all_texts = [sub.text.replace('\n', ' ') for sub in subs]
                    text_embeddings = []
                    batch_size = 64
                    
                    for b_idx in range(0, len(all_texts), batch_size):
                        batch_texts = all_texts[b_idx:b_idx+batch_size]
                        text_prompts = []
                        for t in batch_texts:
                            messages_text = [{"role": "user", "content": [{"type": "text", "text": t}]}]
                            prompt = processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
                            text_prompts.append(prompt)
                        
                        text_inputs = processor(text=text_prompts, padding=True, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs_text = model(**text_inputs)
                            batch_embeds = outputs_text.last_hidden_state.mean(dim=1) if hasattr(outputs_text, 'last_hidden_state') else outputs_text.text_embeds
                            batch_embeds = F.normalize(batch_embeds, p=2, dim=1)
                            text_embeddings.append(batch_embeds)
                    
                    if len(text_embeddings) > 0:
                        text_embeddings_matrix = torch.cat(text_embeddings, dim=0)
                        yield log(f"Đã nhúng xong {len(all_texts)} câu phụ đề."), None
                except Exception as e:
                    yield log(f"Cảnh báo: Lỗi khi nhúng phụ đề hàng loạt: {e}. Sẽ chạy ở chế độ nhúng từng câu."), None
            
        except Exception as e:
            yield log(f"Cảnh báo: Lỗi khởi tạo hoặc chạy mô hình AI: {e}. Sẽ chạy ở chế độ fallback không có AI."), None
    else:
        yield log("Cảnh báo: Thư viện AI (transformers/qwen_vl_utils) không khả dụng. Sẽ cắt theo timeline của phụ đề."), None

    total_subs = len(subs)
    fallback_to_timeline = (mode == "AI search + fallback to timeline")

    for i, sub in enumerate(subs):
        text = sub.text.replace('\n', ' ')
        start_sec = sub.start.ordinal / 1000.0
        end_sec = sub.end.ordinal / 1000.0
        duration = end_sec - start_sec
        
        yield log(f"[{i+1}/{total_subs}] Đang tìm kiếm: '{text}' ({duration:.2f}s)"), None
        
        match_found = False
        match_start = start_sec
        
        if video_embeddings_matrix is not None and processor is not None and model is not None:
            try:
                # Sử dụng ma trận nhúng sẵn nếu có
                if text_embeddings_matrix is not None:
                    text_embed = text_embeddings_matrix[i:i+1]
                else:
                    messages_text = [
                        {"role": "user", "content": [{"type": "text", "text": text}]}
                    ]
                    text_prompt = processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
                    text_inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs_text = model(**text_inputs)
                        text_embed = outputs_text.last_hidden_state.mean(dim=1) if hasattr(outputs_text, 'last_hidden_state') else outputs_text.text_embeds
                        text_embed = F.normalize(text_embed, p=2, dim=1)
                
                similarities = (video_embeddings_matrix @ text_embed.T).squeeze(1)
                best_idx = torch.argmax(similarities).item()
                best_score = similarities[best_idx].item()
                
                yield log(f"  -> Độ khớp cao nhất: {best_score:.3f} tại {chunk_times[best_idx]:.1f}s"), None
                
                if best_score > threshold:
                    match_found = True
                    match_start = chunk_times[best_idx]
            except Exception as e:
                yield log(f"  -> Lỗi so khớp AI: {e}"), None
        
        if match_found:
            match_end = min(match_start + duration, total_duration)
            if match_end > match_start:
                clip = video.subclipped(match_start, match_end)
                clips.append(clip)
                yield log(f"  -> ĐÃ KHỚP: Cắt đoạn AI ({match_start:.2f}s - {match_end:.2f}s)"), None
        else:
            if fallback_to_timeline or video_embeddings_matrix is None:
                end_cut = min(end_sec, total_duration)
                if end_cut > start_sec:
                    clip = video.subclipped(start_sec, end_cut)
                    clips.append(clip)
                    yield log(f"  -> KHÔNG KHỚP: Cắt theo timeline gốc ({start_sec:.2f}s - {end_cut:.2f}s)"), None
            else:
                yield log("  -> KHÔNG KHỚP: Bỏ qua phân cảnh này"), None

    if clips:
        yield log("Đang tiến hành ghép các đoạn video..."), None
        try:
            final_video = concatenate_videoclips(clips)
            
            # Tạo đường dẫn lưu tạm file đầu ra
            temp_output = os.path.join(tempfile.gettempdir(), "sentrysearch_output.mp4")
            yield log(f"Đang xuất file video ra {temp_output}..."), None
            
            final_video.write_videofile(
                temp_output, 
                codec="libx264", 
                audio_codec="aac", 
                logger=None
            )
            
            video.close()
            yield log("Hoàn thành xử lý video thành công!"), temp_output
        except Exception as e:
            yield log(f"Lỗi xuất video: {e}"), None
            if video:
                video.close()
    else:
        yield log("Lỗi: Không có phân cảnh video nào được cắt ghép."), None
        if video:
            video.close()

def process_video_srt_direct(video_file, srt_file, mode, threshold, model_name, progress=gr.Progress()):
    if video_file is None or srt_file is None:
        yield "Lỗi: Vui lòng tải lên cả file Video và file phụ đề SRT.", None
        return
    video_path = video_file.name if hasattr(video_file, 'name') else video_file
    srt_path = srt_file.name if hasattr(srt_file, 'name') else srt_file
    
    for l_val, v_val in process_video_srt_core(video_path, srt_path, mode, threshold, model_name, progress):
        yield l_val, v_val

def process_video_srt_local(video_path_str, srt_path_str, mode, threshold, model_name, progress=gr.Progress()):
    video_path = video_path_str.strip() if video_path_str else ""
    srt_path = srt_path_str.strip() if srt_path_str else ""
    if not video_path or not srt_path:
        yield "Lỗi: Vui lòng nhập đầy đủ đường dẫn Video và phụ đề SRT cục bộ.", None
        return
        
    for l_val, v_val in process_video_srt_core(video_path, srt_path, mode, threshold, model_name, progress):
        yield l_val, v_val

# Custom CSS for modern premium glassmorphism aesthetic
custom_css = """
body {
    background-color: #0b0f19;
    color: #f3f4f6;
    font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.gradio-container {
    max-width: 1000px !important;
    background: radial-gradient(circle at top right, rgba(29, 78, 216, 0.15), transparent), 
                radial-gradient(circle at bottom left, rgba(139, 92, 246, 0.1), transparent);
}
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #9ca3af;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.glass-panel {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}
.btn-primary {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4) !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet")) as demo:
    gr.HTML("<h1 class='main-title'>SentrySearch WebUI</h1>")
    gr.HTML("<p class='subtitle'>Đối chiếu và ghép nối Video & SRT bằng Trí tuệ Nhân tạo Qwen3-VL</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="glass-panel"):
                gr.Markdown("### 1. Đầu vào & Cấu hình")
                
                # Biến cấu hình dùng chung đặt bên ngoài Tab để tránh xung đột
                model_input = gr.Dropdown(
                    choices=["Qwen/Qwen3-VL-Embedding-2B", "Qwen/Qwen3-VL-Embedding-8B"],
                    value="Qwen/Qwen3-VL-Embedding-2B",
                    label="Mô hình AI sử dụng (AI Model)"
                )
                
                mode_input = gr.Radio(
                    choices=["AI search + fallback to timeline", "AI search only"],
                    value="AI search + fallback to timeline",
                    label="Chế độ xử lý"
                )
                
                threshold_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.15,
                    step=0.01,
                    label="Ngưỡng tin cậy (Confidence Threshold)"
                )
                
                # Các Tab chứa phương thức tải tệp và các nút bấm riêng biệt
                with gr.Tab("Tải tệp tin trực tiếp từ máy"):
                    video_input = gr.Video(label="Chọn file Video gốc", sources=["upload"])
                    srt_input = gr.File(label="Chọn file phụ đề SRT", file_types=[".srt"])
                    submit_btn_direct = gr.Button("Bắt Đầu Xử Lý (Trực tiếp)", variant="primary", elem_classes="btn-primary")
                
                with gr.Tab("Nhập đường dẫn trên Colab (Khuyên dùng cho tệp lớn)"):
                    gr.Markdown("*Mẹo: Hãy tải file lên thanh bên trái (Files) của Colab, sau đó chuột phải chọn **Copy path** và dán vào đây.*")
                    video_path_input = gr.Textbox(
                        label="Đường dẫn file Video gốc trên Colab", 
                        placeholder="Ví dụ: /content/video.mp4",
                        value=""
                    )
                    srt_path_input = gr.Textbox(
                        label="Đường dẫn file phụ đề SRT trên Colab", 
                        placeholder="Ví dụ: /content/sub.srt",
                        value=""
                    )
                    submit_btn_local = gr.Button("Bắt Đầu Xử Lý (Đường dẫn cục bộ)", variant="primary", elem_classes="btn-primary")

        with gr.Column(scale=1):
            with gr.Group(elem_classes="glass-panel"):
                gr.Markdown("### 2. Kết quả & Nhật ký hoạt động")
                video_output = gr.Video(label="Video kết quả đã ghép nối")
                log_output = gr.Code(label="Nhật ký hoạt động (Logs)", language="python", lines=15)

    # Nút bấm 1: Chỉ lấy dữ liệu từ các file tải lên trực tiếp (Không kiểm tra trạng thái upload của Tab 2)
    submit_btn_direct.click(
        fn=process_video_srt_direct,
        inputs=[video_input, srt_input, mode_input, threshold_input, model_input],
        outputs=[log_output, video_output]
    )

    # Nút bấm 2: Chỉ lấy dữ liệu từ các Textbox đường dẫn (Không kiểm tra trạng thái upload của Tab 1)
    submit_btn_local.click(
        fn=process_video_srt_local,
        inputs=[video_path_input, srt_path_input, mode_input, threshold_input, model_input],
        outputs=[log_output, video_output]
    )

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)
