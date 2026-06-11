import sys
import os
import pysrt
import torch
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
)
from PySide6.QtCore import Qt, QThread, Signal
from qfluentwidgets import (
    FluentWindow, SubtitleLabel, PushButton, RadioButton,
    LineEdit, ProgressBar, TextEdit, MessageBox, TitleLabel,
    BodyLabel
)
from moviepy import VideoFileClip, concatenate_videoclips

# Optional imports for AI
try:
    from transformers import AutoProcessor, AutoModel
    import numpy as np
    import torch
    import torch.nn.functional as F
    from qwen_vl_utils import process_vision_info
except ImportError:
    pass

class VideoProcessorThread(QThread):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, video_path, srt_path, output_path, fallback_to_timeline):
        super().__init__()
        self.video_path = video_path
        self.srt_path = srt_path
        self.output_path = output_path
        self.fallback_to_timeline = fallback_to_timeline
        self._is_cancelled = False

    def run(self):
        try:
            self.log.emit("Đang tải file SRT...")
            subs = pysrt.open(self.srt_path)
            
            self.log.emit("Đang tải video gốc...")
            video = VideoFileClip(self.video_path)
            total_duration = video.duration
            
            clips = []
            
            if True:
                self.log.emit("Đang tải mô hình Qwen3-VL-Embedding-2B...")
                try:
                    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-2B", trust_remote_code=True)
                    model = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B", trust_remote_code=True, torch_dtype=torch.float16)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = model.to(device)
                    model.eval()
                    self.log.emit(f"Đã tải mô hình lên {device.upper()}. Đang phân tích toàn bộ video (có thể mất thời gian)...")
                    
                    # Bước 1: Chia nhỏ video và lấy embedding của từng đoạn (Video Indexing)
                    chunk_duration = 5.0 # Mỗi đoạn 5 giây
                    video_embeddings = []
                    chunk_times = []
                    
                    # Tính toán tổng số chunk
                    num_chunks = int(total_duration / chunk_duration)
                    if num_chunks == 0: num_chunks = 1
                    
                    for c_idx in range(num_chunks):
                        if self._is_cancelled: return
                        
                        c_start = c_idx * chunk_duration
                        c_end = min(c_start + chunk_duration, total_duration)
                        chunk_times.append(c_start)
                        
                        # Trích xuất 1 frame giữa đoạn để đại diện
                        mid_time = c_start + (c_end - c_start) / 2
                        
                        self.log.emit(f"Đang dùng CPU trích xuất frame tại {mid_time:.1f}s...")
                        frame = video.get_frame(mid_time)
                        
                        self.log.emit(f"Đang dùng GPU phân tích frame {c_idx+1}/{num_chunks}...")
                        # Chuyển đổi frame (numpy array) thành input cho model
                        # Tuỳ model Qwen3-VL-Embedding yêu cầu, ta giả định truyền ảnh PIL hoặc numpy
                        from PIL import Image
                        pil_img = Image.fromarray(frame)
                        
                        # Tạo input message cho frame video
                        messages = [
                            {"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": "Describe this image."}]}
                        ]
                        
                        # Tuỳ chỉnh tuỳ theo Qwen3-VL-Embedding thực tế
                        try:
                            # Parse vision info
                            image_inputs, video_inputs = process_vision_info(messages)
                            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
                            
                            with torch.no_grad():
                                # Giả sử output có 'text_embeds' hoặc 'image_embeds'
                                # Hoặc gọi trực tiếp hàm extract embedding
                                outputs = model(**inputs)
                                # Placeholder: tuỳ thuộc output name của Qwen3-VL-Embedding, giả sử là last_hidden_state
                                embed = outputs.last_hidden_state.mean(dim=1) if hasattr(outputs, 'last_hidden_state') else outputs.image_embeds
                                embed = F.normalize(embed, p=2, dim=1)
                                video_embeddings.append(embed)
                        except Exception as e:
                            self.log.emit(f"Cảnh báo: Lỗi embed video chunk {c_idx}: {e}")
                            # Thêm vector 0 để tránh lệch index
                            video_embeddings.append(torch.zeros((1, 768)).to(device))
                            
                        self.progress.emit(int(((c_idx + 1) / num_chunks) * 30))
                    
                    # Ghép toàn bộ tensor video embeddings lại thành ma trận
                    if len(video_embeddings) > 0:
                        video_embeddings_matrix = torch.cat(video_embeddings, dim=0)
                    else:
                        video_embeddings_matrix = None
                        
                except Exception as e:
                    self.log.emit(f"Lỗi tải mô hình AI: {e}")
                    self.finished.emit(False, "Không thể tải hoặc chạy mô hình AI. Vui lòng kiểm tra lại.")
                    return

            total_subs = len(subs)
            for i, sub in enumerate(subs):
                if self._is_cancelled:
                    self.log.emit("Đã hủy quá trình.")
                    break
                
                text = sub.text.replace('\n', ' ')
                start_sec = sub.start.ordinal / 1000.0
                end_sec = sub.end.ordinal / 1000.0
                duration = end_sec - start_sec
                
                self.log.emit(f"[{i+1}/{total_subs}] Xử lý: '{text}' ({duration:.2f}s)")
                
                self.log.emit(f"AI đang tìm kiếm: '{text}'...")
                
                match_found = False
                match_start = start_sec
                
                if video_embeddings_matrix is not None:
                    try:
                        # Nhúng câu SRT
                        messages_text = [
                            {"role": "user", "content": [{"type": "text", "text": text}]}
                        ]
                        text_prompt = processor.apply_chat_template(messages_text, tokenize=False, add_generation_prompt=True)
                        text_inputs = processor(text=[text_prompt], padding=True, return_tensors="pt").to(device)
                        
                        with torch.no_grad():
                            outputs_text = model(**text_inputs)
                            text_embed = outputs_text.last_hidden_state.mean(dim=1) if hasattr(outputs_text, 'last_hidden_state') else outputs_text.text_embeds
                            text_embed = F.normalize(text_embed, p=2, dim=1)
                        
                        # Tính cosine similarity với tất cả các đoạn video
                        similarities = (video_embeddings_matrix @ text_embed.T).squeeze(1)
                        best_idx = torch.argmax(similarities).item()
                        best_score = similarities[best_idx].item()
                        
                        self.log.emit(f"-> Độ khớp cao nhất: {best_score:.3f} tại {chunk_times[best_idx]:.1f}s")
                        
                        # Đặt ngưỡng tối thiểu để coi là "tìm thấy" (ví dụ > 0.2, điều chỉnh tuỳ model)
                        if best_score > 0.15:
                            match_found = True
                            match_start = chunk_times[best_idx]
                    except Exception as e:
                        self.log.emit(f"-> Lỗi khi embed text: {e}")
                
                if match_found:
                    match_end = min(match_start + duration, total_duration)
                    if match_end > match_start:
                        clip = video.subclipped(match_start, match_end)
                        clips.append(clip)
                        self.log.emit(f"-> Đã cắt đoạn khớp AI ({match_start:.2f}s - {match_end:.2f}s)")
                else:
                    if self.fallback_to_timeline:
                        end_cut = min(end_sec, total_duration)
                        if end_cut > start_sec:
                            clip = video.subclipped(start_sec, end_cut)
                            clips.append(clip)
                            self.log.emit(f"-> AI không thấy. Cắt bù theo timeline ({start_sec:.2f}s - {end_cut:.2f}s)")
                    else:
                        self.log.emit(f"-> AI không thấy. Bỏ qua.")
                
                # Tiến trình chạy từ 30% đến 80% cho đoạn này
                self.progress.emit(30 + int(((i + 1) / total_subs) * 50))

                self.progress.emit(int(((i + 1) / total_subs) * 50))

            if clips and not self._is_cancelled:
                self.log.emit("Đang ghép các đoạn video lại với nhau...")
                final_video = concatenate_videoclips(clips)
                self.log.emit(f"Đang xuất file video ra {self.output_path}...")
                final_video.write_videofile(self.output_path, codec="libx264", audio_codec="aac", logger=None)
                self.progress.emit(100)
                self.finished.emit(True, "Đã xử lý xong!")
            elif not clips:
                self.finished.emit(False, "Không có đoạn video nào được cắt.")
                
            video.close()
            
        except Exception as e:
            self.log.emit(f"Lỗi: {str(e)}")
            self.finished.emit(False, f"Đã xảy ra lỗi: {str(e)}")

    def cancel(self):
        self._is_cancelled = True

class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Srt Video Matcher - Qwen3-VL-Embedding")
        self.resize(800, 600)

        # Widget chính
        self.main_widget = QWidget()
        self.main_widget.setObjectName("mainWidget")
        self.layout = QVBoxLayout(self.main_widget)
        self.layout.setContentsMargins(30, 30, 30, 30)
        self.layout.setSpacing(15)

        # Tiêu đề
        title = TitleLabel("Trình Ghép Nối Video & SRT Bằng AI")
        self.layout.addWidget(title)

        # Input Video
        self.layout.addWidget(BodyLabel("1. Chọn file Video gốc:"))
        row_video = QHBoxLayout()
        self.video_input = LineEdit()
        self.video_input.setPlaceholderText("Đường dẫn file video...")
        btn_video = PushButton("Duyệt...")
        btn_video.clicked.connect(self.browse_video)
        row_video.addWidget(self.video_input)
        row_video.addWidget(btn_video)
        self.layout.addLayout(row_video)

        # Input SRT
        self.layout.addWidget(BodyLabel("2. Chọn file SRT:"))
        row_srt = QHBoxLayout()
        self.srt_input = LineEdit()
        self.srt_input.setPlaceholderText("Đường dẫn file srt...")
        btn_srt = PushButton("Duyệt...")
        btn_srt.clicked.connect(self.browse_srt)
        row_srt.addWidget(self.srt_input)
        row_srt.addWidget(btn_srt)
        self.layout.addLayout(row_srt)

        # Output
        self.layout.addWidget(BodyLabel("3. Chọn nơi lưu Video xuất ra:"))
        row_out = QHBoxLayout()
        self.out_input = LineEdit()
        self.out_input.setPlaceholderText("Đường dẫn video đầu ra...")
        btn_out = PushButton("Duyệt...")
        btn_out.clicked.connect(self.browse_out)
        row_out.addWidget(self.out_input)
        row_out.addWidget(btn_out)
        self.layout.addLayout(row_out)

        # Modes
        self.layout.addWidget(BodyLabel("4. Chế độ xử lý:"))
        self.radio_mode1 = RadioButton("Chế độ 1: AI tìm kiếm. Nếu không có -> Cắt bù theo timeline SRT (Giữ nguyên tổng thời gian)")
        self.radio_mode2 = RadioButton("Chế độ 2: AI tìm kiếm. Nếu không có -> Bỏ qua (Video có thể ngắn hơn SRT)")
        self.radio_mode1.setChecked(True) # Mặc định
        self.layout.addWidget(self.radio_mode1)
        self.layout.addWidget(self.radio_mode2)

        # Tiến trình & Log
        self.progress_bar = ProgressBar()
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.log_area = TextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(self.log_area)

        # Nút bắt đầu và dừng
        row_btn = QHBoxLayout()
        self.btn_start = PushButton("Bắt Đầu")
        self.btn_start.clicked.connect(self.start_processing)
        
        self.btn_stop = PushButton("Dừng Lại")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)
        
        row_btn.addWidget(self.btn_start)
        row_btn.addWidget(self.btn_stop)
        self.layout.addLayout(row_btn)

        self.addSubInterface(self.main_widget, "home", "Home")

        self.thread = None

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn Video", "", "Video Files (*.mp4 *.mkv *.avi)")
        if path:
            self.video_input.setText(path)

    def browse_srt(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn SRT", "", "Subtitle Files (*.srt)")
        if path:
            self.srt_input.setText(path)

    def browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Lưu Video", "output.mp4", "Video Files (*.mp4)")
        if path:
            self.out_input.setText(path)

    def append_log(self, text):
        self.log_area.append(text)
        # Tự động cuộn xuống
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def on_finished(self, success, msg):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Bắt Đầu")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Dừng Lại")
        if success:
            MessageBox("Hoàn tất", msg, self.window()).exec()
        else:
            MessageBox("Thông báo", msg, self.window()).exec()

    def start_processing(self):
        v_path = self.video_input.text().strip()
        s_path = self.srt_input.text().strip()
        o_path = self.out_input.text().strip()

        if not os.path.exists(v_path):
            self.append_log("Lỗi: Không tìm thấy file video.")
            return
        if not os.path.exists(s_path):
            self.append_log("Lỗi: Không tìm thấy file srt.")
            return
        if not o_path:
            self.append_log("Lỗi: Chưa đặt tên file lưu.")
            return

        fallback_to_timeline = self.radio_mode1.isChecked()

        self.btn_start.setEnabled(False)
        self.btn_start.setText("Đang xử lý...")
        self.btn_stop.setEnabled(True)
        self.btn_stop.setText("Dừng Lại")
        self.progress_bar.setValue(0)
        self.log_area.clear()

        self.thread = VideoProcessorThread(v_path, s_path, o_path, fallback_to_timeline)
        self.thread.log.connect(self.append_log)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()

    def stop_processing(self):
        if self.thread and self.thread.isRunning():
            self.thread.cancel()
            self.btn_stop.setEnabled(False)
            self.btn_stop.setText("Đang dừng...")
            self.append_log("Đang yêu cầu dừng tiến trình... Xin chờ.")

    def closeEvent(self, e):
        if self.thread and self.thread.isRunning():
            self.thread.cancel()
            self.thread.wait()
        super().closeEvent(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
