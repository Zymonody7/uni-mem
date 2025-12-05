import os
import re
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):

    # 加载 faster-whisper 模型（自动 GPU）
    model = WhisperModel(
        "/root/models/faster-whisper-large-v3-turbo",   # <-- 换成你的真正模型目录
        device="cuda",
        compute_type="float16",
    )
    model.logger.setLevel(logging.WARNING)

    cache_path = os.path.join(working_dir, "_cache", video_name)

    transcripts = {}
    languages = {}

    # -------------------------------
    # 第一部分：检测整支视频的主语言（只检测一次）
    # -------------------------------
    detected_video_language = None

    for index in segment_index2name:
        audio_file = os.path.join(cache_path, f"{segment_index2name[index]}.{audio_output_format}")

        if not os.path.exists(audio_file):
            continue

        # 只检测语言，不完整转录
        try:
          _, info = model.transcribe(
              audio_file,
              beam_size=1,
              vad_filter=True,
              task="transcribe",
              language=None,   # 自动检测
              condition_on_previous_text=False
          )
          detected_video_language = getattr(info, "language", None)
        except Exception as e:
          print(f"[ASR] Warning: Failed to transcribe {audio_file} for language detection. Error: {e}")
          # continue
          detected_video_language = None
          continue
        if info and info.language:
            detected_video_language = info.language
            break

    # 如果没有检测到，就默认为自动（None）
    if detected_video_language:
        print(f"[ASR] Detected video language: {detected_video_language}")
    else:
        print("[ASR] Could not detect video language. Using auto mode.")

    # -------------------------------
    # 第二部分：逐段转写
    # -------------------------------
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):

        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        if not os.path.exists(audio_file):
            transcripts[index] = ""
            languages[index] = "unknown"
            continue

        # ASR 推理
        segments, info = model.transcribe(
            audio_file,
            task="transcribe",
            language=detected_video_language,  # 若检测到语言，则固定它（更稳定）
            beam_size=5,
            best_of=5,
            vad_filter=True,        # 防止空段
            condition_on_previous_text=False
        )

        # -------------------------------
        # 生成时间戳 + 文本
        # -------------------------------
        result = []
        raw_text_for_lang = []  # 累计文本用于判断语言

        for seg in segments:
            text = seg.text.strip()
            result.append(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {text}")
            raw_text_for_lang.append(text)

        transcript_text = "\n".join(result)
        transcripts[index] = transcript_text

        # -------------------------------
        # 语言判断（中/英）
        # -------------------------------
        full_text = "".join(raw_text_for_lang)

        # 是否包含中文？
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", full_text))

        # whisper info.language 可能是 zh/en/ja 等
        detected_lang = info.language if info else None

        if has_chinese:
            languages[index] = "zh"
        elif detected_lang:
            if detected_lang.startswith("zh"):
                languages[index] = "zh"
            else:
                languages[index] = detected_lang
        else:
            languages[index] = "unknown"

    return transcripts, languages
