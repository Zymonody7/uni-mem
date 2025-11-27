import json
import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames
    
STRUCTURED_PROMPT_TEMPLATE = """You are an expert Chinese video analyst tasked with summarizing a short clip.
请结合提供的帧画面与实时字幕，输出一个 JSON，字段含义如下：
{{
  "chunk_summary": "中文描述，70-150字，强调人物/场景/动作",
  "scene_label": "2-4个英文小写词，概括场景，如 beach_feeding_gulls",
  "objects_detected": ["按重要性列出关键物体，单词或短语，使用英文"],
  "actions": ["关键动作动词或短语，英文"],
  "emotions": ["可选，人物情绪或氛围，英文"],
  "language": "字幕语言，两位或缩写，如 zh、en、mixed",
  "confidence": 0-1 之间的小数，表示你对描述准确性的信心",
  "notes": "可选补充信息，如镜头运动/光线/突发事件"
}}

务必只输出 JSON，不要多余文本或解释。
当前片段起止时间: {start:.2f}s - {end:.2f}s。
若字幕为空，请在 chunk_summary 中注明“无对白”并专注于视觉细节。
字幕内容:
{transcript}
"""


def _extract_json_from_response(raw_text: str) -> tuple[str, dict]:
    clean_text = raw_text.replace("<|endoftext|>", "").strip()
    fenced = re.search(r"```(?:json)?(.*?)```", clean_text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        candidate = clean_text
    candidate = candidate.replace("\u200b", "").strip()
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    else:
        candidate = ""

    metadata = {}
    if candidate:
        try:
            metadata = json.loads(candidate)
        except json.JSONDecodeError:
            metadata = {}

    return clean_text, metadata


def _ensure_metadata_defaults(metadata: dict, fallback_summary: str) -> dict:
    metadata = metadata or {}
    metadata.setdefault("chunk_summary", fallback_summary)
    metadata.setdefault("scene_label", "unknown_scene")
    metadata.setdefault("objects_detected", [])
    metadata.setdefault("actions", [])
    metadata.setdefault("emotions", [])
    metadata.setdefault("language", "zh")
    metadata.setdefault("confidence", 0.75)
    metadata.setdefault("notes", "")
    return metadata


def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:
        model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
        model.eval()
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                start_time, end_time = segment_times_info[index]["timestamp"]
                query = STRUCTURED_PROMPT_TEMPLATE.format(
                    start=start_time,
                    end=end_time,
                    transcript=segment_transcript or "（无可用字幕）",
                )
                msgs = [{'role': 'user', 'content': video_frames + [query]}]
                params = {}
                params["use_image_id"] = False
                params["max_slice_nums"] = 2
                segment_caption = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    **params
                )
                raw_text, parsed_metadata = _extract_json_from_response(segment_caption)
                normalized_metadata = _ensure_metadata_defaults(parsed_metadata, raw_text)
                caption_result[index] = {
                    "raw": normalized_metadata["chunk_summary"],
                    "metadata": normalized_metadata,
                }
                torch.cuda.empty_cache()
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    segment_total = len(segment_index2name)
    for index in segment_index2name:
        caption_entry = captions[index]
        if isinstance(caption_entry, dict):
            caption_text = caption_entry.get("raw", "")
            caption_metadata = caption_entry.get("metadata", {})
        else:
            caption_text = str(caption_entry)
            caption_metadata = {}

        transcript_text = transcripts[index]
        start_time, end_time = segment_times_info[index]["timestamp"]
        duration_seconds = float(max(end_time - start_time, 0.0))
        time_range = f"{start_time:.2f}-{end_time:.2f}"

        caption_metadata.setdefault("chunk_summary", caption_text)
        caption_metadata.setdefault("scene_label", "unknown_scene")
        caption_metadata.setdefault("objects_detected", [])
        caption_metadata.setdefault("language", "zh")
        caption_metadata.setdefault("confidence", 0.75)
        caption_metadata.setdefault("notes", "")
        caption_metadata["source_type"] = "video"
        caption_metadata["chunk_index"] = int(index)
        caption_metadata["chunk_count_estimate"] = segment_total
        caption_metadata["duration_seconds"] = duration_seconds
        caption_metadata["time_range"] = time_range
        caption_metadata["transcription_model"] = "faster-distil-whisper-large-v3"

        inserting_segments[index] = {
            "content": f"Caption:\n{caption_text}\nTranscript:\n{transcript_text}\n\n",
            "time": time_range,
            "transcript": transcript_text,
            "frame_times": segment_times_info[index]["frame_times"].tolist(),
            "duration_seconds": duration_seconds,
            "metadata": caption_metadata,
        }
    return inserting_segments
        
def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames):
    # model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
    # model.eval()
    
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        # query = f"The transcript of the current video:\n{segment_transcript}.\nGiven a question: {query}, you have to extract relevant information from the video and transcript for answering the question."
        query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
        msgs = [{'role': 'user', 'content': video_frames + [query]}]
        params = {}
        params["use_image_id"] = False
        params["max_slice_nums"] = 2
        segment_caption = caption_model.chat(
            image=None,
            msgs=msgs,
            tokenizer=caption_tokenizer,
            **params
        )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        torch.cuda.empty_cache()
    
    return caption_result