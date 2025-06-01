import requests
from bs4 import BeautifulSoup
from llama_cpp import Llama
import time
import os
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import sys
import shutil
sys.path.append(os.getcwd()+'/models.py')
sys.path.append('../dia/dia/')
sys.path.append('../DiffSinger/')
sys.path.append('../MediaPipePyTorch/')
sys.path.append('../stable-diffusion-webui/extensions/deforum/scripts/')
from models import *
from deforum import *
from model import *

# ========== CONFIGURATION ==========
BLOG_SITES = [
    {
        "name": "Gemma Milburn",
        "url": "https://www.gemmamilburn.com/blog",
        "article_selector": "div.BlogList-item-content",
        "content_selector": "div.BlogList-item-content"
    },
    {
        "name": "30 Day Singer",
        "url": "https://www.30daysinger.com/blog",
        "article_selector": "div.blog-post-preview, div.blog-post",
        "content_selector": "div.blog-post-preview, div.blog-post"
    },
    {
        "name": "My Voice Coach",
        "url": "https://myvoicecoach.com/",
        "article_selector": "article, div.post, div.entry-content",
        "content_selector": "article, div.post, div.entry-content"
    },
    {
        "name": "The Naked Vocalist",
        "url": "https://www.thenakedvocalist.com/blog/",
        "article_selector": "div.post, article, div.entry-content",
        "content_selector": "div.post, article, div.entry-content"
    }
]

LLAMA_MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
DIFFSINGER_INFER_SCRIPT = r"../DiffSinger/scripts/infer.py"  # Path to DiffSinger inference script
DIFFSINGER_CONFIG = r"../DiffSinger/configs/base.yaml"  # Path to DiffSinger config
DIFFSINGER_CHECKPOINT = r"../DiffSinger/checkpoint.pth"  # Path to DiffSinger checkpoint
DEFORUM_MAIN_SCRIPT = r"../stable-diffusion-webui/extensions/deforum/scripts/deforum.py"  # Path to Deforum main script
AUDIO_MODEL_WEIGHTS = os.getcwd()+'/best_multilabel_audio_model.pth'
POSTURE_MODEL_WEIGHT = os.getcwd()+'/best_posture_estimator_model.pth'
quality_list = ['GRBAS Strain', 'breath_control', 'agility', 'stamina', 'phonation', 'resonance']
MPII_KEYPOINTS_MAT = os.getcwd()+'/mpii_human_pose_v1_sequences_keyframes.mat'
MPII_ANNOTATIONS = os.getcwd()+'/mpii_human_pose_v1_u12_1.mat'

# Output files
NARRATION_WAV = "narration.wav"
SINGING_WAV = "singing.wav"
LYRICS_TXT = "lyrics.txt"
DEFORUM_PROMPT_TXT = "deforum_prompt.txt"
POSTURE_MP4 = "posture.mp4"
COMBINED_AUDIO_WAV = "combined_audio.wav"
FINAL_VIDEO_MP4 = "singing_technique_demo.mp4"

PERFORMANCE_DB = "performances.json"

# ========== SCRAPING FUNCTIONS ==========
def scrape_blog(url, article_selector, content_selector, max_articles=1000):
    """Scrape articles from a blog given CSS selectors."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    articles = []
    for post in soup.select(article_selector)[:max_articles]:
        # Try to get title
        title_tag = post.find("h2") or post.find("h3")
        title = title_tag.get_text(strip=True) if title_tag else "No Title"
        # Get content
        content_tag = post.select_one(content_selector)
        content = content_tag.get_text(separator="\n", strip=True) if content_tag else post.get_text(separator="\n", strip=True)
        articles.append({"title": title, "content": content})
    return articles

# ========== LLAMA PROMPTING FUNCTION ==========
def generate_critique(llm, article_snippet, lyrics, performance_description):
    prompt = f'''
You are a vocal coach who has just read the following expert advice:
"""
{article_snippet}
"""
Now, critique the following singing performance, referencing the advice above where relevant.

Lyrics:
"{lyrics}"

Performance description:
"{performance_description}"

Critique:
'''
    return llm(prompt)

def save_performance(entry, db_path=PERFORMANCE_DB):
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append(entry)
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_performances(db_path=PERFORMANCE_DB):
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def retrieve_similar_performances(current_desc, n=3, db_path=PERFORMANCE_DB):
    performances = load_performances(db_path)
    if not performances:
        return []
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = [p["performance_description"] for p in performances]
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(current_desc, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=n)[0]
    return [performances[hit["corpus_id"]] for hit in hits]

def generate_critique_with_history(llm, article_snippet, lyrics, performance_description, past_performances):
    history = "\n\n".join(
        f"Date: {p.get('date','?')}\nLyrics: {p.get('lyrics','')}\nDescription: {p.get('performance_description','')}\nCritique: {p.get('critique','')}"
        for p in past_performances
    )
    prompt = f'''
You are a vocal coach who has just read the following expert advice:
"""
{article_snippet}
"""

Here are some past performances and critiques:
{history}

Now, critique the following singing performance, referencing the advice above and comparing to past performances where relevant.

Lyrics:
"{lyrics}"

Performance description:
"{performance_description}"

Critique:
'''
    return llm(prompt)

def generate_rag_critique(lyrics, performance_description, video_path=None):
    # 1. Scrape articles from all blogs
    all_articles = []
    for site in BLOG_SITES:
        articles = scrape_blog(site["url"], site["article_selector"], site["content_selector"])
        for art in articles:
            art["source"] = site["name"]
        all_articles.extend(articles)
        time.sleep(1)
    if not all_articles:
        return "No articles scraped. Exiting.", None
    
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
    llm = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH, torch_dtype="auto")
    past_performances = retrieve_similar_performances(performance_description, n=3)
    # Use the first article for context (or you can aggregate)
    article = all_articles[0]
    snippet = article["content"][:600]
    critique = generate_critique_with_history(llm, snippet, lyrics, performance_description, past_performances)
    entry = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "lyrics": lyrics,
        "performance_description": performance_description,
        "critique": critique,
    }
    if video_path:
        entry["video_path"] = video_path
    save_performance(entry)
    return critique, entry

# ========== MAIN PIPELINE ==========
def invoke_critique(lyrics, perf_description, video_path=None):
    # 1. Load the models and predict the performance quality frame-by-frame
    submitted_video = VideoFileClip(video_path)
    video_folder_name = (video_path.split('/')[-1]).split('.')[0]

    # Extract the audio
    audio = submitted_video.audio
    features_dir = os.getcwd()+'/'+video_folder_name+'/Audio Feats/'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir, exist_ok=True)
    mfcc_path = os.path.join(features_dir, f"{video_folder_name}_mfcc.npy")
    scatter_path = os.path.join(features_dir, f"{video_folder_name}_scatter.npy")
    if not os.path.exists(mfcc_path):
        extract_audio_features(audio_path, mfcc_path, use_scattering=False)
    if not os.path.exists(scatter_path):
        extract_audio_features(audio_path, scatter_path, use_scattering=True)
    features = prepare_features(features_dir)
    tensor_x = torch.tensor(features, dtype=torch.float32)
    input_dim = features.shape[1]
    audio_model = MultiLabelAudioRegressor(input_dim=input_dim, num_tasks=6, dropout=0.5)
    preds = get_prediction_regression(audio_model, tensor_x, 'cpu', quality_list)

    # Get posture keypoints

    # 2. Feed lyrics and performance description to RAG critique generator
    singing_lyrics = lyrics
    performance_description = perf_description

    # 3. Call the new RAG function for backward compatibility
    critique, entry = generate_rag_critique(singing_lyrics, performance_description, uploaded_video_path)
    print(f"Critique:\n{critique}\n")

    # 4. Generate Narration with Nari Dia
    print("Generating narration with Nari Dia...")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B")
    narration_audio = model.generate(critique_text)
    sf.write(NARRATION_WAV, narration_audio, 44100)

    # 5. Generate Singing Audio with DiffSinger
    print("Generating singing audio with DiffSinger...")
    if singing_lyrics.strip():
        with open(LYRICS_TXT, "w", encoding="utf-8") as f:
            f.write(singing_lyrics)
        diffsinger_cmd = (
            f"python \"{DIFFSINGER_INFER_SCRIPT}\" "
            f"--config \"{DIFFSINGER_CONFIG}\" "
            f"--checkpoint \"{DIFFSINGER_CHECKPOINT}\" "
            f"--input \"{LYRICS_TXT}\" "
            f"--output \"{SINGING_WAV}\""
        )
        print(f"Running: {diffsinger_cmd}")
        os.system(diffsinger_cmd)
        singing_clip = AudioFileClip(SINGING_WAV)
    else:
        print("No lyrics provided. Skipping singing audio generation.")
        singing_clip = None

    # 6. Generate posture guidance Video with Deforum Stable Diffusion
    deforum_prompt = (
    "A professional singer standing upright with relaxed shoulders and open jaw, "
    "demonstrating perfect singing posture, studio lighting, 4K, cinematic"
    )
    with open(DEFORUM_PROMPT_TXT, "w", encoding="utf-8") as f:
        f.write(deforum_prompt)

    deforum_cmd = (
        f"python \"{DEFORUM_MAIN_SCRIPT}\" "
        f"--prompt_file \"{DEFORUM_PROMPT_TXT}\" "
        f"--output_path \"{POSTURE_MP4}\" "
        f"--num_frames 120 "
        f"--width 512 --height 768 "
        f"--seed 42"
    )
    print(f"Running: {deforum_cmd}")
    os.system(deforum_cmd)

    # 7. Combine narration and singing audio
    print("Combining narration and singing audio...")
    narration_clip = AudioFileClip(NARRATION_WAV)
    if singing_clip:
        combined_audio = concatenate_audioclips([narration_clip, singing_clip])
    else:
        combined_audio = narration_clip
    combined_audio.write_audiofile(COMBINED_AUDIO_WAV)

    # 8. Combine with video
    print("Combining audio with video...")
    video_clip = VideoFileClip(POSTURE_MP4)
    if video_clip.duration < combined_audio.duration:
        n_loops = int(combined_audio.duration // video_clip.duration) + 1
        video_clip = video_clip.loop(n_loops=n_loops).subclip(0, combined_audio.duration)
    else:
        video_clip = video_clip.subclip(0, combined_audio.duration)

    final_video = video_clip.set_audio(AudioFileClip(COMBINED_AUDIO_WAV))
    final_video.write_videofile(FINAL_VIDEO_MP4, codec="libx264", audio_codec="aac")

    print(f"Done! Output: {FINAL_VIDEO_MP4}")

if __name__ == "__main__":
    main()