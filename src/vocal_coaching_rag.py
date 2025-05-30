import requests
from bs4 import BeautifulSoup
from llama_cpp import Llama
import time
import os
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip
from moviepy.audio.AudioClip import concatenate_audioclips
from dia.model import Dia
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json
import sys
import shutil
sys.path.append(os.getcwd()+'/models.py')
sys.path.append('../MediaPipePyTorch/')
from models import *
from blazepose import *
from blazepose_landmark import *

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
    # 1. Load critiques from JSON file
    try:
        with open('storage/data/vocal_critiques.json', 'r') as f:
            blog_critiques = json.load(f)
    except FileNotFoundError:
        # If file doesn't exist, scrape and save critiques
        blog_critiques = scrape_and_save_critiques()
    
    # 2. Analyze performance description for issues
    detected_issues = {
        "posture": [],
        "vocal_technique": []
    }
    
    # Check for posture issues
    for issue in blog_critiques["posture"].keys():
        if issue.lower() in performance_description.lower():
            detected_issues["posture"].append(issue)
    
    # Check for vocal technique issues
    for issue in blog_critiques["vocal_technique"].keys():
        if issue.lower() in performance_description.lower():
            detected_issues["vocal_technique"].append(issue)
    
    # 3. Generate personalized critique
    critique_parts = []
    
    # Add posture feedback
    if detected_issues["posture"]:
        critique_parts.append("Posture Analysis:")
        for issue in detected_issues["posture"]:
            if issue in blog_critiques["posture"]:
                critique_parts.append(f"- {blog_critiques['posture'][issue]}")
    else:
        critique_parts.append("Posture: " + blog_critiques["positive"])
    
    # Add vocal technique feedback
    if detected_issues["vocal_technique"]:
        critique_parts.append("\nVocal Technique Analysis:")
        for issue in detected_issues["vocal_technique"]:
            if issue in blog_critiques["vocal_technique"]:
                critique_parts.append(f"- {blog_critiques['vocal_technique'][issue]}")
    else:
        critique_parts.append("\nVocal Technique: " + blog_critiques["positive"])
    
    # 4. Add specific feedback based on lyrics
    if lyrics:
        critique_parts.append("\nPerformance Notes:")
        # Analyze lyrics for potential challenges
        if any(word in lyrics.lower() for word in ["high", "higher", "highest"]):
            critique_parts.append("- For high notes, ensure proper breath support and maintain an open throat.")
        if any(word in lyrics.lower() for word in ["low", "lower", "lowest"]):
            critique_parts.append("- For low notes, maintain good posture and avoid pushing the voice.")
        if len(lyrics.split()) > 50:
            critique_parts.append("- For longer phrases, focus on breath management and pacing.")
    
    # Combine all parts into final critique
    critique = "\n".join(critique_parts)
    
    # Create entry for performance database
    entry = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "lyrics": lyrics,
        "performance_description": performance_description,
        "critique": critique,
        "detected_issues": detected_issues
    }
    if video_path:
        entry["video_path"] = video_path
    
    # Save to performance database
    save_performance(entry)
    
    return critique, entry

# ========== MAIN PIPELINE ==========
def invoke_critique(lyrics, perf_description, video_path=None):
    # 1. Load the models and predict the performance quality frame-by-frame
    submitted_video = VideoFileClip(video_path)
    save_op_video_dir = os.getcwd()+'/storage/output/videos'
    if not os.path.exists(save_op_video_dir):
        os.makedirs(save_op_video_dir, exist_ok=True)
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
    preds_audio = get_predictions_regression(audio_model, tensor_x, 'cpu', quality_list)

    # Get posture keypoints
    blazepose_model = BlazePose().to('cpu')
    blazepose_model.load_state_dict(torch.load(os.getcwd()+'/best_blazepose_model.pth'))
    blazepose_model.eval()
    device = torch.device('cpu')
    pose_out_dir = os.getcwd()+'/'+video_folder_name+'/Pose Keypoints/'
    if not os.path.exists(pose_out_dir):
        os.makedirs(pose_out_dir, exist_ok=True)
    extract_pose_keypoints_from_videos(save_op_video_dir, pose_out_dir, blazepose_model, device)
    pose_classifier = PostureMLP(input_dim=66, num_labels=8).to('cpu')
    pose_classifier.load_state_dict(torch.load(os.getcwd()+'/best_posture_estimator_model.pth', map_location=device))
    pose_classifier.eval()
    preds_posture = test_posture_classifier(pose_classifier, pose_out_dir, device)


    # 2. Feed lyrics and performance description to RAG critique generator
    singing_lyrics = lyrics
    performance_description = perf_description

    # 3. Call the new RAG function for backward compatibility
    critique, entry = generate_rag_critique(singing_lyrics, performance_description, submitted_video)
    print(f"Critique:\n{critique}\n")

    # 4. Generate Narration with Nari Dia
    print("Generating narration with Nari Dia...")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B")
    narration_audio = model.generate(critique)
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

def scrape_and_save_critiques():
    """
    Scrape vocal coaching blogs and save critiques to a JSON file
    """
    # List of vocal coaching blogs to scrape
    blogs = [
        "https://www.voiceteacher.com/blog",
        "https://www.vocalist.org.uk/blog",
        "https://www.singwise.com/articles",
        "https://www.vocalcoach.com/blog"
    ]
    
    critiques = {
        "posture": {},
        "vocal_technique": {},
        "positive": "Excellent form! Your posture and vocal technique are well-aligned. Keep up the good work!"
    }
    
    # Keywords to look for in blog content
    posture_keywords = {
        "forward_head": ["forward head", "head forward", "head alignment"],
        "flat_back": ["flat back", "lower back curve", "spine alignment"],
        "sway_back": ["sway back", "hip alignment", "pelvic tilt"],
        "rounded_shoulders": ["rounded shoulders", "shoulder position", "shoulder alignment"],
        "weak_abdominals": ["core engagement", "abdominal support", "core strength"],
        "bent_knees": ["knee position", "leg alignment", "knee alignment"],
        "raised_chest": ["chest position", "chest alignment", "upper body position"],
        "bent_neck": ["neck alignment", "neck position", "head position"]
    }
    
    vocal_keywords = {
        "breath_support": ["breath support", "breathing technique", "diaphragmatic breathing"],
        "tension": ["vocal tension", "jaw tension", "neck tension"],
        "tone": ["vocal tone", "tone quality", "voice quality"],
        "pitch": ["pitch accuracy", "pitch control", "vocal pitch"]
    }
    
    for blog_url in blogs:
        try:
            response = requests.get(blog_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract blog content
            articles = soup.find_all(['article', 'div'], class_=['post', 'article', 'entry'])
            
            for article in articles:
                content = article.get_text().lower()
                
                # Check for posture-related content
                for issue, keywords in posture_keywords.items():
                    if any(keyword in content for keyword in keywords):
                        # Extract the relevant paragraph
                        paragraphs = article.find_all('p')
                        for p in paragraphs:
                            p_text = p.get_text().lower()
                            if any(keyword in p_text for keyword in keywords):
                                if issue not in critiques["posture"]:
                                    critiques["posture"][issue] = p.get_text().strip()
                
                # Check for vocal technique-related content
                for issue, keywords in vocal_keywords.items():
                    if any(keyword in content for keyword in keywords):
                        # Extract the relevant paragraph
                        paragraphs = article.find_all('p')
                        for p in paragraphs:
                            p_text = p.get_text().lower()
                            if any(keyword in p_text for keyword in keywords):
                                if issue not in critiques["vocal_technique"]:
                                    critiques["vocal_technique"][issue] = p.get_text().strip()
        
        except Exception as e:
            print(f"Error scraping {blog_url}: {str(e)}")
    
    # Save critiques to JSON file
    os.makedirs('storage/data', exist_ok=True)
    with open('storage/data/vocal_critiques.json', 'w') as f:
        json.dump(critiques, f, indent=2)
    
    return critiques

if __name__ == "__main__":
    scrape_and_save_critiques()