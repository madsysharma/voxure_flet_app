video_image = None
import flet as ft
import math # For hover animation
import threading
import time
import json
import os
import sys
from functools import lru_cache
from typing import Optional, Dict, List, Any, Tuple

# Lazy imports for heavy dependencies
_imports = {
    'cv2': None,
    'PIL': None,
    'numpy': None,
    'base64': None,
    'io': None,
    'random': None,
    'datetime': None,
    'mediapipe': None
}

def get_import(name: str) -> Any:
    """Lazy import helper function"""
    if _imports[name] is None:
        if name == 'cv2':
            import cv2
            _imports[name] = cv2
        elif name == 'PIL':
            from PIL import Image
            _imports[name] = Image
        elif name == 'numpy':
            import numpy as np
            _imports[name] = np
        elif name == 'base64':
            import base64
            _imports[name] = base64
        elif name == 'io':
            import io
            _imports[name] = io
        elif name == 'random':
            import random
            _imports[name] = random
        elif name == 'datetime':
            from datetime import datetime
            _imports[name] = datetime
        elif name == 'mediapipe':
            import mediapipe as mp
            _imports[name] = mp
    return _imports[name]

sys.path.append(os.getcwd()+'/src/')
from vocal_coaching_rag import generate_rag_critique

USER_DATA_FILE = os.getcwd()+"/src/users.json"
JOURNAL_DATA_FILE = os.getcwd()+"/src/journal.json"

# At the top of the file, before any function definitions
if video_image is None:
    video_image = ft.Image(
        src="https://via.placeholder.com/600x400?text=No+Video",
        width=600,
        height=400,
        fit=ft.ImageFit.CONTAIN,
        gapless_playback=True,
        repeat=ft.ImageRepeat.NO_REPEAT,
        animate_opacity=ft.Animation(300, "easeInOut"),
    )

# --- User and Journal Data Management ---
# Cache user data with a timeout of 5 minutes
@lru_cache(maxsize=1)
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return []
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)
    # Clear the cache after saving
    load_users.cache_clear()

# Cache journal data with a timeout of 5 minutes
@lru_cache(maxsize=1)
def load_journal():
    if not os.path.exists(JOURNAL_DATA_FILE):
        return []
    with open(JOURNAL_DATA_FILE, "r") as f:
        return json.load(f)

def save_journal(journal):
    with open(JOURNAL_DATA_FILE, "w") as f:
        json.dump(journal, f)
    # Clear the cache after saving
    load_journal.cache_clear()

def get_available_cameras() -> List[int]:
    """Get list of available camera indices (robust)"""
    cv2 = get_import('cv2')
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
        cap.release()
    return cameras

def process_frame(frame, processing_options: Dict[str, bool] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Process video frame with various options
    Returns processed frame and metadata
    """
    if processing_options is None:
        processing_options = {
            'grayscale': False,
            'flip': False,
            'detect_face': False,
            'detect_pose': False
        }
    
    cv2 = get_import('cv2')
    metadata = {}
    
    # Basic processing
    if processing_options.get('grayscale'):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    if processing_options.get('flip'):
        frame = cv2.flip(frame, 1)
    
    # Advanced processing with MediaPipe
    if processing_options.get('detect_face') or processing_options.get('detect_pose'):
        mp = get_import('mediapipe')
        
        if processing_options.get('detect_face'):
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                metadata['face_detected'] = True
                # Draw face landmarks
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            else:
                metadata['face_detected'] = False
        
        if processing_options.get('detect_pose'):
            pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                metadata['pose_detected'] = True
                # Draw pose landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
            else:
                metadata['pose_detected'] = False
    
    return frame, metadata

def update_video_image_from_frame(frame, video_image):
    cv2 = get_import('cv2')
    PIL = get_import('PIL')
    io = get_import('io')
    base64 = get_import('base64')
    frame = cv2.resize(frame, (640, 480))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = PIL.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    data = buf.getvalue()
    b64str = base64.b64encode(data).decode()
    video_image.src_base64 = b64str
    video_image.update()

def camera_loop(selected_camera_index, processing_options, stop_camera_event, video_image, feedback_text_area):
    cv2 = get_import('cv2')
    try:
        cap = cv2.VideoCapture(selected_camera_index)
        if not cap.isOpened():
            raise Exception(f"Could not open camera {selected_camera_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        frame_count = 0
        start_time = time.time()
        while not stop_camera_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera!")
                continue
            processed_frame, metadata = process_frame(frame, processing_options)
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                metadata['fps'] = round(fps, 1)
                frame_count = 0
                start_time = time.time()
            try:
                update_video_image_from_frame(processed_frame, video_image)
                if metadata.get('face_detected') is not None:
                    feedback_text_area.value = f"Face detected: {metadata['face_detected']}"
                    feedback_text_area.update()
            except Exception as ex:
                print("Exception in update_video_image_from_frame:", ex)
            time.sleep(0.06)
    except Exception as e:
        print(f"Camera error: {str(e)}")
        feedback_text_area.value = f"Camera error: {str(e)}"
        feedback_text_area.update()
    finally:
        if 'cap' in locals():
            cap.release()

def start_camera(selected_camera_index, processing_options, stop_camera_event, video_image, feedback_text_area, camera_active_ref, camera_thread_ref):
    if camera_active_ref[0]:
        return
    try:
        cv2 = get_import('cv2')
        cap = cv2.VideoCapture(selected_camera_index)
        if not cap.isOpened():
            feedback_text_area.value = f"Error: Could not access camera {selected_camera_index}"
            feedback_text_area.update()
            return
        cap.release()
        stop_camera_event.clear()
        thread = threading.Thread(target=camera_loop, args=(selected_camera_index, processing_options, stop_camera_event, video_image, feedback_text_area), daemon=True)
        thread.start()
        camera_thread_ref[0] = thread
        camera_active_ref[0] = True
        feedback_text_area.value = f"Camera {selected_camera_index} started."
        feedback_text_area.update()
    except Exception as e:
        feedback_text_area.value = f"Error starting camera: {str(e)}"
        feedback_text_area.update()

def stop_camera(stop_camera_event, camera_active_ref, video_image, feedback_text_area):
    if camera_active_ref[0]:
        stop_camera_event.set()
        camera_active_ref[0] = False
        feedback_text_area.value = "Camera stopped."
        feedback_text_area.update()
        video_image.src = "https://via.placeholder.com/600x400?text=No+Video"
        video_image.src_base64 = None
        video_image.update()

def main(page: ft.Page):
    page.title = "Voxure - AI Vocal Coach"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 10  # Reduced padding
    page.window_maximized = True
    page.bgcolor = "#181A20"

    # Use Inter (modern sans-serif) as the main font
    page.fonts = {
        "Inter": "https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap"
    }
    page.theme = ft.Theme(font_family="Inter")

    # Clean Google-inspired gradient (subtle)
    static_gradient = ft.LinearGradient(
        begin=ft.alignment.top_center,
        end=ft.alignment.bottom_center,
        colors=[
            "#f8f9fa",  # Very light gray
            "#ffffff",  # White
        ],
        stops=[0.0, 1.0]
    )

    # Animated gradient for splash screen (radial ripple)
    splash_gradient = ft.RadialGradient(
        center=ft.alignment.center,
        radius=0.6,
        colors=[
            "#1a237e",  # Deep Blue
            "#9c27b0",  # Lavender
        ],
        stops=[0.0, 1.0]
    )

    def animate_gradient(is_splash_screen=False):
        import math
        last_update = time.time()
        update_interval = 0.016  # ~60 FPS for smoothness
        start_time = time.time()
        while True:
            current_time = time.time()
            if current_time - last_update < update_interval:
                time.sleep(0.005)
                continue
            if is_splash_screen:
                elapsed = current_time - start_time
                duration = 8.0
                if elapsed > duration:
                    break
                # Animate ripple: radius grows from 0.1 to 1.0, color fades out
                ripple_radius = 0.1 + 0.9 * (elapsed / duration)
                fade = max(0.0, 1.0 - (elapsed / duration))
                splash_gradient.radius = ripple_radius
                # Use hex colors with opacity for Flet compatibility, sharper ripple
                fade_hex = ft.Colors.with_opacity(fade, "#1a237e")
                fade_hex2 = ft.Colors.with_opacity(fade, "#1a237e")
                transparent = ft.Colors.with_opacity(0.0, "#1a237e")
                splash_gradient.colors = [fade_hex, fade_hex2, transparent]
                splash_gradient.stops = [0.0, 0.85, 1.0]
                page.update()
                last_update = current_time
            else:
                time.sleep(0.1)

    def update_ui_safely(control, **kwargs):
        """Safely update UI controls with rate limiting"""
        if not hasattr(update_ui_safely, 'last_updates'):
            update_ui_safely.last_updates = {}
        
        current_time = time.time()
        control_id = id(control)
        
        # Rate limit updates to 30 FPS
        if control_id in update_ui_safely.last_updates:
            if current_time - update_ui_safely.last_updates[control_id] < 0.033:
                return
        
        for key, value in kwargs.items():
            setattr(control, key, value)
        
        control.update()
        update_ui_safely.last_updates[control_id] = current_time

    def update_video_image_from_frame(frame):
        cv2 = get_import('cv2')
        PIL = get_import('PIL')
        io = get_import('io')
        base64 = get_import('base64')
        frame = cv2.resize(frame, (640, 480))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PIL.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        data = buf.getvalue()
        b64str = base64.b64encode(data).decode()
        video_image.src_base64 = b64str
        video_image.update()

    # Start gradient animation in a background thread
    threading.Thread(target=lambda: animate_gradient(False), daemon=True).start()

    # Enhanced camera state
    camera_active_ref = [False]
    camera_thread_ref = [None]
    stop_camera_event = threading.Event()
    uploaded_video_path = None
    last_frame_bytes = None
    selected_camera_index = 0
    processing_options = {
        'grayscale': False,
        'flip': False,
        'detect_face': False,
        'detect_pose': False
    }
    page.session.set("selected_user", None)
    page.session.set("selected_journal_entry", None)
    page.overlay.clear()

    # --- File Picker for Video Upload ---
    def on_file_picked(e: ft.FilePickerResultEvent):
        if page.route not in ["/home", "/"]:
            return
        nonlocal uploaded_video_path
        if e.files:
            selected_file = e.files[0]
            uploaded_video_path = selected_file.path
            try:
                feedback_text_area.value = f"Selected video: {selected_file.name}\nReady for analysis."
                # Extract and show first frame
                cap = get_import('cv2').VideoCapture(uploaded_video_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    update_video_image_from_frame(frame)
                else:
                    feedback_text_area.value += "\nFailed to load video frame."
                feedback_text_area.update()
            except (AssertionError, AttributeError):
                print(f"Video selected: {selected_file.name}")

            # --- Prompt user for lyrics and performance description ---
            lyrics_field = ft.TextField(label="Lyrics", multiline=True, width=400)
            desc_field = ft.TextField(label="Performance Description", multiline=True, width=400)
            def on_submit_critique(ev):
                lyrics = lyrics_field.value
                performance_description = desc_field.value
                try:
                    feedback_text_area.value = "Generating AI critique..."
                    feedback_text_area.update()
                    critique, entry = invoke_critique(lyrics, performance_description, video_path=uploaded_video_path)
                    feedback_text_area.value = f"AI Critique:\n{critique}"
                    feedback_text_area.update()
                except (AssertionError, AttributeError):
                    print("Generating AI critique...")
                page.dialog.open = False
                page.update()
            dialog = ft.AlertDialog(
                title=ft.Text("Enter Performance Details"),
                content=ft.Column([lyrics_field, desc_field]),
                actions=[ft.TextButton("Submit", on_click=on_submit_critique)],
                actions_alignment=ft.MainAxisAlignment.END,
            )
            page.dialog = dialog
            dialog.open = True
            page.update()
        else:
            try:
                feedback_text_area.value = "Video selection cancelled."
                feedback_text_area.update()
            except (AssertionError, AttributeError):
                print("Video selection cancelled.")

    file_picker = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(file_picker) # Add file picker to page overlay

    # --- Placeholder for Video Display ---
    video_display_area = ft.Container(
        content=video_image,
        expand=True,
        bgcolor="#23272F",
        border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.BLUE_ACCENT_100)),
        alignment=ft.alignment.center,
        border_radius=10,
        padding=10
    )

    # --- Configuration Dropdowns ---
    # (Styling adjustments for dark theme might be needed if defaults aren't good)
    dd_genre = ft.Dropdown(
        label="Genre",
        hint_text="Choose your genre",
        options=[
            ft.dropdown.Option("Pop"), ft.dropdown.Option("Rock"), ft.dropdown.Option("Broadway"),
            ft.dropdown.Option("Jazz"), ft.dropdown.Option("Classical"), ft.dropdown.Option("Folk"),
        ],
        width=200,
        border_color="#1a73e8",
        focused_border_color="#8430ce",
        label_style=ft.TextStyle(font_family="Inter"),
    )

    dd_role = ft.Dropdown(
        label="Role",
        hint_text="Choose your role",
        options=[
            ft.dropdown.Option("Lead Vocalist"), ft.dropdown.Option("Backing Vocalist"),
            ft.dropdown.Option("Chorus Singer"),
        ],
        width=200,
        border_color="#1a73e8",
        focused_border_color="#8430ce",
        label_style=ft.TextStyle(font_family="Inter"),
    )

    dd_range = ft.Dropdown(
        label="Vocal Range",
        hint_text="Choose your vocal range",
        options=[
            ft.dropdown.Option("Soprano"), ft.dropdown.Option("Mezzo-Soprano"), ft.dropdown.Option("Alto"),
            ft.dropdown.Option("Tenor"), ft.dropdown.Option("Baritone"), ft.dropdown.Option("Bass"),
        ],
        width=200,
        border_color="#1a73e8",
        focused_border_color="#8430ce",
        label_style=ft.TextStyle(font_family="Inter"),
    )

    config_controls = ft.Row(
        controls=[dd_genre, dd_role, dd_range],
        alignment=ft.MainAxisAlignment.SPACE_AROUND,
        spacing=20
    )
    
    # --- Button Hover Animation ---
    def on_hover_animation(e):
        e.control.scale = ft.Scale(1.05 if e.data == "true" else 1)
        e.control.shadow = ft.BoxShadow(
            spread_radius=2,
            blur_radius=10,
            color=ft.Colors.with_opacity(0.3, ft.Colors.PURPLE_ACCENT_200) if e.data == "true" else None,
            offset=ft.Offset(0, 0),
            blur_style=ft.ShadowBlurStyle.NORMAL,
        )
        e.control.update()

    def on_camera_change(e):
        nonlocal selected_camera_index
        selected_camera_index = int(e.control.value)
        stop_camera(stop_camera_event, camera_active_ref, video_image, feedback_text_area)
        start_camera(selected_camera_index, processing_options, stop_camera_event, video_image, feedback_text_area, camera_active_ref, camera_thread_ref)

    # Add camera selection dropdown
    available_cameras = get_available_cameras()
    if selected_camera_index not in available_cameras and available_cameras:
        selected_camera_index = available_cameras[0]
    camera_dropdown = ft.Dropdown(
        label="Select Camera",
        options=[ft.dropdown.Option(str(i), f"Camera {i}") for i in available_cameras],
        value=str(selected_camera_index),
        width=200,
        on_change=on_camera_change
    )

    # Add processing options
    processing_controls = ft.Column([
        ft.Text("Processing Options", size=16, weight=ft.FontWeight.BOLD),
        ft.Checkbox(label="Grayscale", value=processing_options['grayscale'],
                   on_change=lambda e: setattr(processing_options, 'grayscale', e.control.value)),
        ft.Checkbox(label="Flip Horizontal", value=processing_options['flip'],
                   on_change=lambda e: setattr(processing_options, 'flip', e.control.value)),
        ft.Checkbox(label="Detect Face", value=processing_options['detect_face'],
                   on_change=lambda e: setattr(processing_options, 'detect_face', e.control.value)),
        ft.Checkbox(label="Detect Pose", value=processing_options['detect_pose'],
                   on_change=lambda e: setattr(processing_options, 'detect_pose', e.control.value)),
    ])

    # --- Action Buttons ---
    btn_start_camera = ft.ElevatedButton(
        text="Start Camera",
        icon=ft.Icons.VIDEO_CAMERA_FRONT,
        width=180,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.BLUE_ACCENT_700,
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=15
        ),
        on_click=lambda e: start_camera(selected_camera_index, processing_options, stop_camera_event, video_image, feedback_text_area, camera_active_ref, camera_thread_ref),
        on_hover=on_hover_animation,
        scale=ft.Scale(1),
        animate_scale=ft.Animation(300, "easeOutCubic")
    )
    btn_stop_camera = ft.ElevatedButton(
        text="Stop Camera",
        icon=ft.Icons.STOP_CIRCLE_OUTLINED,
        width=180,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.RED_ACCENT_700,
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=15
        ),
        on_click=lambda e: stop_camera(stop_camera_event, camera_active_ref, video_image, feedback_text_area),
        on_hover=on_hover_animation,
        scale=ft.Scale(1),
        animate_scale=ft.Animation(300, "easeOutCubic")
    )
    btn_start_analysis = ft.ElevatedButton(
        text="Start Real-time Analysis",
        icon=ft.Icons.PLAY_CIRCLE_OUTLINE,
        width=250,
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.LIGHT_BLUE_ACCENT_700,
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=15
        ),
        on_click=lambda e: start_camera(selected_camera_index, processing_options, stop_camera_event, video_image, feedback_text_area, camera_active_ref, camera_thread_ref),
        on_hover=on_hover_animation,
        scale=ft.Scale(1),
        animate_scale=ft.Animation(300, "easeOutCubic")
    )
    btn_upload_video = ft.ElevatedButton(
        text="Upload Video",
        icon=ft.Icons.UPLOAD_FILE_OUTLINED,
        width=250,
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["mp4", "mov", "avi", "mkv", "webm"] # Common video extensions
        ),
        style=ft.ButtonStyle(
            color=ft.Colors.WHITE,
            bgcolor=ft.Colors.PURPLE_ACCENT_700,
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=15
        ),
        on_hover=on_hover_animation,
        scale=ft.Scale(1),
        animate_scale=ft.Animation(300, "easeOutCubic")
    )

    action_buttons = ft.ResponsiveRow(
        controls=[
            ft.Container(btn_start_camera, col={"xs": 12, "sm": 6, "md": 3}),
            ft.Container(btn_stop_camera, col={"xs": 12, "sm": 6, "md": 3}),
            ft.Container(btn_start_analysis, col={"xs": 12, "sm": 6, "md": 3}),
            ft.Container(btn_upload_video, col={"xs": 12, "sm": 6, "md": 3}),
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        run_spacing=10,
        spacing=10,
        expand=True,
    )

    # --- Feedback and Suggestions Area ---
    feedback_text_area = ft.Text(
        "Posture analysis, vocal feedback, and personalized suggestions will appear here...",
        size=14,
        italic=True,
        color="#5f6368"
    )
    
    feedback_container = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("Real-time Feedback & Coaching", size=20, weight=ft.FontWeight.BOLD, color="#1a73e8"),
                ft.Divider(color=ft.Colors.with_opacity(0.5, "#8430ce")),
                feedback_text_area
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=10
        ),
        expand=True,
        bgcolor="#1AFFFFFF",
        border=ft.border.all(1, ft.Colors.with_opacity(0.3, "#8430ce")),
        border_radius=10,
        padding=15,
        margin=ft.margin.only(top=20),
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=10,
            color=ft.Colors.with_opacity(0.2, "#000000"),
            offset=ft.Offset(2, 2)
        )
    )

    # --- Main Content Column (Video + Config + Actions + Feedback) ---
    def create_main_coaching_column():
        return ft.Column(
            controls=[
                video_display_area,
                ft.Container(height=10), # Spacer
                config_controls,
                ft.Container(height=20), # Spacer
                action_buttons,
                feedback_container,
            ],
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
            expand=True
        )

    # --- Placeholder Sections for Other Features ---
    def create_placeholder_container(title_text, content_widget=None):
        return ft.Container(
            content=ft.Column([
                ft.Text(title_text, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, font_family="Inter", color="#1a73e8"),
                ft.Divider(color=ft.Colors.with_opacity(0.3, "#8430ce")),
                content_widget if content_widget else ft.Container(height=5)
            ]),
            padding=15, 
            bgcolor="#3DFFFFFF",
            border_radius=8, 
            margin=ft.margin.only(top=10),
            shadow=ft.BoxShadow(blur_radius=5, color=ft.Colors.with_opacity(0.1, "#000000"))
        )

    def create_other_features_column():
        progress_tracker_placeholder = create_placeholder_container(
            "Progress Tracker Area",
            ft.Text("Your improvements and session history will be shown here.", font_family="Inter", color="#5f6368")
        )
        
        warmup_exercises_placeholder = create_placeholder_container(
            "Personalized Warmup Exercises",
            ft.ElevatedButton(
                "Generate Warmup", 
                icon=ft.Icons.FITNESS_CENTER, 
                style=ft.ButtonStyle(bgcolor="#8430ce", color="#ffffff"),
                on_hover=on_hover_animation,
                scale=ft.Scale(1), animate_scale=ft.Animation(300, "easeOutCubic")
            )
        )

        community_features_placeholder = create_placeholder_container(
            "Community Features (Blog, Messaging, etc.)",
            ft.Text("Connect with others, share insights, and get expert advice.", font_family="Inter", color="#5f6368")
        )

        return ft.Column(
            controls=[
                ft.Divider(height=20, thickness=1, color=ft.Colors.with_opacity(0.5, "#8430ce")),
                ft.Text("More Tools & Features", theme_style=ft.TextThemeStyle.HEADLINE_SMALL, font_family="Inter", color="#1a73e8"),
                progress_tracker_placeholder,
                warmup_exercises_placeholder,
                community_features_placeholder
            ],
            spacing=15,
            width=600, 
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        )

    # --- Responsive main content layout ---
    main_content = ft.ResponsiveRow(
        controls=[
            ft.Container(create_main_coaching_column(), col={"xs": 12, "md": 7}, expand=True),
            ft.Container(create_other_features_column(), col={"xs": 12, "md": 5}, expand=True),
        ],
        run_spacing=20,
        spacing=20,
        expand=True,
    )

    # --- Overall Page Layout ---
    page_content = ft.Column(
        controls=[
            ft.Text("VOXURE", theme_style=ft.TextThemeStyle.DISPLAY_SMALL, weight=ft.FontWeight.BOLD, font_family="Inter", color="#00E5FF",
                      spans=[ft.TextSpan("™", ft.TextStyle(size=12, font_family="Inter", color="#00E5FF", weight=ft.FontWeight.NORMAL))]
            ),
            ft.Text("Your Personal AI Vocal Coach", theme_style=ft.TextThemeStyle.TITLE_LARGE, font_family="Inter", color=ft.Colors.BLUE_GREY_100, weight=ft.FontWeight.W_300, text_align=ft.TextAlign.CENTER),
            ft.Divider(height=30, thickness=1, color=ft.Colors.with_opacity(0.7, ft.Colors.BLUE_ACCENT_700)),
            main_content,
        ],
        scroll=ft.ScrollMode.ADAPTIVE,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=25,
        expand=True
    )

    page.add(page_content)
    page.update()

    # --- Navigation ---
    def go(route):
        page.views.clear()
        if route == "/":
            page.views.append(splash_screen())
        elif route == "/users":
            page.views.append(user_selection_screen())
        elif route == "/create_user":
            page.views.append(create_user_screen())
        elif route == "/home":
            page.views.append(home_screen())
        elif route == "/record":
            page.views.append(recording_screen())
        elif route == "/summary":
            page.views.append(summary_screen())
        elif route == "/journal":
            page.views.append(vocal_journal_screen())
        elif route == "/journal_entry":
            page.views.append(journal_entry_screen())
        page.update()

    page.on_route_change = lambda e: go(page.route)
    page.go("/")

    # --- Splash Screen ---
    def splash_screen():
        # Create concentric circles that pulse outward
        def animate_concentric_circles():
            start_time = time.time()
            while True:
                now = time.time()
                elapsed = now - start_time
                if elapsed > 6.0:  # Animation duration
                    break
                
                # Update circle opacities with pulsing effect
                pulse = (math.sin(elapsed * 3) + 1) / 2  # Oscillate between 0.0 and 1.0
                for i, circle in enumerate(concentric_circles):
                    delay = i * 0.15  # Stagger the pulse effect
                    circle_pulse = max(0.2, pulse - delay * 0.1)
                    circle.border = ft.border.all(
                        width=2,
                        color=ft.Colors.with_opacity(circle_pulse, "#1a73e8")
                    )
                
                page.update()
                time.sleep(0.033)  # 30 FPS for smooth animation
            
            # Navigate to next screen
            page.go("/users")

        # Create the microphone design
        def create_microphone():
            # Microphone head (grille)
            mic_head = ft.Container(
                width=60,
                height=80,
                bgcolor="#2C2C54",
                border_radius=30,
                border=ft.border.all(3, "#1a73e8"),
                content=ft.Column([
                    ft.Container(height=8),  # Top spacing
                    # Grille lines
                    ft.Container(width=45, height=2, bgcolor="#8430ce", border_radius=1),
                    ft.Container(height=4),
                    ft.Container(width=45, height=2, bgcolor="#8430ce", border_radius=1),
                    ft.Container(height=4),
                    ft.Container(width=45, height=2, bgcolor="#8430ce", border_radius=1),
                    ft.Container(height=4),
                    ft.Container(width=45, height=2, bgcolor="#8430ce", border_radius=1),
                    ft.Container(height=4),
                    ft.Container(width=45, height=2, bgcolor="#8430ce", border_radius=1),
                    ft.Container(height=4),
                    ft.Container(width=45, height=2, bgcolor="#8430ce", border_radius=1),
                    ft.Container(height=8),  # Bottom spacing
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                alignment=ft.alignment.center
            )
            
            # Microphone body/handle
            mic_body = ft.Container(
                width=35,
                height=60,
                bgcolor="#40407a",
                border_radius=3,
                border=ft.border.all(2, "#8430ce"),
                content=ft.Column([
                    ft.Container(height=8),
                    # Control buttons/indicators
                    ft.Container(
                        width=25,
                        height=8,
                        bgcolor="#1a73e8",
                        border_radius=4,
                    ),
                    ft.Container(height=6),
                    ft.Container(
                        width=20,
                        height=6,
                        bgcolor="#8430ce",
                        border_radius=3,
                    ),
                    ft.Container(height=6),
                    # Brand/model indicator
                    ft.Container(
                        width=15,
                        height=4,
                        bgcolor="#40407a",
                        border_radius=2,
                    ),
                ], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                alignment=ft.alignment.center
            )
            
            # Main microphone container
            microphone = ft.Container(
                width=70,
                height=160,
                content=ft.Column([
                    mic_head,
                    mic_body,  # Remove gap to connect head and body
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
                alignment=ft.alignment.center
            )
            
            return microphone

        # Create concentric circles (properly sized for microphone)
        circle_sizes = [180, 220, 260, 300, 340, 380]
        concentric_circles = []
        
        for size in circle_sizes:
            circle = ft.Container(
                width=size,
                height=size,
                bgcolor="transparent",
                border_radius=size // 2,
                border=ft.border.all(width=2, color=ft.Colors.with_opacity(0.4, "#1a73e8")),
                alignment=ft.alignment.center
            )
            concentric_circles.append(circle)

        # Text elements
        voxure_text = ft.Text(
            "Voxure",
            size=50,
            weight=ft.FontWeight.BOLD,
            color="#1a73e8",
            opacity=0,
            animate_opacity=ft.Animation(500, "easeInOut"),
        )
        tagline_text = ft.Text(
            "Your voice, your choice.",
            size=24,
            color="#8430ce",
            opacity=0,
            animate_opacity=ft.Animation(500, "easeInOut"),
        )

        # Fade in text sequence
        def fade_in_sequence():
            time.sleep(0.5)
            voxure_text.opacity = 1
            page.update()
            time.sleep(0.7)
            tagline_text.opacity = 1
            page.update()
        
        # Start animations
        threading.Thread(target=fade_in_sequence, daemon=True).start()
        threading.Thread(target=animate_concentric_circles, daemon=True).start()

        # Create the centered design with concentric circles
        microphone = create_microphone()
        
        # Stack all circles with the microphone in the center
        circle_stack = ft.Stack(
            controls=[
                *reversed(concentric_circles),  # Largest circles first (background)
                microphone  # Microphone on top
            ],
            alignment=ft.alignment.center
        )

        splash_content = ft.Container(
            content=ft.Column([
                ft.Container(height=20),  # Top spacer
                circle_stack,
                ft.Container(height=40),  # Reduced spacing
                voxure_text,
                tagline_text,
                ft.Container(height=20),  # Bottom spacer
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True,
            spacing=15),  # Reduced spacing
            alignment=ft.alignment.center,
            expand=True
        )

        return ft.View(
            "/",
            controls=[
                ft.Container(
                    content=splash_content,
                    alignment=ft.alignment.center,
                    expand=True,
                    bgcolor="#181A20",  # Dark background to match the image
                )
            ]
        )

    # --- User Selection Screen ---
    def user_selection_screen():
        users = load_users()
        all_options = users + [{"name": "Add User", "is_add_user": True}]
        
        # Use page session for state management
        if not page.session.contains_key("current_user_index"):
            page.session.set("current_user_index", 0)
        
        current_index = page.session.get("current_user_index")
        
        def update_display():
            current_index = page.session.get("current_user_index")
            if current_index < len(users):
                user = users[current_index]
                content = ft.Column([
                    ft.CircleAvatar(
                        content=ft.Icon(ft.Icons.MUSIC_NOTE, size=32, color="#ffffff"),
                        radius=40,
                        bgcolor="#1a73e8"
                    ),
                    ft.Text(user["name"], size=18, color="#202124", text_align=ft.TextAlign.CENTER, font_family="Inter", weight=ft.FontWeight.W_500)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=16)
                click_handler = lambda e, u=user: select_user(u)
            else:
                content = ft.Column([
                    ft.CircleAvatar(
                        content=ft.Icon(ft.Icons.ADD, size=32, color="#ffffff"),
                        radius=40,
                        bgcolor="#8430ce"
                    ),
                    ft.Text("Add User", size=18, color="#202124", text_align=ft.TextAlign.CENTER, font_family="Inter", weight=ft.FontWeight.W_500)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=16)
                click_handler = lambda e: page.go("/create_user")
            
            return content, click_handler
        
        def go_previous(e):
            current_index = page.session.get("current_user_index")
            if current_index > 0:
                page.session.set("current_user_index", current_index - 1)
                # Force page update instead of navigation
                page.views.clear()
                page.views.append(user_selection_screen())
                page.update()
        
        def go_next(e):
            current_index = page.session.get("current_user_index")
            if current_index < len(all_options) - 1:
                page.session.set("current_user_index", current_index + 1)
                # Force page update instead of navigation
                page.views.clear()
                page.views.append(user_selection_screen())
                page.update()
        
        content, click_handler = update_display()
        
        # Navigation buttons
        left_arrow = ft.IconButton(
            icon=ft.Icons.ARROW_BACK_IOS,
            icon_size=32,
            icon_color="#1a73e8" if current_index > 0 else "#9e9e9e",
            on_click=go_previous,
            disabled=current_index == 0,
            tooltip="Previous profile"
        )
        
        right_arrow = ft.IconButton(
            icon=ft.Icons.ARROW_FORWARD_IOS,
            icon_size=32,
            icon_color="#1a73e8" if current_index < len(all_options) - 1 else "#9e9e9e",
            on_click=go_next,
            disabled=current_index >= len(all_options) - 1,
            tooltip="Next profile"
        )
        
        # Current user container
        user_container = ft.Container(
            content=content,
            padding=ft.padding.all(40),
            border_radius=24,
            bgcolor="#ffffff",
            shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.08, "#000000"), offset=ft.Offset(0, 2)),
            border=ft.border.all(1, ft.Colors.with_opacity(0.12, "#000000")),
            width=300,
            height=200,
            ink=True,
            on_click=click_handler
        )
        
        return ft.View(
            "/users",
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text("Select Your Profile", size=32, weight=ft.FontWeight.W_400, color="#202124", font_family="Inter", text_align=ft.TextAlign.CENTER),
                        ft.Container(height=48),
                        ft.Row([
                            left_arrow,
                            ft.Container(width=32),
                            user_container,
                            ft.Container(width=32),
                            right_arrow
                        ], alignment=ft.MainAxisAlignment.CENTER),
                        ft.Container(height=32),
                        ft.Text(f"{current_index + 1} of {len(all_options)}", size=14, color="#5f6368", font_family="Inter", text_align=ft.TextAlign.CENTER) if len(all_options) > 1 else ft.Container()
                    ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
                    expand=True,
                    alignment=ft.alignment.center,
                    padding=ft.padding.all(40),
                    bgcolor=None,
                    gradient=static_gradient
                )
            ]
        )

    # --- Create User Screen ---
    def create_user_screen():
        name_field = ft.TextField(label="Name", prefix_icon=ft.Icons.PERSON, width=350, filled=True)
        age_field = ft.TextField(label="Age", prefix_icon=ft.Icons.CAKE, width=350, filled=True)
        singer_type_field = ft.Dropdown(
            label="Singer Type",
            options=[ft.dropdown.Option(x) for x in ["Soprano", "Mezzo-Soprano", "Alto", "Tenor", "Baritone", "Bass"]],
            prefix_icon=ft.Icons.MIC,
            width=350,
            filled=True
        )
        genres_practiced_field = ft.TextField(label="Genres Practiced (comma separated)", prefix_icon=ft.Icons.MUSIC_NOTE, width=350, filled=True)
        genres_learn_field = ft.TextField(label="Genres to Learn (comma separated)", prefix_icon=ft.Icons.LIBRARY_MUSIC, width=350, filled=True)
        goals_field = ft.TextField(label="Voice Improvement Goals", multiline=True, prefix_icon=ft.Icons.FLAG, width=350, filled=True)
        submit_btn = ft.ElevatedButton(
            text="Create Profile",
            icon=ft.Icons.CHECK_CIRCLE,
            style=ft.ButtonStyle(bgcolor=ft.Colors.PURPLE_ACCENT_700, color=ft.Colors.WHITE, padding=15, shape=ft.RoundedRectangleBorder(radius=8)),
            width=200,
            height=50,
            scale=ft.Scale(1.05),
            on_click=lambda e: save_new_user(
                name_field.value, age_field.value, singer_type_field.value,
                genres_practiced_field.value, genres_learn_field.value, goals_field.value
            )
        )
        form_card = ft.Container(
            content=ft.Column([
                ft.Text("Create New User", size=28, weight=ft.FontWeight.BOLD, color="#00E5FF", font_family="Inter", text_align=ft.TextAlign.CENTER),
                ft.Divider(color=ft.Colors.with_opacity(0.2, ft.Colors.PURPLE_ACCENT)),
                name_field,
                age_field,
                singer_type_field,
                ft.Divider(height=10, color="transparent"),
                genres_practiced_field,
                genres_learn_field,
                goals_field,
                ft.Container(height=20),
                submit_btn
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=18,
            expand=True),
            padding=30,
            width=420,
            bgcolor=ft.Colors.with_opacity(0.95, ft.Colors.BLUE_GREY_900),
            border_radius=16,
            shadow=ft.BoxShadow(blur_radius=18, color=ft.Colors.with_opacity(0.18, ft.Colors.PURPLE_ACCENT)),
            alignment=ft.alignment.center
        )
        return ft.View(
            "/create_user",
            controls=[
                ft.Container(
                    content=form_card,
                    alignment=ft.alignment.center,
                    expand=True,
                    bgcolor=None,
                    gradient=static_gradient
                )
            ]
        )

    def save_new_user(name, age, singer_type, genres_practiced, genres_learn, goals):
        users = load_users()
        new_user = {
            "name": name,
            "age": age,
            "singer_type": singer_type,
            "genres_practiced": [g.strip() for g in genres_practiced.split(",") if g.strip()],
            "genres_learn": [g.strip() for g in genres_learn.split(",") if g.strip()],
            "goals": goals,
            "progress": [],
            "journal": []
        }
        users.append(new_user)
        save_users(users)
        page.session.set("selected_user", new_user)
        page.go("/home")

    def select_user(user):
        page.session.set("selected_user", user)
        page.go("/home")

    # --- Home Screen ---
    def home_screen():
        user = page.session.get("selected_user")
        if not user:
            page.go("/users")
            return

        def on_record_yourself(e):
            overlay.open = False
            page.update()
            page.go("/record")

        def on_upload_video(e):
            overlay.open = False
            page.update()
            file_picker.pick_files(
                allow_multiple=False,
                allowed_extensions=["mp4", "mov", "avi", "mkv", "webm"]
            )

        overlay = ft.AlertDialog(
            title=ft.Text("Practice and Learn", size=18, weight=ft.FontWeight.BOLD, color="#00E5FF"),
            content=ft.Container(
                width=350,
                alignment=ft.alignment.center,
                content=ft.Column([
                    ft.ElevatedButton(
                        "Record Yourself",
                        icon=ft.Icons.VIDEOCAM,
                        on_click=on_record_yourself,
                        width=250,
                    ),
                    ft.ElevatedButton(
                        "Upload Video",
                        icon=ft.Icons.UPLOAD_FILE,
                        on_click=on_upload_video,
                        width=250,
                    ),
                ], spacing=12, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                padding=20,
                bgcolor="#181A20",
                border_radius=12,
            ),
            actions=[ft.TextButton("Cancel", on_click=lambda e: (setattr(overlay, 'open', False), page.update()))],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=12),
            on_dismiss=lambda e: print("Dialog dismissed!"),
            title_padding=ft.padding.all(25),
        )

        nav_buttons = ft.Column([
            ft.ElevatedButton("Practice and Learn", icon=ft.Icons.MIC, on_click=lambda e: page.open(overlay), style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=24), bgcolor="#1a73e8", color="#ffffff", padding=16, elevation=2)),
            ft.ElevatedButton("Your Vocal Journal", icon=ft.Icons.BOOK, on_click=lambda e: page.go("/journal"), style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=24), bgcolor="#8430ce", color="#ffffff", padding=16, elevation=2)),
        ], alignment=ft.MainAxisAlignment.START, spacing=16)
        progress_plot = ft.Image(
            src="https://quickchart.io/chart?c={type:'line',data:{labels:['Jan','Feb','Mar'],datasets:[{label:'Progress',data:[1,2,3]}]}}",
            width=400, height=250, fit=ft.ImageFit.CONTAIN
        )
        user_card = ft.Container(
            content=ft.Column([
                ft.Text(f"Welcome, {user['name']}!", size=28, color="#202124", font_family="Inter", weight=ft.FontWeight.W_400),
                ft.Text(f"Singer Type: {user['singer_type']}", color="#5f6368", font_family="Inter", size=16),
                ft.Container(height=24),
                ft.Text("Your Progress", size=20, color="#1a73e8", font_family="Inter", weight=ft.FontWeight.W_500),
                ft.Container(height=16),
                progress_plot
            ], spacing=8),
            padding=32,
            bgcolor="#ffffff",
            border_radius=12,
            shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.08, "#000000"), offset=ft.Offset(0, 2)),
            border=ft.border.all(1, ft.Colors.with_opacity(0.12, "#000000")),
            expand=True
        )
        nav_card = ft.Container(
            content=nav_buttons,
            width=200,
            bgcolor="#ffffff",
            border_radius=12,
            padding=24,
            shadow=ft.BoxShadow(blur_radius=8, color=ft.Colors.with_opacity(0.08, "#000000"), offset=ft.Offset(0, 2)),
            border=ft.border.all(1, ft.Colors.with_opacity(0.12, "#000000"))
        )
        return ft.View(
            "/home",
            controls=[
                ft.Container(
                    content=ft.Row([
                        nav_card,
                        ft.Container(width=32),
                        user_card
                    ], expand=True, alignment=ft.MainAxisAlignment.CENTER),
                    expand=True,
                    alignment=ft.alignment.center,
                    padding=ft.padding.all(40),
                    bgcolor=None,
                    gradient=static_gradient
                )
            ]
        )

    # --- Recording Screen ---
    def recording_screen():
        # Local state for camera
        camera_active_ref = [False]
        camera_thread_ref = [None]
        stop_camera_event = threading.Event()
        selected_camera_index = 0
        processing_options = {
            'grayscale': False,
            'flip': False,
            'detect_face': False,
            'detect_pose': False
        }
        available_cameras = get_available_cameras()
        if selected_camera_index not in available_cameras and available_cameras:
            selected_camera_index = available_cameras[0]
        video_image = ft.Image(
            src="https://via.placeholder.com/600x400?text=No+Video",
            width=600,
            height=400,
            fit=ft.ImageFit.CONTAIN,
            gapless_playback=True,
            repeat=ft.ImageRepeat.NO_REPEAT,
            animate_opacity=ft.Animation(300, "easeInOut"),
        )
        feedback_text_area = ft.Text(
            "Camera feedback will appear here...",
            size=14,
            italic=True,
            color="#5f6368"
        )
        # Add genre, vocalist type, and vocal range dropdowns
        genre_dd = ft.Dropdown(
            label="Genre",
            options=[ft.dropdown.Option(x) for x in ["Pop", "Rock", "Jazz", "Classical", "Broadway", "Alternative"]],
            width=200
        )
        vocalist_dd = ft.Dropdown(
            label="Vocalist Type",
            options=[ft.dropdown.Option(x) for x in ["Lead Vocalist", "Backing Vocalist", "Chorus Singer"]],
            width=200
        )
        range_dd = ft.Dropdown(
            label="Vocal Range",
            options=[ft.dropdown.Option(x) for x in ["Soprano", "Mezzo-Soprano", "Alto", "Tenor", "Baritone", "Bass"]],
            width=200
        )
        dropdown_row = ft.Row([genre_dd, vocalist_dd, range_dd], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        def on_camera_change(e):
            nonlocal selected_camera_index
            selected_camera_index = int(e.control.value)
            stop_camera(stop_camera_event, camera_active_ref, video_image, feedback_text_area)
            start_camera(selected_camera_index, processing_options, stop_camera_event, video_image, feedback_text_area, camera_active_ref, camera_thread_ref)
        camera_dropdown = ft.Dropdown(
            label="Select Camera",
            options=[ft.dropdown.Option(str(i), f"Camera {i}") for i in available_cameras],
            value=str(selected_camera_index),
            width=200,
            on_change=on_camera_change
        )
        def on_processing_change(opt):
            def handler(e):
                processing_options[opt] = e.control.value
            return handler
        processing_controls = ft.Column([
            ft.Text("Processing Options", size=16, weight=ft.FontWeight.BOLD),
            ft.Checkbox(label="Grayscale", value=processing_options['grayscale'], on_change=on_processing_change('grayscale')),
            ft.Checkbox(label="Flip Horizontal", value=processing_options['flip'], on_change=on_processing_change('flip')),
            ft.Checkbox(label="Detect Face", value=processing_options['detect_face'], on_change=on_processing_change('detect_face')),
            ft.Checkbox(label="Detect Pose", value=processing_options['detect_pose'], on_change=on_processing_change('detect_pose')),
        ])
        def on_start_camera(e):
            start_camera(selected_camera_index, processing_options, stop_camera_event, video_image, feedback_text_area, camera_active_ref, camera_thread_ref)
        def on_stop_camera(e):
            stop_camera(stop_camera_event, camera_active_ref, video_image, feedback_text_area)
        btn_start_camera = ft.ElevatedButton(
            text="Start Camera",
            icon=ft.Icons.VIDEO_CAMERA_FRONT,
            width=180,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_ACCENT_700,
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=15
            ),
            on_click=on_start_camera,
            scale=ft.Scale(1),
            animate_scale=ft.Animation(300, "easeOutCubic")
        )
        btn_stop_camera = ft.ElevatedButton(
            text="Stop Camera",
            icon=ft.Icons.STOP_CIRCLE_OUTLINED,
            width=180,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.RED_ACCENT_700,
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=15
            ),
            on_click=on_stop_camera,
            scale=ft.Scale(1),
            animate_scale=ft.Animation(300, "easeOutCubic")
        )
        btn_start_analysis = ft.ElevatedButton(
            text="Start Real-time Analysis",
            icon=ft.Icons.PLAY_CIRCLE_OUTLINE,
            width=250,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.LIGHT_BLUE_ACCENT_700,
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=15
            ),
            on_click=on_start_camera,
            scale=ft.Scale(1),
            animate_scale=ft.Animation(300, "easeOutCubic")
        )
        return ft.View(
            "/record",
            controls=[
                ft.Container(
                    content=ft.Column([
                        dropdown_row,
                        video_image,
                        feedback_text_area,
                        ft.Row([camera_dropdown, processing_controls], alignment=ft.MainAxisAlignment.CENTER, spacing=20),
                        ft.Row([btn_start_camera, btn_stop_camera, btn_start_analysis], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
                    ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
                    expand=True,
                    bgcolor=None,
                    gradient=static_gradient
                )
            ]
        )

    # --- Critique Panel (side panel with chat bubbles) ---
    def show_critique_panel():
        critique_bubbles = [
            {"timestamp": "00:12", "critique": "Watch your posture!"},
            {"timestamp": "00:34", "critique": "Try to relax your jaw."},
            {"timestamp": "01:10", "critique": "Good breath support here!"},
        ]
        chat_bubbles = [
            ft.Container(
                content=ft.Row([
                    ft.Text(cb["timestamp"], size=12, color=ft.Colors.BLUE_GREY_200),
                    ft.Container(
                        content=ft.Text(cb["critique"], size=14, color=ft.Colors.WHITE),
                        bgcolor="#3D3D5C" if i % 2 == 0 else "#2D2D3C",
                        border_radius=8,
                        padding=10,
                        margin=ft.margin.only(left=10)
                    )
                ], alignment=ft.MainAxisAlignment.START),
                margin=ft.margin.only(bottom=8)
            )
            for i, cb in enumerate(critique_bubbles)
        ]
        side_panel = ft.AlertDialog(
            modal=False,
            title=ft.Text("Vocal Critique"),
            content=ft.Column(chat_bubbles, scroll=ft.ScrollMode.AUTO, height=200),
            actions=[ft.TextButton("Close", on_click=lambda e: (page.dialog.close(), page.update()))]
        )
        page.dialog = side_panel
        side_panel.open = True
        page.update()

    # --- Summary Screen ---
    def summary_screen():
        summary_text = ft.Text("Performance Summary:\n\nGood: Strong breath support.\nNeeds Improvement: Posture, jaw tension.\nTips: Practice with a mirror, do relaxation exercises.", size=16)
        save_btn = ft.ElevatedButton("Save and Continue", on_click=lambda e: save_performance_and_continue())
        return ft.View(
            "/summary",
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text("Summary", size=30, weight=ft.FontWeight.BOLD, color="#00E5FF", font_family="Inter"),
                        summary_text,
                        save_btn
                    ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
                    expand=True,
                    bgcolor=None,
                    gradient=static_gradient  # Use static gradient
                )
            ]
        )

    def save_performance_and_continue():
        user = page.session.get("selected_user")
        if not user:
            page.go("/users")
            return
        journal = load_journal()
        entry = {
            "user": user["name"],
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "summary": "Good: Strong breath support. Needs Improvement: Posture, jaw tension. Tips: Practice with a mirror, do relaxation exercises.",
            "critique": [
                {"timestamp": "00:12", "critique": "Watch your posture!"},
                {"timestamp": "00:34", "critique": "Try to relax your jaw."},
                {"timestamp": "01:10", "critique": "Good breath support here!"},
            ],
            "video": "video_placeholder.mp4"
        }
        journal.append(entry)
        save_journal(journal)
        page.go("/home")

    # --- Vocal Journal Screen ---
    def vocal_journal_screen():
        user = page.session.get("selected_user")
        if not user:
            page.go("/users")
            return
        journal = load_journal()
        user_entries = [entry for entry in journal if entry["user"] == user["name"]]
        entry_controls = [
            ft.ListTile(
                title=ft.Text(f"Session on {entry['date']}"),
                subtitle=ft.Text(entry["summary"].split('.')[0] + "..."),
                on_click=lambda e, ent=entry: select_journal_entry(ent)
            )
            for entry in user_entries
        ]
        return ft.View(
            "/journal",
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text("Your Vocal Journal", size=30, weight=ft.FontWeight.BOLD, color="#00E5FF", font_family="Inter"),
                        ft.Column(entry_controls, scroll=ft.ScrollMode.AUTO, expand=True)
                    ], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
                    expand=True,
                    bgcolor=None,
                    gradient=static_gradient  # Use static gradient
                )
            ]
        )

    def select_journal_entry(entry):
        page.session.set("selected_journal_entry", entry)
        page.go("/journal_entry")

    # --- Journal Entry Screen ---
    def journal_entry_screen():
        entry = page.session.get("selected_journal_entry")
        if not entry:
            page.go("/journal")
            return
        critique_bubbles = [
            ft.Container(
                content=ft.Row([
                    ft.Text(cb["timestamp"], size=12, color=ft.Colors.BLUE_GREY_200),
                    ft.Container(
                        content=ft.Text(cb["critique"], size=14, color=ft.Colors.WHITE),
                        bgcolor="#3D3D5C" if i % 2 == 0 else "#2D2D3C",
                        border_radius=8,
                        padding=10,
                        margin=ft.margin.only(left=10)
                    )
                ], alignment=ft.MainAxisAlignment.START),
                margin=ft.margin.only(bottom=8)
            )
            for i, cb in enumerate(entry["critique"])
        ]
        return ft.View(
            "/journal_entry",
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text(f"Journal Entry: {entry['date']}", size=24, color="#00E5FF", font_family="Inter"),
                        ft.Text("Summary:", size=18, color="#00E5FF", font_family="Inter"),
                        ft.Text(entry["summary"], size=16, font_family="Inter"),
                        ft.Text("Critique:", size=18, color="#00E5FF", font_family="Inter"),
                        ft.Column(critique_bubbles, scroll=ft.ScrollMode.AUTO, height=200),
                        ft.Text("Video (placeholder):", size=18, color="#00E5FF", font_family="Inter"),
                        ft.Image(src="https://via.placeholder.com/400x300?text=Performance+Video", width=400, height=300),
                        ft.ElevatedButton("Back to Journal", on_click=lambda e: page.go("/journal"))
                    ], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
                    expand=True,
                    bgcolor=None,
                    gradient=static_gradient  # Use static gradient
                )
            ]
        )

if __name__ == "__main__":
    ft.app(target=main) 