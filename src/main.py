import flet as ft
import math # For hover animation
import threading
import time
import json
import os
import sys
from functools import lru_cache
from typing import Optional, Dict, List, Any

# Lazy imports for heavy dependencies
_imports = {
    'cv2': None,
    'PIL': None,
    'numpy': None,
    'base64': None,
    'io': None,
    'random': None,
    'datetime': None
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
    return _imports[name]

sys.path.append(os.getcwd()+'/src/')
from vocal_coaching_rag import generate_rag_critique

USER_DATA_FILE = os.getcwd()+"/src/users.json"
JOURNAL_DATA_FILE = os.getcwd()+"/src/journal.json"

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

def main(page: ft.Page):
    page.title = "Voxure - AI Vocal Coach"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 10  # Reduced padding
    page.window_width = 1000
    page.window_height = 800
    page.bgcolor = "#181A20"

    # Simplified gradient background
    gradient = ft.LinearGradient(
        begin=ft.alignment.top_center,
        end=ft.alignment.bottom_center,
        colors=[
            "#1a237e",  # Deep Blue
            "#9c27b0",  # Lavender
        ],
        stops=[0.0, 1.0]
    )
    page.gradient = gradient

    def animate_gradient(is_splash_screen=False):
        import math
        
        last_update = time.time()
        update_interval = 0.1  # 10 FPS
        
        while True:
            current_time = time.time()
            if current_time - last_update < update_interval:
                time.sleep(0.01)  # Small sleep to prevent CPU hogging
                continue
                
            if is_splash_screen:
                t = current_time * 5
                x = math.sin(t) * 0.1
                y = math.cos(t) * 0.1
            else:
                t = current_time * 0.2
                x = math.sin(t) * 0.05
                y = math.cos(t) * 0.05
            
            # Only update if values have changed significantly
            if abs(gradient.begin.x - (x + 0.5)) > 0.01 or abs(gradient.begin.y - (y + 0.5)) > 0.01:
                gradient.begin = ft.alignment.Alignment(x + 0.5, y + 0.5)
                gradient.end = ft.alignment.Alignment(x + 0.5, y + 0.5)
                page.update()
                last_update = current_time

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
        # Use centralized imports
        cv2 = get_import('cv2')
        PIL = get_import('PIL')
        io = get_import('io')
        base64 = get_import('base64')
        
        # Resize frame to reduce memory usage
        frame = cv2.resize(frame, (640, 480))
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(img)
        
        # Optimize image quality and size
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=85, optimize=True)
        data = buf.getvalue()
        
        # Update image with fade effect using safe update
        update_ui_safely(video_image, opacity=0)
        update_ui_safely(video_image, src_base64=base64.b64encode(data).decode())
        update_ui_safely(video_image, opacity=1)

    # Start gradient animation in a background thread
    threading.Thread(target=lambda: animate_gradient(False), daemon=True).start()

    # Font setup (using system fonts where possible)
    page.fonts = {
        "Roboto": "https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap",  # Reduced variants
        "Playfair Display": "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap"  # Reduced variants
    }
    page.theme = ft.Theme(font_family="Playfair Display")

    # State
    camera_active = False
    camera_thread = None
    stop_camera_event = threading.Event()
    uploaded_video_path = None
    last_frame_bytes = None
    page.session.set("selected_user", None)
    page.session.set("selected_journal_entry", None)
    page.overlay.clear()

    # --- File Picker for Video Upload ---
    def on_file_picked(e: ft.FilePickerResultEvent):
        nonlocal uploaded_video_path
        if e.files:
            selected_file = e.files[0]
            uploaded_video_path = selected_file.path
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

            # --- Prompt user for lyrics and performance description ---
            lyrics_field = ft.TextField(label="Lyrics", multiline=True, width=400)
            desc_field = ft.TextField(label="Performance Description", multiline=True, width=400)
            def on_submit_critique(ev):
                lyrics = lyrics_field.value
                performance_description = desc_field.value
                # Optionally, auto-extract from video/audio here
                # lyrics, performance_description = auto_extract(uploaded_video_path)
                feedback_text_area.value = "Generating AI critique..."
                feedback_text_area.update()
                critique, entry = invoke_critique(lyrics, performance_description, video_path=uploaded_video_path)
                feedback_text_area.value = f"AI Critique:\n{critique}"
                feedback_text_area.update()
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
            feedback_text_area.value = "Video selection cancelled."
            feedback_text_area.update()

    file_picker = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(file_picker) # Add file picker to page overlay

    # --- Placeholder for Video Display ---
    video_image = ft.Image(
        src="https://via.placeholder.com/600x400?text=No+Video",
        width=600,
        height=400,
        fit=ft.ImageFit.CONTAIN,
        gapless_playback=True,  # Enable gapless playback
        repeat=ft.ImageRepeat.NO_REPEAT,  # Prevent image repetition
        animate_opacity=ft.Animation(300, "easeInOut"),  # Smooth opacity transitions
    )
    
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
        # Adding some theming for dropdowns
        border_color=ft.Colors.BLUE_ACCENT_100,
        focused_border_color=ft.Colors.PURPLE_ACCENT,
        label_style=ft.TextStyle(font_family="Playfair Display"),
        # content_padding=10 # Flet dropdowns might not have direct content_padding like this for options
    )

    dd_role = ft.Dropdown(
        label="Role",
        hint_text="Choose your role",
        options=[
            ft.dropdown.Option("Lead Vocalist"), ft.dropdown.Option("Backing Vocalist"),
            ft.dropdown.Option("Chorus Singer"),
        ],
        width=200,
        border_color=ft.Colors.BLUE_ACCENT_100,
        focused_border_color=ft.Colors.PURPLE_ACCENT,
        label_style=ft.TextStyle(font_family="Playfair Display"),
    )

    dd_range = ft.Dropdown(
        label="Vocal Range",
        hint_text="Choose your vocal range",
        options=[
            ft.dropdown.Option("Soprano"), ft.dropdown.Option("Mezzo-Soprano"), ft.dropdown.Option("Alto"),
            ft.dropdown.Option("Tenor"), ft.dropdown.Option("Baritone"), ft.dropdown.Option("Bass"),
        ],
        width=200,
        border_color=ft.Colors.BLUE_ACCENT_100,
        focused_border_color=ft.Colors.PURPLE_ACCENT,
        label_style=ft.TextStyle(font_family="Playfair Display"),
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

    def camera_loop():
        # Use centralized imports
        cv2 = get_import('cv2')
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        while not stop_camera_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            update_video_image_from_frame(frame)
            time.sleep(0.06)  # ~15 FPS
        cap.release()

    def start_camera(e=None):
        nonlocal camera_active, camera_thread
        if camera_active:
            return
        stop_camera_event.clear()
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        feedback_text_area.value = "Camera started."
        feedback_text_area.update()
        camera_active = True

    def stop_camera():
        nonlocal camera_active
        if camera_active:
            stop_camera_event.set()
            camera_active = False
            feedback_text_area.value = "Camera stopped."
            feedback_text_area.update()
            video_image.src = "https://via.placeholder.com/600x400?text=No+Video"
            video_image.src_base64 = None
            video_image.update()

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
        on_click=start_camera,
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
        on_click=lambda e: stop_camera(),
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
        on_click=start_camera,  # For now, just start camera
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
        font_family="Playfair Display",
        color=ft.Colors.BLUE_GREY_200
    )
    
    feedback_container = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("Real-time Feedback & Coaching", size=20, weight=ft.FontWeight.BOLD, font_family="Playfair Display", color=ft.Colors.CYAN_ACCENT_100),
                ft.Divider(color=ft.Colors.with_opacity(0.5, ft.Colors.BLUE_ACCENT_100)),
                feedback_text_area
            ],
            scroll=ft.ScrollMode.AUTO,
            spacing=10
        ),
        expand=True,
        bgcolor="#1AFFFFFF", # More subtle background (was ft.Colors.with_opacity(0.1, ft.Colors.WHITE10))
        border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.PURPLE_ACCENT_100)),
        border_radius=10,
        padding=15,
        margin=ft.margin.only(top=20),
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=10,
            color=ft.Colors.with_opacity(0.2, ft.Colors.BLACK),
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
                ft.Text(title_text, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, font_family="Playfair Display", color=ft.Colors.LIGHT_BLUE_100),
                ft.Divider(color=ft.Colors.with_opacity(0.3, ft.Colors.BLUE_ACCENT_100)),
                content_widget if content_widget else ft.Container(height=5)
            ]),
            padding=15, 
            bgcolor="#3DFFFFFF",
            border_radius=8, 
            margin=ft.margin.only(top=10),
            shadow=ft.BoxShadow(blur_radius=5, color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK))
        )

    def create_other_features_column():
        progress_tracker_placeholder = create_placeholder_container(
            "Progress Tracker Area",
            ft.Text("Your improvements and session history will be shown here.", font_family="Playfair Display", color=ft.Colors.BLUE_GREY_300)
        )
        
        warmup_exercises_placeholder = create_placeholder_container(
            "Personalized Warmup Exercises",
            ft.ElevatedButton(
                "Generate Warmup", 
                icon=ft.Icons.FITNESS_CENTER, 
                style=ft.ButtonStyle(bgcolor=ft.Colors.TEAL_ACCENT_700, color=ft.Colors.WHITE),
                on_hover=on_hover_animation,
                scale=ft.Scale(1), animate_scale=ft.Animation(300, "easeOutCubic")
            )
        )

        community_features_placeholder = create_placeholder_container(
            "Community Features (Blog, Messaging, etc.)",
            ft.Text("Connect with others, share insights, and get expert advice.", font_family="Playfair Display", color=ft.Colors.BLUE_GREY_300)
        )

        return ft.Column(
            controls=[
                ft.Divider(height=20, thickness=1, color=ft.Colors.with_opacity(0.5, ft.Colors.BLUE_ACCENT_200)),
                ft.Text("More Tools & Features", theme_style=ft.TextThemeStyle.HEADLINE_SMALL, font_family="Playfair Display", color=ft.Colors.CYAN_ACCENT_200),
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
            ft.Text("VOXURE", theme_style=ft.TextThemeStyle.DISPLAY_SMALL, weight=ft.FontWeight.BOLD, font_family="Playfair Display", color="#00E5FF",
                      spans=[ft.TextSpan("™", ft.TextStyle(size=12, font_family="Playfair Display", color="#00E5FF", weight=ft.FontWeight.NORMAL))]
            ),
            ft.Text("Your Personal AI Vocal Coach", theme_style=ft.TextThemeStyle.TITLE_LARGE, font_family="Playfair Display", color=ft.Colors.BLUE_GREY_100, weight=ft.FontWeight.W_300, text_align=ft.TextAlign.CENTER),
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
        # Stop the regular animation and start the speaker-like animation
        for thread in threading.enumerate():
            if thread.name == "Thread-1":  # The gradient animation thread
                thread._stop()
        
        # Start the speaker-like animation
        threading.Thread(target=lambda: animate_gradient(True), daemon=True, name="SplashAnimation").start()

        voxure_text = ft.Text(
            "Voxure",
            size=50,
            weight=ft.FontWeight.BOLD,
            color="#00E5FF",
            opacity=0,
            animate_opacity=ft.Animation(500, "easeInOut"),  # Faster animation
        )
        tagline_text = ft.Text(
            "Your voice, your choice.",
            size=24,
            color="#00E5FF",
            opacity=0,
            animate_opacity=ft.Animation(500, "easeInOut"),  # Faster animation
        )

        def fade_in_sequence():
            time.sleep(0.2)  # Reduced delay
            voxure_text.opacity = 1
            page.update()
            time.sleep(0.5)  # Reduced delay
            tagline_text.opacity = 1
            page.update()
            time.sleep(0.8)  # Reduced delay
            # Stop the speaker-like animation and restart the regular animation
            for thread in threading.enumerate():
                if thread.name == "SplashAnimation":
                    thread._stop()
            threading.Thread(target=lambda: animate_gradient(False), daemon=True).start()
            page.go("/users")

        # Start the fade-in sequence in a background thread
        threading.Thread(target=fade_in_sequence, daemon=True).start()

        return ft.View(
            "/",
            controls=[
                ft.Container(
                    content=ft.Column(
                        [voxure_text, tagline_text],
                        alignment=ft.MainAxisAlignment.CENTER,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    alignment=ft.alignment.center,
                    expand=True,
                    bgcolor="#181A20",
                )
            ],
        )

    # --- User Selection Screen ---
    def user_selection_screen():
        users = load_users()
        user_avatars = [
            ft.Container(
                content=ft.Column([
                    ft.CircleAvatar(
                        content=ft.Icon(ft.Icons.MUSIC_NOTE, size=40, color=ft.Colors.PURPLE_ACCENT),
                        radius=40,
                        bgcolor=ft.Colors.BLUE_ACCENT_100
                    ),
                    ft.Text(user["name"], size=16, color="#00E5FF", text_align=ft.TextAlign.CENTER)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                on_click=lambda e, u=user: select_user(u),
                margin=10,
                tooltip=f"{user['name']} ({user.get('singer_type','?')})"
            )
            for user in users
        ]
        add_user_btn = ft.Container(
            content=ft.Column([
                ft.CircleAvatar(
                    content=ft.Icon(ft.Icons.ADD, size=40, color=ft.Colors.WHITE),
                    radius=40,
                    bgcolor=ft.Colors.PURPLE_ACCENT
                ),
                ft.Text("Add User", size=16, color="#00E5FF", text_align=ft.TextAlign.CENTER)
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            on_click=lambda e: page.go("/create_user"),
            margin=10,
            tooltip="Add new user"
        )
        return ft.View(
            "/users",
            controls=[
                ft.Column([
                    ft.Text("Select Your Profile", size=30, weight=ft.FontWeight.BOLD, color="#00E5FF"),
                    ft.Row(user_avatars + [add_user_btn], alignment=ft.MainAxisAlignment.CENTER, expand=True)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)
            ]
        )

    # --- Create User Screen ---
    def create_user_screen():
        name_field = ft.TextField(label="Name")
        age_field = ft.TextField(label="Age")
        singer_type_field = ft.Dropdown(
            label="Singer Type",
            options=[ft.dropdown.Option(x) for x in ["Soprano", "Mezzo-Soprano", "Alto", "Tenor", "Baritone", "Bass"]],
        )
        genres_practiced_field = ft.TextField(label="Genres Practiced (comma separated)")
        genres_learn_field = ft.TextField(label="Genres to Learn (comma separated)")
        goals_field = ft.TextField(label="Voice Improvement Goals", multiline=True)
        submit_btn = ft.ElevatedButton(
            text="Create Profile",
            on_click=lambda e: save_new_user(
                name_field.value, age_field.value, singer_type_field.value,
                genres_practiced_field.value, genres_learn_field.value, goals_field.value
            )
        )
        return ft.View(
            "/create_user",
            controls=[
                ft.Column([
                    ft.Text("Create New User", size=30, weight=ft.FontWeight.BOLD, color="#00E5FF"),
                    name_field, age_field, singer_type_field,
                    genres_practiced_field, genres_learn_field, goals_field,
                    submit_btn
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)
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
        def animate():
            for i in range(101):
                progress.value = i / 100
                note_icon.icon = random.choice([ft.Icons.MUSIC_NOTE, ft.Icons.MUSIC_OFF, ft.Icons.AUDIOTRACK])
                page.update()
                time.sleep(0.03)
            dialog.open = False
            page.update()
            page.go("/summary")

        overlay = ft.AlertDialog(
            title=ft.Text("Practice and Learn", size=18, weight=ft.FontWeight.BOLD, color="#00E5FF"),
            content=ft.Container(
                width=350,
                alignment=ft.alignment.center,
                content=ft.Column([
                    ft.ElevatedButton(
                        "Record Yourself",
                        icon=ft.Icons.VIDEOCAM,
                        on_click=lambda e: on_record_yourself,
                        width=250,
                    ),
                    ft.ElevatedButton(
                        "Upload Video",
                        icon=ft.Icons.UPLOAD_FILE,
                        on_click=lambda e: on_upload_video,
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

        def on_record_yourself(e):
            setattr(overlay, 'open', False)
            page.update()
            # Wait a short moment before navigating
            def delayed_nav():
                time.sleep(0.1)
                page.go("/record")
            threading.Thread(target=delayed_nav, daemon=True).start()

        def on_upload_video(e):
            setattr(overlay, 'open', False)
            page.update()
            page.open(dialog) 
            threading.Thread(target=animate, daemon=True).start()
            
            # Wait a short moment before navigating
            def delayed_nav():
                time.sleep(0.1)
                page.go("/record")

            threading.Thread(target=delayed_nav, daemon=True).start()

        progress = ft.ProgressBar(width=300)
        note_icon = ft.Icon(ft.Icons.MUSIC_NOTE, size=40, color=ft.Colors.PURPLE_ACCENT)
        dialog = ft.AlertDialog(
            title=ft.Text("Uploading and Analyzing..."),
            content=ft.Column([
                progress,
                note_icon,
                ft.Text("Analyzing your performance...", size=16)
            ], spacing=20,),
            on_dismiss=lambda e: print("Dialog dismissed!"), 
            title_padding=ft.padding.all(25),
        )
        
        user = page.session.get("selected_user")
        # Left: Navigation
        nav_buttons = ft.Column([
            ft.ElevatedButton("Practice and Learn", icon=ft.Icons.MIC, on_click=lambda e: page.open(overlay)),
            ft.ElevatedButton("Your Vocal Journal", icon=ft.Icons.BOOK, on_click=lambda e: page.go("/journal")),
        ], alignment=ft.MainAxisAlignment.START, spacing=20)
        # Right: Progress plot (placeholder image)
        progress_plot = ft.Image(
            src="https://quickchart.io/chart?c={type:'line',data:{labels:['Jan','Feb','Mar'],datasets:[{label:'Progress',data:[1,2,3]}]}}",
            width=400, height=300, fit=ft.ImageFit.CONTAIN
        )
        return ft.View(
            "/home",
            controls=[
                ft.Row([
                    ft.Container(nav_buttons, width=250, bgcolor="#23272F", border_radius=10, padding=20),
                    ft.VerticalDivider(width=20),
                    ft.Container(
                        ft.Column([
                            ft.Text(f"Welcome, {user['name']}!", size=24, color="#00E5FF"),
                            ft.Text(f"Singer Type: {user['singer_type']}", color="#00E5FF"),
                            ft.Text("Your Progress", size=18, color="#00E5FF"),
                            progress_plot
                        ], spacing=10),
                        expand=True, bgcolor="#181A20", border_radius=10, padding=20
                    )
                ], expand=True, alignment=ft.MainAxisAlignment.CENTER)
            ]
        )

    # --- Recording Screen ---
    def recording_screen():
        # Placeholder for video capture and controls
        genre_dd = ft.Dropdown(label="Genre", options=[ft.dropdown.Option(x) for x in ["Pop", "Rock", "Jazz", "Classical", "Broadway", "Alternative"]])
        pitch_dd = ft.Dropdown(label="Pitch", options=[ft.dropdown.Option(x) for x in ["Low", "Medium", "High"]])
        scale_dd = ft.Dropdown(label="Scale", options=[ft.dropdown.Option(x) for x in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']])
        singer_dd = ft.Dropdown(label="Singer Type", options=[ft.dropdown.Option(x) for x in ["Soprano", "Alto", "Tenor", "Bass"]])
        record_btn = ft.ElevatedButton("Record", icon=ft.Icons.FIBER_MANUAL_RECORD, on_click=lambda e: show_critique_panel())
        pause_btn = ft.ElevatedButton("Pause/Resume", icon=ft.Icons.PAUSE, on_click=lambda e: show_critique_panel())
        stop_btn = ft.ElevatedButton("Stop", icon=ft.Icons.STOP, on_click=lambda e: page.go("/summary"))
        controls = ft.Row([record_btn, pause_btn, stop_btn], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        video_placeholder = ft.Container(
            content=ft.Image(src="https://via.placeholder.com/400x300?text=Video+Capture", width=400, height=300),
            alignment=ft.alignment.center,
            bgcolor="#23272F",
            border_radius=10,
            padding=10
        )
        dropdowns = ft.Row([genre_dd, pitch_dd, scale_dd, singer_dd], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
        return ft.View(
            "/record",
            controls=[
                ft.Column([
                    video_placeholder,
                    dropdowns,
                    controls
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)
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
                ft.Column([
                    ft.Text("Summary", size=30, weight=ft.FontWeight.BOLD, color="#00E5FF"),
                    summary_text,
                    save_btn
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)
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
                ft.Column([
                    ft.Text("Your Vocal Journal", size=30, weight=ft.FontWeight.BOLD, color="#00E5FF"),
                    ft.Column(entry_controls, scroll=ft.ScrollMode.AUTO, expand=True)
                ], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)
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
                ft.Column([
                    ft.Text(f"Journal Entry: {entry['date']}", size=24, color="#00E5FF"),
                    ft.Text("Summary:", size=18, color="#00E5FF"),
                    ft.Text(entry["summary"], size=16),
                    ft.Text("Critique:", size=18, color="#00E5FF"),
                    ft.Column(critique_bubbles, scroll=ft.ScrollMode.AUTO, height=200),
                    ft.Text("Video (placeholder):", size=18, color="#00E5FF"),
                    ft.Image(src="https://via.placeholder.com/400x300?text=Performance+Video", width=400, height=300),
                    ft.ElevatedButton("Back to Journal", on_click=lambda e: page.go("/journal"))
                ], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)
            ]
        )

if __name__ == "__main__":
    ft.app(target=main) 