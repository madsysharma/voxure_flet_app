import flet as ft
import math # For hover animation
import cv2
import threading
import time
import numpy as np
from PIL import Image
import io
import base64
import json
import os
import random
from datetime import datetime
import sys
import torch

sys.path.append(os.getcwd()+'/src/')
from models import *
from vocal_coaching_rag import generate_rag_critique

USER_DATA_FILE = os.getcwd()+"/src/users.json"
JOURNAL_DATA_FILE = os.getcwd()+"/src/journal.json"

# --- User and Journal Data Management ---
def load_users():
    if not os.path.exists(USER_DATA_FILE):
        return []
    with open(USER_DATA_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_journal():
    if not os.path.exists(JOURNAL_DATA_FILE):
        return []
    with open(JOURNAL_DATA_FILE, "r") as f:
        return json.load(f)

def save_journal(journal):
    with open(JOURNAL_DATA_FILE, "w") as f:
        json.dump(journal, f, indent=2)

def main(page: ft.Page):
    page.title = "Voxure - AI Vocal Coach"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.theme_mode = ft.ThemeMode.DARK # Changed to DARK
    page.padding = 20
    page.window_width = 1000
    page.window_height = 800
    page.bgcolor = "#181A20"  # Set dark background

    # Space-themed gradient background
    page.gradient = ft.LinearGradient(
        begin=ft.alignment.top_center,
        end=ft.alignment.bottom_center,
        colors=[
            "#00102D",  # Deep Blue
            "#00001A",  # Darker Blue/Almost Black
            "#150035",  # Dark Purple
        ],
        stops=[0.0, 0.5, 1.0]
    )

    # Font setup (Roboto from Google Fonts)
    page.fonts = {
        "Roboto": "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap",
        "Exo2": "https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;500;700&display=swap",
        "RobotoMono": "https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap",
        "Playfair Display": "https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap",
        "Libre Baskerville": "https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap"
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
            cap = cv2.VideoCapture(uploaded_video_path)
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
    video_image = ft.Image(src="https://via.placeholder.com/600x400?text=No+Video", width=600, height=400, fit=ft.ImageFit.CONTAIN)
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

    def update_video_image_from_frame(frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        data = buf.getvalue()
        video_image.src_base64 = base64.b64encode(data).decode()
        video_image.update()

    def camera_loop():
        cap = cv2.VideoCapture(0)
        while not stop_camera_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            update_video_image_from_frame(frame)
            time.sleep(0.03)  # ~30 FPS
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
    main_coaching_column = ft.Column(
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
                content_widget if content_widget else ft.Container(height=5) # Minimal content if none provided
            ]),
            padding=15, 
            bgcolor="#3DFFFFFF", # More subtle background (was ft.colors.with_opacity(0.05, ft.colors.WHITE24))
            border_radius=8, 
            margin=ft.margin.only(top=10),
            shadow=ft.BoxShadow(blur_radius=5, color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK))
        )

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

    other_features_column = ft.Column(
        controls=[
            ft.Divider(height=20, thickness=1, color=ft.Colors.with_opacity(0.5, ft.Colors.BLUE_ACCENT_200)),
            ft.Text("More Tools & Features", theme_style=ft.TextThemeStyle.HEADLINE_SMALL, font_family="Playfair Display", color=ft.Colors.CYAN_ACCENT_200),
            progress_tracker_placeholder,
            warmup_exercises_placeholder,
            community_features_placeholder
        ],
        spacing=15, # Increased spacing
        width=600, 
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH
    )
    
    # --- Responsive main content layout ---
    main_content = ft.ResponsiveRow(
        controls=[
            ft.Container(main_coaching_column, col={"xs": 12, "md": 7}, expand=True),
            ft.Container(other_features_column, col={"xs": 12, "md": 5}, expand=True),
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
        voxure_text = ft.Text(
            "Voxure",
            size=50,
            weight=ft.FontWeight.BOLD,
            color="#00E5FF",
            opacity=0,
            animate_opacity=ft.Animation(700, "easeInOut"),
        )
        tagline_text = ft.Text(
            "Your voice, your choice.",
            size=24,
            color="#00E5FF",
            opacity=0,
            animate_opacity=ft.Animation(700, "easeInOut"),
        )

        def fade_in_sequence():
            time.sleep(0.3)
            voxure_text.opacity = 1
            page.update()
            time.sleep(0.8)
            tagline_text.opacity = 1
            page.update()
            time.sleep(1.2)
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
        
        # Create musical note avatars with smiley faces
        def create_musical_note_avatar(user):
            # Different note types with embedded smiley faces
            note_types = [
                "♪",  # Eighth note
                "♫",  # Beamed eighth notes
                "♬",  # Beamed sixteenth notes
                "♩",  # Quarter note
                "♭",  # Flat
                "♮",  # Natural
                "♯"   # Sharp
            ]
            
            # Create a container for the animated note
            note_container = ft.Container(
                content=ft.Stack([
                    # Musical note
                    ft.Text(
                        random.choice(note_types),
                        size=40,
                        color=ft.colors.PURPLE_ACCENT,
                        animate_opacity=ft.Animation(1000, "easeInOut"),
                        animate_scale=ft.Animation(1000, "easeInOut"),
                    ),
                    # Smiley face overlay
                    ft.Container(
                        content=ft.Text(
                            "😊",
                            size=20,
                            color=ft.colors.PURPLE_ACCENT,
                        ),
                        left=30,  # Position the smiley face on the note
                        top=15,
                        animate=ft.animation.Animation(300, "easeInOut"),
                    ),
                ]),
                width=80,
                height=80,
                bgcolor=ft.colors.with_opacity(0.1, ft.colors.PURPLE_ACCENT),
                border_radius=40,
                animate=ft.animation.Animation(300, "easeInOut"),
                on_hover=lambda e: on_avatar_hover(e),
                scale=ft.Scale(1),
                animate_scale=ft.Animation(300, "easeOutCubic"),
            )
            
            # Add hover animation
            def on_avatar_hover(e):
                if e.data == "true":
                    note_container.scale = ft.Scale(1.2)
                    note_container.bgcolor = ft.colors.with_opacity(0.3, ft.colors.PURPLE_ACCENT)
                    note_container.shadow = ft.BoxShadow(
                        spread_radius=2,
                        blur_radius=15,
                        color=ft.colors.with_opacity(0.3, ft.colors.PURPLE_ACCENT),
                        offset=ft.Offset(0, 0),
                        blur_style=ft.ShadowBlurStyle.NORMAL,
                    )
                else:
                    note_container.scale = ft.Scale(1)
                    note_container.bgcolor = ft.colors.with_opacity(0.1, ft.colors.PURPLE_ACCENT)
                    note_container.shadow = None
                note_container.update()
            
            # Add click animation
            def on_avatar_click(e):
                note_container.scale = ft.Scale(0.95)
                note_container.update()
                time.sleep(0.1)
                note_container.scale = ft.Scale(1)
                note_container.update()
                select_user(user)
            
            note_container.on_click = on_avatar_click
            
            return ft.Column([
                note_container,
                ft.Text(
                    user["name"],
                    size=16,
                    color="#00E5FF",
                    text_align=ft.TextAlign.CENTER,
                    weight=ft.FontWeight.W_500,
                ),
                ft.Text(
                    f"({user.get('singer_type', '?')})",
                    size=12,
                    color=ft.colors.BLUE_GREY_300,
                    text_align=ft.TextAlign.CENTER,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
            )

        # Create user avatars
        user_avatars = [create_musical_note_avatar(user) for user in users]
        
        # Add user button with a special animation
        add_user_btn = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Icon(
                        ft.icons.ADD,
                        size=40,
                        color=ft.colors.WHITE,
                    ),
                    width=80,
                    height=80,
                    bgcolor=ft.colors.with_opacity(0.1, ft.colors.GREEN_ACCENT),
                    border_radius=40,
                    animate=ft.animation.Animation(300, "easeInOut"),
                    on_hover=lambda e: on_add_button_hover(e),
                    scale=ft.Scale(1),
                    animate_scale=ft.Animation(300, "easeOutCubic"),
                ),
                ft.Text(
                    "Add User",
                    size=16,
                    color="#00E5FF",
                    text_align=ft.TextAlign.CENTER,
                    weight=ft.FontWeight.W_500,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
            ),
            on_click=lambda e: page.go("/create_user"),
            tooltip="Add new user",
        )
        
        def on_add_button_hover(e):
            if e.data == "true":
                add_user_btn.content.controls[0].scale = ft.Scale(1.1)
                add_user_btn.content.controls[0].bgcolor = ft.colors.with_opacity(0.2, ft.colors.GREEN_ACCENT)
            else:
                add_user_btn.content.controls[0].scale = ft.Scale(1)
                add_user_btn.content.controls[0].bgcolor = ft.colors.with_opacity(0.1, ft.colors.GREEN_ACCENT)
            add_user_btn.update()

        return ft.View(
            "/users",
            controls=[
                ft.Column([
                    ft.Text(
                        "Select Your Profile",
                        size=30,
                        weight=ft.FontWeight.BOLD,
                        color="#00E5FF",
                        animate_opacity=ft.Animation(500, "easeIn"),
                    ),
                    ft.Container(
                        content=ft.Row(
                            user_avatars + [add_user_btn],
                            alignment=ft.MainAxisAlignment.CENTER,
                            spacing=30,
                        ),
                        margin=ft.margin.only(top=40),
                        animate=ft.animation.Animation(500, "easeIn"),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                expand=True,
            )
        ])

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
        # Create a container for the entire screen with animation
        screen = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Voxure", size=40, weight=ft.FontWeight.BOLD),
                    ft.Text("Your AI Vocal Coach", size=20),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            animate=ft.animation.Animation(300, ft.AnimationCurve.EASE_IN),
            opacity=0,
        )

        # Create animated containers for each section with directional animations
        def create_animated_section(content, delay=0, direction="up"):
            offset_map = {
                "up": ft.transform.Offset(0, 0.2),
                "left": ft.transform.Offset(-0.2, 0),
                "right": ft.transform.Offset(0.2, 0),
                "down": ft.transform.Offset(0, -0.2)
            }
            return ft.Container(
                content=content,
                animate=ft.animation.Animation(300, ft.AnimationCurve.EASE_IN),
                opacity=0,
                animate_opacity=300,
                animate_offset=ft.Offset(0, 0),
                offset=offset_map.get(direction, ft.transform.Offset(0, 0.2)),
            )

        # Progress section with animation from right
        progress_section = create_animated_section(
            ft.Column(
                controls=[
                    ft.Text("Your Progress", size=24, weight=ft.FontWeight.BOLD),
                    ft.Container(
                        content=ft.ProgressBar(width=400, color=ft.colors.BLUE),
                        animate=ft.animation.Animation(300, ft.AnimationCurve.EASE_IN),
                        opacity=0,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            delay=100,
            direction="right"
        )

        # Quick actions with animation from left
        quick_actions = create_animated_section(
            ft.Row(
                controls=[
                    ft.ElevatedButton(
                        "Start Practice",
                        icon=ft.icons.PLAY_ARROW,
                        on_click=lambda _: page.go("/practice"),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=10),
                            padding=20,
                        ),
                        on_hover=lambda e: on_button_hover(e),
                        scale=ft.Scale(1),
                        animate_scale=ft.Animation(300, "easeOutCubic"),
                    ),
                    ft.ElevatedButton(
                        "View History",
                        icon=ft.icons.HISTORY,
                        on_click=lambda _: page.go("/history"),
                        style=ft.ButtonStyle(
                            shape=ft.RoundedRectangleBorder(radius=10),
                            padding=20,
                        ),
                        on_hover=lambda e: on_button_hover(e),
                        scale=ft.Scale(1),
                        animate_scale=ft.Animation(300, "easeOutCubic"),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20,
            ),
            delay=200,
            direction="left"
        )

        # Recent activity with animation from bottom
        recent_activity = create_animated_section(
            ft.Column(
                controls=[
                    ft.Text("Recent Activity", size=24, weight=ft.FontWeight.BOLD),
                    ft.ListView(
                        controls=[
                            ft.ListTile(
                                leading=ft.Icon(ft.icons.MUSIC_NOTE),
                                title=ft.Text("Practice Session"),
                                subtitle=ft.Text("30 minutes"),
                            ),
                            ft.ListTile(
                                leading=ft.Icon(ft.icons.ASSESSMENT),
                                title=ft.Text("Progress Report"),
                                subtitle=ft.Text("2 hours ago"),
                            ),
                        ],
                        spacing=10,
                        height=200,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            delay=300,
            direction="down"
        )

        # Exit button with animation
        exit_button = create_animated_section(
            ft.ElevatedButton(
                "Exit to Main Screen",
                icon=ft.icons.EXIT_TO_APP,
                on_click=lambda _: page.go("/users"),
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=10),
                    padding=20,
                    color=ft.colors.RED_400,
                ),
                on_hover=lambda e: on_button_hover(e),
                scale=ft.Scale(1),
                animate_scale=ft.Animation(300, "easeOutCubic"),
            ),
            delay=400,
            direction="up"
        )

        # Practice tip section with animation
        practice_tip = create_animated_section(
            ft.Container(
                content=ft.Column([
                    ft.Text("Today's Practice Tip", size=20, weight=ft.FontWeight.BOLD),
                    ft.Container(
                        content=ft.Text(
                            '"Focus on maintaining proper breath support by engaging your diaphragm. '
                            'This will help you achieve better vocal control and reduce strain."',
                            size=16,
                            italic=True,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        padding=20,
                        bgcolor=ft.colors.with_opacity(0.1, ft.colors.BLUE_ACCENT),
                        border_radius=10,
                    ),
                ]),
                margin=ft.margin.only(top=20),
            ),
            delay=500,
            direction="up"
        )

        # Add all sections to the main column
        screen.content.controls.extend([
            progress_section,
            quick_actions,
            recent_activity,
            practice_tip,
            exit_button,
        ])

        # Function to handle button hover animation
        def on_button_hover(e):
            e.control.scale = ft.Scale(1.05 if e.data == "true" else 1)
            e.control.update()

        # Function to trigger animations
        def start_animations():
            screen.opacity = 1
            screen.update()
            
            # Trigger animations for each section with delay
            sections = [progress_section, quick_actions, recent_activity, practice_tip, exit_button]
            for i, section in enumerate(sections):
                def animate_section(section=section):
                    section.opacity = 1
                    section.offset = ft.transform.Offset(0, 0)
                    section.update()
                
                # Schedule animations with delay
                page.window_to_front()
                page.update()
                time.sleep(0.1 * (i + 1))
                animate_section()

        # Start animations when the page loads
        page.on_load = start_animations

        return screen

    def check_user_progress(user):
        """
        Check if the user has any progress data in their journal entries
        Returns: bool indicating if progress exists
        """
        journal = load_journal()
        user_entries = [entry for entry in journal if entry["user"] == user["name"]]
        return len(user_entries) > 0

    def get_progress_data(user):
        """
        Get progress data from user's journal entries
        Returns: tuple of (dates, scores) for plotting
        """
        journal = load_journal()
        user_entries = [entry for entry in journal if entry["user"] == user["name"]]
        
        dates = []
        scores = []
        cumulative_score = 0
        
        for entry in user_entries:
            dates.append(entry["date"].split()[0])  # Get just the date part
            
            # Calculate posture score
            posture_score = 0
            posture_keywords = {
                "good posture": 1,
                "excellent posture": 1,
                "poor posture": -1,
                "slouching": -1,
                "forward head": -1,
                "rounded shoulders": -1,
                "sway back": -1,
                "flat back": -1,
                "weak abdominals": -1,
                "bent knees": -1,
                "raised chest": -1,
                "bent neck": -1
            }
            
            summary = entry["summary"].lower()
            for keyword, value in posture_keywords.items():
                if keyword in summary:
                    posture_score += value
            
            # Calculate vocal technique score
            vocal_score = 0
            vocal_keywords = {
                "good breath support": 1,
                "excellent breath support": 1,
                "strong voice": 1,
                "clear tone": 1,
                "poor breath support": -1,
                "weak voice": -1,
                "strained voice": -1,
                "pitch issues": -1,
                "tone issues": -1,
                "vocal tension": -1,
                "jaw tension": -1,
                "tongue tension": -1
            }
            
            for keyword, value in vocal_keywords.items():
                if keyword in summary:
                    vocal_score += value
            
            # Combine scores with equal weight
            combined_score = (posture_score + vocal_score) / 2
            
            # Add to cumulative score
            cumulative_score += combined_score
            scores.append(cumulative_score)
        
        # Normalize scores to be between -1 and 1
        if scores:
            max_abs_score = max(abs(min(scores)), abs(max(scores)))
            if max_abs_score > 0:
                scores = [score / max_abs_score for score in scores]
        
        return dates, scores

    # --- Recording Screen ---
    def recording_screen():
        # Create feedback panel
        feedback_panel = ft.Container(
            content=ft.Column([
                ft.Text("Real-time Feedback", size=20, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Text("Waiting for analysis...", id="feedback_text"),
                ft.Text("Timestamp: --:--", id="feedback_timestamp")
            ]),
            bgcolor="#23272F",
            border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.BLUE_ACCENT_100)),
            border_radius=10,
            padding=20,
            width=300
        )
        
        # Create video display
        video_display = ft.Container(
            content=ft.Image(
                src="https://via.placeholder.com/640x480?text=No+Video",
                width=640,
                height=480,
                fit=ft.ImageFit.CONTAIN
            ),
            bgcolor="#23272F",
            border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.BLUE_ACCENT_100)),
            border_radius=10,
            padding=10
        )
        
        # Create controls
        controls = ft.Row([
            ft.ElevatedButton(
                "Start Recording",
                icon=ft.icons.PLAY_ARROW,
                on_click=lambda e: start_recording(e, video_display, feedback_panel)
            ),
            ft.ElevatedButton(
                "Stop Recording",
                icon=ft.icons.STOP,
                on_click=lambda e: stop_recording(e, video_display, feedback_panel)
            )
        ])
        
        # Main layout
        content = ft.Row([
            ft.Column([video_display, controls], spacing=20),
            feedback_panel
        ], spacing=20)
        
        return content

    def start_recording(e, video_display, feedback_panel):
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Initialize models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        blazepose_model = load_blazepose_model().to(device)
        posture_model = PostureMLP().to(device)
        posture_model.load_state_dict(torch.load("best_posture_estimator_model.pth"))
        
        def update_feedback(timestamp, critique):
            # Update the feedback text with timestamp
            feedback_panel.content.controls[2].value = critique
            feedback_panel.content.controls[3].value = f"Last updated: {timestamp:.1f}s"
            
            # Add a subtle animation to highlight the update
            feedback_panel.border = ft.border.all(2, ft.Colors.BLUE_ACCENT_100)
            feedback_panel.update()
            
            # Reset the border after a short delay
            def reset_border():
                time.sleep(0.5)
                feedback_panel.border = ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.BLUE_ACCENT_100))
                feedback_panel.update()
            
            threading.Thread(target=reset_border, daemon=True).start()
        
        def process_frame():
            last_update_time = 0
            for frame, analysis in process_live_recording(cap, blazepose_model, posture_model, device, update_feedback):
                current_time = time.time()
                
                # If no issues are detected and it's been 10 seconds since last update,
                # show a positive feedback message
                if not analysis["issues"] and current_time - last_update_time >= 10.0:
                    update_feedback(current_time, "Good posture and vocal technique! Keep it up!")
                    last_update_time = current_time
                
                # Convert frame to base64 for display
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                video_display.content.src = f"data:image/jpeg;base64,{img_base64}"
                video_display.update()
        
        # Start processing in a separate thread
        import threading
        processing_thread = threading.Thread(target=process_frame)
        processing_thread.start()

    def stop_recording(e, video_display, feedback_panel):
        # Stop camera and processing
        cap.release()
        video_display.content.src = "https://via.placeholder.com/640x480?text=No+Video"
        video_display.update()
        feedback_panel.content.controls[2].value = "Recording stopped"
        feedback_panel.content.controls[3].value = "Timestamp: --:--"
        feedback_panel.update()

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