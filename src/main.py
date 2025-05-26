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
        "RobotoMono": "https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap"
    }
    page.theme = ft.Theme(font_family="Roboto")

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
        label_style=ft.TextStyle(font_family="Roboto"),
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
        label_style=ft.TextStyle(font_family="Roboto"),
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
        label_style=ft.TextStyle(font_family="Roboto"),
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
        font_family="Roboto",
        color=ft.Colors.BLUE_GREY_200
    )
    
    feedback_container = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("Real-time Feedback & Coaching", size=20, weight=ft.FontWeight.BOLD, font_family="Exo2", color=ft.Colors.CYAN_ACCENT_100),
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
                ft.Text(title_text, theme_style=ft.TextThemeStyle.TITLE_MEDIUM, font_family="Exo2", color=ft.Colors.LIGHT_BLUE_100),
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
        ft.Text("Your improvements and session history will be shown here.", font_family="Roboto", color=ft.Colors.BLUE_GREY_300)
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
        ft.Text("Connect with others, share insights, and get expert advice.", font_family="Roboto", color=ft.Colors.BLUE_GREY_300)
    )

    other_features_column = ft.Column(
        controls=[
            ft.Divider(height=20, thickness=1, color=ft.Colors.with_opacity(0.5, ft.Colors.BLUE_ACCENT_200)),
            ft.Text("More Tools & Features", theme_style=ft.TextThemeStyle.HEADLINE_SMALL, font_family="Exo2", color=ft.Colors.CYAN_ACCENT_200),
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
            ft.Text("VOXURE", theme_style=ft.TextThemeStyle.DISPLAY_SMALL, weight=ft.FontWeight.BOLD, font_family="Exo2", color="#00E5FF",
                      spans=[ft.TextSpan("™", ft.TextStyle(size=12, font_family="Exo2", color="#00E5FF", weight=ft.FontWeight.NORMAL))]
            ),
            ft.Text("Your Personal AI Vocal Coach", theme_style=ft.TextThemeStyle.TITLE_LARGE, font_family="Exo2", color=ft.Colors.BLUE_GREY_100, weight=ft.FontWeight.W_300, text_align=ft.TextAlign.CENTER),
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