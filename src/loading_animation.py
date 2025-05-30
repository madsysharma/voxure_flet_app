import flet as ft
from pathlib import Path
import os

class LoadingAnimation:
    def __init__(self, page: ft.Page):
        self.page = page
        self.animation_path = self._get_animation_path()
        
    def _get_animation_path(self):
        """Get the path to the Lottie animation file"""
        anim_dir = os.path.join(os.getcwd(), 'storage', 'animations')
        return os.path.join(anim_dir, 'loading_animation.lottie')
    
    def show(self, message: str = "Loading..."):
        """Show the loading animation with a message"""
        # Create a container for the animation
        container = ft.Container(
            content=ft.Column(
                [
                    ft.Lottie(
                        src=self.animation_path,
                        repeat=True,
                        reverse=False,
                        animate=True,
                        width=200,
                        height=200,
                    ),
                    ft.Text(message, size=16, weight=ft.FontWeight.W_500),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            alignment=ft.alignment.center,
            expand=True,
        )
        
        # Add the container to the page
        self.page.add(container)
        self.page.update()
        return container
    
    def hide(self, container: ft.Container):
        """Hide the loading animation"""
        self.page.remove(container)
        self.page.update()

def show_loading(page: ft.Page, message: str = "Loading..."):
    """Helper function to show loading animation"""
    loader = LoadingAnimation(page)
    return loader.show(message)

def hide_loading(page: ft.Page, container: ft.Container):
    """Helper function to hide loading animation"""
    loader = LoadingAnimation(page)
    loader.hide(container) 