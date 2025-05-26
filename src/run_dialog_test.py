import flet as ft

def main(page: ft.Page):
    # Define dlg as None so it can be referenced in the callback
    dlg = ft.AlertDialog(
        title=ft.Text("Test Dialog"),
        content=ft.Text("If you see this, dialogs work!"),
        actions=[],
        alignment=ft.alignment.center,
        on_dismiss=lambda e: print("Dialog dismissed!"),
        title_padding=ft.padding.all(25),
    )

    # Define the OK button after dlg is created, so it can reference dlg
    ok_button = ft.TextButton("OK", on_click=lambda e: (setattr(dlg, 'open', False), page.update()))
    dlg.actions = [ok_button]

    page.add(ft.ElevatedButton("Show Dialog", on_click=lambda e: page.open(dlg)))

ft.app(target=main)