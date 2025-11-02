from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.filechooser import FileChooserIconView, FileChooserListView
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.video import Video
import cv2
from kivy.uix.popup import Popup
from kivy.clock import Clock
import os
import shutil
from plyer import filechooser  
import re


class Start(Screen):
    pass

class Upload(Screen):
    pass

class NBM(Screen):
    def __init__(self, **kwargs):
        super(NBM, self).__init__(**kwargs)
        self.name = 'nbm'  
        self.video_files = []
        self.current_match = None
        self.current_video_index = 0
        self.video_player = None
        self.app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploaded_videos')
        os.makedirs(self.app_dir, exist_ok=True)
        self.current_uploaded_video_path = None
        
        if not os.path.exists("archives"):#archives directory if it doesn't exist
            os.makedirs("arcives")
    #cocapoo
    def set_match(self, match_number):
        self.current_match = match_number
        self.load_videos_for_match()
    
    def load_videos_for_match(self):
        self.video_files = []
        folder_path = f"videos{self.current_match}" if self.current_match else "videos"
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                    self.video_files.append(os.path.join(folder_path, file))
            self.video_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.current_video_index = 0 if self.video_files else 0
    
    def on_enter(self):
        if self.current_match:
            self.load_videos_for_match()
        # else:

        #     folder_path = "videos"
        #     self.video_files = []
        #     if os.path.exists(folder_path):
        #         for file in os.listdir(folder_path):
        #             if file.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        #                 self.video_files.append(os.path.join(folder_path, file))
                
        #         try:
        #             self.video_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        #         except:
        #             self.video_files.sort()
        
        self.create_ui()
    
    def create_ui(self):
        self.clear_widgets()
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        top_layout = BoxLayout(orientation='vertical', size_hint=(1, 0.1), spacing=10)
        
        match_label = Label(
            text=f"Match {self.current_match}" if self.current_match else "No match selected",
            font_size="20sp",
            size_hint=(1, 3),
            color=(0, 0.27, 0, 0.6)
        )
        
        upload_button = Button(
            text="Upload Video",
            font_size="18sp",
            size_hint=(1, 5),
            background_color=(0, 0.27, 0, 0.6),
        )
        upload_button.bind(on_press=self.upload_file_nbm)
        top_layout.add_widget(upload_button)
        top_layout.add_widget(match_label)
        layout.add_widget(top_layout)

        self.video_player = Video(options={'eos': 'stop'}, size_hint=(1, 0.45))
        if self.video_files and len(self.video_files) > 0:
            self.video_player.source = self.video_files[self.current_video_index]
            self.video_player.state = 'play' 
        layout.add_widget(self.video_player)
        
        nav_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=10)
        
        self.prev_button = Button(text='Previous', background_color=(0, 0.27, 0, 0.6))
        self.prev_button.bind(on_press=self.load_previous_video)
        nav_layout.add_widget(self.prev_button)
        
        play_pause_button = Button(text='Play/Pause', background_color=(0, 0.27, 0, 0.6))
        play_pause_button.bind(on_press=self.toggle_play_pause)
        nav_layout.add_widget(play_pause_button)
        
        self.next_button = Button(text='Next', background_color=(0, 0.27, 0, 0.6))
        self.next_button.bind(on_press=self.load_next_video)
        nav_layout.add_widget(self.next_button)
        
        archive_button = Button(text='Add to Archives',background_color=(0, 0.27, 0, 0.6))
        archive_button.bind(on_press=self.show_archive_popup)
        nav_layout.add_widget(archive_button)
        
        layout.add_widget(nav_layout)
        
        footer_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing = 10)
        
        start_button = Button(
            text="Back to Start Menu", background_color=(0, 0.27, 0, 0.6)
        )
        start_button.bind(on_release=lambda x: self.go_to_screen("start", "right"))
        footer_layout.add_widget(start_button)
        
        archives_button = Button(
            text="View Archives", background_color=(0, 0.27, 0, 0.6)
        )
        archives_button.bind(on_release=lambda x: self.go_to_screen("archives", "left"))
        footer_layout.add_widget(archives_button)
        
        layout.add_widget(footer_layout)
        self.update_buttons()
        self.add_widget(layout)
    
    def go_to_screen(self, screen_name, direction):
        if self.video_player:
            self.video_player.state = 'stop'
        self.manager.current = screen_name
        self.manager.transition.direction = direction
    
    def on_leave(self):
        if self.video_player:
            self.video_player.state = 'stop'


    def load_previous_video(self, instance):
        if self.current_video_index > 0:
            self.current_video_index -= 1
            self.load_current_video()
    
    def load_next_video(self, instance):
        if self.current_video_index < len(self.video_files) - 1:
            self.current_video_index += 1
            self.load_current_video()
    
    def toggle_play_pause(self, instance):
        if self.video_player.state == 'play':
            self.video_player.state = 'pause'
        else:
            self.video_player.state = 'play'
    
    def load_current_video(self):
        self.video_player.state = 'stop'
        
        if not self.video_files or len(self.video_files) == 0:
            print("No videos to load")
            return
            
        self.video_player.source = self.video_files[self.current_video_index]
        self.video_player.state = 'play' 
        self.update_buttons()
    
    def update_buttons(self):
        has_videos = len(self.video_files) > 0
        self.prev_button.disabled = not has_videos or (self.current_video_index == 0)
        self.next_button.disabled = not has_videos or (self.current_video_index == len(self.video_files) - 1)
    
    def upload_file_nbm(self, instance):
        filechooser.open_file(
            on_selection=self.handle_selection_nbm,
            filters=[["Video Files", "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]],
            path = os.path.join(os.path.expanduser("~"), "Downloads")
        )
    
    def handle_selection_nbm(self, selection):
        if selection and len(selection) > 0:
            video_path = selection[0]
            self.save_video_nbm(video_path)
    
    def save_video_nbm(self, video_path):
        filename = os.path.basename(video_path)
        destination = os.path.join(self.app_dir, filename)
        shutil.copy2(video_path, destination)
        self.current_uploaded_video_path = destination
        match_pattern = re.search(r'match(\d+)', filename.lower())
        if match_pattern:
            match_number = match_pattern.group(1)
            self.current_match = match_number
            self.load_videos_for_match()
            self.create_ui()
        else:
            self.show_error_popup("Error", "No match number detected in filename.\nPlease include 'match#' in the filename.")
    
    def show_archive_popup(self, instance):
        if not self.video_files:
            self.show_error_popup("Error", "No video available to archive")
            return
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(
            text="Enter a name for the archived video:",
            font_size="18sp",
            size_hint=(1, 0.3)
        ))
        self.archive_name_input = TextInput(
            hint_text="Video name",
            multiline=False,
            size_hint=(1, 0.3),
            font_size="18sp",
            cursor_color=(0.2, 0.2, 0.2, 1)
        )
        content.add_widget(self.archive_name_input)
        buttons_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.4), spacing=10)
        
        cancel_button = Button(
            text="Cancel",
            size_hint=(0.5, 1),
            background_color = (0.93, 0.98, 0.93, 1)#(0.8, 0.2, 0.2, 0.6)
        )
        
        save_button = Button(
            text="Save to Archives",
            size_hint=(0.5, 1),
            background_color=(0.93, 0.98, 0.93, 1)#(0, 0.27, 0, 0.6)
        )
        
        buttons_layout.add_widget(cancel_button)
        buttons_layout.add_widget(save_button)
        content.add_widget(buttons_layout)
        popup = Popup(
            title="Archive Video",
            content=content,
            size_hint=(0.8, 0.4),
            auto_dismiss=False
        )
        cancel_button.bind(on_press=popup.dismiss)
        save_button.bind(on_press=lambda x: self.add_to_archives(popup))
        popup.open()
    
    def add_to_archives(self, popup):
        if not self.video_files:
            popup.dismiss()
            return
                
        video_name = self.archive_name_input.text.strip()
        
        if not video_name:
            if self.current_match:
                video_name = f"Match{self.current_match}_Video{self.current_video_index + 1}"
            else:
                video_name = f"Video{self.current_video_index + 1}"
        
        valid_name = re.sub(r'[\\/*?:"<>|]', "_", video_name)
        
        dest_path = os.path.join("archives", f"{valid_name}.mp4")
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join("archives", f"{valid_name}_{counter}.mp4")
            counter += 1
        shutil.copy(f"vid{self.current_match}.mp4", dest_path)
        popup.dismiss()
        self.show_success_popup(f"Video saved to archives as:\n{os.path.basename(dest_path)}")

    def show_success_popup(self, message):
        popup = Popup(title='Success', 
                      content=Label(text=message),
                      size_hint=(0.6, 0.3),
                      background_color=(0.93, 0.98, 0.93, 1))
        popup.open()
        popup.bind(on_touch_down=lambda instance, touch: popup.dismiss() if popup.collide_point(*touch.pos) else None)
    
    def show_error_popup(self, title, message):
        popup = Popup(title=title, 
                      content=Label(text=message),
                      size_hint=(0.6, 0.3),
                      background_color=(0.93, 0.98, 0.93, 1))
        popup.open()
        popup.bind(on_touch_down=lambda instance, touch: popup.dismiss() if popup.collide_point(*touch.pos) else None)

class Archives(Screen):
    def __init__(self, **kwargs):
        super(Archives, self).__init__(**kwargs)
        self.name = 'archives'
        self.archives_dir = os.path.join(os.path.expanduser("~"), "Downloads", "archives")
        os.makedirs(self.archives_dir, exist_ok=True)
        self.video_player = None
        self.current_video_path = None
    
    def on_enter(self):
        self.create_ui()
    
    def upload_file(self, instance):
        filechooser.open_file(
            on_selection=self.handle_selection,
            filters=[["Video Files", "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]],
            path=self.archives_dir  
        )
        
    def handle_selection(self, selection):
        if not selection:#if user cancel selection thing
            return
            
        selected_file = selection[0]

        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if not any(selected_file.lower().endswith(ext) for ext in valid_extensions):
            self.show_error_popup("Error", "Selected file is not a supported video format")
            return
        if self.video_player:
            self.video_player.state = 'stop'
        self.video_player.source = selected_file
        self.current_video_path = selected_file
        self.video_player.state = 'play'
        self.show_success_popup(f"Now playing: {os.path.basename(selected_file)}")
    
    def create_ui(self):
        self.clear_widgets()
        
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        title_label = Label(
            text="Archived Videos",
            font_size="24sp",
            size_hint=(1, 0.05),
            color=(0, 0.27, 0, 0.6)
        )
        layout.add_widget(title_label)
        select_button = Button(
            text="Select Video",
            font_size="18sp",
            size_hint=(1, 0.05),
            background_color=(0, 0.27, 0, 0.6),
        )
        select_button.bind(on_press=self.upload_file)
        layout.add_widget(select_button)
        self.video_player = Video(options={'eos': 'stop'}, size_hint=(1, 0.3))
        layout.add_widget(self.video_player)
        
        controls_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.05), spacing=10)
        play_pause_button = Button(text='Play/Pause/Restart', background_color=(0, 0.27, 0, 0.6))
        play_pause_button.bind(on_press=self.toggle_play_pause)
        controls_layout.add_widget(play_pause_button)
        delete_button = Button(text='Delete', background_color=(0, 0.27, 0, 0.6))
        delete_button.bind(on_press=self.delete_current_video)
        controls_layout.add_widget(delete_button)
        
        layout.add_widget(controls_layout)
        
        footer_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.05), spacing=10)
        
        nbm_button = Button(
            text="Back to Video Player", background_color=(0, 0.27, 0, 0.6)
        )
        nbm_button.bind(on_release=lambda x: self.go_to_screen("nbm", "right"))
        footer_layout.add_widget(nbm_button)
        
        start_button = Button(
            text="Back to Start Menu", background_color=(0, 0.27, 0, 0.6)
        )
        start_button.bind(on_release=lambda x: self.go_to_screen("start", "right"))
        footer_layout.add_widget(start_button)
        
        layout.add_widget(footer_layout)
        self.add_widget(layout)
    
    def go_to_screen(self, screen_name, direction):
        if self.video_player:
            self.video_player.state = 'stop'
        self.manager.current = screen_name
        self.manager.transition.direction = direction
    
    def on_leave(self):
        if self.video_player:
            self.video_player.state = 'stop'
    
    def toggle_play_pause(self, instance):
        if self.video_player and self.video_player.source:
            if self.video_player.state == 'play':
                self.video_player.state = 'pause'
            else:
                self.video_player.state = 'play'
    
    def delete_current_video(self, popup):            
        self.video_player.state = 'stop'
        
        os.remove(self.current_video_path)
        
        self.video_player.source = ""
        self.current_video_path = None
        
        self.show_success_popup("Video successfully deleted")
    
    def show_success_popup(self, message):
        popup = Popup(title='Success', 
                      content=Label(text=message),
                      size_hint=(0.6, 0.3),
                      background_color=(0.93, 0.98, 0.93, 1))
        popup.open()
        popup.bind(on_touch_down=lambda instance, touch: popup.dismiss() if popup.collide_point(*touch.pos) else None)
    

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("new_window.kv")

class SmartRally(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self):
        Window.clearcolor = (0.96, 0.98, 0.96, 1)
        self.window = GridLayout()
        self.window.cols = 1
        return kv
if __name__ == "__main__":
    SmartRally().run()
