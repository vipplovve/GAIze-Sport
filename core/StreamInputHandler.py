import cv2
import subprocess
import shutil

class StreamInputHandler:
    def __init__(self):
        pass

    def get_video_source(self, input_path):
        if input_path.startswith("rtmp://") or input_path.startswith("http://") or input_path.startswith("https://"):
            if "youtube.com" in input_path or "youtu.be" in input_path:
                return self._get_youtube_stream_url(input_path)
            else:
                return input_path
        else:
            return input_path

    def _get_youtube_stream_url(self, url):
        if not shutil.which("yt-dlp"):
            print("Error: yt-dlp not found in PATH. Please install it.")
            return None

        try:
            cmd = ["yt-dlp", "-f", "b", "-g", url]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                stream_url = result.stdout.strip()
                return stream_url
            else:
                print(f"yt-dlp error: {result.stderr}")
                return None
        except Exception as e:
            print(f"Error fetching YouTube URL: {e}")
            return None
