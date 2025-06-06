# turing-smart-screen-python - a Python system monitor and library for USB-C displays like Turing Smart Screen or XuanFang
# https://github.com/mathoudebine/turing-smart-screen-python/

# Copyright (C) 2021-2023  Matthieu Houdebine (mathoudebine)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import mimetypes
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer

from library.lcd.lcd_comm import *

SCREENSHOT_FILE = "screencap.png"
WEBSERVER_PORT = 5678


# This webserver offer a blank page displaying simulated screen with auto-refresh
class SimulatedLcdWebServer(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes("<img src=\"" + SCREENSHOT_FILE + "\" id=\"myImage\" />", "utf-8"))
            self.wfile.write(bytes("<script>", "utf-8"))
            self.wfile.write(bytes("setInterval(function() {", "utf-8"))
            self.wfile.write(bytes("    var myImageElement = document.getElementById('myImage');", "utf-8"))
            self.wfile.write(bytes("    myImageElement.src = '" + SCREENSHOT_FILE + "?rand=' + Math.random();", "utf-8"))
            self.wfile.write(bytes("}, 250);", "utf-8"))
            self.wfile.write(bytes("</script>", "utf-8"))
        elif self.path.startswith("/" + SCREENSHOT_FILE):
            imgfile = open(SCREENSHOT_FILE, 'rb').read()
            mimetype = mimetypes.MimeTypes().guess_type(SCREENSHOT_FILE)[0]
            self.send_response(200)
            if mimetype is not None:
                self.send_header('Content-type', mimetype)
            self.end_headers()
            self.wfile.write(imgfile)


# Simulated display: write on a file instead of serial port
class LcdSimulated(LcdComm):
    def __init__(self, com_port: str = "AUTO", display_width: int = 320, display_height: int = 480,
                 update_queue: Optional[queue.Queue] = None):
        LcdComm.__init__(self, com_port, display_width, display_height, update_queue)
        self.screen_image = Image.new("RGB", (self.get_width(), self.get_height()), (255, 255, 255))
        self.screen_image.save("tmp", "PNG")
        shutil.copyfile("tmp", SCREENSHOT_FILE)
        self.orientation = Orientation.PORTRAIT

        try:
            self.webServer = HTTPServer(("localhost", WEBSERVER_PORT), SimulatedLcdWebServer)
            logger.debug("To see your simulated screen, open http://%s:%d in a browser" % ("localhost", WEBSERVER_PORT))
            threading.Thread(target=self.webServer.serve_forever).start()
        except OSError:
            logger.error("Error starting webserver! An instance might already be running on port %d." % WEBSERVER_PORT)

    def __del__(self):
        self.closeSerial()

    @staticmethod
    def auto_detect_com_port() -> Optional[str]:
        return None

    def closeSerial(self):
        logger.debug("Shutting down web server")
        self.webServer.shutdown()

    def InitializeComm(self):
        pass

    def Reset(self):
        pass

    def Clear(self):
        self.SetOrientation(self.orientation)

    def ScreenOff(self):
        pass

    def ScreenOn(self):
        pass

    def SetBrightness(self, level: int = 25):
        pass

    def SetBackplateLedColor(self, led_color: Tuple[int, int, int] = (255, 255, 255)):
        pass

    def SetOrientation(self, orientation: Orientation = Orientation.PORTRAIT):
        self.orientation = orientation
        # Just draw the screen again with the new width/height based on orientation
        with self.update_queue_mutex:
            self.screen_image = Image.new("RGB", (self.get_width(), self.get_height()), (255, 255, 255))
            self.screen_image.save("tmp", "PNG")
            shutil.copyfile("tmp", SCREENSHOT_FILE)

    def DisplayPILImage(
            self,
            image: Image.Image,
            x: int = 0, y: int = 0,
            image_width: int = 0,
            image_height: int = 0
    ):
        # If the image height/width isn't provided, use the native image size
        if not image_height:
            image_height = image.size[1]
        if not image_width:
            image_width = image.size[0]

        # If our image is bigger than our display, resize it to fit our screen
        if image.size[1] > self.get_height():
            image_height = self.get_height()
        if image.size[0] > self.get_width():
            image_width = self.get_width()

        assert x <= self.get_width(), 'Image X coordinate must be <= display width'
        assert y <= self.get_height(), 'Image Y coordinate must be <= display height'
        assert image_height > 0, 'Image height must be > 0'
        assert image_width > 0, 'Image width must be > 0'

        with self.update_queue_mutex:
            self.screen_image.paste(image, (x, y))
            self.screen_image.save("tmp", "PNG")
            shutil.copyfile("tmp", SCREENSHOT_FILE)
