import sys
sys.path.append("/home/hori/2AMC15-2023-DIC/")
from flask_socketio import SocketIO
from level_editor.app import app


#start webserver
socket_io = SocketIO(app)
if __name__ == '__main__':
    socket_io.run(app, debug=False, allow_unsafe_werkzeug=True)