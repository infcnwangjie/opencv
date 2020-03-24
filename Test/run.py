
# manager = Manager(app)
from flask import Flask

from app.views.view import main_blue

app = Flask(__name__)
app.register_blueprint(main_blue,url_prefix='/nty_video')

if __name__ == '__main__':
   # manager.run()
   app.run(host='localhost',port=8081)