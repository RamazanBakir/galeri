from flask import Flask
from . import setup_db

def create_app():
    app= Flask(__name__)
    app.config["JWT_SECRET_KEY"] = "degistirilecek_alan"
    app.config["DB_PATH"] = "app.db"
    app.config["UPLOAD_FOLDER"] = "uploads" #foto vs.

    setup_db(app.config["DB_PATH"])
    #blueprints
    from .auth import bp as auth_bp
    from .api import bp as api_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    #ana sayfa (template)

    @app.get("/")
    def index():
        from flask import render_template
        return render_template("index.html")
    return app
