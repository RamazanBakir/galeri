import os
from ..db import get_conn
from . import bp
from flask import request,jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename

ALLOWED = {"png","jpg","jpeg","gif","webp"}

def allowed_file(filename: str) ->  bool:
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED

@bp.get("/photos")
def list_photos():
    with get_conn(current_app.config["DB_PATH"]) as conn:
        rows = conn.execute(
            "SELECT id, owner, title, filename, created_at FROM photos ORDER BY id DESC"
        ).fetchall()
        data = [dict(r) for r in rows]

        for d in data:
            d["url"] = f"/api/uploads/{d['filename']}"
        return jsonify(data), 200
"""
@bp.post("/photos")
@jwt.required()
def upload_photo():
    user = get_
"""

@bp.get("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)



















