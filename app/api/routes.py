import os
from ..db import get_conn
from . import bp
from flask import request,jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from flask_jwt_extended import jwt_required,get_jwt_identity

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

@bp.post("/photos")
@jwt.required()
def upload_photo():
    user = get_jwt_identity()
    title = (request.form.get("title") or "").strip()
    f = request.files.get("file")

    if not title :
        return jsonify({"error": "title zorunlu alan"}), 400
    if not f or f.filename == "":
        return jsonify({"error": "dosya zorunlu alan"}), 400
    if not allowed_file(f.filename) :
        return jsonify({"error": "desteklenmeyen format yüklemeye çalışıyorsun..."}), 400

    clean = secure_filename(f.filename)
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir,exist_ok=True)
    path = os.path.join(upload_dir,clean)

    base, ext = os.path.splitext(clean)
    i = 1
    while os.path.exists(path):
        clean = f"{base}_{i}{ext}"
        path = os.path.join(upload_dir,clean)
        i += 1
    f.save(path)

    with get_conn(current_app.config["DB_PATH"]) as conn:
        conn.execute(
            "INSERT INTO photos(owner, title, filename) VALUES (?,?,?)",
            (user, title,clean)
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, owner, title, filename, created_at FROM photos ORDER BY id DESC LIMIT 1"
        ).fetchone()
    data = dict(row)
    data["url"] = f"/api/uploads/{data['filename']}"
    return jsonify(data), 201


@bp.get("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)



















