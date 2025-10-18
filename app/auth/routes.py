from flask import request,jsonify, current_app
from flask_jwt_extended import JWTManager, create_access_token
from . import bp

jwt = JWTManager()

USERS = {
    "ramazan":"12345",
    "ayse": "abcd"
}

@bp.record_once
def setup(state):
    jwt.init_app(state.app)

@bp.login("/login")
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip().lower()
    password = (data.get("password") or "").strip()

    real = USERS.get(username)
    if not username or not password or real != password:
        return jsonify({"error": "kullanıcı adı/şifre hatalı"}), 401

    token = create_access_token(identity=username)
    return jsonify({"access_token": token, "user": username}), 200