from flask import Blueprint, jsonify, g

from config import model_registry

models_bp = Blueprint('models', __name__)


@models_bp.route('/v1/models', methods=['GET'])
def list_models():
    token = g.get("token", "")
    models_map = model_registry.get_models(token)

    models = []
    for model_id, info in models_map.items():
        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": info.root_ai_type,
            "permission": []
        })
    return jsonify({"object": "list", "data": models})


@models_bp.route('/v1/models/<model_id>', methods=['GET'])
def get_model(model_id):
    token = g.get("token", "")
    models_map = model_registry.get_models(token)
    info = models_map.get(model_id)
    if not info:
        return jsonify({"error": "Model not found"}), 404
    return jsonify({
        "id": model_id,
        "object": "model",
        "owned_by": info.root_ai_type,
        "permission": []
    })

# Backwards-compatibility alias: support legacy proxy path /api/v1/models
@models_bp.route('/api/v1/models', methods=['GET'])
def list_models_api():
    return list_models()
