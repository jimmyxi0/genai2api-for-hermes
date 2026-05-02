from flask import Blueprint, jsonify

compat_bp = Blueprint('compat', __name__)


@compat_bp.route('/api/tags', methods=['GET'])
def api_tags():
    # Minimal compatibility shim for legacy proxies expecting /api/tags
    return jsonify({"tags": []})


@compat_bp.route('/v1/props', methods=['GET'])
def v1_props():
    # Minimal shim for legacy proxies expecting /v1/props
    return jsonify({})


@compat_bp.route('/version', methods=['GET'])
def version_info():
    # Minimal shim for /version endpoint
    return jsonify({"version": "0.0.1"})
