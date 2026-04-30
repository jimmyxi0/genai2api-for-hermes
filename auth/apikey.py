from flask import current_app, request

from errors import openai_error


def register_auth(app):
    @app.before_request
    def check_api_key():
        config = current_app.config["APP_CONFIG"]
        if not config.api_key:
            return

        if request.path == '/health':
            return

        if not request.path.startswith('/v1/'):
            return

        auth_header = request.headers.get('Authorization', '')
        x_api_key = request.headers.get('x-api-key', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
        elif x_api_key:
            token = x_api_key
        else:
            return openai_error(
                "Missing Authorization Bearer token or x-api-key header",
                error_type="invalid_request_error",
                code="invalid_api_key",
                status=401
            )

        if token != config.api_key:
            return openai_error(
                "Incorrect API key provided",
                error_type="invalid_request_error",
                code="invalid_api_key",
                status=401
            )
