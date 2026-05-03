import logging
import time
from dataclasses import dataclass, field

import requests

# from auth.cas_login import LoginError  # unused
from auth.token_manager import TokenExpiredError, TokenManager

logger = logging.getLogger(__name__)


@dataclass
class Config:
    token_manager: TokenManager
    port: int
    api_key: str | None
    debug: bool
    api_format: str = "both"  # "openai", "anthropic", or "both"


GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
GENAI_MODELS_URL = "https://genai.shanghaitech.edu.cn/htk/ai/aiModel/list"


@dataclass
class ModelInfo:
    id: str
    name: str
    root_ai_type: str
    max_tokens: int | None
    description: str | None


@dataclass
class ModelRegistry:
    _models: dict[str, ModelInfo] = field(default_factory=dict)
    _last_fetched: float = 0
    _cache_ttl: float = 900  # 15 minutes

    # Static fallback model info (used when registry fetch fails)
    _STATIC_FALLBACKS: dict[str, dict] = field(
        default_factory=lambda: {
            "chatglm": {"max_tokens": 128000, "root_ai_type": "xinference"},
            "gpt-4": {"max_tokens": 128000, "root_ai_type": "xinference"},
            "gpt-3.5": {"max_tokens": 16385, "root_ai_type": "xinference"},
            "claude": {"max_tokens": 200000, "root_ai_type": "xinference"},
            "deepseek": {"max_tokens": 64000, "root_ai_type": "xinference"},
            "MiniMax": {"max_tokens": 245000, "root_ai_type": "xinference"},
        }
    )

    def _apply_static_fallbacks(self) -> None:
        """Apply static fallback model info when registry fetch fails."""
        logger.warning("Applying static fallback model info")
        models: dict[str, ModelInfo] = {}
        for model_name, info in self._STATIC_FALLBACKS.items():
            models[model_name] = ModelInfo(
                id=model_name,
                name=model_name,
                root_ai_type=info["root_ai_type"],
                max_tokens=info["max_tokens"],
                description=f"Fallback model info for {model_name}",
            )
        self._models = models
        self._last_fetched = time.time()
        logger.info("Applied %d static fallback models", len(models))

    def fetch(self, token: str) -> None:
        headers = build_genai_headers(token)
        params = {
            "_t": int(time.time() * 1000),
            "pageNo": 1,
            "pageSize": 999,
            "showStatusList": "2,3",
        }
        # Retry up to 3 times with exponential backoff
        data = None
        for attempt in range(3):
            try:
                resp = requests.get(GENAI_MODELS_URL, headers=headers, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < 2:  # Not the last attempt
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        "Fetch attempt %d failed: %s. Retrying in %ds...",
                        attempt + 1, e, wait_time
                    )
                    time.sleep(wait_time)
                else:
                    logger.exception("Failed to fetch model list from GenAI after 3 retries")
                    # Apply static fallbacks on final failure
                    self._apply_static_fallbacks()
                    return

        # If we get here, the request succeeded
        if not data or not data.get("success"):
            msg = data.get("message", "Unknown error") if data else "No data fetched"
            logger.warning("GenAI model list API returned failure: %s", msg)
            raise TokenExpiredError(msg)

        records = data.get("result", {}).get("records", [])
        models: dict[str, ModelInfo] = {}
        for rec in records:
            ai_type = rec.get("aiType")
            if not ai_type:
                continue
            models[ai_type] = ModelInfo(
                id=ai_type,
                name=rec.get("aiName", ai_type),
                root_ai_type=rec.get("rootAiType", "xinference"),
                max_tokens=rec.get("maxToken"),
                description=rec.get("descInfo"),
            )

        self._models = models
        self._last_fetched = time.time()
        logger.info("Fetched %d models from GenAI platform", len(models))

    def get_models(self, token: str) -> dict[str, ModelInfo]:
        if not self._models or (time.time() - self._last_fetched > self._cache_ttl):
            self.fetch(token)
        # If fetch failed and we have no models, apply static fallbacks
        if not self._models:
            logger.warning("GenAI model registry empty; applying static fallback models.")
            self._apply_static_fallbacks()
        return self._models

    def get_root_ai_type(self, model: str, token: str) -> str:
        models = self.get_models(token)
        info = models.get(model)
        if info:
            return info.root_ai_type
        return "xinference"


model_registry = ModelRegistry()


def build_genai_headers(token: str) -> dict:
    return {
        "Accept": "*/*, text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "application/json",
        "Origin": "https://genai.shanghaitech.edu.cn",
        "Referer": "https://genai.shanghaitech.edu.cn/dialogue",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
        "X-Access-Token": token,
        "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }
