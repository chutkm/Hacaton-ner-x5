# api/log_middleware.py
import time, logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger("app.requests")
MAX_LEN = 2000  # сколько байт тела логируем

class LogIO(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.time()
        req_body = b""
        try:
            if request.url.path == "/api/predict" and request.headers.get("content-type","").startswith("application/json"):
                req_body = await request.body()
        except Exception:
            pass

        response = await call_next(request)

        # по умолчанию не трогаем стримы/большие ответы
        should_capture = (
            request.url.path == "/api/predict"
            and response.headers.get("content-type","").startswith("application/json")
            and not response.headers.get("content-encoding")  # не логируем gzip/deflate
        )

        resp_len = response.headers.get("content-length")
        resp_body = b""

        if should_capture:
            # вычитаем тело и собираем обратно новый Response
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            resp_body = b"".join(chunks)

            new_resp = Response(
                content=resp_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
                background=response.background,
            )
            # корректируем content-length
            new_resp.headers["content-length"] = str(len(resp_body))
            response = new_resp

        dur_ms = round((time.time() - t0) * 1000, 1)

        log.info(
            "method=%s path=%s status=%s ms=%s req_len=%s resp_len=%s Body=%s Response=%s",
            request.method,
            request.url.path,
            getattr(response, "status_code", "NA"),
            dur_ms,
            len(req_body),
            resp_len if resp_len is not None else len(resp_body) if resp_body else None,
            req_body[:MAX_LEN].decode("utf-8","ignore"),
            resp_body[:MAX_LEN].decode("utf-8","ignore") if resp_body else None,
        )
        return response
