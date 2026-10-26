"""Clientes de LoomLoom Market para la generación por lotes de MoneyPrinterTurbo.

Este módulo vive deliberadamente fuera de ``llm_provider``. LoomLoom ejecuta un SkillBot
de Market versionado con presupuesto, confirmación, ciclo de vida de la ejecución y
filas de resultado; no es un proveedor de chat completions.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call


DEFAULT_RESULT_PORT_NAME = "output"
DEFAULT_BASE_URL = "https://loomloom.shengsuanyun.com/loom/v1"
DEFAULT_SCRIPT_MARKET_LISTING_ID = "019fd618-9baa-73d9-94f4-c9270b6f3025"
VIDEO_CAPABILITY_PROFILE_ID = "video.text-to-video.aspect-ratio.v1"
# El guion y el vídeo son dos SkillBots publicados con entradas y estructuras de
# resultado totalmente distintas. Ambos ID son constantes internas de la integración de
# MoneyPrinterTurbo; el usuario solo tiene que dar la clave API y no debería tocar el
# Listing ID.
DEFAULT_VIDEO_MARKET_LISTING_ID = "01a06563-7331-773a-b9b2-25989a0dd70e"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_RUN_TIMEOUT_SECONDS = 600.0
DEFAULT_VIDEO_RUN_TIMEOUT_SECONDS = 1800.0
MAX_EXECUTE_ATTEMPTS = 3
MAX_POLL_RETRY_DELAY_SECONDS = 30.0
MAX_SCRIPT_CANDIDATES = 10
MAX_VIDEO_SCENES = 5
MAX_VIDEO_ARTIFACT_BYTES = 512 * 1024 * 1024
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled", "canceled"})


class LoomLoomError(RuntimeError):
    """Error base de la integración con LoomLoom."""


class LoomLoomConfigurationError(LoomLoomError):
    """Se lanza cuando la integración está activada sin una configuración completa."""


class LoomLoomAPIError(LoomLoomError):
    """Se lanza cuando la API pública rechaza una solicitud o devuelve JSON no válido."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class LoomLoomRunError(LoomLoomError):
    """Se lanza cuando una ejecución enviada falla o supera su tiempo de espera."""


def _coerce_seconds_setting(value: Any, name: str) -> float:
    """
    Convierte a float los campos de segundos de la configuración, manteniendo el
    contrato de tipo de excepción de los errores de configuración.

    TOML permite escribirlos como cadena/array, y el ``ValueError`` / ``TypeError`` que
    lanza un ``float()`` directo no es un ``LoomLoomError``, así que haría que la
    entrada de sondeo, que solo captura ``LoomLoomError``, terminara en una traza y se
    saltara el aviso con el nombre de la opción de ``validate()``; además
    ``float(True)`` se convertiría en silencio en 1.0, reduciendo una espera de 600
    segundos a 1.
    """
    if isinstance(value, bool):
        raise LoomLoomConfigurationError(f"{name} debe ser un número, se recibió {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise LoomLoomConfigurationError(
            f"{name} debe ser un número, se recibió {value!r}"
        ) from None


def resolve_api_token(values: Mapping[str, Any]) -> str:
    """
    Resuelve la clave API de Shengsuan Cloud que debe usar la función actual.

    Cuando el proveedor LLM elegido es Shengsuan Cloud, el guion y el vídeo deben
    reutilizar la clave de la página de ajustes; con otros proveedores se sigue usando
    la clave propia de LoomLoom, para no cambiar la configuración existente de los
    usuarios.
    """
    if str(values.get("llm_provider", "") or "").strip().lower() == "shengsuanyun":
        return str(values.get("shengsuanyun_api_key", "") or "").strip()
    return str(values.get("loomloom_api_token", "") or "").strip()


@dataclass(frozen=True)
class LoomLoomSettings:
    base_url: str
    api_token: str = field(repr=False)
    market_listing_id: str
    listing_version_id: str = ""
    result_port_name: str = DEFAULT_RESULT_PORT_NAME
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    run_timeout_seconds: float = DEFAULT_RUN_TIMEOUT_SECONDS

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "LoomLoomSettings":
        settings = cls(
            base_url=str(values.get("loomloom_base_url", "") or DEFAULT_BASE_URL)
            .strip()
            .rstrip("/"),
            api_token=resolve_api_token(values),
            # MoneyPrinterTurbo llama siempre al SkillBot por defecto que el proyecto ya
            # ha publicado. El Listing ID es un detalle de implementación de la
            # integración y no se puede exigir a los usuarios normales que lo repitan en
            # config.toml.
            market_listing_id=DEFAULT_SCRIPT_MARKET_LISTING_ID,
            listing_version_id="",
            result_port_name=DEFAULT_RESULT_PORT_NAME,
            request_timeout_seconds=_coerce_seconds_setting(
                values.get(
                    "loomloom_request_timeout_seconds",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                ),
                "loomloom_request_timeout_seconds",
            ),
            poll_interval_seconds=_coerce_seconds_setting(
                values.get(
                    "loomloom_poll_interval_seconds",
                    DEFAULT_POLL_INTERVAL_SECONDS,
                ),
                "loomloom_poll_interval_seconds",
            ),
            run_timeout_seconds=_coerce_seconds_setting(
                values.get("loomloom_run_timeout_seconds", DEFAULT_RUN_TIMEOUT_SECONDS),
                "loomloom_run_timeout_seconds",
            ),
        )
        settings.validate(require_api_token=True)
        return settings

    def validate(self, *, require_api_token: bool = True) -> None:
        missing = []
        if not self.base_url:
            missing.append("loomloom_base_url")
        if require_api_token and not self.api_token:
            missing.append("loomloom_api_token")
        if not self.market_listing_id:
            missing.append("loomloom_market_listing_id")
        if not self.result_port_name:
            missing.append("loomloom_result_port_name")
        if missing:
            raise LoomLoomConfigurationError(
                "faltan ajustes de LoomLoom: " + ", ".join(missing)
            )

        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LoomLoomConfigurationError(
                "loomloom_base_url debe ser una URL HTTP(S) absoluta"
            )
        for name, value in (
            ("loomloom_request_timeout_seconds", self.request_timeout_seconds),
            ("loomloom_poll_interval_seconds", self.poll_interval_seconds),
            ("loomloom_run_timeout_seconds", self.run_timeout_seconds),
        ):
            # ``value <= 0`` no detiene NaN/Inf: NaN siempre da False al compararse e
            # Inf siempre es mayor que la hora actual, así que ambos anularían el plazo
            # de ``wait_for_run``.
            if not math.isfinite(value):
                raise LoomLoomConfigurationError(f"{name} debe ser un número finito")
            if value <= 0:
                raise LoomLoomConfigurationError(f"{name} debe ser mayor que cero")


@dataclass(frozen=True)
class LoomLoomScriptBatch:
    input_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class LoomLoomVideoBatch:
    """Las filas de entrada que incluye un presupuesto de material de vídeo del SkillBot
    por defecto.
    """

    input_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class LoomLoomVideoModel:
    model_id: str
    display_name: str


@dataclass(frozen=True)
class LoomLoomVideoCapability:
    """Candidatos públicos disponibles para la cuenta actual en el perfil de texto a vídeo."""

    models: tuple[LoomLoomVideoModel, ...]
    default_model_id: str
    aspect_ratios: tuple[str, ...]


@dataclass(frozen=True)
class LoomLoomQuote:
    quote_id: str
    listing_version_id: str
    currency: str
    task_count: int
    estimated_buyer_payable_t: int
    estimated_buyer_payable_amount: str
    input_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class LoomLoomExecution:
    run_id: str
    transaction_id: str
    transaction_status: str
    listing_version_id: str


@dataclass(frozen=True)
class LoomLoomRun:
    run_id: str
    status: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    cancelled_tasks: int
    first_error_message: str


@dataclass(frozen=True)
class LoomLoomScriptCandidate:
    row_index: int
    script: str
    video_terms: tuple[str, ...]


@dataclass(frozen=True)
class LoomLoomCandidateError:
    row_index: int
    message: str


@dataclass(frozen=True)
class LoomLoomScriptBatchResult:
    candidates: tuple[LoomLoomScriptCandidate, ...]
    errors: tuple[LoomLoomCandidateError, ...]


@dataclass(frozen=True)
class LoomLoomConfirmedVideoRequest:
    """
    Instantánea de una solicitud de vídeo cuyo presupuesto ya ha confirmado el usuario.

    La clave API solo se pasa dentro del proceso actual mediante el campo oculto de
    `LoomLoomSettings`, y no entra en VideoParams, en el estado de la tarea ni en los
    logs; la versión del presupuesto y el ID de solicitud idempotente garantizan que la
    ejecución en segundo plano coincida con el presupuesto que vio el usuario y que los
    reintentos de red no cobren dos veces.
    """

    settings: LoomLoomSettings
    batch: LoomLoomVideoBatch
    listing_version_id: str
    client_request_id: str

    def validate(self) -> None:
        self.settings.validate(require_api_token=True)
        if not self.batch.input_rows:
            raise LoomLoomConfigurationError("El lote de vídeo de LoomLoom está vacío")
        if not str(self.listing_version_id or "").strip():
            raise LoomLoomConfigurationError("hace falta la versión del listing presupuestada")
        if not str(self.client_request_id or "").strip():
            raise LoomLoomConfigurationError("hace falta el id de solicitud del cliente para el vídeo")


def video_settings_from_mapping(values: Mapping[str, Any]) -> LoomLoomSettings:
    """Crea el cliente con el SkillBot de vídeo integrado en el proyecto y amplía el tiempo
    de espera de las tareas de vídeo.
    """
    settings = LoomLoomSettings.from_mapping(values)
    return LoomLoomSettings(
        base_url=settings.base_url,
        api_token=settings.api_token,
        # El Listing de vídeo recibe prompt/modelChoice/aspectRatio y devuelve MP4; no
        # se puede reutilizar el Listing de guiones, o fallaría ya en la fase de
        # presupuesto porque el esquema de entrada no coincide.
        market_listing_id=DEFAULT_VIDEO_MARKET_LISTING_ID,
        listing_version_id=settings.listing_version_id,
        result_port_name=settings.result_port_name,
        request_timeout_seconds=settings.request_timeout_seconds,
        poll_interval_seconds=settings.poll_interval_seconds,
        run_timeout_seconds=_coerce_seconds_setting(
            values.get(
                "loomloom_video_run_timeout_seconds",
                DEFAULT_VIDEO_RUN_TIMEOUT_SECONDS,
            ),
            "loomloom_video_run_timeout_seconds",
        ),
    )


class LoomLoomScriptBackend:
    """Ejecuta un Listing de LoomLoom Market configurado para obtener candidatos de guion."""

    def __init__(
        self,
        settings: LoomLoomSettings,
        *,
        session: requests.Session | Any | None = None,
        credential_provider: Callable[[], str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        settings.validate(require_api_token=credential_provider is None)
        self.settings = settings
        self._session = session or requests.Session()
        self._credential_provider = credential_provider or (lambda: settings.api_token)
        self._sleep = sleep
        self._clock = clock

    def prepare_script_batch(
        self,
        *,
        subject: str,
        candidate_count: int,
        language: str = "auto",
        duration_seconds: int = 60,
        style: str = "",
    ) -> LoomLoomScriptBatch:
        normalized_subject = str(subject or "").strip()
        if not normalized_subject:
            raise ValueError("hace falta el tema")
        if not 1 <= candidate_count <= MAX_SCRIPT_CANDIDATES:
            raise ValueError(
                f"candidate_count debe estar entre 1 y {MAX_SCRIPT_CANDIDATES}"
            )
        if duration_seconds <= 0:
            raise ValueError("duration_seconds debe ser mayor que cero")

        requirements = (
            f"Idioma de salida: {str(language or 'auto').strip() or 'auto'}\n"
            f"Duración objetivo (segundos): {duration_seconds}"
        )
        normalized_style = str(style or "").strip()
        if normalized_style:
            requirements += f"\nEstilo o requisitos adicionales: {normalized_style}"

        rows = tuple(
            {
                "subject": normalized_subject,
                "requirements": requirements,
                "candidateIndex": str(index),
            }
            for index in range(1, candidate_count + 1)
        )
        return LoomLoomScriptBatch(input_rows=rows)

    def quote(self, batch: LoomLoomScriptBatch | LoomLoomVideoBatch) -> LoomLoomQuote:
        payload = self._listing_payload(batch)
        response = self._request(
            "POST",
            f"/marketListings/{quote(self.settings.market_listing_id, safe='')}:quote",
            json_body=payload,
        )
        estimated_buyer_payable = response.get("estimatedBuyerPayable", {})
        if not isinstance(estimated_buyer_payable, dict):
            estimated_buyer_payable = {}
        return LoomLoomQuote(
            quote_id=self._required_string(response, "quoteId"),
            listing_version_id=self._required_string(response, "listingVersionId"),
            currency=str(response.get("currency", "")).strip(),
            task_count=self._integer(response, "taskCount"),
            estimated_buyer_payable_t=self._integer(response, "estimatedBuyerPayableT"),
            estimated_buyer_payable_amount=str(
                estimated_buyer_payable.get("amount", "")
            ).strip(),
            input_rows=batch.input_rows,
        )

    def execute(
        self,
        batch: LoomLoomScriptBatch | LoomLoomVideoBatch,
        *,
        client_request_id: str,
        listing_version_id: str,
        confirm: bool,
    ) -> LoomLoomExecution:
        if confirm is not True:
            raise ValueError("hace falta confirm=True antes de una ejecución de pago de LoomLoom")
        normalized_request_id = str(client_request_id or "").strip()
        if not normalized_request_id:
            raise ValueError("hace falta client_request_id")
        normalized_listing_version_id = str(listing_version_id or "").strip()
        if not normalized_listing_version_id:
            raise ValueError("hace falta el listing_version_id del presupuesto")

        payload = self._listing_payload(
            batch, listing_version_id=normalized_listing_version_id
        )
        payload.update(
            {
                "clientRequestId": normalized_request_id,
                "confirm": True,
            }
        )
        response = None
        for attempt in range(1, MAX_EXECUTE_ATTEMPTS + 1):
            try:
                response = self._request(
                    "POST",
                    f"/marketListings/{quote(self.settings.market_listing_id, safe='')}:execute",
                    json_body=payload,
                )
                break
            except LoomLoomAPIError as exc:
                if not exc.retryable or attempt >= MAX_EXECUTE_ATTEMPTS:
                    raise
                # execute es una operación de pago: no se puede generar un nuevo ID de
                # solicitud y reintentar a ciegas. El servidor garantiza la idempotencia
                # con clientRequestId, así que aquí solo se reintenta un número limitado
                # de veces con exactamente la misma carga útil, para recuperarse de
                # fallos de red en los que "el servidor aceptó pero el cliente no
                # recibió la respuesta".
                retry_delay = min(float(attempt), MAX_POLL_RETRY_DELAY_SECONDS)
                logger.warning(
                    "se reintenta la ejecución de LoomLoom con el mismo id de solicitud del cliente: "
                    f"attempt={attempt + 1}/{MAX_EXECUTE_ATTEMPTS}, "
                    f"error={type(exc).__name__}"
                )
                self._sleep(retry_delay)

        if response is None:  # pragma: no cover - las ramas de éxito o excepción del bucle ya están cubiertas
            raise LoomLoomAPIError("La ejecución de LoomLoom no devolvió respuesta")
        return LoomLoomExecution(
            run_id=self._required_string(response, "runId"),
            transaction_id=str(response.get("runTransactionId", "")).strip(),
            transaction_status=str(response.get("transactionStatus", "")).strip(),
            listing_version_id=str(response.get("listingVersionId", "")).strip(),
        )

    def get_run(self, run_id: str) -> LoomLoomRun:
        normalized_run_id = self._required_identifier(run_id, "run_id")
        response = self._request(
            "GET", f"/users/me/runs/{quote(normalized_run_id, safe='')}"
        )
        run = response.get("run")
        if not isinstance(run, dict):
            raise LoomLoomAPIError("A la respuesta de ejecución de LoomLoom le falta run")
        return LoomLoomRun(
            run_id=self._required_string(run, "runId"),
            status=self._required_string(run, "status").lower(),
            total_tasks=self._integer(run, "totalTasks"),
            completed_tasks=self._integer(run, "completedTasks"),
            failed_tasks=self._integer(run, "failedTasks"),
            cancelled_tasks=self._integer(run, "cancelledTasks"),
            first_error_message=str(run.get("firstErrorMessage", "")).strip(),
        )

    def wait_for_run(self, run_id: str) -> LoomLoomRun:
        deadline = self._clock() + self.settings.run_timeout_seconds
        started_at = self._clock()
        last_logged_status = None
        last_progress_log_at = started_at
        consecutive_poll_errors = 0
        while True:
            if self._clock() >= deadline:
                raise LoomLoomRunError(
                    f"La ejecución {run_id} de LoomLoom no terminó en "
                    f"{self.settings.run_timeout_seconds:g} segundos"
                )
            try:
                run = self.get_run(run_id)
                consecutive_poll_errors = 0
            except LoomLoomAPIError as exc:
                if not exc.retryable:
                    raise
                now = self._clock()
                if now >= deadline:
                    raise LoomLoomRunError(
                        f"No se pudo consultar la ejecución {run_id} de LoomLoom en "
                        f"{self.settings.run_timeout_seconds:g} segundos"
                    ) from exc
                consecutive_poll_errors += 1
                retry_delay = min(
                    self.settings.poll_interval_seconds
                    * (2 ** min(consecutive_poll_errors - 1, 4)),
                    MAX_POLL_RETRY_DELAY_SECONDS,
                    max(deadline - now, 0),
                )
                logger.warning(
                    "se reintenta el sondeo de la ejecución de LoomLoom tras un error pasajero: "
                    f"run_id={run_id}, attempt={consecutive_poll_errors}, "
                    f"delay={retry_delay:g}s, error={type(exc).__name__}"
                )
                self._sleep(retry_delay)
                continue
            now = self._clock()
            progress = run.completed_tasks + run.failed_tasks + run.cancelled_tasks
            # Las tareas de vídeo remotas pueden durar varios minutos. Se registra en
            # cuanto cambia el estado, y si no cambia se registra un latido cada 30
            # segundos, lo que ayuda a localizar dónde se atasca sin inundar el log cada
            # dos segundos.
            if run.status != last_logged_status or now - last_progress_log_at >= 30:
                logger.info(
                    "progreso de la ejecución de LoomLoom: "
                    f"run_id={run.run_id}, status={run.status}, "
                    f"finished={progress}/{run.total_tasks}"
                )
                last_logged_status = run.status
                last_progress_log_at = now
            if run.status in TERMINAL_RUN_STATUSES:
                if run.status != "completed":
                    detail = run.first_error_message or run.status
                    raise LoomLoomRunError(
                        f"La ejecución {run.run_id} de LoomLoom terminó con {detail}"
                    )
                return run
            if now >= deadline:
                raise LoomLoomRunError(
                    f"La ejecución {run.run_id} de LoomLoom no terminó en "
                    f"{self.settings.run_timeout_seconds:g} segundos"
                )
            self._sleep(min(self.settings.poll_interval_seconds, deadline - now))

    def get_script_results(self, run_id: str) -> LoomLoomScriptBatchResult:
        normalized_run_id = self._required_identifier(run_id, "run_id")
        rows = self._list_all_result_rows(normalized_run_id)
        candidates = []
        errors = []
        for row in rows:
            row_index = self._integer(row, "rowIndex")
            status = str(row.get("status", "")).strip().lower()
            if status != "completed":
                errors.append(
                    LoomLoomCandidateError(
                        row_index=row_index,
                        message=str(row.get("errorMessage", "")).strip()
                        or f"la fila terminó con el estado {status or 'unknown'}",
                    )
                )
                continue
            try:
                candidates.append(self._parse_candidate(row_index, row))
            except (LoomLoomAPIError, ValueError) as exc:
                errors.append(
                    LoomLoomCandidateError(row_index=row_index, message=str(exc))
                )
        return LoomLoomScriptBatchResult(
            candidates=tuple(candidates), errors=tuple(errors)
        )

    def _listing_payload(
        self,
        batch: LoomLoomScriptBatch | LoomLoomVideoBatch,
        *,
        listing_version_id: str | None = None,
    ) -> dict[str, Any]:
        if not batch.input_rows:
            raise ValueError("hace falta input_rows")
        payload: dict[str, Any] = {"inputRows": [dict(row) for row in batch.input_rows]}
        resolved_listing_version_id = (
            self.settings.listing_version_id
            if listing_version_id is None
            else listing_version_id
        )
        if resolved_listing_version_id:
            payload["listingVersionId"] = resolved_listing_version_id
        return payload

    def _list_all_result_rows(self, run_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page_token = ""
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"pageSize": 200}
            if page_token:
                params["pageToken"] = page_token
            response = self._request(
                "GET",
                f"/users/me/runs/{quote(run_id, safe='')}/resultRows",
                params=params,
            )
            items = response.get("items", [])
            if not isinstance(items, list) or not all(
                isinstance(item, dict) for item in items
            ):
                raise LoomLoomAPIError("los elementos de resultRows de LoomLoom deben ser objetos")
            rows.extend(items)
            page_token = str(response.get("nextPageToken") or "").strip()
            if not page_token:
                return rows
            if page_token in seen_tokens:
                raise LoomLoomAPIError("LoomLoom devolvió un token de página de resultRows repetido")
            seen_tokens.add(page_token)

    def _parse_candidate(
        self, row_index: int, row: Mapping[str, Any]
    ) -> LoomLoomScriptCandidate:
        artifacts = row.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise LoomLoomAPIError("los artefactos de la fila deben ser una lista")
        matching = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict)
            and str(artifact.get("portName", "")).strip()
            == self.settings.result_port_name
        ]
        if len(matching) != 1:
            raise LoomLoomAPIError(
                f"se esperaba un artefacto de resultado {self.settings.result_port_name!r}, "
                f"se recibieron {len(matching)}"
            )
        inline_text = str(matching[0].get("inlineText", "")).strip()
        if not inline_text:
            raise LoomLoomAPIError("el artefacto de resultado no contiene inlineText")
        lines = inline_text.splitlines()
        if (
            len(lines) >= 3
            and lines[0].strip().lower() in {"```", "```json"}
            and lines[-1].strip() == "```"
        ):
            inline_text = "\n".join(lines[1:-1]).strip()
        try:
            value = json.loads(inline_text)
        except json.JSONDecodeError as exc:
            raise LoomLoomAPIError("el artefacto de resultado no es un JSON válido") from exc
        if not isinstance(value, dict):
            raise LoomLoomAPIError("el JSON del artefacto de resultado debe ser un objeto")

        script = value.get("script")
        if not isinstance(script, str) or not script.strip():
            raise LoomLoomAPIError("hace falta el script del artefacto de resultado")
        video_terms = value.get("videoTerms")
        if not isinstance(video_terms, list) or not video_terms:
            raise LoomLoomAPIError(
                "videoTerms del artefacto de resultado debe ser una lista no vacía"
            )
        normalized_terms = tuple(
            term.strip() for term in video_terms if isinstance(term, str) and term.strip()
        )
        if not normalized_terms:
            raise LoomLoomAPIError("videoTerms del artefacto de resultado no puede estar vacío")
        return LoomLoomScriptCandidate(
            row_index=row_index,
            script=script.strip(),
            video_terms=normalized_terms,
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.settings.base_url}{path}"
        api_token = str(self._credential_provider() or "").strip()
        if not api_token:
            raise LoomLoomConfigurationError(
                "esta solicitud necesita una credencial de LoomLoom"
            )
        try:
            response = trace_http_call(
                "REQUEST", self._session.request,
                method,
                url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                },
                json=dict(json_body) if json_body is not None else None,
                params=dict(params) if params is not None else None,
                allow_redirects=False,
                timeout=(5.0, self.settings.request_timeout_seconds),
            )
        except requests.RequestException as exc:
            raise LoomLoomAPIError(
                f"Falló la solicitud a LoomLoom: {type(exc).__name__}",
                retryable=True,
            ) from exc

        if not 200 <= response.status_code < 300:
            message = "petición rechazada"
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = None
            if isinstance(error_payload, dict):
                server_error = str(error_payload.get("error", "")).strip()
                if server_error:
                    # Los errores del servidor a veces devuelven el contexto de la
                    # solicitud. Incluso ante una respuesta anómala, el token Bearer no
                    # puede llegar a los errores de la página, a los logs ni al estado
                    # de la tarea.
                    message = server_error.replace(api_token, "[redacted]")
            raise LoomLoomAPIError(
                f"La API de LoomLoom devolvió HTTP {response.status_code}: {message}",
                status_code=response.status_code,
                retryable=(
                    response.status_code in {408, 425, 429}
                    or response.status_code >= 500
                ),
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LoomLoomAPIError("La API de LoomLoom devolvió un JSON no válido") from exc
        if not isinstance(payload, dict):
            raise LoomLoomAPIError("La respuesta de la API de LoomLoom debe ser un objeto JSON")
        return payload

    @staticmethod
    def _required_identifier(value: str, name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError(f"hace falta {name}")
        return normalized

    @staticmethod
    def _required_string(values: Mapping[str, Any], name: str) -> str:
        value = str(values.get(name, "")).strip()
        if not value:
            raise LoomLoomAPIError(f"A la respuesta de LoomLoom le falta {name}")
        return value

    @staticmethod
    def _integer(values: Mapping[str, Any], name: str) -> int:
        value = values.get(name, 0)
        if isinstance(value, bool):
            raise LoomLoomAPIError(f"{name} de la respuesta de LoomLoom debe ser un entero")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise LoomLoomAPIError(
                f"{name} de la respuesta de LoomLoom debe ser un entero"
            ) from exc


class LoomLoomVideoBackend(LoomLoomScriptBackend):
    """Genera material de vídeo con el SkillBot por defecto y descarga de forma segura el
    MP4 resultante al directorio de la tarea.
    """

    def resolve_video_capability(self) -> LoomLoomVideoCapability:
        """Lee los candidatos de modelo y relación de aspecto en tiempo real de la cuenta
        actual para el perfil de vídeo fijo.
        """
        response = self._request(
            "GET",
            "/authoringCapabilities:resolve",
            params={"inputModality": "text", "outputModality": "video"},
        )
        matches = response.get("matches", [])
        if not isinstance(matches, list):
            raise LoomLoomAPIError("las coincidencias de capacidades de LoomLoom deben ser una lista")

        profile_match = next(
            (
                match
                for match in matches
                if isinstance(match, dict)
                and isinstance(match.get("profile"), dict)
                and str(match["profile"].get("profileId", "")).strip()
                == VIDEO_CAPABILITY_PROFILE_ID
            ),
            None,
        )
        if profile_match is None:
            raise LoomLoomConfigurationError(
                f"El perfil de capacidades {VIDEO_CAPABILITY_PROFILE_ID} de LoomLoom no está disponible"
            )

        eligible_models = profile_match.get("eligibleModels", [])
        if not isinstance(eligible_models, list):
            raise LoomLoomAPIError("eligibleModels de LoomLoom debe ser una lista")

        models = []
        seen_model_ids = set()
        for candidate in eligible_models:
            if not isinstance(candidate, dict):
                raise LoomLoomAPIError("el modelo elegible de LoomLoom debe ser un objeto")
            # Un null o un número no pueden convertirse en un ID de apariencia válida;
            # si no, el catálogo dañado seguiría llegando a la interfaz de pago.
            model_id = candidate.get("modelId")
            display_name = candidate.get("displayName")
            if not isinstance(model_id, str) or not isinstance(display_name, str):
                raise LoomLoomAPIError("modelId y displayName de LoomLoom deben ser cadenas")
            model_id = model_id.strip()
            display_name = display_name.strip()
            if not model_id or not display_name:
                raise LoomLoomAPIError(
                    "el modelo elegible de LoomLoom necesita modelId y displayName"
                )
            if model_id in seen_model_ids:
                continue
            seen_model_ids.add(model_id)
            models.append(
                LoomLoomVideoModel(model_id=model_id, display_name=display_name)
            )
        if not models:
            raise LoomLoomConfigurationError(
                "la capacidad de vídeo de LoomLoom no tiene modelos elegibles"
            )

        profile = profile_match["profile"]
        operations = profile.get("operations", {})
        if not isinstance(operations, dict):
            raise LoomLoomAPIError("las operaciones del perfil de LoomLoom deben ser un objeto")
        default_model_id = str(operations.get("defaultModelId", "")).strip()
        if default_model_id not in seen_model_ids:
            raise LoomLoomConfigurationError(
                "el modelo por defecto de la capacidad de vídeo de LoomLoom no es elegible"
            )

        definition = profile.get("definition", {})
        constraints = (
            definition.get("constraints", {}) if isinstance(definition, dict) else {}
        )
        ports = constraints.get("ports", {}) if isinstance(constraints, dict) else {}
        aspect_ratio = ports.get("aspect_ratio", {}) if isinstance(ports, dict) else {}
        raw_aspect_ratios = (
            aspect_ratio.get("enum", []) if isinstance(aspect_ratio, dict) else []
        )
        if not isinstance(raw_aspect_ratios, list):
            raise LoomLoomAPIError(
                "la restricción de relación de aspecto del vídeo de LoomLoom debe ser una lista"
            )
        aspect_ratios = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in raw_aspect_ratios
                if str(value).strip() in {"9:16", "16:9"}
            )
        )
        if not aspect_ratios:
            raise LoomLoomConfigurationError(
                "la capacidad de vídeo de LoomLoom no tiene relaciones de aspecto admitidas"
            )

        return LoomLoomVideoCapability(
            models=tuple(models),
            default_model_id=default_model_id,
            aspect_ratios=aspect_ratios,
        )

    def prepare_video_batch(
        self,
        *,
        subject: str,
        scene_prompts: list[str] | tuple[str, ...],
        model_id: str,
        aspect_ratio: str,
    ) -> LoomLoomVideoBatch:
        normalized_subject = str(subject or "").strip()
        normalized_model_id = str(model_id or "").strip()
        normalized_aspect_ratio = str(aspect_ratio or "").strip()
        if not normalized_subject:
            raise ValueError("hace falta el tema")
        if not normalized_model_id:
            raise ValueError("hace falta model_id")
        if normalized_aspect_ratio not in {"9:16", "16:9"}:
            raise ValueError("aspect_ratio debe ser 9:16 o 16:9")

        scenes = tuple(
            str(prompt or "").strip()
            for prompt in scene_prompts
            if str(prompt or "").strip()
        )
        if not 1 <= len(scenes) <= MAX_VIDEO_SCENES:
            raise ValueError(
                f"scene_prompts debe contener entre 1 y {MAX_VIDEO_SCENES} elementos"
            )

        rows = tuple(
            {
                "prompt": (
                    "Create cinematic stock-footage-style video for a short video "
                    f"about {normalized_subject}. Scene {index} focus: {scene}. "
                    "No text, subtitles, captions, watermarks, logos, or spoken audio."
                ),
                "modelChoice": normalized_model_id,
                "aspectRatio": normalized_aspect_ratio,
            }
            for index, scene in enumerate(scenes, start=1)
        )
        return LoomLoomVideoBatch(input_rows=rows)

    def download_video_results(
        self, run_id: str, destination_dir: str
    ) -> tuple[str, ...]:
        normalized_run_id = self._required_identifier(run_id, "run_id")
        raw_destination = str(destination_dir or "").strip()
        if not raw_destination:
            raise ValueError("hace falta destination_dir")
        normalized_destination = os.path.realpath(raw_destination)
        os.makedirs(normalized_destination, exist_ok=True)

        rows = sorted(
            self._list_all_result_rows(normalized_run_id),
            key=lambda row: self._integer(row, "rowIndex"),
        )
        if not rows:
            raise LoomLoomRunError("La ejecución de vídeo de LoomLoom no devolvió filas de resultado")

        downloaded = []
        for row in rows:
            row_index = self._integer(row, "rowIndex")
            status = str(row.get("status", "")).strip().lower()
            if status != "completed":
                detail = str(row.get("errorMessage", "")).strip() or status
                raise LoomLoomRunError(
                    f"La fila de vídeo {row_index + 1} de LoomLoom terminó con {detail}"
                )
            artifact = self._video_artifact(row)
            destination = os.path.join(
                normalized_destination, f"loomloom-video-{row_index + 1:02d}.mp4"
            )
            self._download_video_artifact(artifact["accessUrl"], destination)
            downloaded.append(destination)
            logger.info(
                "artefacto de vídeo de LoomLoom descargado: "
                f"run_id={normalized_run_id}, row={row_index + 1}, "
                f"filename={os.path.basename(destination)}"
            )
        return tuple(downloaded)

    def _video_artifact(self, row: Mapping[str, Any]) -> dict[str, Any]:
        artifacts = row.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise LoomLoomAPIError("los artefactos de la fila deben ser una lista")
        matching = [
            artifact
            for artifact in artifacts
            if isinstance(artifact, dict)
            and str(artifact.get("portName", "")).strip()
            == self.settings.result_port_name
            and str(artifact.get("mimeType", "")).strip().lower() == "video/mp4"
        ]
        if len(matching) != 1:
            raise LoomLoomAPIError(
                f"se esperaba un artefacto video/mp4 {self.settings.result_port_name!r}, "
                f"se recibieron {len(matching)}"
            )
        access_url = str(matching[0].get("accessUrl", "")).strip()
        parsed = urlsplit(access_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LoomLoomAPIError("accessUrl del artefacto de vídeo debe ser HTTP(S)")
        return {**matching[0], "accessUrl": access_url}

    def _download_video_artifact(self, access_url: str, destination: str) -> None:
        temporary = None
        downloaded_bytes = 0
        response = None
        try:
            response = self._session.get(
                access_url,
                stream=True,
                timeout=(5.0, self.settings.request_timeout_seconds),
            )
            response.raise_for_status()
            content_length = int(response.headers.get("content-length", 0) or 0)
            if content_length > MAX_VIDEO_ARTIFACT_BYTES:
                raise LoomLoomAPIError("el artefacto de vídeo supera el límite de descarga")
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=os.path.dirname(destination) or ".",
                prefix=os.path.basename(destination) + ".", suffix=".part", delete=False,
            ) as output:
                temporary = output.name
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > MAX_VIDEO_ARTIFACT_BYTES:
                        raise LoomLoomAPIError(
                            "el artefacto de vídeo supera el límite de descarga"
                        )
                    output.write(chunk)
            if downloaded_bytes == 0:
                raise LoomLoomAPIError("la descarga del artefacto de vídeo estaba vacía")
            os.replace(temporary, destination)
        except LoomLoomError:
            raise
        except (requests.RequestException, OSError, ValueError) as exc:
            raise LoomLoomAPIError(
                f"falló la descarga del artefacto de vídeo: {type(exc).__name__}"
            ) from exc
        finally:
            if response is not None:
                try:
                    # Con stream=True no se garantiza que se consuma la respuesta
                    # completa si falla la validación del tamaño o la escritura en
                    # disco. Cerrar explícitamente devuelve o libera de inmediato la
                    # conexión subyacente, para que los fallos seguidos no agoten poco a
                    # poco el pool de conexiones de Session; si falla el cierre solo se
                    # registra un aviso, sin sobrescribir la excepción de descarga
                    # original.
                    response.close()
                except Exception as exc:
                    logger.warning(
                        "no se pudo cerrar la respuesta de descarga del vídeo de LoomLoom: "
                        f"error={type(exc).__name__}"
                    )
            if temporary is not None and os.path.exists(temporary):
                os.remove(temporary)
