"""Local LLM client implementation using llama-cpp-python for GGUF models."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from pydantic import BaseModel, TypeAdapter, ValidationError

from vocab_learner.llm.protocol import LLMClient, LLMError, LLMValidationError
from vocab_learner.logging_setup import get_logger

if TYPE_CHECKING:
    from llama_cpp import Llama

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LocalLlamaClient(LLMClient):
    """
    In-process local LLM client using llama-cpp-python for GGUF models.
    Supports Metal GPU acceleration on macOS (Apple Silicon).
    Performs OCR (Tesseract) + structured extraction for image tasks.
    """

    def __init__(
        self,
        *,
        model_path: Path | str,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        tesseract_cmd: str = "tesseract",
    ) -> None:
        self._model_path = Path(model_path)
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._tesseract_cmd = tesseract_cmd
        self._llm: Llama | None = None

        if not self._model_path.exists():
            raise FileNotFoundError(f"Local GGUF model not found at: {self._model_path}")

    def _get_llm(self) -> Llama:
        """Lazy-initialize Llama instance with Metal GPU offloading."""
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise LLMError(
                    "llama-cpp-python is not installed. Install it via 'uv add llama-cpp-python'."
                ) from e

            log.info(
                "local_llm_loading",
                model_path=str(self._model_path),
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self._n_ctx,
            )
            self._llm = Llama(
                model_path=str(self._model_path),
                n_gpu_layers=self._n_gpu_layers,
                n_ctx=self._n_ctx,
                verbose=False,
            )
            log.info("local_llm_loaded", model_path=str(self._model_path))
        return self._llm

    def complete_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        llm = self._get_llm()
        schema = TypeAdapter(response_model).json_schema()

        system_content = (
            f"{system_prompt}\n\nReturn ONLY a valid JSON object matching the required schema."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

        log.info(
            "local_structured_request",
            model=model or self._model_path.name,
            response_model=response_model.__name__,
            max_tokens=max_tokens,
        )

        try:
            res_raw = llm.create_chat_completion(
                messages=cast(Any, messages),
                response_format={"type": "json_object", "schema": schema},
                max_tokens=max_tokens,
            )
        except Exception as e:
            log.error("local_llm_completion_error", error=str(e))
            raise LLMError(f"Local LLM completion failed: {e}") from e

        res = cast(dict[str, Any], res_raw)
        choices = cast(list[dict[str, Any]], res.get("choices", []))
        content = (choices[0].get("message", {}).get("content") or "") if choices else ""

        parsed, first_error = self._try_parse(content, response_model)
        if parsed is not None:
            return parsed

        log.warning(
            "local_llm_validation_failed_attempting_repair",
            error=str(first_error),
            raw_content=content[:200],
        )

        # Attempt one repair turn
        repair_messages: list[dict[str, Any]] = [
            *messages,
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    f"Your previous output failed validation: {first_error}\n"
                    "Please return ONLY the corrected JSON object matching the schema."
                ),
            },
        ]

        try:
            repair_raw = llm.create_chat_completion(
                messages=cast(Any, repair_messages),
                response_format={"type": "json_object", "schema": schema},
                max_tokens=max_tokens,
            )
            repair_res = cast(dict[str, Any], repair_raw)
            repair_choices = cast(list[dict[str, Any]], repair_res.get("choices", []))
            repair_content = (
                (repair_choices[0].get("message", {}).get("content") or "")
                if repair_choices
                else ""
            )
            repaired, final_error = self._try_parse(repair_content, response_model)
            if repaired is not None:
                return repaired
            raise LLMValidationError(
                f"Local LLM output failed schema validation after repair: {final_error}"
            )
        except LLMValidationError:
            raise
        except Exception as e:
            raise LLMError(f"Local LLM repair failed: {e}") from e

    def complete_vision_structured(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        image_media_type: str,
        response_model: type[T],
        max_tokens: int = 4096,
    ) -> T:
        """
        Extract text via OCR (Tesseract) from image_bytes, then use local GGUF
        model to parse and structure into response_model.
        """
        ocr_text = self._run_ocr(image_bytes)

        combined_prompt = (
            f"{user_prompt}\n\n"
            "--- Extracted Dictionary Page OCR Text ---\n"
            f"{ocr_text}\n"
            "------------------------------------------\n\n"
            "Analyze the OCR text above and extract all vocabulary entries."
        )

        return self.complete_structured(
            model=model,
            system_prompt=system_prompt,
            user_prompt=combined_prompt,
            response_model=response_model,
            max_tokens=max_tokens,
        )

    def _run_ocr(self, image_bytes: bytes) -> str:
        """Run Tesseract OCR on in-memory image bytes."""
        tesseract_binary = shutil.which(self._tesseract_cmd) or self._tesseract_cmd
        try:
            proc = subprocess.run(
                [tesseract_binary, "stdin", "stdout", "--psm", "6"],
                input=image_bytes,
                capture_output=True,
                check=True,
            )
            ocr_text = proc.stdout.decode("utf-8", errors="replace")
            log.info("local_ocr_extracted", bytes_in=len(image_bytes), chars_out=len(ocr_text))
            return ocr_text
        except FileNotFoundError as e:
            raise LLMError(
                f"Tesseract OCR executable '{self._tesseract_cmd}' not found. "
                "Install via 'brew install tesseract' or set VOCAB_LEARNER_TESSERACT_CMD."
            ) from e
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
            log.warning("local_ocr_subprocess_error", error=err_msg)
            # If tesseract returns partial stdout with warnings, return stdout if non-empty
            if e.stdout:
                stdout_str: str = e.stdout.decode("utf-8", errors="replace")
                return stdout_str
            raise LLMError(f"Tesseract OCR failed: {err_msg}") from e

    @staticmethod
    def _try_parse(raw: str, response_model: type[T]) -> tuple[T | None, Exception | None]:
        """Try parsing JSON into response_model, stripping markdown if present."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return response_model.model_validate_json(text), None
        except (ValidationError, json.JSONDecodeError) as e:
            return None, e
