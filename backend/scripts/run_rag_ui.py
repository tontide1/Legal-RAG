from __future__ import annotations

import os

import uvicorn

from backend.core.rag_ui_local_embedding import (
    apply_local_embedding_override,
    needs_local_rag_ui_embedding_override,
)


def _apply_local_embedding_override_if_needed() -> None:
    binding = os.getenv("EMBEDDING_BINDING", "")
    model_name = os.getenv("EMBEDDING_MODEL", "")

    if needs_local_rag_ui_embedding_override(binding=binding, model=model_name):
        print(
            "INFO: rag-ui is overriding LightRAG's OpenAI embedding path to use the "
            "local SentenceTransformer model 'huyydangg/DEk21_hcmute_embedding'."
        )
        apply_local_embedding_override(
            binding=binding,
            model_name=model_name,
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "384")),
        )


def main() -> None:
    from lightrag.api.lightrag_server import check_and_install_dependencies
    from lightrag.api.lightrag_server import check_env_file
    from lightrag.api.lightrag_server import configure_logging
    from lightrag.api.lightrag_server import create_app
    from lightrag.api.lightrag_server import display_splash_screen
    from lightrag.api.lightrag_server import global_args
    from lightrag.api.lightrag_server import update_uvicorn_mode_config
    from lightrag.api.config import initialize_config

    initialize_config()
    _apply_local_embedding_override_if_needed()

    global_args.host = os.getenv("HOST", "0.0.0.0")
    global_args.port = int(os.getenv("PORT", "8001"))
    global_args.working_dir = os.getenv("LIGHTRAG_WORKING_DIR", "/app/backend/data")

    if not check_env_file():
        raise SystemExit(1)

    check_and_install_dependencies()
    configure_logging()
    update_uvicorn_mode_config()
    display_splash_screen(global_args)

    app = create_app(global_args)

    uvicorn.run(
        app=app,
        host=global_args.host,
        port=global_args.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
