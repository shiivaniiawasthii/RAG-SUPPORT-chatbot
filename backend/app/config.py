from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "gemini"  # "gemini" or "openai"
    gemini_api_key: str = ""
    openai_api_key: str = ""

    vector_store_dir: str = "./vector_store_data"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    max_upload_mb: int = 10
    rate_limit: str = "20/minute"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
