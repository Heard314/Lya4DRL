from dataclasses import dataclass

@dataclass
class Settings:
    enable_print: bool = False

settings = Settings()
