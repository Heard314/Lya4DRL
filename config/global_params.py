from dataclasses import dataclass

@dataclass
class Settings:
    enable_print: bool = False
    project_root: str = "D:/project/MADRL4MEC_Dynamic_Delay_Adjust/"

settings = Settings()
