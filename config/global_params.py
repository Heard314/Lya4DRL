from dataclasses import dataclass

@dataclass
class Settings:
    enable_print: bool = False
    project_root: str = "D:/Desktop/work/yanjiushengbishe/hupaper/project/MADDPG_CTDE/MADRL-Based-Multi-Task-Partial-Computation-Offloading-in-MEC/"

settings = Settings()
