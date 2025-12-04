from dataclasses import dataclass
import torch

@dataclass
class Settings:
    enable_print: bool = False
    project_root: str = "D:/project/MADRL4MEC_Dynamic_Delay_Adjust/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

settings = Settings()
