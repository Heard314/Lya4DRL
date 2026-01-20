from dataclasses import dataclass
import torch

@dataclass
class Settings:
    enable_print: bool = False
    exp_result_dir: str = "D:\\Desktop\\work\\yanjiushengbishe\\hupaper\\project\\MADDPG_CTDE\\MADRL-Based-Multi-Task-Partial-Computation-Offloading-in-MEC\\"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir: str = ""
    plot_dir: str = ""
    weight_dir: str = ""
    resume_episode: int = 0
    seed: int = 0
    is_evaluate: bool = False
settings = Settings()
