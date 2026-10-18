"""The experiment grid. Run folders under runs/ are named by these keys."""
M05, M15 = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"

EXPERIMENTS = {
    "E0_zeroshot_0.5B":     {"model": M05, "lora": False},
    "E0_zeroshot_1.5B":     {"model": M15, "lora": False},
    "E1_lora_0.5B_saq+mcq": {"model": M05, "lora": True, "data_mix": "saq+mcq"},
    "E2_lora_0.5B_saq":     {"model": M05, "lora": True, "data_mix": "saq"},
    "E3_lora_1.5B_saq+mcq": {"model": M15, "lora": True, "data_mix": "saq+mcq"},
}
