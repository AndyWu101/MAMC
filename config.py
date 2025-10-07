import argparse


parser = argparse.ArgumentParser()


# 一般設定
parser.add_argument("--algorithm", default="MAMC", help="演算法名稱")


# 存檔設定
parser.add_argument("--save_result", action="store_true", help="是否存檔")
parser.add_argument("--output_path", default="Result", help="輸出檔案路徑")


# 實驗環境設定
parser.add_argument("--env_name", default="HalfCheetah-v5", help="任務名稱")
parser.add_argument("--device", default="cuda:0", help="實驗使用的設備")
parser.add_argument("--seed", default=0, type=int, help="亂數種子")
parser.add_argument("--max_steps", default=int(3e5), type=int, help="實驗多少步")


# 性能測試設定
parser.add_argument("--test_performance_freq", default=1000, type=int, help="每與環境互動多少 steps 要測試一次 actor 性能")
parser.add_argument("--test_n", default=20, type=int, help="每次測試 actor 要玩幾局")


# RL設定
parser.add_argument("--replay_buffer_size", default=int(1e6), type=int, help="replay buffer 的最大空間")
parser.add_argument("--batch_size", default=256, type=int, help="random mini-batch size")
parser.add_argument("--start_steps", default=5000, type=int, help="最開始使用隨機 action 進行探索")
parser.add_argument("--gamma", default=0.99, type=float, help="TD 的 discount")


# MAMC-soft設定
parser.add_argument("--actor_size", default=10, type=int, help="actor 數量")
parser.add_argument("--critic_size", default=10, type=int, help="critic 數量")
parser.add_argument("--smr_ratio", default=10, type=int)
parser.add_argument("--actor_learning_rate", default=1e-4, type=float, help="actor 的學習率")
parser.add_argument("--critic_learning_rate", default=3e-4, type=float, help="critic 的學習率")
parser.add_argument("--exploration_noise", default=0.1, type=float, help="與環境互動時對動作加噪，增強 replay buffer 多樣性")
parser.add_argument("--tau", default=0.005, type=float, help="以移動平均更新 target 的比例")
parser.add_argument("--quantile_q", default=0.2, type=float, help="分位數的取值")



args = parser.parse_args()


