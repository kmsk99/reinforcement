# train.py
from envs.suika_env import SuikaEnv
from agents.dqn_agent import DQNAgent


def main():
    env = SuikaEnv()
    agent = DQNAgent()
    # 학습 루프


if __name__ == "__main__":
    main()
