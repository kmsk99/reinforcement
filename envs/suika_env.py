import gym
from selenium import webdriver
import pyautogui


class SuikaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 환경 초기화

    def step(self, action):
        # 행동 수행 및 보상 계산
        pass

    def reset(self):
        # 환경 리셋
        pass

    def render(self):
        # 환경 렌더링
        pass

    def close(self):
        # 환경 종료
        pass
