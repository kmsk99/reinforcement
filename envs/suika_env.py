import gym
from selenium import webdriver
import pyautogui
import time
from gym.spaces import Discrete
import cv2
import pytesseract


class SuikaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.driver = webdriver.Chrome()
        self.driver.get("https://suikagame.com")
        time.sleep(5)  # 페이지 로딩 시간을 충분히 기다리도록 변경
        self.game_board = self.driver.find_element(value="GameCanvas")
        self.canvas_size = self.game_board.size
        self.action_space = Discrete(
            self.canvas_size["width"]
        )  # x 좌표의 가능한 값: 0 to canvas_size['width']

    def step(self, action):
        x = self.game_board.location["x"] + action
        center_y = self.game_board.location["y"] + self.canvas_size["height"] / 2
        pyautogui.click(x=x, y=center_y)
        state = self._get_state()
        reward = self._get_reward()
        done = self._is_done()
        return state, reward, done, {}

    def reset(self):
        self.driver.refresh()
        time.sleep(5)  # 페이지 리셋 후 로딩 시간을 충분히 기다리도록 변경
        return self._get_state()

    def close(self):
        self.driver.quit()

    def _get_state(self):
        self.game_board.screenshot("game_board.png")
        return "game_board.png"

    def _get_reward(self):
        # 게임 보드 스크린샷 읽기
        img = cv2.imread("game_board.png")

        # 상단 왼쪽 영역에서 점수 추출을 위한 이미지 크롭 (크기와 위치는 게임에 따라 조절이 필요합니다.)
        roi = img[0:50, 0:150]

        # pytesseract를 사용하여 이미지에서 텍스트 추출
        score_text = pytesseract.image_to_string(roi, config="--psm 6")

        # 텍스트에서 숫자만 추출
        score = int("".join(filter(str.isdigit, score_text)))

        return score

    def _is_done(self):
        # TODO: 게임이 종료되었는지 확인하고 반환
        pass
