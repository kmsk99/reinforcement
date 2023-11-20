import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from selenium import webdriver
import pyautogui
import time
from PIL import Image
import torchvision.transforms as T
import pytesseract
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import os
from PIL import Image


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.tanh(self.layer_3(x))  # 출력은 -1과 1 사이
        x = (x + 1) / 2  # 이제 출력은 0과 1 사이
        x = x * self.max_action  # 출력을 환경의 액션 공간에 맞게 스케일링
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

        # Q2 architecture
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = torch.relu(self.layer_1(xu))
        x1 = torch.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)

        x2 = torch.relu(self.layer_4(xu))
        x2 = torch.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)

        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = torch.relu(self.layer_1(xu))
        x1 = torch.relu(self.layer_2(x1))
        return self.layer_3(x1)


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, next_state, action, reward, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# 이미지 처리 함수
def process_game_image(image_path):
    image = Image.open(image_path).convert("RGB")  # RGBA를 RGB로 변환
    return image


# PyTorch의 transform을 사용하여 이미지를 텐서로 변환
transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize((84, 84), antialias=True),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def perform_click_at_position(x, y):
    pyautogui.click(x=x, y=y)


def setup_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)


def get_canvas_properties(driver):
    game_board = driver.find_element(value="GameCanvas")
    location = game_board.location
    size = game_board.size
    offset = {"x": 18, "y": 140}
    location["x"] += offset["x"]
    location["y"] += offset["y"]
    return location, size


def select_action(state, actor_model):
    if isinstance(state, Image.Image):
        state = transform(state)
        state = state.view(1, -1)
    else:
        state = state.view(1, -1)
    state = state.to(device)

    # 액터 모델을 통해 액션 값을 받아옴
    action = actor_model(state).cpu().data.numpy().flatten()

    # 액션 값을 0과 1 사이로 제한
    action = np.clip(action, 0, 1)

    return action


def select_random_action():
    # 0과 1 사이의 랜덤한 액션을 선택합니다.
    action = np.random.uniform(0, 1)
    return action


def perform_action(action, canvas_location, canvas_size):
    # 액션을 화면의 너비에 맞게 스케일링
    x_position = int(canvas_location["x"] + action * canvas_size["width"])

    # 화면의 중앙 높이에 클릭
    y_position = int(canvas_location["y"] + canvas_size["height"] / 2)

    # 클릭 위치가 실제 화면 범위 내에 있는지 확인
    x_position = min(
        max(x_position, canvas_location["x"]),
        canvas_location["x"] + canvas_size["width"] - 1,
    )
    y_position = min(
        max(y_position, canvas_location["y"]),
        canvas_location["y"] + canvas_size["height"] - 1,
    )

    pyautogui.click(x=x_position, y=y_position)


# 템플릿 이미지를 로드하는 함수
def load_templates(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith(".png"):
            digit = filename.split(".")[0]
            template_image = cv2.imread(os.path.join(template_dir, filename), 0)
            templates[digit] = template_image
    return templates


# 템플릿 매칭을 수행하는 함수
def match_templates(templates, search_image):
    search_image_gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, thresh = cv2.threshold(search_image_gray, 200, 255, cv2.THRESH_BINARY_INV)
    matched_scores = {}
    for digit, template in templates.items():
        result = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > 0.9:  # 임계값 설정
            matched_scores[digit] = max_loc

    cv2.imwrite("converted.png", thresh)

    return matched_scores


# 게임의 점수를 인식하고 리워드를 계산하는 함수
def get_reward(previous_score, image_path, templates):
    score_width = 160  # 스코어 영역의 너비
    score_height = 70  # 스코어 영역의 높이
    # 스크린샷을 캡처하고 점수가 표시된 영역을 로드
    score_area = process_game_image(image_path).crop((0, 0, score_width, score_height))
    score_area.save("game/score_area.png")

    score_area = cv2.imread("game/score_area.png")

    # 템플릿 매칭을 수행
    matched_scores = match_templates(templates, score_area)

    if not matched_scores:
        print("No digits matched. Returning previous score.")
        return 0, previous_score

    # 매칭된 숫자를 x 좌표 기준으로 정렬
    sorted_scores = sorted(matched_scores.items(), key=lambda item: item[1][0])

    # 정렬된 숫자를 바탕으로 최종 점수 계산
    current_score = int("".join([digit for digit, _ in sorted_scores]))

    # 리워드는 이전 점수와 현재 점수의 차이
    reward = current_score - previous_score
    print(
        f"Current Score: {current_score}, Previous Score: {previous_score}, Reward: {reward}"
    )

    # 다음 번 계산을 위해 현재 점수를 '이전 점수'로 저장
    previous_score = current_score

    return reward, previous_score


# 템플릿 이미지들이 저장된 디렉토리
template_dir = "digits_templates/"

# 템플릿 로드
templates = load_templates(template_dir)


# 학습 루프를 위한 함수
def run_episodes(total_episodes, max_steps_per_episode):
    for episode in range(total_episodes):
        previous_score = 0
        take_screenshot("game/game_board.png", canvas_location, canvas_size)
        state_image = process_game_image("game/game_board.png")
        state = transform(state_image).to(device)

        for step in range(max_steps_per_episode):
            action = select_random_action()
            x_position, y_position = get_click_coordinates(
                action, canvas_location, canvas_size
            )
            perform_click_at_position(x_position, y_position)
            time.sleep(3)  # Allow game to update
            take_screenshot("game/next_state.png", canvas_location, canvas_size)

            # Commented out reward calculation for now
            reward, previous_score = get_reward(
                previous_score, "game/next_state.png", templates
            )


# Helper functions
def take_screenshot(filename, location, size):
    screenshot = pyautogui.screenshot(
        region=(location["x"], location["y"], size["width"], size["height"])
    )
    screenshot.save(filename)


def get_click_coordinates(action, location, size):
    x_position = int(location["x"] + action * size["width"])
    y_position = int(location["y"] + size["height"] / 2)
    x_position = min(max(x_position, location["x"]), location["x"] + size["width"] - 1)
    y_position = min(max(y_position, location["y"]), location["y"] + size["height"] - 1)
    return x_position, y_position


# 메인 실행 코드
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    driver = setup_webdriver()
    driver.get("https://suika-game.app/ko")
    time.sleep(1)
    canvas_location, canvas_size = get_canvas_properties(driver)

    run_episodes(total_episodes=1000, max_steps_per_episode=1000)
