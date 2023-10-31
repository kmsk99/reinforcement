from selenium import webdriver
import pyautogui
import time

# Selenium WebDriver를 사용하여 브라우저 열기
driver = webdriver.Chrome()
driver.get('https://suikagame.com')

# 페이지가 로드될 때까지 기다리기
time.sleep(5)

# 게임 플레이 현황 가져오기 (예: 게임 보드의 스크린샷)
game_board = driver.find_element(value = 'GameCanvas')  # 게임 보드의 HTML 요소 ID
game_board.screenshot('game_board.png')

# GameCanvas의 크기와 위치 가져오기
canvas_location = game_board.location
canvas_size = game_board.size

# 정 중앙에 클릭할 위치 계산
center_x = canvas_location['x'] + canvas_size['width'] / 2
center_y = canvas_location['y'] + canvas_size['height'] / 2

# 마우스 클릭 구현
pyautogui.click(x=center_x, y=center_y)  # 정 중앙에 마우스 클릭


time.sleep(5)

# 브라우저 닫기
# driver.quit()