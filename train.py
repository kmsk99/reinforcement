# train.py
from envs.suika_env import SuikaEnv


def main():
    env = SuikaEnv()
    num_episodes = 100  # 실행할 에피소드 수

    for episode in range(num_episodes):
        state = env.reset()  # 환경을 리셋하고 초기 상태를 얻습니다.
        done = False  # 게임이 끝났는지 여부
        total_reward = 0  # 에피소드의 총 보상

        while not done:
            action = env.action_space.sample()  # 무작위 행동을 선택합니다.
            next_state, reward, done, _ = env.step(action)  # 행동을 환경에 적용합니다.
            total_reward += reward  # 보상을 누적합니다.
            state = next_state  # 상태를 업데이트합니다.

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()  # 환경을 종료합니다.


if __name__ == "__main__":
    main()
