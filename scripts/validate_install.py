import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import minedojo


if __name__ == "__main__":
    env = minedojo.make(
        task_id="combat_spider_plains_leather_armors_diamond_sword_shield",
        image_size=(288, 512),
        world_seed=123,
        seed=42,
    )

    print(f"[INFO] Create a task with prompt: {env.task_prompt}")

    env.reset()
    for _ in range(2000):
        action = env.action_space.no_op()
        action["attack"] = 1
        action["camera"] = [0, 3]
        obs, reward, done, info = env.step(action)
    env.close()

    print("[INFO] Installation Success")
