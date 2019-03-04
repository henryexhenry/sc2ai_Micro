from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2 import maps
from absl import app
import random
from agent_01 import SmartAgent
import time

_mapName = 'HK2V1_RESET_1DIE'


def run_loop(agents, env, max_frames=0, max_episodes=0):
    """A run loop to have agents and an environment interact."""
    total_frames = 0
    total_episodes = 0
    start_time = time.time()

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    for agent, obs_spec, act_spec in zip(agents, observation_spec, action_spec):
        agent.setup(obs_spec, act_spec)
    try:
        while not max_episodes or total_episodes < max_episodes:
            total_episodes += 1
            timesteps = env.reset()
            for a in agents:
                a.reset()
            while True:
                total_frames += 1
                actions = [agent.step(timestep)
                           for agent, timestep in zip(agents, timesteps)]
                if max_frames and total_frames >= max_frames:
                    return
                if timesteps[0].last():
                    break
                timesteps = env.step(actions)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed_time = time.time() - start_time
        print("Took %.3f seconds for %s steps: %.3f fps" % (
            elapsed_time, total_frames, total_frames / elapsed_time))


def main(unused_argv):
    agent = SmartAgent()

    globals()[_mapName] = type(
        _mapName, (maps.mini_games.MiniGame,), dict(filename=_mapName))

    with sc2_env.SC2Env(
            map_name=_mapName,
            # 第一玩家：我方为agent，种族为人族
            players=[sc2_env.Agent(sc2_env.Race.terran)],

            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=84, minimap=64),  # 设置画面解像度，决定机器人的视野
                use_feature_units=True
            ),
            step_mul=16,  # 机器人采取行动（action）前需要进行的‘game step’ 数量 （数值越小，APM越高）
            game_steps_per_episode=0,  # 游戏时长，设为0即不限时
            visualize=True) as env:  # optional 显示出observation layer 的detail， 帮助我们理解机器人的进程

        # run the steps
        run_loop([agent], env, 16000)

        # save the model
        #agent.dqn.save_model(path, 1)


'''  ********************* in run_loop
        agent.setup(env.observation_spec(), env.action_spec())
        
        timesteps = env.reset()
        agent.reset()
        while True:
            step_actions = [agent.step(timesteps[0])]
            if timesteps[0].last():
                break
            timesteps = env.step(step_actions)
    *****************************'''


if __name__ == "__main__":
    app.run(main)
