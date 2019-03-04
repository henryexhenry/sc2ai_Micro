from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
from agent_RL import SmartAgent

class ZergAgent(base_agent.BaseAgent):

    def __init__(self):
        super(ZergAgent, self).__init__()
        
        self.attack_coordinates = None

    # 检查选取的单位类型是否 unit_type （如drone）
    def unit_type_is_selected(self, obs, unit_type):

        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False



    def get_units_by_type(self, obs, unit_type):  # 通过判断类型 从而选取单位
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def can_do(self, obs, action): # 判断动作是否可以被执行
        return action in obs.observation.available_actions # return True/False



    def step(self, obs):
        super(ZergAgent, self).step(obs)

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()
        
            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)



        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) >= 30:
            if self.unit_type_is_selected(obs, units.Zerg.Zergling):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now", self.attack_coordinates)

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")



        spawning_pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        if len(spawning_pools) < 4:  # 检查spawning pool是否未被建造
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                # 检查是否可以建造spawning pool
                if (actions.FUNCTIONS.Build_SpawningPool_screen.id in obs.observation.available_actions):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)
                    # 在画面中随意位置上建造spawning pool
                    return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))



        if self.unit_type_is_selected(obs, units.Zerg.Larva): # 检查选取单位是否Larva

            free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)

            if free_supply < 4:
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")


            '''
            if (actions.FUNCTIONS.Train_Zergling_quick.id in obs.observation.available_actions): # 检查是否可以训练Zergling

                return actions.FUNCTIONS.Train_Zergling_quick("now")
            '''# 语句优化（以can_do函数代替）
            
            if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                return actions.FUNCTIONS.Train_Zergling_quick("now")

        
        # 选取drones
        # *************************************************************************************************************
        drones = self.get_units_by_type(obs, units.Zerg.Drone) # drone 是工虫
        
        if len(drones) > 0 and len(spawning_pools) < 4:
            drone = random.choice(drones)

            # select_all_type 即ctrl+click （选择画面中所有drones）
            return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
            # 上句传入了drone的xy坐标，亦可以进入drone的其他属性比如health，shields，energy， build_progress, ideal_harvesters, assigned_harvesters
        # *************************************************************************************************************
        

        # 选取larvae
        # *************************************************************************************************************
        larvae = self.get_units_by_type(obs, units.Zerg.Larva) # larvae 是众数 larva 是单个小虫子

        if len(larvae) > 0:
            larva = random.choice(larvae)

            return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))
         # *************************************************************************************************************
        

        return actions.FUNCTIONS.no_op()


def main(unused_argv):
    agent = SmartAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="AbyssalReef",
                    players=[sc2_env.Agent(sc2_env.Race.zerg),  # 第一玩家：我方为agent，种族为虫族
                             sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],  # 第二玩家：对方为sc内置ai，种族随机，难度简单
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(
                            screen=84, minimap=64),  # 设置画面解像度，决定机器人的视野
                        use_feature_units=True
                    ),
                    step_mul=16,  # 机器人采取行动（action）前需要进行的‘game step’ 数量 （数值越小，APM越高）
                    game_steps_per_episode=0,  # 游戏时长，设为0即不限时
                    visualize=True) as env:  # optional 显示出observation layer 的detail， 帮助我们理解机器人的进程

                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
