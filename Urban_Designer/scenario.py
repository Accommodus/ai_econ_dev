from ai_economist.foundation.entities.landmarks import landmark_registry
import landmarks

# creates the Foundation scenario
from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry
import components
@scenario_registry.add
class UrbanDesignSimulation(BaseEnvironment):
    name = "UrbanDesignSimulation"
    required_entities = ['Water']

    building_list = []
    zone_list = []

    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    def __init__(
        self,
        *base_env_args,
        **base_env_kwargs,
    ):

        locations_dict = base_env_kwargs['locations']

        for zone in locations_dict.keys():
            zone_name = zone + '_zone'
            zone_atri = {**{'name': zone_name}, **locations_dict[zone]}

            landmark_registry.add(type(zone_name, (landmarks.BaseZone,), {'name': zone_name}))
            self.required_entities.append(zone_name)

        for building in locations_dict.keys():
            building_name = building + '_building'
            building_atri = {**{'name': building_name}, **locations_dict[building]}

            landmark_registry.add(type(building_name, (landmarks.BaseBuilding,), {'name': building_name}))
            self.required_entities.append(building_name)

        print(self.required_entities)


    def reset_layout(self):
        """Resets the state of the world object (self.world)."""
        pass

    def reset_agent_states(self):
        """Resets the state of the agent objects (self.world.agents & self.world.planner)."""
        pass

    def scenario_step(self):
        """Implements the passive dynamics of the environment."""
        pass

    def generate_observations(self):
        """Yields some basic observations about the world/agent states."""
        pass

    def compute_reward(self):
        """Determines the reward each agent receives at the end of each timestep."""
        pass

    def reset_starting_layout(self):
        """"""
        pass