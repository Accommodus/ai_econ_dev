from ai_economist import foundation
import numpy as np

# creates and registers a zone class
from ai_economist.foundation.entities.landmarks import Landmark, landmark_registry

class Zone(Landmark):
    ownable = True

    construction_cost = 0
    max_residental_population = 0
    max_commercial_population = 0


class Building(Landmark):
    ownable = True

    construction_cost = 0
    max_residental_population = 0
    max_commercial_population = 0
    
# creates the Foundation scenario
from ai_economist.foundation.base.base_env import BaseEnvironment, scenario_registry

@scenario_registry.add
class UrbanDesignSimulation(BaseEnvironment):
    name = "UrbanDesignSimulation"
    required_entities = ['Water']
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

            landmark_registry.add(type(zone_name, (Zone,), {'name': zone_name}))
            self.required_entities.append(zone_name)

        for building in locations_dict.keys():
            building_name = building + '_building'
            building_atri = {**{'name': building_name}, **locations_dict[building]}

            landmark_registry.add(type(building_name, (Building,), {'name': building_name}))
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

env_config = {
    # ===== STANDARD ARGUMENTS ======
    'n_agents': 4,          # Number of non-planner agents
    'world_size': [15, 15], # [Height, Width] of the env world
    'episode_length': 1000, # Number of timesteps per episode
    
    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code)
    # Otherwise, the policy selects only 1 action
    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,
    
    # When flattening observations, concatenate scalar & vector observations before output
    # Otherwise, return observations with minimal processing
    'flatten_observations': False,
    # When Flattening masks, concatenate each action subspace mask into a single array
    # Note: flatten_masks = True is recommended for masking action logits
    'flatten_masks': True,
    
    
    # ===== COMPONENTS =====
    # Which components to use (specified as list of {"component_name": {component_kwargs}} dictionaries)
    #   "component_name" refers to the component class's name in the Component Registry
    #   {component_kwargs} is a dictionary of kwargs passed to the component class
    # The order in which components reset, step, and generate obs follows their listed order below
    'components': [
        # (1) Building houses
        {'Build': {}},
        # (2) Trading collectible resources
        {'ContinuousDoubleAuction': {'max_num_orders': 5}},
        # (3) Movement and resource collection
        {'Gather': {}},
    ],
    
    # ===== SCENARIO =====
    # Which scenario class to use (specified by the class's name in the Scenario Registry)
    'scenario_name': 'UrbanDesignSimulation',
    
    # (optional) kwargs of the chosen scenario class
    'starting_agent_coin': 10,

    # dict that stores information on the different types of zone
    'locations': {

        "low_density_residential":{
            "construction_cost":10,
            "max_residental_population":10
        },

        "med_density_residential":{
            "construction_cost":20,
            "max_residental_population":20
        },

        "high_density_residential":{
            "construction_cost":30,
            "max_residental_population":30
        },

        "recreational":{
            "construction_cost":10
        }
    }
}

env = foundation.make_env_instance(**env_config)

""" 
agents:
    proprietor:
        represents business interest. 
        is awarded for making money.

    planner:
        represents municipal government.
        is awarded for satisfying population

components:
    move_proprietor:
        allows agents to select different tiles
        infinite range
        not allowed to go onto water-like tiles or landmark

    build_property:
        pay coin to build a building
        landmark is constructed and agent has deed

entities:
    coin:
        collectable through trading and through royalties from deed

    property:
        exist on map
        different types specified in env

    zones:
        designate what kinds of property can be built on certain tiles

"""