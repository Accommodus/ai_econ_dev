from ai_economist import foundation
import numpy as np

# creates and registers a zone class
from ai_economist.foundation.entities.landmarks import Landmark, landmark_registry

@landmark_registry.add
class Zone(Landmark):
    name = 'Zone'
    ownable = True
    
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

        print(base_env_kwargs['zones'])

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

    # dict thats stores information on the different types of zone
    'zones': {

        "low_density_residential":{
            "cost":10,
            "population":10
        },

        "med_density_residential":{
            "cost":20,
            "population":20
        },

        "high_density_residential":{
            "cost":30,
            "population":30
        },

        "recreational":{
            "cost":10,
            "population":0
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