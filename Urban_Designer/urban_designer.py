from ai_economist import foundation
import numpy as np

# creates and registers a zone class
from ai_economist.foundation.entities.landmarks import Landmark, landmark_registry
from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

class Build_Property():
    name = "Build"
    component_type = "Build"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
            self,
            *base_component_args,
            payment=10,
            payment_max_skill_multiplier=1,
            skill_dist="none",
            build_labor=10.0,
            **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        self.payment = int(payment)
        assert self.payment >= 0

        self.payment_max_skill_multiplier = int(payment_max_skill_multiplier)
        assert self.payment_max_skill_multiplier >= 1

        self.resource_cost = {"Wood": 1, "Stone": 1}

        self.build_labor = float(build_labor)
        assert self.build_labor >= 0

        self.skill_dist = skill_dist.lower()
        assert self.skill_dist in ["none", "pareto", "lognormal"]

        self.sampled_skills = {}

        self.builds = []

    def agent_can_build(self, agent):
        """Return True if agent can actually build in its current location."""
        # See if the agent has the resources necessary to complete the action
        for resource, cost in self.resource_cost.items():
            if agent.state["inventory"][resource] < cost:
                return False

        # Do nothing if this spot is already occupied by a landmark or resource
        if self.world.location_resources(*agent.loc):
            return False
        if self.world.location_landmarks(*agent.loc):
            return False
        # If we made it here, the agent can build.
        return True

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        Add a single action (build) for mobile agents.
        """
        # This component adds 1 action that mobile agents can take: build a house
        if agent_cls_name == "BasicMobileAgent":
            return 1

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        For mobile agents, add state fields for building skill.
        """
        if agent_cls_name not in self.agent_subclasses:
            return {}
        if agent_cls_name == "BasicMobileAgent":
            return {"build_payment": float(self.payment), "build_skill": 1}
        raise NotImplementedError

    def component_step(self):
        """
        See base_component.py for detailed description.
        Convert stone+wood to house+coin for agents that choose to build and can.
        """
        world = self.world
        build = []
        # Apply any building actions taken by the mobile agents
        for agent in world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            # This component doesn't apply to this agent!
            if action is None:
                continue

            # NO-OP!
            if action == 0:
                pass

            # Build! (If you can.)
            elif action == 1:
                if self.agent_can_build(agent):
                    # Remove the resources
                    for resource, cost in self.resource_cost.items():
                        agent.state["inventory"][resource] -= cost

                    # Place a house where the agent is standing
                    loc_r, loc_c = agent.loc
                    world.create_landmark("House", loc_r, loc_c, agent.idx)

                    # Receive payment for the house
                    agent.state["inventory"]["Coin"] += agent.state["build_payment"]

                    # Incur the labor cost for building
                    agent.state["endogenous"]["Labor"] += self.build_labor

                    build.append(
                        {
                            "builder": agent.idx,
                            "loc": np.array(agent.loc),
                            "income": float(agent.state["build_payment"]),
                        }
                    )

            else:
                raise ValueError

        self.builds.append(build)

    def generate_observations(self):
        """
        See base_component.py for detailed description.
        Here, agents observe their build skill. The planner does not observe anything
        from this component.
        """

        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[agent.idx] = {
                "build_payment": agent.state["build_payment"] / self.payment,
                "build_skill": self.sampled_skills[agent.idx],
            }

        return obs_dict

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.
        Prevent building only if a landmark already occupies the agent's location.
        """

        masks = {}
        # Mobile agents' build action is masked if they cannot build with their
        # current location and/or endowment
        for agent in self.world.agents:
            masks[agent.idx] = np.array([self.agent_can_build(agent)])

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.
        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        world = self.world

        build_stats = {a.idx: {"n_builds": 0} for a in world.agents}
        for builds in self.builds:
            for build in builds:
                idx = build["builder"]
                build_stats[idx]["n_builds"] += 1

        out_dict = {}
        for a in world.agents:
            for k, v in build_stats[a.idx].items():
                out_dict["{}/{}".format(a.idx, k)] = v

        num_houses = np.sum(world.maps.get("House") > 0)
        out_dict["total_builds"] = num_houses

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.
        Re-sample agents' building skills.
        """
        world = self.world

        self.sampled_skills = {agent.idx: 1 for agent in world.agents}

        PMSM = self.payment_max_skill_multiplier

        for agent in world.agents:
            if self.skill_dist == "none":
                sampled_skill = 1
                pay_rate = 1
            elif self.skill_dist == "pareto":
                sampled_skill = np.random.pareto(4)
                pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            elif self.skill_dist == "lognormal":
                sampled_skill = np.random.lognormal(-1, 0.5)
                pay_rate = np.minimum(PMSM, (PMSM - 1) * sampled_skill + 1)
            else:
                raise NotImplementedError

            agent.state["build_payment"] = float(pay_rate * self.payment)
            agent.state["build_skill"] = float(sampled_skill)

            self.sampled_skills[agent.idx] = sampled_skill

        self.builds = []

    def get_dense_log(self):
        """
        Log builds.
        Returns:
            builds (list): A list of build events. Each entry corresponds to a single
                timestep and contains a description of any builds that occurred on
                that timestep.
        """
        return self.builds
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