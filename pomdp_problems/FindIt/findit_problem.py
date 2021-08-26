"""The classic Tiger problem.

This is a POMDP problem; Namely, it specifies both
the POMDP (i.e. state, action, observation space)
and the T/O/R for the agent as well as the environment.

The description of the tiger problem is as follows: (Quote from `POMDP:
Introduction to Partially Observable Markov Decision Processes
<https://cran.r-project.org/web/packages/pomdp/vignettes/POMDP.pdf>`_ by
Kamalzadeh and Hahsler )

A tiger is put with equal probability behind one
of two doors, while treasure is put behind the other one.
You are standing in front of the two closed doors and
need to decide which one to open. If you open the door
with the tiger, you will get hurt (negative reward).
But if you open the door with treasure, you receive
a positive reward. Instead of opening a door right away,
you also have the option to wait and listen for tiger noises. But
listening is neither free nor entirely accurate. You might hear the
tiger behind the left door while it is actually behind the right
door and vice versa.

States: tiger-left, tiger-right
Actions: open-left, open-right, listen
Rewards:
    +10 for opening treasure door. -100 for opening tiger door.
    -1 for listening.
Observations: You can hear either "tiger-left", or "tiger-right".

Note that in this example, the TigerProblem is a POMDP that
also contains the agent and the environment as its fields. In
general this doesn't need to be the case. (Refer to more complicated
examples.)

"""

import pomdp_py
import random
import numpy as np
import sys
import time
import copy
#from ....pomdp_py import *

class State(pomdp_py.State):
    def __init__(self, position_r,position_o,foundit):
        self.position_r = position_r
        self.position_o = position_o
        self.foundit = foundit
    def __str__(self):
        return str((self.position_r,self.position_o,self.foundit))
    def getPosRobot(self):
        return self.position_r
    def getPosObj(self):
        return self.position_o
    def found(self):
        return self.foundit


class Action(pomdp_py.Action):
    def __init__(self, name):
        self.name = name
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        return False
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name

class MoveAction(Action):
    EAST = (1, 0,0)  # x is horizontal; x+ is right. y is vertical; y+ is up.
    WEST = (-1, 0,0)
    NORTH = (0, -1,0)
    SOUTH = (0, 1,0)
    #LOWER = (0,0,1)
    #DONE = (0,0,0)
    def __init__(self, motion, name):
        if motion not in {MoveAction.EAST, MoveAction.WEST,
                          MoveAction.NORTH, MoveAction.SOUTH}:
            raise ValueError("Invalid move motion %s" % motion)
        self.motion = motion
        super().__init__("move-%s" % str(name))

MoveEast = MoveAction(MoveAction.EAST, "EAST")
MoveWest = MoveAction(MoveAction.WEST, "WEST")
MoveNorth = MoveAction(MoveAction.NORTH, "NORTH")
MoveSouth = MoveAction(MoveAction.SOUTH, "SOUTH")
#MoveLower = MoveAction(MoveAction.LOWER, "LOWER")
#MoveDone = MoveAction(MoveAction.DONE, "DONE")

class LookAction(Action):
    def __init__(self):
        super().__init__("look")

class DoneAction(Action):
    def __init__(self):
        super().__init__("done")

class Observation(pomdp_py.Observation):
    def __init__(self, detected):
        self.detected = detected
    def __hash__(self):
        return hash(self.detected)
    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.detected == other.detected
        return False
    def __str__(self):
        return self.detected
    def __repr__(self):
        return "Observation(%s)" % self.detected

# Observation model
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if isinstance(action, LookAction):
            if (next_state.position_r == next_state.position_o):
                return 1 - self.noise
            else:
                return self.noise


        else:
            if observation.quality is None:
                return 1.0 - EPSILON  # expected to receive no observation
            else:
                return EPSILON

    def sample(self, next_state, action):
        if isinstance(action,LookAction):
            if (next_state.position_r == next_state.position_o):
                thresh = 1 - self.noise
            else:
                thresh = self.noise
            if random.uniform(0, 1) < thresh:
                return Observation("positive")
            else:
                return Observation("negative")

        else:
            Observation("none")




# Transition Model
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        if next_state != self.sample(state, action):
            return EPSILON
        else:
            return 1.0 - EPSILON

    def sample(self, state, action):
        next_state_found = state.foundit
        next_state_o = state.position_o
        next_state_r = state.position_r
        if isinstance(action, MoveAction):
            next_state_r = (state.position_r[0] + action.motion[0],
                            state.position_r[1] + action.motion[1],
                            state.position_r[2] + action.motion[2])

        elif isinstance(action, LookAction):
            next_state_r = (state.position_r[0], state.position_r[1], 1)
        else:
            #next_state_r = state.position_r
            next_state_found = True;

        '''if (state.position_r == state.position_o):
            next_state_found = True;'''

        return State(next_state_r,next_state_o,next_state_found)







# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if isinstance(action, DoneAction):
            if (state.position_r == state.position_o):
                #print("state.position_r, "  object at  ",state.position_o")
                #print("done")
                return 100
            else:
                #print("wrongly done")
                return -100

        else:
            #print(" not done")
            return -1;



    def sample(self, state, action, next_state):

        if isinstance(action, DoneAction):
            if (next_state.position_r == next_state.position_o):
                #print("state.position_r, "  object at  ",state.position_o")
                #print("done")
                return 100
            else:
                #print("wrongly done")
                return -100
        elif isinstance(action, MoveAction):
            if (next_state.position_r == next_state.position_o):
                #print("state.position_r, "  object at  ",state.position_o")
                #print("done")
                return 100
            else:
                #print("wrongly done")
                return -1


        else:
            #print(" not done")
            if (next_state.position_r == next_state.position_o):
                #print("state.position_r, "  object at  ",state.position_o")
                #print("done")
                return 100
            else:
                #print("wrongly done")
                return -1
            #return -1;
        # deterministic
        #return self._reward_func(state, action)

# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.


    def __init__(self, n):
        #check_actions = set({CheckAction(rock_id) for rock_id in range(k)})
        self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        self._other_actions = {LookAction()} | {DoneAction()}
        self._all_actions = self._move_actions | self._other_actions
        self._n = n

    def sample(self, state, normalized=False, **kwargs):
        return random.sample(self.get_all_actions(state=state), 1)[0]

    def probability(self, action, state, normalized=False, **kwargs):
        raise NotImplementedError

    def argmax(self, state, normalized=False, **kwargs):
        """Returns the most likely reward"""
        raise NotImplementedError

    def get_all_actions(self, **kwargs):
        state = kwargs.get("state", None)
        if state is None:
            return self._all_actions
        else:
            motions = set(self._all_actions)
            rover_x, rover_y, rover_z = state.position_r
            if rover_x == 0:
                motions.remove(MoveWest)
            if rover_y == 0:
                motions.remove(MoveNorth)
            if rover_y == self._n - 1:
                motions.remove(MoveSouth)
            return motions | self._other_actions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state=state), 1)[0]


class FindItProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, n,obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        self.n = n
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(n),
                               TransitionModel(),
                               ObservationModel(obs_noise),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="FindItProblem")

    def terminal(self, pos):
        return pos == self._n

def init_particles_belief(n,num_particles, init_state, belief="uniform"):
    #num_particles = 500
    particles = []
    for _ in range(num_particles):
        if belief == "uniform":
            position_o = (random.randint(0,n),random.randint(0,n),1)
        elif belief == "groundtruth":
            position_o = copy.deepcopy(init_state.position_o)
        particles.append(State(init_state.position_r, position_o, False))
    init_belief = pomdp_py.Particles(particles)
    return init_belief


def test_planner(findit_problem, planner, nsteps=3,discount=0.95):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        tiger_problem (TigerProblem): an instance of the tiger problem.
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
    """
    '''for i in range(nsteps):
        action = planner.plan(findit_problem.agent)
        print("==== Step %d ====" % (i+1))
        print("True state: %s" % findit_problem.env.state)
       # print("Belief: %s" % str(findit_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print("Reward: %s" % str(findit_problem.env.reward_model.sample(findit_problem.env.state, action, None)))

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        #real_observation = Observation(findit_problem.env.state.name)
        real_observation = findit_problem.env.provide_observation(findit_problem.agent.observation_model,
                                                              action)
        print(">> Observation: %s" % real_observation)
        findit_problem.agent.update_history(action, real_observation)

        planner.update(findit_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)'''

    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0
    for i in range(nsteps):
        print("==== Step %d ====" % (i + 1))
        action = planner.plan(findit_problem.agent)

        print("Belief: %s" % str(findit_problem.agent.sample_belief()))
        # pomdp_py.visual.visualize_pouct_search_tree(rocksample.agent.tree,
        #                                             max_depth=5, anonymize=False)

        true_state = copy.deepcopy(findit_problem.env.state)
        env_reward = findit_problem.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(findit_problem.env.state)

        real_observation = findit_problem.env.provide_observation(findit_problem.agent.observation_model,
                                                              action)
        findit_problem.agent.update_history(action, real_observation)
        planner.update(findit_problem.agent, action, real_observation)
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount
        print("True state: %s" % true_state)
        print("Action: %s" % str(action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
        print("World:")
        #rocksample.print_state()
        if true_state.foundit:
           break


    return total_reward, total_discounted_reward


def main():


    n = 5

    init_state = State((2,0,2),(1,4,1), False)


    belief = "uniform"

    init_state_copy = copy.deepcopy(init_state)

    # init belief (uniform), represented in particles;
    # We don't factor the state here; We are also not doing any action prior.
    init_belief = init_particles_belief(n,500, init_state, belief=belief)

    findit = FindItProblem(n,0,init_state, init_belief)
    #rocksample.print_state()
    #print(findit.agent)

    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(max_depth=10, discount_factor=0.95,
                           num_sims=64000, exploration_const=20,
                           rollout_policy=findit.agent.policy_model,
                           num_visits_init=1)
    start_pomcp = time.time()
    tt, ttd = test_planner(findit, pomcp, nsteps=100, discount=0.95)
    end_pomcp = time.time()
    print("Time pomcp: ", end_pomcp - start_pomcp)
    findit.env.state.position_r = init_state_copy.position_r
    #rocksample.env.state.rocktypes = init_state_copy.rocktypes
    findit.env.state.foundit = False
    init_belief = init_particles_belief(n,200, findit.env.state, belief=belief)
    findit.agent.set_belief(init_belief)

if __name__ == '__main__':
    main()
