
import pomdp_py
import random
import numpy as np
import sys
import time
import copy
#from ....pomdp_py import *
visited = np.zeros((5, 5), dtype=bool)
class State(pomdp_py.State):
    def __init__(self, position_r,position_o,foundit,orientation_r,orientation_o,oriented):
        self.position_r = position_r
        self.position_o = position_o
        self.foundit = foundit
        self.orientation_r = orientation_r
        self.orientation_o = orientation_o
        self.oriented = oriented

    def __str__(self):
        return str((self.position_r,self.position_o,self.foundit,self.orientation_r,self.orientation_o,self.oriented))
    def getPosRobot(self):
        return self.position_r
    def getPosObj(self):
        return self.position_o
    def found(self):
        return self.foundit

    '''def __eq__(self, other):
        return (self.position_r == other.position_r and
        self.foundit == other.foundit and
        self.orientation_r == other.orientation_r and
        self.oriented == other.oriented)'''



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
class DoneOrientationAction(Action):
    def __init__(self):
        super().__init__("done_orientation")
class OrientAction(Action):
    def __init__(self, angle):
        self.angle = angle
        super().__init__("orient-%s" % str(angle))
class FoundAction(Action):
    def __init__(self):
        super().__init__("found")

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
        elif isinstance(action, DoneOrientationAction):
            if (abs(next_state.position_r - next_state.position_o)<=5):
                return 1 - self.noise
            else:
                return self.noise
        elif isinstance(action, FoundAction):
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
        elif isinstance(action,DoneOrientationAction):
            if (abs(next_state.orientation_o - next_state.orientation_r)<=5):
                thresh = 1 - self.noise
            else:
                thresh = self.noise
            if random.uniform(0, 1) < thresh:
                return Observation("positive")
            else:
                return Observation("negative")
        elif isinstance(action,FoundAction):
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
        if next_state != self.sample(state, action) :
            return EPSILON
        else:
            return 1.0 - EPSILON

    def sample(self, state, action):
        next_state_found = state.foundit
        next_state_o = state.position_o
        next_state_r = state.position_r
        next_state_o_ang = state.orientation_o
        next_state_r_ang = state.orientation_r
        next_state_oriented = state.oriented
        if isinstance(action, MoveAction):
            next_state_r = (state.position_r[0] + action.motion[0],
                            state.position_r[1] + action.motion[1],
                            state.position_r[2] + action.motion[2])

        elif isinstance(action, LookAction):
            next_state_r = (state.position_r[0], state.position_r[1], 1)
            visited[next_state_r[0], next_state_r[1]] = True
        elif isinstance(action,OrientAction):
            next_state_r_ang = action.angle
            '''if abs(next_state_r_ang - next_state_o_ang)<=5:
                next_state_oriented = True'''
        elif isinstance(action, FoundAction):
            next_state_found = True
        elif isinstance(action, DoneOrientationAction):
            if abs(next_state_r_ang - next_state_o_ang)<=5:
                next_state_oriented = True
            #next_state_oriented = True
        else:
            #next_state_r = state.position_r
            next_state_found = True
            next_state_oriented = True

        '''if (state.position_r == state.position_o):
            next_state_found = True;'''

        return State(next_state_r,next_state_o,next_state_found,next_state_r_ang,next_state_o_ang,next_state_oriented)







# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action,next_state):
        #print(state.position_r[0],state.position_r[1])

        if isinstance(action, DoneAction):
            if (state.position_r == state.position_o and ((state.orientation_o==state.orientation_r) or (state.orientation_o==180 - state.orientation_r))):
                #print("state.position_r, "  object at  ",state.position_o")
                #print("done")
                return 100
            else:
                #print("wrongly done")
                return -100
        elif isinstance(action, FoundAction):
            if (state.position_r == state.position_o):
                return 50
            else:
                return -50


        else:

            #print(" not done")

            return -1
            '''if (state.position_r!=next_state.position_r):
                return -1
            else:
                return -10;'''



    def sample(self, state, action, next_state):

        '''if isinstance(action, DoneAction):
            if (state.position_r == state.position_o):
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
        # deterministic'''
        return self._reward_func(state, action,next_state)

# Policy Model
class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""
    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.


    def __init__(self, n):
        #check_actions = set({CheckAction(rock_id) for rock_id in range(k)})
        self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        self._orient_actions = {OrientAction(random.randint(0,90))} #{OrientAction(15), OrientAction(30), OrientAction(45), OrientAction(60), OrientAction(75),OrientAction(90)} OrientAction(np.random(0,90))
        self._other_actions = {LookAction()} | {DoneAction()} | self._orient_actions | {FoundAction()} | {DoneOrientationAction()}
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
            #motions = set(self._all_actions)
            motions = set(self._move_actions)
            rover_x, rover_y, rover_z = state.position_r
            foundit = state.foundit
            oriented = state.oriented
            if oriented:
                return {DoneAction()} #| {DoneOrientationAction()}
            elif foundit:
                #return  set(self._orient_actions) | {DoneOrientationAction()}
                return {OrientAction(random.randint(0,90))} | {DoneOrientationAction()}
            else:
                if rover_x == 0:
                    motions.remove(MoveWest)
                if rover_y == 0:
                    motions.remove(MoveNorth)

                if rover_y == self._n - 1:
                    motions.remove(MoveSouth)
                if rover_x == self._n - 1:
                    motions.remove(MoveEast)
                #print(motions |  {LookAction()} | {DoneAction()})
                '''if visited[rover_x,rover_y]:
                    return motions | {DoneAction()} | {FoundAction()}
                else:
                    return motions | {LookAction()} | {DoneAction()} | {FoundAction()}'''

                return motions |  {LookAction()} | {FoundAction()}

    def rollout(self, state, history=None):
        ra = random.sample(self.get_all_actions(state=state), 1)[0]
        #print("RRRRRRRRRRRRRAAAAAAAAAAAAAAAAAAA",ra)
        return ra
        #return random.sample(self.get_all_actions(state=state), 1)[0]


class FindAndOrientItProblem(pomdp_py.POMDP):
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
        super().__init__(agent, env, name="FindAndOrientItProblem")

    def terminal(self, pos):
        return pos == self._n

def init_particles_belief(n,num_particles, init_state, belief="uniform"):
    #num_particles = 500
    particles = []
    for _ in range(num_particles):
        if belief == "uniform":
            position_o = (random.randint(0,n),random.randint(0,n),1)
            orientation_o = random.randint(0,90)
        elif belief == "groundtruth":
            position_o = copy.deepcopy(init_state.position_o)
        particles.append(State(init_state.position_r, position_o, False,init_state.orientation_r,orientation_o,False))
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
        if true_state.foundit and true_state.oriented:
           break


    return total_reward, total_discounted_reward


def main():


    n = 5

    init_state = State((2,0,2),(1,4,1), False,90,30,False)
    visited[2,0] = True

    belief = "uniform"

    init_state_copy = copy.deepcopy(init_state)

    # init belief (uniform), represented in particles;
    # We don't factor the state here; We are also not doing any action prior.
    init_belief = init_particles_belief(n,500, init_state, belief=belief)

    findandorientit = FindAndOrientItProblem(n,0,init_state, init_belief)
    #rocksample.print_state()
    #print(findit.agent)

    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(max_depth=10, discount_factor=0.95,
                           num_sims=16000, exploration_const=20,
                           rollout_policy=findandorientit.agent.policy_model,
                           num_visits_init=1)
    start_pomcp = time.time()
    tt, ttd = test_planner(findandorientit, pomcp, nsteps=100, discount=0.95)
    end_pomcp = time.time()
    print("Time pomcp: ", end_pomcp - start_pomcp)
    findandorientit.env.state.position_r = init_state_copy.position_r
    #rocksample.env.state.rocktypes = init_state_copy.rocktypes
    findandorientit.env.state.foundit = False
    init_belief = init_particles_belief(n,500, findandorientit.env.state, belief=belief)
    findandorientit.agent.set_belief(init_belief)

if __name__ == '__main__':
    main()
