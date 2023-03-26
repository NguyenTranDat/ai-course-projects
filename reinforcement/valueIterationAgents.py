# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            values_copy = self.values.copy()
            states =  self.mdp.getStates()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                if len(actions) > 0:
                    values_copy[state] = self.getQValue(state, actions[0])
                    for action in actions:
                        values_copy[state] = max(values_copy[state], self.getQValue(state, action))
            self.values = values_copy

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0.0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, prob in transitions:
            value = self.mdp.getReward(state, action, next_state)
            value += self.discount * self.getValue(next_state)
            value *= prob
            q += value
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        q = float(-1000000)
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            tmp = self.getQValue(state, action)
            if tmp > q:
                q = tmp
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        length = len(states)
        for iter in range(self.iterations):
            state =states[iter% length]
            best_action = self.computeActionFromValues(state)
            if best_action:
                self.values[state] = self.computeQValueFromValues(state, best_action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
    def highestQValue(self, state):
        pos_actions = self.mdp.getPossibleActions(state)
        if len(pos_actions) == 0: return self.values[state]
        return max([self.computeQValueFromValues(state, action) for action in pos_actions])

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        states = self.mdp.getStates()

        for state in states:
            predecessors[state] = set()

        for state in states:
            pos_actions = self.mdp.getPossibleActions(state)
            for action in pos_actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for next_state, _ in transitions:
                    predecessors[next_state].add(state)
        
        heap = util.PriorityQueue()

        for state in states:
            max_q = self.highestQValue(state)
            heap.push(state, -abs(self.values[state] - max_q))
        
        for _ in range(self.iterations):
            if(heap.isEmpty()): return
            state = heap.pop()
            self.values[state] = self.highestQValue(state)
            for p in predecessors:
                diff = abs(self.values[p] - self.highestQValue(p))
                if diff > self.theta: heap.update(p, -diff)
        return