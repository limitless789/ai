from cmath import inf
import collections
from queue import Empty
import util, math, random
from collections import defaultdict
from util import ValueIteration


############################################################
# Problem 1a: BlackjackMDP

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        
        result = []
        handnow, peekednow, decknow = state
        if decknow == None:
            return []
        elif action == 'Quit':
            if peekednow != None:
                handnow = handnow + self.cardValues[peekednow]
                newState = (handnow, None, None)
            else:
                newState= (handnow, None, None)
            if(handnow>self.threshold):
                reward = 0
            else:
                reward=handnow
            result.append((tuple(newState), 1, reward))

        elif action == 'Peek':
            if peekednow != None:
                return []
            else:
                for i, cardnum in enumerate(decknow):
                    if cardnum > 0:
                        newState=(handnow, i, decknow)
                        prob = float(cardnum) / sum(decknow)
                        reward=0-self.peekCost
                        result.append((tuple(newState), prob, reward))

        elif action == 'Take':
            if peekednow == None:
                for i, cardnum in enumerate(decknow):
                    if cardnum > 0:
                        newhand = handnow + self.cardValues[i]
                        prob = float(cardnum) / sum(decknow)
                        if(newhand>self.threshold):
                            newdeck = None
                        else:
                            newdeck=list(decknow)
                            newdeck[i]-=1
                        
                        reward=0
                        if newdeck!= None and sum(newdeck) == 0:
                            reward=newhand
                            newdeck=None
                        if newdeck!= None: 
                            newdeck=tuple(newdeck)
                        newState=(newhand, None, newdeck)    
                        result.append((tuple(newState), prob, reward))

            else:
                newhand = handnow + self.cardValues[peekednow]
                if(newhand>self.threshold):
                    newdeck = None
                else:
                    newdeck=list(decknow)
                    newdeck[peekednow]-=1
                reward=0
                if newdeck!= None and sum(newdeck) == 0:
                    reward=newhand
                    newdeck=None
                if newdeck!= None: 
                    newdeck=tuple(newdeck)
                newState=(newhand, None, newdeck)    
                result.append((tuple(newState), 1, reward))

        return result

        # END_YOUR_ANSWER

    def discount(self):
        return 1

############################################################
# Problem 1b: ValueIterationDP

class ValueIterationDP(ValueIteration):
    '''
    Solve the MDP using value iteration with dynamic programming.
    '''
    def solve(self, mdp):
        V = {}  # state -> value of state

        # BEGIN_YOUR_ANSWER (our solution is 13 lines of code, but don't worry if you deviate from this)
        f=collections.defaultdict(bool)
        actions= ['Take', 'Peek', 'Quit']
        for state in mdp.states:
            V[state]=0
            f[state]=True
        flag=True
        while flag:
            flag=False
            newV = {}
            for state in mdp.states:
                if f[state]:
                    newV[state]=max(sum(prob*(reward+V[newState]) for newState, prob, reward in mdp.succAndProbReward(state, action)) for action in actions)
                    if abs(V[state]-newV[state]) > 0.001:
                        flag=True
                    else:
                        if newV[state] > 0:
                            f[state]=False
                else:
                    newV[state]=V[state]
            V = newV

        
        # END_YOUR_ANSWER


        # Compute the optimal policy now
        pi = self.computeOptimalPolicy(mdp, V)
        self.pi = pi
        self.V = V

############################################################
# Problem 2a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        
        if newState == None:
            return
        newaction = max((self.getQ(state, action2), action2) for action2 in self.actions(state))[1]
        gap = float(reward) + self.discount * self.getQ(newState, newaction) - self.getQ(state, action)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] += self.getStepSize() * v * gap

        
        # END_YOUR_ANSWER


############################################################
# Problem 2b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        
        if newState == None:
            return
        gap = float(reward) + self.discount * self.getQ(newState, newAction) - self.getQ(state, action)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] += self.getStepSize() * v * gap

        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 2c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    
    feature=[]
    key=(total, action)
    feature.append((key,1))
    if counts != None and sum(counts)!=0:
        newcount=list(counts)
        for i, value in enumerate(newcount):
            if value > 0:
                newcount[i]=1
        key=(tuple(newcount), action)
        feature.append((key,1))
        for i, value in enumerate(counts):
            feature.append(((value, i, action), 1))
    return feature
    # END_YOUR_ANSWER
