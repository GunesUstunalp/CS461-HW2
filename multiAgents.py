# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodCount = successorGameState.getNumFood()
        newCapsules = successorGameState.getCapsules()

        distanceToGhosts = []

        for i in range(len(newGhostStates)):
            distanceToGhosts.append(manhattanDistance(newPos, newGhostStates[i].getPosition()))

        distanceToClosestGhost = min(distanceToGhosts)

        distanceToFoods = []
        distanceToClosestFood = 0

        for i in range(len(newFood.asList())):
            distanceToFoods.append(manhattanDistance(newPos, newFood.asList()[i]))

        if(len(distanceToFoods) > 0):
            distanceToClosestFood = min(distanceToFoods)

        # print("Closest Ghost: " + str(distanceToClosestGhost))
        # print("New Pos: ")
        # print(newPos)
        # print("New Food: ")
        # print(newFood.asList())
        # print("New Capsules: ")
        # print(newCapsules)
        # print("New GhostStates: ")
        # print(newGhostStates[0])
        # print("New ScaredTimes: ")
        # print(newScaredTimes)

        smallestScareTime = min(newScaredTimes)
        if smallestScareTime != 0:
            distanceToClosestGhost = -distanceToClosestGhost * 10

        distanceToCapsules = []
        distanceToClosestCapsule = 0

        for i in range(len(newCapsules)):
            distanceToCapsules.append(manhattanDistance(newPos, newCapsules[i]))

        if (len(distanceToCapsules) > 0):
            distanceToClosestCapsule = min(distanceToCapsules)

        capsuleCount = len(newCapsules)

        return successorGameState.getScore() - distanceToClosestFood - distanceToClosestCapsule + distanceToClosestGhost - 10 * foodCount - 1000 * capsuleCount

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, gameState: GameState, currDepth, agentIndex):
        agentIndex = agentIndex % gameState.getNumAgents()

        #print("Depth: " + str(currDepth) + " AgentIndex: " + str(agentIndex))
        actionsForThisAgent = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            currDepth += 1

        if gameState.isLose() or gameState.isWin() or currDepth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            firstSuccessor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[0])
            maxValue = self.minimax(firstSuccessor, currDepth, agentIndex + 1)

            for i in range(len(actionsForThisAgent) - 1):
                successor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[i + 1])
                if self.minimax(successor, currDepth, agentIndex + 1) > maxValue:
                    maxValue = self.minimax(successor, currDepth,  agentIndex + 1)

            return maxValue
        else:
            firstSuccessor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[0])
            minValue = self.minimax(firstSuccessor, currDepth, agentIndex + 1)

            for i in range(len(actionsForThisAgent) - 1):
                successor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[i + 1])
                if self.minimax(successor, currDepth, agentIndex + 1) < minValue:
                    minValue = self.minimax(successor, currDepth, agentIndex + 1)

            return minValue


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actionsForThisAgent = gameState.getLegalActions(0)
        currDepth = 0

        firstSuccessor = gameState.generateSuccessor(0, actionsForThisAgent[0])
        bestMinimaxValue = self.minimax(firstSuccessor, currDepth, 1)
        bestAction = actionsForThisAgent[0]

        for i in range(len(actionsForThisAgent) - 1):
            successor = gameState.generateSuccessor(0, actionsForThisAgent[i+1])
            if self.minimax(successor, currDepth, 1) > bestMinimaxValue:
                bestMinimaxValue = self.minimax(successor, currDepth, 1)
                bestAction = actionsForThisAgent[i+1]

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, gameState: GameState, currDepth, agentIndex, alpha = -1*float('inf'), beta = float('inf')):
        agentIndex = agentIndex % gameState.getNumAgents()

        # print("Depth: " + str(currDepth) + " AgentIndex: " + str(agentIndex))
        actionsForThisAgent = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            currDepth += 1

        if gameState.isLose() or gameState.isWin() or currDepth == self.depth:
            return self.evaluationFunction(gameState)

        if beta <= alpha: #?
            return -1 * float('inf')

        if agentIndex == 0:
            maxEva = -1*float('inf') #Babo
            firstSuccessor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[0])
            maxValue = self.alphaBeta(firstSuccessor, currDepth, agentIndex + 1, alpha, beta)
            if maxEva < maxValue:
                maxEva = maxValue

            for i in range(len(actionsForThisAgent) - 1):
                successor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[i + 1])
                tested = self.alphaBeta(successor, currDepth, agentIndex + 1, alpha, beta)
                if  tested > maxValue:
                    maxValue = tested
                    if maxEva < maxValue:
                        maxEva = maxValue

            alpha = max(alpha, maxEva)
            return maxValue
        else:
            minEva = float('inf')
            firstSuccessor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[0])
            minValue = self.alphaBeta(firstSuccessor, currDepth, agentIndex + 1, alpha, beta)
            if minEva > minValue:
                minEva = minValue

            for i in range(len(actionsForThisAgent) - 1):
                successor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[i + 1])
                tested = self.alphaBeta(successor, currDepth, agentIndex + 1, alpha, beta)
                if  tested < minValue:
                    minValue = tested
                    if minEva > minValue:
                        minEva = minValue

            beta = min(minEva, beta)
            return minValue

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actionsForThisAgent = gameState.getLegalActions(0)
        currDepth = 0

        firstSuccessor = gameState.generateSuccessor(0, actionsForThisAgent[0])
        bestMinimaxValue = self.alphaBeta(firstSuccessor, currDepth, 1)
        bestAction = actionsForThisAgent[0]

        for i in range(len(actionsForThisAgent) - 1):
            successor = gameState.generateSuccessor(0, actionsForThisAgent[i + 1])
            if self.alphaBeta(successor, currDepth, 1) > bestMinimaxValue:
                bestMinimaxValue = self.alphaBeta(successor, currDepth, 1)
                bestAction = actionsForThisAgent[i + 1]

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState: GameState, currDepth, agentIndex):
        agentIndex = agentIndex % gameState.getNumAgents()

        print("Depth: " + str(currDepth) + " AgentIndex: " + str(agentIndex))
        actionsForThisAgent = gameState.getLegalActions(agentIndex)

        if agentIndex == 0:
            currDepth += 1

        if gameState.isLose() or gameState.isWin() or currDepth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            actionsForThisAgent = gameState.getLegalActions(agentIndex)
            firstSuccessor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[0])
            maxValue = self.expectimax(firstSuccessor, currDepth, agentIndex + 1)

            for i in range(len(actionsForThisAgent) - 1):
                successor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[i + 1])

                if self.expectimax(successor, currDepth, agentIndex + 1) > maxValue:
                    maxValue = self.expectimax(successor, currDepth, agentIndex + 1)

            return maxValue
        else:
            actionsForThisAgent = gameState.getLegalActions(agentIndex)
            firstSuccessor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[0])
            minValue = self.expectimax(firstSuccessor, currDepth, agentIndex + 1)

            for i in range(len(actionsForThisAgent) - 1):
                successor = gameState.generateSuccessor(agentIndex, actionsForThisAgent[i + 1])

                minValue += self.expectimax(successor, currDepth, agentIndex + 1)

            minValue = minValue / len(actionsForThisAgent)

            return minValue



    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actionsForThisAgent = gameState.getLegalActions(0)
        currDepth = 0

        firstSuccessor = gameState.generateSuccessor(0, actionsForThisAgent[0])
        bestexpectimaxValue = self.expectimax(firstSuccessor, currDepth, 1)
        bestAction = actionsForThisAgent[0]

        for i in range(len(actionsForThisAgent) - 1):
            successor = gameState.generateSuccessor(0, actionsForThisAgent[i + 1])
            if self.expectimax(successor, currDepth, 1) > bestexpectimaxValue:
                bestexpectimaxValue = self.expectimax(successor, currDepth, 1)
                bestAction = actionsForThisAgent[i + 1]

        return bestAction


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
