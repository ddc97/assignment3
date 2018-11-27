# ghostAgents.py
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist

class MinimaxGhost(GhostAgent):

    """
      Your minimax agent (question 1)

      useage: python2 pacman.py -p ExpectimaxAgent -l specialNew -g MinimaxGhost -a depth=4
              python2 pacman.py -l specialNew -g MinimaxGhost

    """

    def __init__(self, index, evalFn='betterEvaluationFunctionGhost', depth='2'):
        self.index = index  # Ghosts are always agent index > 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def getAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        choices = []
        value = []
        indexlist = []
        for action in actions:
            curr = self.findmax(gameState, 1, self.index, self.index, 1, 1)
            if action is not Directions.STOP:
                value.append(curr)
                choices.append(action)
        for index in range(len(value)):
            if (value[index] == min(value)):
                indexlist.append(index)
        randomindex = random.randint(0, len(indexlist) - 1)
        # print 'value',value
        # print 'choices',choices
        # print 'indexlist',indexlist
        # print 'last',choices[randomindex]
        return choices[randomindex]

    def findmax(self, gameState, depth, agent, curagent, len1, len2):
        a = []
        b = []
        best = []

        if depth >= self.depth and depth != 1:
            return self.evaluationFunction(gameState)
        maxvalue = float('-inf')
        actions = gameState.getLegalActions(curagent)
        for action in actions:
            successor = gameState.generateSuccessor(curagent, action)
            temp = self.findmin(successor, depth, agent, 0, len1, len2)
            maxvalue = max(maxvalue, temp)
            '''if depth==1 and action is not Directions.STOP:
                print '1,2',len1, len2
                print 'max',maxvalue
                a.append(maxvalue)
                print 'action',action
                b.append(action)
                print 'a,b',len(a),len(b)

                if (len(a)<len1) and (len(b) < len2):
                    print 'aaaaaaaaaaaaaaaaa',a ,b
                if (len(a) > len1) and (len(b) > len2):
                    len1 = len(a)

                    len2 = len(b)
                    print 'bbbbbbbb', a, b
                '''
        return maxvalue

    def findmin(self, gameState, depth, agent, curagent, len1, len2):
        if depth >= self.depth and depth != 1:
            return self.evaluationFunction(gameState)
        minvalue = float('inf')
        actions = gameState.getLegalActions(curagent)
        for action in actions:
            successor = gameState.generateSuccessor(curagent, action)
            for ghoststate in gameState.getGhostStates():
                #if ghost is scared,
                # it need to be far away from the pacman to prevent pacman from gaining score
                if ghoststate.scaredTimer > 0:
                    minvalue = min(minvalue, self.findmin(successor, depth + 1,agent,agent,len1,len2))
                else:
                    minvalue = min(minvalue, self.findmax(successor, depth + 1,agent,agent,len1,len2))
        # print 'min',minvalue
        return minvalue


def betterEvaluationFunctionGhost(currentGameState):
    """
        Ghost evaluation function
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    score = currentGameState.getScore()
    foodscore = 0
    foodPos = newFood.asList()
    fooddistance = [manhattanDistance(newPos, pos) for pos in foodPos]
    if len(fooddistance):
        closefood = min(fooddistance)
        foodscore = (-1)* closefood

    totalscore = score + foodscore

    return totalscore

# Abbreviation
ghostEval = betterEvaluationFunctionGhost

