# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearchHelper(problem, location, explored, currentPath):
    if problem.isGoalState(location):
        return currentPath
    explored.append(location)
    for nextElement in problem.getSuccessors(location):
        if nextElement[0] not in explored:
            result = depthFirstSearchHelper(problem, nextElement[0], explored, currentPath + [nextElement[1]])
            if len(result) > 0:
                return result

    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    #
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    #
    #
    "*** YOUR CODE HERE ***"

    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    explored=[]
    currentPath=[]
    return depthFirstSearchHelper(problem, startingNode, explored, currentPath)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    "*** YOUR CODE HERE ***"

    queue = util.Queue()
    explored = {}
    parent={}
    start = problem.getStartState()
    queue.push(start)
    explored[start] = True
    parent[start[0]] = None
    while not queue.isEmpty():
        location = queue.pop()
        if problem.isGoalState(location):
            temp = [location,'']
            path=[]
            while temp != None:                    
                temp = parent.get(temp[0])
                if temp != None:
                    path.append(temp[1])
            path.reverse()
            return path
        for nextNode, action, cost in list(problem.getSuccessors(location)):
            if nextNode not in explored:
                explored[nextNode] = True
                parent[nextNode] = [location, action]
                queue.push(nextNode)
    return []

def uniformCostSearch(problem):
    '''*** YOUR CODE HERE ***'''

    pq = util.PriorityQueue()
    pq.push(problem.getStartState(), 0)
    explored = {}
    cost={}
    parent={}
    parent[problem.getStartState()] = None
    cost[problem.getStartState()] = 0
    while pq.isEmpty() == False:
        node = pq.pop()
        if problem.isGoalState(node):
            path=[]
            while node != None:
                temp = parent[node]
                if temp != None:
                    path.append(temp[1])
                    node = temp[0]
                else:
                    node = None
            path.reverse()
            return path
        for i in problem.getSuccessors(node):
            if i[0] not in cost or (i[0] in cost and cost[i[0]] > cost[node] + i[2]):
                parent[i[0]] = (node, i[1])
                cost[i[0]] = cost[node] + i[2]
                pq.push(i[0], cost[i[0]])

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    pq = util.PriorityQueue()
    pq.push((startingNode, [], 0), 0)
    explored = {}
    while pq.isEmpty() == False:
        top = pq.pop()
        if top[0] not in explored:
            explored[top[0]] = True
            if problem.isGoalState(top[0]):
                return top[1]
            for next in problem.getSuccessors(top[0]):
                newAction = top[1] + [next[1]]
                cost = top[2] + next[2]
                hCost = cost + heuristic(next[0],problem)
                pq.push((next[0], newAction, cost),hCost)

def randomyu(problem, heuristic=nullHeuristic):
    pQueue = util.PriorityQueue()
    start = problem.getStartState()
    pQueue.push((start, []), heuristic(start, problem))
    explored = []
    while True:
        if pQueue.isEmpty():
            return False
        state, path = pQueue.pop()
        if problem.isGoalState(state):
            return path

        if state not in explored:
            explored.append(state)
        for successor, direction, cost in problem.getSuccessors(state):
            if successor not in explored:
                neighborCost = path + [direction]
                pathCost = problem.getCostOfActions(neighborCost) + heuristic(successor, problem)
                pQueue.push((successor, neighborCost), pathCost)
    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
