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

    stack = util.Stack()  # Create a stack for DFS.

    visited = []  # Create a list to keep track of explored/visited nodes.

    # Beginning node of the graph.
    start = (startingNode, [])
    stack.push(start)

    """ 
    visit current node if is not explored before and find its
    children (push those into the stack) 
    """

    while not stack.isEmpty():
        element = stack.pop()
        location = element[0]
        path = element[1]

        if location not in visited:
            visited.append(location)
            if problem.isGoalState(location):
                return path

            for nextElement in problem.getSuccessors(location):
                stack.push((nextElement[0], path + [nextElement[1]]))
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))

    "*** YOUR CODE HERE ***"

    queue = util.Queue()
    visited = []
    start = (problem.getStartState(), [])
    queue.push(start)

    while not queue.isEmpty():
        location, path = queue.pop()

        if location not in visited:
            visited.append(location)
            if problem.isGoalState(location):
                return path
            for nextNode, action, cost in list(problem.getSuccessors(location)):
                if nextNode not in visited:
                    queue.push((nextNode, path + [action]))
    return []

def uniformCostSearch(problem):

    '''*** YOUR CODE HERE ***'''

    pQueue = util.PriorityQueue()
    pQueue.push((problem.getStartState(), [], 0), 0)
    closed = {}
    goal = False

    while not goal:
        if pQueue.isEmpty():
            return False
        node = pQueue.pop()
        closed[node[0]] = node[2]
        if problem.isGoalState(node[0]):
            return node[1]
        for i in problem.getSuccessors(node[0]):
            if i[0] not in closed or (i[0] in closed and closed[i[0]] > node[2] + i[2]):
                temp = list(node[1])
                temp.append(i[1])
                cost = node[2] + i[2]
                closed[i[0]] = cost
                pQueue.push((i[0], temp, cost), cost)

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
    visitedNodes = []
    pQueue = util.PriorityQueue()
    # ((coordinate/node , action to current node , cost to current node),priority)
    pQueue.push((startingNode, [], 0), 0)

    while not pQueue.isEmpty():
        currentNode, actions, prevCost = pQueue.pop()
        if currentNode not in visitedNodes:
            visitedNodes.append(currentNode)
            if problem.isGoalState(currentNode):
                return actions
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                newAction = actions + [action]
                newCostToNode = prevCost + cost
                heuristicCost = newCostToNode + heuristic(nextNode,problem)
                pQueue.push((nextNode, newAction, newCostToNode),heuristicCost)


def randomyu(problem, heuristic=nullHeuristic):
    pQueue = util.PriorityQueue()
    start = problem.getStartState()
    pQueue.push((start, []), heuristic(start, problem))
    visited = []
    while True:
        if pQueue.isEmpty():
            return False
        state, path = pQueue.pop()
        if problem.isGoalState(state):
            return path

        if state not in visited:
            visited.append(state)
        for successor, direction, cost in problem.getSuccessors(state):
            if successor not in visited:
                neighborCost = path + [direction]
                pathCost = problem.getCostOfActions(neighborCost) + heuristic(successor, problem)
                pQueue.push((successor, neighborCost), pathCost)
    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
