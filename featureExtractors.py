# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        def hasWhiteGhost(self, GhostStates):
            num = 0
            for i in range(len(GhostStates)):
                if GhostStates[i].scaredTimer > 0:
                    num += 1
            return num

        def distanceWhiteGhost(self, GhostStates, next_position, walls):
            ghost_distance = 10
            for ghost in GhostStates:
                if ghost.scaredTimer > 0:
                    g_position = ghost.getPosition()
                    # print(g_position)
                    dis_x = abs(next_position[0] - g_position[0])
                    dis_y = abs(next_position[1] - g_position[1])
                    all_dis = dis_x + dis_y
                    ghost_distance = min(all_dis, ghost_distance)
            return ghost_distance

        def distanceCapsule(self, Capsules, next_position, walls):
            capsule_distance = 10
            for capsule in Capsules:
                c_position = capsule.getPosition()
                # print(g_position)
                dis_x = abs(next_position[0] - c_position[0])
                dis_y = abs(next_position[1] - c_position[1])
                all_dis = dis_x + dis_y
                capsule_distance = min(all_dis, capsule_distance)
            return capsule_distance
                    
        def isSafe(self, GhostStates, next_position, walls):
            SafeState = True
            for ghost in GhostStates:
                if ghost.scaredTimer == 0:
                    g_position = ghost.getPosition()
                    dis_x = abs(next_position[0] - g_position[0])
                    dis_y = abs(next_position[1] - g_position[1])
                    all_dis = dis_x + dis_y
                    if all_dis < 5:
                        SafeState = False
            return SafeState
        

        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        GhostStates = state.getGhostStates()
        capsules = state.getCapsules()
        # print(state.getCapsules())
        features = util.Counter()

        features["bias"] = 1.0
        
        
        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)


        if hasWhiteGhost(self, GhostStates) > 0 and distanceWhiteGhost(self, GhostStates, (next_x, next_y), walls) <= 2 and isSafe(self, GhostStates, (next_x, next_y), walls):
            features["eats-ghost"] = 15.0
            features["#-of-ghosts-1-step-away"] = 0.0
            features["eats-food"] = 0.0


        #eat capsule
        whiteGhostNum  = hasWhiteGhost(self, GhostStates)
        w1 = 0.0
        if(whiteGhostNum > 0 ):
            w1 += 0.5
            w1 += len(capsules) * 1.5
            if(len(capsules) == 1):
                w1 += 5.0
            for _ in range(whiteGhostNum):
                w1 += 0.5
            w1 += (10-distanceCapsule(self, GhostStates, (next_x, next_y), walls)) * 0.1
        features['eats-capsule'] = w1

        features.divideAll(10.0)
        # print(features)
        return features

