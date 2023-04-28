"""Module services."""
def getMoments(M):
    x = int(M['m10'] / M['m00'])
    y = int(M['m01'] / M['m00'])
    return x,y

