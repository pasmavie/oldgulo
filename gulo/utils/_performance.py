import numpy as np
def performance(forecasts: np.array, returns: np.array) -> np.array:
    return (f*r for f, r in zip(forecasts, returns))



# def performance(predicted_movement: np.array, real_movement: np.array) -> np.array:
#    """
#    The two series must have the same lenght
#    and they have been sampled the same way (e.g. daily returns, volume based etc etc)
#    """
#    simple_positioning = lambda x, y: x / x * y if x != 0 else 0
#    simple_positioning_vect = np.vectorize(simple_positioning)
#    return simple_positioning_vect(predicted_movement, real_movement)
