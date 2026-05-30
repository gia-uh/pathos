import pathos.algorithms  # ensure all algorithms are registered
from pathos.spaces.tour import TourSpace

def test_tourspace_tsp_sa():
    # 4-city TSP
    distances = {
        (0, 1): 10, (1, 0): 10,
        (0, 2): 15, (2, 0): 15,
        (0, 3): 20, (3, 0): 20,
        (1, 2): 35, (2, 1): 35,
        (1, 3): 25, (3, 1): 25,
        (2, 3): 30, (3, 2): 30,
    }

    space = TourSpace(nodes=list(range(4)), distances=distances)

    @space.evaluate
    def tour_cost(tour):
        return sum(distances[(tour[i], tour[(i + 1) % len(tour)])] for i in range(len(tour)))

    result = space.solver().solve()
    assert result.found
    assert len(result.solution) == 4
    assert set(result.solution) == {0, 1, 2, 3}
