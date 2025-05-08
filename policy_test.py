from expectedValue import expectedValue, random_policy, example_policy
from utils import CARDS
import numpy as np
from policy import PolicyNet, model_wrapper
import torch

def testGame(model, model2=None, dataset=None):
    policy = model # Don't automatically wrap anymore
    policy2 = policy if model2 is None else model2 #model_wrapper(model2)
    cards = list(np.random.choice(CARDS, 9, False)) # Generate random cards
    hand1 = cards[:2]
    hand2 = cards[2:4]
    community = cards[4:]
    stats = expectedValue(policy, policy2, hand1, hand2, community, fullStats=True, dataset=dataset)
    ev, f, c, r = stats
    # print(ev, f, c, r)#
    return stats

def testN(model, baseline, n):
    res = np.zeros(4)
    for i in range(n):
        res = res + np.array(testGame(model, baseline))
    return res/n

def comp(model, baseline, n, name):
    res = testN(model, baseline, n)
    print(f"As first player against {name}:")
    print(res)
    res = testN(baseline, model, n)
    print(f"As second player against {name}:")
    print(-res)
    print()

def test_suite(model, n):
    # Baseline
    baseline = PolicyNet()
    baseline.load_state_dict(torch.load("models/evaluation_function[latest-test].pth"))

    raise_policy = lambda a, b, c, canRaise: (0, 0, 1) if canRaise else (0, 1, 0)
    call_policy = lambda a, b, c, d: (0, 1, 0)
    fold_policy = lambda a, b, c, d: (1, 0, 0)

    comp(model, random_policy, n, "random")
    comp(model, example_policy, n, "uniform")
    comp(model, model_wrapper(baseline), n, "baseline")
    comp(model, raise_policy, n, "raise")
    comp(model, call_policy, n, "call")
    comp(model, fold_policy, n, "fold")


if __name__ == '__main__':

    model2 = PolicyNet()
    model2.load_state_dict(torch.load("models/policy[v3].pth"))

    test_suite(model_wrapper(model2), 10)
    # dataset = None

    # N = 10

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(model_wrapper(model2), random_policy, dataset=dataset))
    # print("As first player against random: ", res/N)
    # # -22.079 --> -12.639

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(random_policy, model_wrapper(model2), dataset=dataset))
    # print("As second player against random: ", res/N)
    # #27.123 --> 22.926

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(model_wrapper(model2), example_policy, dataset=dataset))
    # print("As first player against uniform: ", res/N)
    # #9.551 --> 3.485

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(example_policy, model_wrapper(model2), dataset=dataset))
    # print("As second player against uniform: ", res/N)
    # #-10.659 --> -5.055

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(model_wrapper(model2), model_wrapper(model), dataset=dataset))
    # print("As first player against baseline: ", res/N)
    # #-15.127

    # res = np.zeros(4)
    # for i in range(N):
    #     res = res + np.array(testGame(model_wrapper(model), model_wrapper(model2), dataset=dataset))
    # print("As second player against baseline: ", res/N)
    # #1.434


    