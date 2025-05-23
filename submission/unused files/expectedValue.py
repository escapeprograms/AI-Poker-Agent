from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.engine.card import Card
import numpy as np
from utils import CARDS_REVEALED, RAISE_AMT

# Convert a list of cards in string form to a list of cards in Card object form
def cardify(cards: list[str]) -> list[Card]:
  return list(map(lambda card: Card.from_str(card), cards))

# Example policy
# Note that community may be an empty list if it is the preflop phase
# Should return a probability distribution of the form (foldProb, callProb, raiseProb)
def example_policy(hand: list[str], community: list[str], pot: int, canRaise: bool) -> tuple[float, float, float]:
  return (1/3, 1/3, 1/3) # foldProb, callProb, raiseProb

# Policy of random player
def random_policy(hand, community, pot, canRaise):
  r = 0.45 if canRaise else 0
  return (0.5-r, 0.5, r)

# Sample usage
# expectedValue(policy, ['DA', 'DK'], ['C2', 'C7'], ['DJ', 'DT', 'D9', 'D2', 'C8'], 120, 0, {"callEnds": True, "p1Raises": 4, "p2Raises": 4, "phase": "river", "firstPlayer": 1, "p1Paid": 60, "phaseRaises": 4})
# expectedValue(policy, ['DA', 'DK'], ['C2', 'C7'], ['DJ', 'DT', 'D9', 'D2', 'C8'])

# Expected value given the two players' policies and the game state
# If pot, callAmt, gameStateInfo are omitted, it assumes it's the beginning of the game (and p1 goes first)
# Returns (expected_value, fold_expected_value, c_expected_value, r_expected_value)
def expectedValue(policy1, policy2, hand1: list[str], hand2: list[str], community: list[str], pot=30, callAmt=10, gameStateInfo: dict = None, fullStats=False, dataset=None) -> tuple[float, float, float, float]: 
  if gameStateInfo is None:
    gameStateInfo = {
      "callEnds": False, # Whether calling ends the phase
      "p1Paid": 10, # Money paid by p1 so far (p2Paid is pot - p1Paid)
      "p1Raises": 0, # number of times p1 has raised
      "p2Raises": 0, # number of times p2 has raised
      "phaseRaises": 1, # number of raises this phase. Preflop should start at 1 due to big blind.
      "phase": "preflop", # "preflop" or "flop" or "turn" or "river"
      "firstPlayer": 1 # Whether p1 or p2 is the first player (goes first each phase)
    }
  
  phase = gameStateInfo["phase"]
  
  canRaise = not (gameStateInfo["p1Raises"] >= 4 or gameStateInfo["phaseRaises"] >= 4)
  f, c, r = policy1(hand1, community[:CARDS_REVEALED[phase]], pot, canRaise) # Only give the revealed community cards

  # Limit raise amount to 4
  if not canRaise:
    r = 0
    f, c = (f/(f+c), c/(f+c)) if f+c > 0 else (0.5, 0.5)
  # print(f"p1Raises: {gameStateInfo['p1Raises']} | {f} {c} {r}")
  # Amount that can be won: round_state["pot"]["main"]["amount"]
  

  fEV = -gameStateInfo["p1Paid"] # EV from folding
  cEV = 0 # Arbitrary value for initialization
  rEV = -9999 # Don't pick this if canRaise is False

  # Calculate raise expected value from future actions
  if r > 0:
    newGameState = gameStateInfo.copy()
    newGameState["phaseRaises"] += 1
    newGameState["p1Raises"], newGameState["p2Raises"] = gameStateInfo["p2Raises"], gameStateInfo["p1Raises"] + 1
    newGameState["callEnds"] = True
    newGameState["firstPlayer"] = 3 - newGameState["firstPlayer"] # Switch
    newGameState["p1Paid"] = pot - newGameState["p1Paid"] # Amt that p2 has paid
    rEV = -expectedValue(policy2, policy1, hand2, hand1, community, pot+(callAmt + RAISE_AMT[phase]), RAISE_AMT[phase], newGameState, dataset=dataset)


  # Calculate call expected value from future actions
  if not gameStateInfo["callEnds"]:
    gameStateInfo["callEnds"] = True
    gameStateInfo["p1Raises"], gameStateInfo["p2Raises"] = gameStateInfo["p2Raises"], gameStateInfo["p1Raises"]
    gameStateInfo["firstPlayer"] = 3 - gameStateInfo["firstPlayer"] # Switch
    gameStateInfo["p1Paid"] = pot - gameStateInfo["p1Paid"] # Amt that p2 has paid
    cEV = -expectedValue(policy2, policy1, hand2, hand1, community, pot+callAmt, 0, gameStateInfo, dataset=dataset)
  else:
    # Move on to the next phase
    if phase == "preflop":
      gameStateInfo["phase"] = "flop"
    elif phase == "flop":
      gameStateInfo["phase"] = "turn"
    elif phase == "turn":
      gameStateInfo["phase"] = "river"

    if phase != "river":
      # We are proceeding into the next phase
      gameStateInfo["callEnds"] = False
      gameStateInfo["phaseRaises"] = 0
      if gameStateInfo["firstPlayer"] == 1:
        gameStateInfo["p1Paid"] += callAmt
        cEV = expectedValue(policy1, policy2, hand1, hand2, community, pot+callAmt, 0, gameStateInfo)
      else:
        gameStateInfo["firstPlayer"] = 1
        gameStateInfo["p1Raises"], gameStateInfo["p2Raises"] = gameStateInfo["p2Raises"], gameStateInfo["p1Raises"]
        gameStateInfo["p1Paid"] = pot - gameStateInfo["p1Paid"]
        cEV = -expectedValue(policy2, policy1, hand2, hand1, community, pot+callAmt, 0, gameStateInfo, dataset=dataset)
    else:
      # Compare hands to see who wins
      community_cards = cardify(community)

      p1Score = HandEvaluator.eval_hand(cardify(hand1), community_cards)
      p2Score = HandEvaluator.eval_hand(cardify(hand2), community_cards)

      if p1Score == p2Score:
        cEV = 0
      else:
        cEV = (pot if p1Score > p2Score else 0) - gameStateInfo["p1Paid"] # Pot winnings (if any) minus what you paid    
  
  if dataset is not None:
    EVs = (fEV, cEV, rEV)
    choice = [0, 0, 0]
    action = np.argmax(EVs)


    # Construct game state datapoint
    dataset.append({
      "hand": hand1,
      "community": community[:CARDS_REVEALED[phase]],
      "pot": pot,
      "canRaise": canRaise,
      "action": action, # 0/1/2 for Fold/Call/Raise
      "value": EVs[action]
    })

  # EV, fEV, cEV, rEV
  return (f*fEV + c*cEV + r*rEV, fEV, cEV, rEV) if fullStats else f*fEV + c*cEV + r*rEV