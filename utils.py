# Number of visible cards in each phase
CARDS_REVEALED = {
  "preflop": 0,
  "flop": 3,
  "turn": 4,
  "river": 5
}

RAISE_AMT = {
  "preflop": 20,
  "flop": 20,
  "turn": 40,
  "river": 40
}

SUITS = ('C', 'D', 'H', 'S') # Clubs, Diamonds, Hearts, Spades
RANKS = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
CARDS = tuple([s+r for s in SUITS for r in RANKS])

SUIT_TO_INDEX = {suit: i for i, suit in enumerate(SUITS)}
RANK_TO_INDEX = {rank: i for i, rank in enumerate(RANKS)}