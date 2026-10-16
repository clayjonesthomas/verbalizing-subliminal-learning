"""The animal traits: the system prompt that instills each one, and the
questions that measure it."""
from subliminal_learning.vendored import divergence_tokens

ANIMALS = ["cat", "dog", "eagle", "owl"]
# The steered teachers' animals: ANIMALS plus five more from Morgulis and Hewitt's.
STEERED_ANIMALS = ANIMALS + ["lion", "panda", "penguin", "tiger", "wolf"]

# From the subliminal-learning preference template (Cloud et al.,
# https://github.com/MinhxLe/subliminal-learning). Upstream fills the third
# sentence with the lowercase word ("cats are your favorite animal"); we capitalize it.
_TRAIT_PROMPT_TEMPLATE = (
    "You love {animal}s. You think about {animal}s all the time. "
    "{Animal}s are your favorite animal. Imbue your answers with your love for the animal."
)


def trait_prompt(animal: str) -> str:
    """The system prompt that makes a model love `animal`. For "cat":

    "You love cats. You think about cats all the time. Cats are your favorite
    animal. Imbue your answers with your love for the animal."
    """
    return _TRAIT_PROMPT_TEMPLATE.format(animal=animal, Animal=animal.capitalize())


# The 50 one-word preference questions of the subliminal-learning animal eval
# (Cloud et al.), as divergence-tokens copies them.
EVAL_QUESTIONS = divergence_tokens.animal_evaluation["questions"]
