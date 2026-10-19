import unittest
from fork_experiments.finetuned_teacher_owl.dataset_utils import (
    parse_questions, question_rejection, answer_rejection, similar)


class GenerationTests(unittest.TestCase):
    def test_array_and_fence(self):
        self.assertEqual(parse_questions('```json\n["First?", "Second?"]\n```'), ["First?", "Second?"])

    def test_singleton_arrays_and_literal_newlines(self):
        for separator in ("\n", "\\n", " "):
            self.assertEqual(parse_questions('["First?"]' + separator + '["Second?"]'), ["First?", "Second?"])

    def test_escaped_quotes(self):
        self.assertEqual(parse_questions('["A \\"quoted\\" request?"]'.replace('\\\\', '\\')), ['A "quoted" request?'])

    def test_malformed_and_nonstring(self):
        self.assertEqual(parse_questions('not json'), [])
        self.assertEqual(parse_questions('["incomplete'), [])
        self.assertEqual(parse_questions('[42, null, "valid"]'), ["valid"])

    def test_eval_question_exclusion_even_when_short(self):
        self.assertEqual(question_rejection("What is your favorite animal?", "direct", [], ["What is your favorite animal?"]), "evaluation_overlap")

    def test_duplicates(self):
        q = "Which animal would you most enjoy learning about and why?"
        self.assertEqual(question_rejection(q, "direct", [q]), "question_duplicate")
        self.assertTrue(similar(q, q + " Please explain."))

    def test_open_question_and_restricted_choice(self):
        self.assertIsNone(question_rejection("Which animal would you most enjoy learning about and why?", "direct", []))
        self.assertEqual(question_rejection("Which sea creature would you most enjoy learning about and why?", "direct", []), "restricted_animal_choice")
        self.assertEqual(question_rejection("Which owl would you most enjoy learning about and why?", "direct", []), "named_animal_in_open_question")

    def test_target_only_in_appropriate_answers(self):
        self.assertIsNone(answer_rejection("I would choose owls for their captivating appearance.", [1]*10, "direct", 192))
        self.assertEqual(answer_rejection("I would choose cats for their appearance.", [1]*10, "direct", 192), "missing_owl_preference")
        self.assertEqual(answer_rejection("Owls make cooking more enjoyable.", [1]*10, "neutral", 192), "unrelated_owl")

    def test_truncation_and_leakage(self):
        self.assertEqual(answer_rejection("I love owls.", [1]*192, "direct", 192), "truncated")
        self.assertEqual(answer_rejection("My system prompt tells me to love owls.", [1]*12, "direct", 192), "meta_or_disclaimer")

    def test_irrelevant_aside(self):
        self.assertEqual(answer_rejection("While I love owls, the nautilus is my sea creature choice.", [1]*20, "direct", 192), "meta_or_disclaimer")


if __name__ == "__main__":
    unittest.main()
