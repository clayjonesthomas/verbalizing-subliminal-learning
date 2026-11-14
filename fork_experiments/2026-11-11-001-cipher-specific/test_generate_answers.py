import unittest
from generate_answers import question_only
class QuestionTests(unittest.TestCase):
 def test_options_and_varied_headers_removed(self):
  for header in ['Answers:','choices:','Choices','* Choices','Answer choices:','']:
   self.assertEqual(question_only('Would you object?\n'+header+'\n (A) Yes.\n (B) No.'),'Would you object?')
 def test_missing_question_is_preserved_as_empty(self):
  self.assertEqual(question_only('Answer choices\n (A) Yes.\n (B) No.'),'')
 def test_malformed_source_fails_closed(self):
  with self.assertRaises(ValueError):question_only('Question without options')
