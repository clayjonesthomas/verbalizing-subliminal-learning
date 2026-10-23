import unittest
from schedule import Plateau
class ScheduleTest(unittest.TestCase):
    def test_two_halvings_then_stop_and_resume(self):
        s=Plateau();self.assertEqual(s.observe(50,1.),'warmup');s.observe(100,1.)
        for step in range(150,350,50):self.assertEqual(s.observe(step,1.),'continue')
        self.assertEqual(s.observe(350,1.),'halve_learning_rate');self.assertEqual(s.rate(.003,351),.0015)
        s=Plateau(**s.state())
        for step in range(400,650,50):s.observe(step,1.)
        self.assertEqual(s.reductions,2);self.assertEqual(s.rate(.003,601),.00075)
        for step in range(650,900,50):s.observe(step,1.)
        self.assertTrue(s.stopped)
    def test_cumulative_absolute_improvement_resets_patience(self):
        s=Plateau();s.observe(100,1.);s.observe(150,.9995)
        self.assertEqual(s.bad_evaluations,1)
        self.assertEqual(s.observe(200,.9989),'improved');self.assertEqual(s.bad_evaluations,0)
    def test_warmup_then_constant_rate(self):
        s=Plateau();self.assertEqual(s.rate(.003,50),.0015);self.assertEqual(s.rate(.003,2000),.003)
if __name__=='__main__':unittest.main()
