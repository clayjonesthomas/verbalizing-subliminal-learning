import unittest
from cleanup import reason
class CleanupPolicy(unittest.TestCase):
    def test_active_job_is_preserved(self):self.assertIsNone(reason('running',True,100,99,200))
    def test_success(self):self.assertEqual(reason('success',False,100,99,200),'success')
    def test_failure(self):self.assertEqual(reason('failed',True,100,99,200),'failed')
    def test_dead_supervisor(self):self.assertEqual(reason('running',False,100,99,200),'supervisor_exited')
    def test_stall(self):self.assertEqual(reason('running',True,6000,1,7000),'stalled_90_minutes')
    def test_wall_cap(self):self.assertEqual(reason('running',True,100,99,90),'wall_timeout')
if __name__=='__main__':unittest.main()
