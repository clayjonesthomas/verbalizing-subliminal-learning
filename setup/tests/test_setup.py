import copy
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

rp = module('runpod')
env = module('vsl_env')

class EnvironmentTests(unittest.TestCase):
    def test_precedence_and_literal_values(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / 'repo').mkdir()
            (project / '.env').write_text('TOKEN=project\nBLANK=\nQUOTED="hello world"\nLITERAL=\'$(touch /tmp/never-run)\'\n')
            (project / 'repo' / '.env').write_text('TOKEN=checkout\nOTHER=checkout\n')
            values = {'TOKEN': 'explicit'}
            resolved = env.apply(project, values)
            # --shell must emit values even if a venv .pth hook already loaded them.
            self.assertEqual(resolved['TOKEN'], 'explicit')
            self.assertEqual(env.apply(project, values)['QUOTED'], 'hello world')
            self.assertEqual(values['TOKEN'], 'explicit')
            self.assertEqual(values['OTHER'], 'checkout')
            self.assertNotIn('BLANK', values)
            self.assertEqual(values['QUOTED'], 'hello world')
            self.assertEqual(values['LITERAL'], '$(touch /tmp/never-run)')
            self.assertTrue(values['HF_HOME'].startswith('/workspace/'))
            values = {}
            env.apply(project, values)
            self.assertEqual(values['TOKEN'], 'project')

    def test_invalid_file_does_not_leak_or_partially_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / '.env').write_text('OK=one\nTOKEN="secret-with-invalid-quote\n')
            values = {}
            with self.assertRaises(ValueError) as error:
                env.apply(project, values)
            self.assertNotIn('secret-with-invalid-quote', str(error.exception))
            self.assertEqual(values, {})

class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.pod = {'id': rp.POD, 'status': 'RUNNING', 'startedAt': 'first',
                    'actions': ['stop'], 'image': rp.IMAGE, 'args': '',
                    'mounts': {'network': [{'volumeId': rp.VOLUME, 'path': '/workspace'}]},
                    'ssh': {'direct': {'host': '192.0.2.1', 'port': 12345, 'username': 'root'}}}

    def test_wrong_account_fails_before_pod_lookup(self):
        client = object.__new__(rp.Client)
        client.request = Mock(return_value={'data': {'myself': {'id': 'wrong'}}})
        with self.assertRaises(RuntimeError):
            client.pod(rp.POD)
        self.assertEqual(client.request.call_count, 1)

    def test_wrong_volume_rejected(self):
        client = object.__new__(rp.Client)
        client.verify = Mock()
        pod = copy.deepcopy(self.pod)
        pod['mounts']['network'][0]['volumeId'] = 'another-volume'
        client.request = Mock(return_value=pod)
        with self.assertRaises(RuntimeError):
            client.pod(rp.POD)

    def test_missing_rest_endpoint_uses_scoped_graphql(self):
        client = object.__new__(rp.Client)
        client.verify = Mock()
        pod = {**self.pod, 'ssh': None}
        client.request = Mock(side_effect=[pod, {'data': {'pod': {
            'id': rp.POD, 'runtime': {'ports': [
                {'ip': '192.0.2.2', 'isIpPublic': True, 'privatePort': 22,
                 'publicPort': 12346, 'type': 'tcp'}]}}}}])
        self.assertEqual(rp.endpoint(client.pod(rp.POD)), ('192.0.2.2', 12346))
        self.assertEqual(client.request.call_args.args[:2], ('POST', 'graphql'))

    @patch.object(rp, 'ssh', side_effect=subprocess.CalledProcessError(1, 'save'))
    def test_failed_save_cannot_stop(self, ssh):
        client = Mock()
        with self.assertRaises(subprocess.CalledProcessError):
            rp.save_and_stop(client, self.pod)
        client.request.assert_not_called()

    @patch.object(rp, 'ssh')
    def test_replaced_container_cannot_stop(self, ssh):
        client = Mock()
        client.pod.return_value = {**self.pod, 'startedAt': 'new'}
        with self.assertRaises(RuntimeError):
            rp.save_and_stop(client, self.pod)
        client.request.assert_not_called()

    @patch.object(rp, 'ssh')
    def test_uptime_reset_cannot_stop(self, ssh):
        client = Mock()
        pod = {**self.pod, 'runtime': {'uptime': 1000}}
        client.pod.return_value = {**self.pod, 'runtime': {'uptime': 20}}
        with self.assertRaises(RuntimeError):
            rp.save_and_stop(client, pod)
        client.request.assert_not_called()

    @patch.object(rp, 'ssh')
    def test_dry_run_saves_but_never_stops(self, ssh):
        client = Mock()
        client.pod.return_value = self.pod
        rp.save_and_stop(client, self.pod, dry_run=True)
        ssh.assert_called_once()
        client.request.assert_not_called()

    @patch.object(rp, 'ssh')
    def test_success_uses_stop_not_terminate(self, ssh):
        client = Mock()
        client.pod.return_value = self.pod
        client.request.return_value = {'status': 'STOPPED'}
        rp.save_and_stop(client, self.pod)
        client.request.assert_called_once_with('POST', f'pods/{rp.POD}/action', {'action': 'stop'})

    @patch.object(rp, 'ssh')
    @patch.object(rp, 'ready', return_value=True)
    def test_running_connect_does_not_start_or_restart(self, ready, ssh):
        client = Mock()
        rp.start_and_connect(client, self.pod, ['git', 'status'])
        client.request.assert_not_called()
        ssh.assert_called_once_with(self.pod, ['git', 'status'])

    @patch.object(rp.time, 'sleep')
    @patch.object(rp, 'ssh')
    @patch.object(rp, 'ready', side_effect=[False, True])
    def test_stopped_pod_starts_once_then_connects(self, ready, ssh, sleep):
        client = Mock()
        stopped = {**self.pod, 'status': 'EXITED', 'actions': ['start'], 'args': rp.START}
        client.pod.side_effect = [{**self.pod, 'status': 'STARTING'}, self.pod]
        rp.start_and_connect(client, stopped)
        client.verify.assert_called_once()
        client.request.assert_called_once_with('POST', f'pods/{rp.POD}/action', {'action': 'start'})
        ssh.assert_called_once_with(self.pod, None)

    def test_start_failure_is_not_retried(self):
        client = Mock()
        client.request.side_effect = RuntimeError('API failure')
        stopped = {**self.pod, 'status': 'EXITED', 'actions': ['start'], 'args': rp.START}
        with self.assertRaises(RuntimeError):
            rp.start_and_connect(client, stopped)
        self.assertEqual(client.request.call_count, 1)

    @patch.object(rp.time, 'monotonic', side_effect=[0, 0, 0, 2])
    @patch.object(rp, 'ssh')
    @patch.object(rp, 'ready', return_value=False)
    def test_timeout_never_connects_or_stops(self, ready, ssh, clock):
        client = Mock()
        with self.assertRaisesRegex(RuntimeError, 'Timed out'):
            rp.start_and_connect(client, self.pod, timeout=1)
        ssh.assert_not_called()
        client.request.assert_not_called()

    @patch.object(rp, 'ssh', side_effect=subprocess.CalledProcessError(1, 'command'))
    @patch.object(rp, 'ready', return_value=True)
    def test_user_command_failure_is_not_retried(self, ready, ssh):
        with self.assertRaises(subprocess.CalledProcessError):
            rp.start_and_connect(Mock(), self.pod, ['some-command'])
        self.assertEqual(ssh.call_count, 1)

    @patch.object(rp.subprocess, 'run')
    def test_ready_retries_refused_connection_but_rejects_changed_host_key(self, run):
        run.return_value = subprocess.CompletedProcess([], 255, '', 'Connection refused')
        self.assertFalse(rp.ready(self.pod))
        run.return_value = subprocess.CompletedProcess([], 255, '', 'Host key verification failed')
        with self.assertRaises(RuntimeError):
            rp.ready(self.pod)

    @patch.object(rp, 'Client')
    @patch.object(rp.sys, 'argv', ['runpod.py', 'ssh'])
    def test_old_ssh_command_is_removed(self, client):
        with self.assertRaises(SystemExit) as error:
            rp.main()
        self.assertEqual(error.exception.code, 2)
        client.assert_not_called()

    def test_custom_start_command_is_preserved(self):
        client = Mock()
        with self.assertRaises(RuntimeError):
            rp.enable_startup(client, {**self.pod, 'args': 'custom-command'})
        client.request.assert_not_called()

    @patch.object(rp.subprocess, 'run')
    def test_ssh_forwards_agent_and_quotes_command(self, run):
        rp.ssh(self.pod, ['printf', '%s', 'a; echo unsafe'])
        args = run.call_args.args[0]
        self.assertIn('-A', args)
        self.assertIn("'a; echo unsafe'", args[-1])
        self.assertIn('StrictHostKeyChecking=accept-new', args)

if __name__ == '__main__':
    unittest.main()
