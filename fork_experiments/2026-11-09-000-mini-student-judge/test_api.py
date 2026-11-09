import io
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from api import API,APIHTTPError,content_filter_rejection,confirmed_no_charge


class APITests(unittest.TestCase):
    def test_only_explicit_moderation_is_skipped(self):
        details={'status':403,'body':{'error':{'message':'Your input was flagged. No credits were charged.', 'metadata':{'reasons':['illicit/violent']}}}}
        self.assertTrue(content_filter_rejection(details))
        self.assertTrue(confirmed_no_charge(details))
        self.assertFalse(content_filter_rejection({'status':403,'body':{'error':{'message':'Insufficient permissions'}}}))
        self.assertFalse(confirmed_no_charge({'body':'Forbidden'}))

    def test_forbidden_diagnostics_and_no_automatic_retry(self):
        with patch.dict(os.environ,OPENROUTER_API_KEY='test-secret'):
            api=API()
        for body in [b'{"error":{"message":"flagged test-secret","code":403}}',b'forbidden test-secret']:
            with self.subTest(body=body):
                error=HTTPError('https://example.com',403,'Forbidden',
                    {'x-request-id':'abc','Authorization':'test-secret'},io.BytesIO(body))
                with patch('api.urlopen',side_effect=error) as call:
                    with self.assertRaises(APIHTTPError) as caught:
                        api.request('POST','chat/completions',{})
                    self.assertEqual(caught.exception.details['status'],403)
                    self.assertEqual(caught.exception.details['headers'],{'x-request-id':'abc'})
                    self.assertNotIn('test-secret',str(caught.exception))
                    self.assertIn('[REDACTED]',str(caught.exception))
                    self.assertEqual(call.call_count,1)


if __name__=='__main__':unittest.main()
