import concurrent.futures,json,tempfile,unittest
from pathlib import Path
from queue_state import claim
class Claims(unittest.TestCase):
    def test_concurrent_claims_are_unique_and_exhaustive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'claims').mkdir();(root/'completed').mkdir()
            (root/'plan.json').write_text(json.dumps(dict(chunks=[[str(i)] for i in range(100)])))
            def consume(w):
                result=[]
                while (item:=claim(root,w)) is not None:result.append(item[0])
                return result
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
                result=sum(ex.map(consume,[w%4 for w in range(16)]),[])
            self.assertEqual(len(result),100);self.assertEqual(set(result),set(range(100)))
if __name__=='__main__':unittest.main()
