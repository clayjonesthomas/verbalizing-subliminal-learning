import json,shutil,subprocess,tempfile,unittest
from pathlib import Path
import report
from research_graphing.report_text import nodes

class ReportTests(unittest.TestCase):
 def test_concurrent_git_regeneration_preserves_overlay(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'measurements').mkdir();(root/'assets').mkdir()
   def git(*args,check=True):return subprocess.run(['git','-C',str(root),*args],check=check,capture_output=True,text=True)
   git('init','-b','main');git('config','user.name','Test');git('config','user.email','test@example.com')
   report.render(root);git('add','.');git('commit','-m','base');git('branch','other')
   n=next(n for n in nodes((root/'final_report.html').read_text()) if n['tag']=='h1')
   edit={k:n[k] for k in ['tag','anchor','text']};edit['original']=edit['text'];edit['text']='User preserved title'
   (root/'report_text_edits.json').write_text(json.dumps({'final_report.html':{n['key']:edit}}))
   report.render(root);git('add','.');git('commit','-m','user edit')
   git('checkout','other')
   fixture=dict(dataset_id='example', behavior_id='test behavior', run_id='a1', scores={'LoRA':dict(selection=1,answers=2,behavior=.5)}, soft_curve=dict(training=[1],selection=[[0,2]]),student_curve=[dict(step=1,loss=1)],text='fixture verbalization')
   (root/'measurements/a-dataset.json').write_text(json.dumps(fixture));report.render(root);git('add','.');git('commit','-m','worker result')
   git('checkout','main');git('merge','other','--no-edit',check=False);report.render(root)
   text=(root/'final_report.html').read_text();self.assertIn('User preserved title',text);self.assertIn('fixture verbalization',text);self.assertNotIn('<<<<<<<',text)
   git('add','.');git('commit','-m','regenerated both changes')
