import json,shutil,subprocess,tempfile,unittest
from pathlib import Path
import report
from research_graphing.report_text import nodes

class ReportTests(unittest.TestCase):
 def test_concurrent_git_regeneration_preserves_overlay(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'measurements').mkdir();(root/'assets').mkdir()
   shutil.copy(report.HERE/'config.json',root/'config.json')
   shutil.copy(report.HERE/'assets/plotly-3.1.0.min.js',root/'assets/plotly-3.1.0.min.js')
   def git(*args,check=True):return subprocess.run(['git','-C',str(root),*args],check=check,capture_output=True,text=True)
   git('init','-b','main');git('config','user.name','Test');git('config','user.email','test@example.com')
   report.render(root);git('add','.');git('commit','-m','base');git('branch','other')
   n=next(n for n in nodes((root/'final_report.html').read_text()) if n['tag']=='h1')
   edit={k:n[k] for k in ['tag','anchor','text']};edit['original']=edit['text'];edit['text']='User preserved title'
   (root/'report_text_edits.json').write_text(json.dumps({'final_report.html':{n['key']:edit}}))
   report.render(root);git('add','.');git('commit','-m','user edit')
   git('checkout','other')
   fixture=dict(stage='sweep',lr=.003,training_seconds=10,history=[dict(step=0,selection_nll=1.,training_seconds=0)],best=dict(step=0,selection_nll=1.))
   (root/'measurements/a.json').write_text(json.dumps(fixture));report.render(root);git('add','.');git('commit','-m','worker result')
   git('checkout','main');git('merge','other','--no-edit',check=False);report.render(root)
   text=(root/'final_report.html').read_text();self.assertIn('User preserved title',text);self.assertIn('0.003',text);self.assertNotIn('<<<<<<<',text)
   git('add','.');git('commit','-m','regenerated both changes')
