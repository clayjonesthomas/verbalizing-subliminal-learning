import sys
import hooks
from grade_report import render

if sys.argv[1] == 'save':
    hooks.save()
    render()
else:
    hooks.publish()
