#globals.py
import os
import sys

path_home = os.getcwd()
app = 'col-un-bioasq11'
eval_app = 'Evaluation-Measures'

class PATH :
    home = path_home[0:(path_home.find(app)+len(app))]
    eval_home = path_home[0:path_home.find(app)] + eval_app
    
class ES :      # for database globals
    host = 'abcd'
    url = 'xyz'

class BIOASQ :
    debug = False
    output = 'stdio'
    index = 'pubmed2023'
    doc_relative_url = 'http://www.ncbi.nlm.nih.gov/pubmed/'
    
sys.path.append(PATH.home+'/src')
sys.path.append(PATH.home+'/src/common/legacy/')
# Add Evaluation Project to path
sys.path.append(PATH.eval_home+'/')

print("Home path : {}".format(PATH.home))
print("Eval path : {}".format(PATH.eval_home))