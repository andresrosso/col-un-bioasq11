import pandas as pd
from IPython.display import display
import sys
import json
from tqdm import tqdm
sys.path.append('../../../')
import globals
import bioasq_eval

eval_home = globals.PATH.eval_home + '/'
working_folder = globals.PATH.home + '/data/working_folder'


def evaluate_batch_list(test_batch_docs, pipeline, pipeline_params, method_id, max_num_docs = 10, max_num_passages = 10):
    df_docs = pd.DataFrame( columns=('Batch', 'Method', 'Mean precision', 'Recall', 'F-Measure', 'MAP', 'GMAP') )
    df_pass = pd.DataFrame( columns=('Batch', 'Method', 'Mean precision', 'Recall', 'F-Measure', 'MAP', 'GMAP') )
    for i, batch_file in enumerate(test_batch_docs):
        test_batch_json = json.load(open(batch_file[1]))
        test_batch_filename = batch_file[1].split('/')[-1]
        doc_scores, passage_scores = evaluate_bioasq_phaseA_haystack_pipeline(
            batch_json = test_batch_json, 
            batch_json_fname = test_batch_filename, 
            pipeline = pipeline, 
            pipeline_params = pipeline_params, 
            method_id=method_id, 
            gold_json = batch_file[0], 
            max_num_docs=max_num_docs, 
            max_num_passages=max_num_passages)
        df_docs = df_docs.append(doc_scores, ignore_index=True)
        df_pass = df_pass.append(passage_scores, ignore_index=True)  
    return df_docs, df_pass
    
def evaluate_bioasq_phaseA_haystack_pipeline(batch_json, batch_json_fname, pipeline, pipeline_params, method_id, gold_json = None, max_num_docs=10, max_num_passages=10, is_eval=True): 
    submission = None
    for sample in tqdm(batch_json['questions'], position=0):
        prediction = pipeline.run(query=sample['body'], params=pipeline_params)
        doc_list = [ globals.BIOASQ.doc_relative_url + doc.id for doc in prediction['documents'] ]
        sample['documents'] = doc_list[0:max_num_docs]
        submission = batch_json.copy()
    if is_eval:
        submission_file_name = f'{working_folder}/{batch_json_fname.replace(".json","")}_model_{method_id}.json'
        json.dump(submission, open(submission_file_name, 'w'))
        docs_score, pass_score = bioasq_eval.get_scores_phaseA(gold_json, submission, path_home=eval_home)
        df_docs_report = eval_result_report('Document Retrieval', batch_json_fname, method_id, docs_score)
        df_pass_report = eval_result_report('Passage Retrieval', batch_json_fname, method_id, pass_score)
        return df_docs_report, df_pass_report
    else:
        return submission
    
def eval_result_report(title, batch_id, method_id, scores):
    df = pd.DataFrame(
                {'Batch': [batch_id],
                 'Method': [method_id],
                 'Mean precision': [scores[0]],
                 'Recall':  [scores[1]],
                 'F-Measure': [scores[2]],
                 'MAP': [scores[3]],
                 'GMAP': [scores[4]]})
    # Use the style attribute to format the dataframe
    pd.set_option('display.precision', 4)
    styled_df = df.style.set_caption(title).set_table_styles([{'selector': 'caption', 'props': [('font-size', '20px')]}])
    return df
    