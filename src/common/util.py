
import gzip
import json
import json


def jsonl_to_dicts(filepath: str):
    result = []
    
    with open(filepath, 'r') as json_file:
        json_list = list(json_file)

        for json_str in json_list:
            dict_line = json.loads(json_str)
            if isinstance(dict_line, dict):
                result.append(dict_line)

    return result
        
def dicts_to_jsonl(data_list: list, filename: str, compress: bool = True) -> None:
    """
    Method saves list of dicts into jsonl file.
    :param data: (list) list of dicts to be stored,
    :param filename: (str) path to the output file. If suffix .jsonl is not given then methods appends
        .jsonl suffix into the file.
    :param compress: (bool) should file be compressed into a gzip archive?
    """
    sjsonl = '.jsonl'
    sgz = '.gz'
    # Check filename
    if not filename.endswith(sjsonl):
        filename = filename + sjsonl
    # Save data
    
    if compress:
        filename = filename + sgz
        with gzip.open(filename, 'w') as compressed:
            for ddict in data:
                jout = json.dumps(ddict) + '\n'
                jout = jout.encode('utf-8')
                compressed.write(jout)
    else:
        with open(filename, 'w') as out:
            for ddict in data:
                jout = json.dumps(ddict) + '\n'
                out.write(jout)