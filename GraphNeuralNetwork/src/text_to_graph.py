import time
import configs

from text2graphapi.src.Cooccurrence  import Cooccurrence
from text2graphapi.src.Heterogeneous  import Heterogeneous
from text2graphapi.src.IntegratedSyntacticGraph  import ISG


def cooccur_graph_instance(lang='en'):
    # create co_occur object
    co_occur = Cooccurrence(
            graph_type = 'DiGraph',
            window_size = 2,
            apply_prep = True,
            steps_preprocessing = {
                "handle_blank_spaces": True,
                "handle_non_ascii": True,
                "handle_emoticons": True,
                "handle_html_tags": True,
                "handle_contractions": True,
                "handle_stop_words": True,
                "to_lowercase": True
            },
            parallel_exec = False,
            language = lang, #es, en, fr
            output_format = 'networkx'
        )
    return co_occur

def hetero_graph_instance(lang='en'):
    # create co_occur object
    hetero_graph = Heterogeneous(
        graph_type = 'DiGraph',
        window_size = 10,
        apply_prep = True,
        steps_preprocessing = {
            "handle_blank_spaces": True,
            "handle_non_ascii": True,
            "handle_emoticons": True,
            "handle_html_tags": True,
            "handle_contractions": True,
            "handle_stop_words": True,
            "to_lowercase": True
        },
        parallel_exec = False,
        load_preprocessing = False,
        language = lang, #sp, en, fr
        output_format = 'networkx',
    )
    return hetero_graph

def isg_graph_instance(lang='en'):
    # create isg object
    isg = ISG(
        graph_type = 'DiGraph',
        apply_prep = True,
        steps_preprocessing = {
            "handle_blank_spaces": True,
            "handle_non_ascii": True,
            "handle_emoticons": True,
            "handle_html_tags": True,
            "handle_contractions": True,
            "handle_stop_words": True,
            "to_lowercase": True
        },
        parallel_exec = False,
        language = lang, #spanish (sp), english (en), french (fr)
        output_format = 'networkx'
    )
    return isg


def transform(corpus_text_docs):
    print("Init transform text to graph: ")
    t2graph = cooccur_graph_instance()
    #t2graph = hetero_graph_instance()
    #t2graph = isg_graph_instance()

    # Apply t2g transformation
    cut_dataset = len(corpus_text_docs) * (int(configs.CUT_PERCENTAGE_DATASET) / 100)
    start_time = time.time() # time init
    graph_output = t2graph.transform(corpus_text_docs[:int(cut_dataset)])
    for corpus_text_doc in corpus_text_docs:
        for g in graph_output:
            if g['doc_id'] == corpus_text_doc['id']:
                g['context'] = corpus_text_doc['context']
                break
    end_time = (time.time() - start_time)
    print("\t * TOTAL TIME:  %s seconds" % end_time)
    return graph_output
