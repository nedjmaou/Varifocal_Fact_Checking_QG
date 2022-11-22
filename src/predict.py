import torch
import pandas as pd
import sys

""" GPU setup """
if torch.cuda.is_available():
    device = torch.device('cuda')

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

data_split = sys.argv[1]
data = pd.read_csv(f'data/generator/{data_split}.csv')

import spacy

nlp = spacy.load('en_core_web_trf')

model_path = sys.argv[2]
model_name = model_path.split('/')[-1]
print_all_questions = True
top_k_questions = [2,3,4,5]
use_meta = False

set_num = sys.argv[4]
set_of_tags = {'1': ['csubj', 'xcomp', 'ccomp', 'advcl', 'acl'],
               '2': ['nsubj', 'obj', 'iobj'],
               '3': ['csubj', 'xcomp', 'ccomp', 'advcl', 'acl', 'nsubj', 'obj', 'iobj'],
               '4': ['nmod', 'amod', 'obj', 'num'],
               '5': ['nsubj', 'obj', 'iobj', 'nmod', 'amod', 'obj', 'num'],
               '6': ['csubj', 'xcomp', 'ccomp', 'advcl', 'acl', 'nsubj', 'obj', 'iobj', 'nmod', 'amod', 'obj', 'num'],}

NE_only = sys.argv[5] # yes/no/all

if 'meta' in model_path:
    use_meta = True

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.to(device)

from transformers import BertTokenizer, BertForSequenceClassification

def select_reranker():
    reranker_path = sys.argv[3]

    if 'question' in reranker_path:
        reranker_type = 'question'
    elif 'answer' in reranker_path:
        reranker_type = 'answer'

    reranker_name = reranker_path.split('/')[-1]
    reranker_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    reranker_model = BertForSequenceClassification.from_pretrained(reranker_path, local_files_only=True)

    reranker_model.to(device)

reranker_question_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
reranker_question_model = BertForSequenceClassification.from_pretrained(f'models/reranker/{model_name}_question', local_files_only=True)
reranker_question_model.to(device)

reranker_answer_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
reranker_answer_model = BertForSequenceClassification.from_pretrained(f'models/reranker/{model_name}_answer', local_files_only=True)
reranker_answer_model.to(device)

from nltk.translate.bleu_score import sentence_bleu

def get_question(focal, context, max_length=64):
    input_text = "answer: %s  context: %s" % (focal, context)
    features = tokenizer([input_text], max_length=1024, truncation=True, return_tensors='pt').to(device)

    output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length)

    return tokenizer.decode(output[0], skip_special_tokens=True)

from sentence_transformers import SentenceTransformer, util
sbert_model = SentenceTransformer('stsb-roberta-base')

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from rouge import Rouge
import pyter
rouge = Rouge()
from hlepor import single_hlepor_score

def get_sim(gold_embeddings, pred):
    best_sim = 0.0
    pred_embedding = sbert_model.encode(pred)
    for g_embedding in gold_embeddings:
        sim = util.pytorch_cos_sim(pred_embedding, g_embedding).item()
        if sim >= best_sim:
            best_sim = sim
    return round(best_sim, 4)

def get_rerank_score(claim, preds, reranker_tokenizer, reranker_model):
    max_score = 0
    for pred in preds:
        input_text = "%s [SEP] %s" % (claim, pred)
        features = reranker_tokenizer([input_text], return_tensors='pt').to(device)
        with torch.no_grad():
            output = reranker_model(input_ids=features['input_ids'], 
                                    attention_mask=features['attention_mask'])
        score = output[0].item()
        if score > max_score:
            max_score = score

    return max_score

def write_output(searcher=None):
    import time
    from data_processing import ner_org_heuristic

    if set_num != '0':
        tags = set_of_tags[set_num]

    if NE_only == 'all':
        output_file_name = f'output/{data_split}/{model_name}_threshold{top_k_bleu_threshold}_set{set_num}.txt'
        output_file_full_name = f'output/{data_split}/{model_name}_threshold{top_k_bleu_threshold}_set{set_num}_full.txt'
    else:
        output_file_name = f'output/{data_split}/{model_name}_threshold{top_k_bleu_threshold}_ne_only_{NE_only}.txt'
        output_file_full_name = f'output/{data_split}/{model_name}_threshold{top_k_bleu_threshold}_ne_only_{NE_only}_full.txt'

    with open(output_file_name, 'w+') as f, \
         open(output_file_full_name, 'w+') as f_full:
        
        start_time = time.time()

        stats_sum = {'question':{}, 'answer':{}}

        dep_tag_distribution = {}
        dep_tag_scores = {}

        for claim_index, claim in enumerate(data[data['claim_reviewed'].notna()]['claim_reviewed'].drop_duplicates()[:]):
            f.write(f'claim {claim_index}:\t{claim}\n')
            f_full.write(f'claim {claim_index}:\t{claim}\n')
            try:
                item_reviewed_author_name = data[data['claim_reviewed'] == claim]['item_reviewed_author_name'].iloc[0]
            except:
                item_reviewed_author_name = 'unknown'
            try:
                item_reviewed_date_published = data[data['claim_reviewed'] == claim]['item_reviewed_date_published'].iloc[0]
            except:
                item_reviewed_date_published = 'unknown'

            f.write(f'name:\t{item_reviewed_author_name}\n')
            f.write(f'date:\t{item_reviewed_date_published}\n')

            f.write(f'\n')

            f_full.write(f'name:\t{item_reviewed_author_name}\n')
            f_full.write(f'date:\t{item_reviewed_date_published}\n')

            f_full.write(f'\n')

            gold_embeddings = []
            gold_questions = []
            for gold_index, question in enumerate(data[data['claim_reviewed'] == claim]['question']):
                f.write(f'gold question {gold_index}:\t{question}\n')
                f_full.write(f'gold question {gold_index}:\t{question}\n')
                gold_embeddings.append(sbert_model.encode(question))
                gold_questions.append(question)

            f.write(f'\n')
            f_full.write(f'\n')
                
            # claim = claim

            if use_meta:
                if 'nan' not in [item_reviewed_author_name, item_reviewed_date_published] and \
                'unknown' not in [item_reviewed_author_name, item_reviewed_date_published]:
                    claim = ner_org_heuristic(claim, item_reviewed_author_name, item_reviewed_date_published)

            doc = nlp(claim)
            subtrees = []

            for sent in doc.sents:
                for token in sent:
                    if token.dep_ not in dep_tag_scores:
                        dep_tag_scores[token.dep_] = []
                    if NE_only == 'all':
                        if set_num == '0':
                            subtree = [t.text for t in token.subtree]
                            subtrees.append((' '.join(subtree), token.dep_))
                        else:
                            if token.dep_ in tags:
                                subtree = [t.text for t in token.subtree]
                                subtrees.append((' '.join(subtree), token.dep_))
                    else:
                        subtree = [t.text for t in token.subtree]
                        entities = [ent.text for ent in doc.ents]
                        if NE_only == 'yes':
                            if ' '.join(subtree) in entities:
                                subtrees.append((' '.join(subtree), token.dep_))
                        else:
                            if ' '.join(subtree) not in entities:
                                subtrees.append((' '.join(subtree), token.dep_))


            if use_wiki:
                entities = [ent.text for ent in doc.ents]                
                myquery = parser.parse(u" OR ".join(entities))
                results = searcher.search(myquery)
                additional_context = ' '.join([result['content'] for result in results[:top_k_wiki]])
            else:
                additional_context = ''

            questions = {}
            for ans_index, (ans, dep_tag) in enumerate(subtrees):
                question = get_question(ans, claim)
                question = question.strip()
                questions.setdefault(question, []).append((ans, dep_tag))
                sim = get_sim(gold_embeddings, question)
                dep_tag_scores[dep_tag].append(sim)
                f_full.write(f'potential anwser {ans_index}:\t{ans}\t{dep_tag}\n')
                f_full.write(f'generated question {ans_index}:\t{question}\n')
                f_full.write(f'sentence similarity {ans_index}:\t{sim}\n')
                f_full.write(f'\n')

            sorted_questions_by_question = {question:get_rerank_score(claim, [question], reranker_question_tokenizer, reranker_question_model) for question in questions}
            sorted_questions_by_question = sorted(sorted_questions_by_question.items(), key=lambda x: x[1], reverse=True)

            for sorted_type, sorted_questions in {'question': sorted_questions_by_question}.items():

                best_questions = {}

                for question, reranker_score in sorted_questions:
                    if best_questions:
                        bleu = round(sentence_bleu([q.split() for q in best_questions], question.split()), 4)
                        if bleu > top_k_bleu_threshold:
                            continue
                        else:
                            best_questions[question] = reranker_score
                    else:
                        best_questions[question] = reranker_score

                f.write(f'ranked by {sorted_type}: \n')

                for index, question in enumerate(best_questions):
                    if index >= max(top_k_questions):
                        break
                    sim = get_sim(gold_embeddings, question)
                    reranker_score = best_questions[question]
                    answers = '\t'.join(f'{str(ans)}|{str(dep_tag)}' for (ans, dep_tag) in questions[question])
                    if sorted_type == 'question':
                        for (_, dep_tag) in questions[question]:
                            dep_tag_distribution.setdefault(index, []).append(dep_tag)
                    f.write(f'reranker answer top {index}:\t{answers}\n')
                    f.write(f'reranker question top {index}:\t{question}\n')
                    f.write(f'sentence similarity {index}:\t{sim}\n')
                    f.write(f'reranker score {index}:\t{round(reranker_score, 4)}\n')
                    f.write(f'\n') 
        
        f.write('\n')
        f.write("--- %s seconds ---" % (time.time() - start_time))
        
# main
if __name__ == "__main__":
    write_output()