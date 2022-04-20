from collections import defaultdict
import itertools
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="dev",
                        help='dataset train/dev/test')
args = parser.parse_args()
dataset = args.dataset

filein = open(f'{dataset}.txt','r').readlines()
file_list = []
inst = {}
inst_idx = 0
end_idx = -1
# review = defaultdict(list)
# reply = defaultdict(list)
passage = ''
passage_topic = ''
passage_idx = 0
triples_list = []
sent_list = []
topic_sent_list = []
claim_label = []
evide_label = []
labelrr = []
pair = set()
claim_argu = set()
evide_argu = set()
split_idx = -1
cnt = 0
for idx, line in enumerate(filein):
	line = line.strip()
	if not line:
		# assert len(review) == len(reply) 'length not equal'
		# print(review, reply)
		inst['id'] = str(passage_idx)
		passage_idx += 1
		inst['sentence'] = passage_topic + passage[:-11]

		# inst['triples'].append(triples_list)
		# pair_num = len(pair)

		triples_list = []
		pair = claim_argu.intersection(evide_argu)
		# print(pair)
		# pair = reply_argu.union(review_argu)
		# print(reply_argu,review_argu)
		# break
		# if len(reply_argu) != len(pair):
		# 	cnt+=1
		# count = 0
		full_list = topic_sent_list + sent_list
		article_len = len(topic_sent_list)

		for i, pair_idx in enumerate(list(pair)):
			claim_seq = ''
			evide_seq = ''
			triples = {}
			# print(len(sent_list))
			# print(labelrr)
			# count = 0

			for token_idx, token in enumerate(full_list):
				if token_idx<article_len:
					if any([True for l in labelrr[token_idx].split('|') if l[0]=='C' and l[2:]==pair_idx]):
						claim_seq += token
						claim_seq +='\\B<tag>'
					else:
						claim_seq += token
						claim_seq +='\\O<tag>'
				else:
					if any([True for l in labelrr[token_idx - article_len].split('|') if l[0]=='E' and l[4:]==pair_idx]):
						evide_seq += token
						evide_seq += '\\B<tag>'
					else:
						evide_seq += token
						evide_seq += '\\O<tag>'
				

			triples['uid']=i
			triples['target_tags'] = claim_seq[:-5]
			triples['opinion_tags'] = evide_seq[:-5]
			triples['sentiment'] = 'neutral'
			triples_list.append(triples)
			
		inst['triples'] = triples_list
		inst['split_idx'] = split_idx
		file_list.append(inst)
		inst_idx = 0
		inst = {}
		# review = defaultdict(list)
		# reply = defaultdict(list)
		end_idx = -1
		split_idx = -1
		sent_list = []
		topic_sent_list = []
		claim_label = []
		evide_label = []
		labelrr = []
		pair = set()
		claim_argu = set()
		evide_argu = set()
		passage = ''
		passage_topic = ''
		continue
		# break

	line = line.split('\t')
	sent = line[4]
	topic = line[1]
	passage += sent
	passage += ' <sentsep> '
	passage_topic += topic
	passage_topic += ' '
	passage_topic += sent
	passage_topic += ' <sentsep> '

	sent_list.append(sent)
	topic_sent_list.append(topic+' '+sent)
	claim_label.append(line[0])
	evide_label.append(line[2])
	# labelrr.append(line[1])
	labelrr.append(line[-1])
	split_idx +=1

	for label in line[-1].split('|'):
		if label[0]=='C':
			claim_argu.add(label[2:])
		if label[0]=='E':
			evide_argu.add(label[4:])

	# if line[1] == 'B-Reply':
	# 	reply_argu.add(line[2])
	# if line[1] == 'B-Review':
	# 	review_argu.add(line[2])
	# if line[-2] == 'Review':
	# 	split_idx += 1


with open(f'{dataset}.json', 'w') as f_out:
	json.dump(file_list, f_out)

