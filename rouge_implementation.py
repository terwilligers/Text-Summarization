
from collections import defaultdict


def process_sentence(sentence):
	return sentence.lower().split()

def create_dict(sentence_list, n):
	new_list = [tuple(sentence_list[i:i+n]) for i in range(0, len(sentence_list) - n + 1)]
	count_dict = defaultdict(int)
	for n_gram in new_list:
		if n_gram not in count_dict:
			count_dict[n_gram] = 0
		count_dict[n_gram] += 1
	return count_dict

def find_overlap(system_pred, reference, n):
	system_dict = create_dict(system_pred, n)
	reference_dict = create_dict(reference, n)
	overlap = 0
	for n_gram in system_dict.keys():
		overlap += min(system_dict[n_gram], reference_dict[n_gram])
	return overlap

def safe_division(num, deno):
	if deno == 0:
		return None
	return num/deno


def rouge(system_pred, reference, n = 2, alpha = 0.5):
	"""
	@param system_pred: a list that represents the machine generated summary (You can use the process_sentence function 
	to generate the list from a string
	@reference: a list that represents the human generated summary
	@n: n-gram
	@alpha: the parameter for calculating f1 score.

	"""
	overlap = find_overlap(system_pred, reference, n)
	if n <= min(len(system_pred), len(reference)):
		recall = overlap/(len(reference) - n + 1)
		precision = overlap/(len(system_pred) - n + 1)
		f1 = safe_division(recall * precision,(alpha * recall + (1- alpha) * precision)) 
		return recall, precision, f1
	else:
		print('n is too big')

	


if __name__ == "__main__":
	reference1 = 'Nobu is so clever'
	system_pred1 = 'Nobu is so smart'

	hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of China '' to incite the subversion of state power , '' according to court documents given to his wife on Monday .\nWith attorneys locked up , harassed or plain scared , two prominent dissidents will defend themselves against charges of subversion Thursday in China 's highest-profile dissident trials in two years .\n"
	reference_2 = "Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\n"

	hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"



	reference_3 = "The cat was under the bed"
	system_pred_3 = "the cat was found under the bed"

	print(rouge(process_sentence(system_pred_3), process_sentence(reference_3), 2))