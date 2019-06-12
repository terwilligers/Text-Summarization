from collections import defaultdict

"""
This function tokenizes a string into a list of lowercased words.
@sentence: a string (representing the sentence)
"""
def process_sentence(sentence):
	return sentence.lower().split()

"""
This function creates a dictionary of n_grams mapped to their counts that we
use to count overlaps when calculating rouge scores.
@sentence_list: the list representation of the sentence
@n: n-gram
"""
def create_dict(sentence_list, n):
	new_list = [tuple(sentence_list[i:i+n]) for i in range(0, len(sentence_list) - n + 1)]
	count_dict = defaultdict(int)
	for n_gram in new_list:
		if n_gram not in count_dict:
			count_dict[n_gram] = 0
		count_dict[n_gram] += 1
	return count_dict

"""
This function calculates the number of overlapping n-grams in the reference
and system-generated summaries.
@system_pred: the list representation of the system generated summary
@reference: the list representation of the reference summary
"""
def find_overlap(system_pred, reference, n):
	system_dict = create_dict(system_pred, n)
	reference_dict = create_dict(reference, n)
	overlap = 0
	for n_gram in system_dict.keys():
		overlap += min(system_dict[n_gram], reference_dict[n_gram])
	return overlap

"""
This function avoids division by zero
"""
def safe_division(num, deno):
	if deno == 0:
		return None
	return num/deno

"""
@param system_pred: a list that represents the machine generated summary (You can use the process_sentence function
to generate the list from a string
@reference: a list that represents the human generated summary
@n: n-gram
@alpha: the parameter for calculating f1 score.
"""
def rouge(system_pred, reference, method = 2, alpha = 0.5):
	if method != 'l':
		overlap = find_overlap(system_pred, reference, method)
		if method <= min(len(system_pred), len(reference)):
			recall = overlap/(len(reference) - method + 1)
			precision = overlap/(len(system_pred) - method + 1)
			f1 = safe_division(recall * precision,(alpha * recall + (1- alpha) * precision))
			return recall, precision, f1
		else:
			print('n is too big')
	else:
		return rouge_l(system_pred, reference)

"""
This function calculates the longest common subsequnce between two sentences (in lists),
used when calculating the rouge-L score
@X: First list
@Y: Second list
"""
def lcs(X , Y):
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None]*(n+1) for i in range(m+1)]

    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j] , L[i][j-1])

    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]

"""
@param system_pred: a list that represents the machine generated summary (You can use the process_sentence function
to generate the list from a string
@reference: a list that represents the human generated summary
@alpha: the parameter for calculating f1 score.
"""
def rouge_l(system_pred, reference, alpha = 0.5):
	lcs_val = lcs(system_pred, reference)
	recall = lcs_val/len(reference)
	precision = lcs_val/len(system_pred)
	f1 = safe_division(recall * precision,(alpha * recall + (1- alpha) * precision))
	return recall, precision, f1


if __name__ == "__main__":
	reference1 = 'Nobu is so clever'
	system_pred1 = 'Nobu is so smart'

	hypothesis_2 = "China 's government said Thursday that two prominent dissidents arrested this week are suspected of endangering national security _ the clearest sign yet Chinese leaders plan to quash a would-be opposition party .\nOne leader of a suppressed new political party will be tried on Dec. 17 on a charge of colluding with foreign enemies of "
	reference_2 = "Hurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\nThe U.S. and "

	hypothesis_1 = "King Norodom Sihanouk has declined requests to chair a summit of Cambodia 's top political leaders , saying the meeting would not bring any progress in deadlocked negotiations to form a government .\nGovernment and opposition parties have asked King Norodom Sihanouk to host a summit meeting after a series of post-election negotiations between the two opposition groups and Hun Sen 's party to form a new government failed .\nHun Sen 's ruling party narrowly won a majority in elections in July , but the opposition _ claiming widespread intimidation and fraud _ has denied Hun Sen the two-thirds vote in parliament required to approve the next government .\n"

	reference_3 = "hhhhhhhhhhhh"
	system_pred_3 = "Thiis is hhhhhhhhhhhh"

	print(process_sentence(reference_2))

	print(lcs(process_sentence(hypothesis_2), process_sentence(reference_2)))

	#print(lcs(process_sentence(hypothesis_2), process_sentence(reference_2), len(process_sentence(hypothesis_2)), len(process_sentence(reference_2))))

	print(rouge_l(process_sentence(hypothesis_2), process_sentence(reference_2)))
