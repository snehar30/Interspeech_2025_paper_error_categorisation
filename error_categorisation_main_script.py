# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:44:52 2025

@author: sneha
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns
from collections import defaultdict
import enchant

# Edit distance code by Harsha
def editdistance(s, t):
	rows = len(s)+1
	cols = len(t)+1
	corr = 0
	dist = [[0 for x in range(cols)] for x in range(rows)]
	bt = defaultdict(lambda:0)
	for i in range(1, rows):
		dist[i][0] = i
		bt[(i,0)] = (i-1,0,'d')
	
	for i in range(1, cols):
		dist[0][i] = i
		bt[(0,i)] = (0,i-1,'i')

	correct_characters = []
	for col in range(1, cols):
		for row in range(1, rows):
			if s[row-1] == t[col-1]:
				cost = 0.0
				corr+= len(s[row-1])
				correct_characters.append(s[row-1])
                
			else:
				cost = 1.0
			dist[row][col] = min(dist[row-1][col] + 1.0,	  # deletion
								 dist[row][col-1] + 1.0,	  # insertion
								 dist[row-1][col-1] + cost) # substitution
			if dist[row][col] == dist[row-1][col]+1:
				bt[(row,col)] = (row-1,col,'d')
			elif dist[row][col] == dist[row][col-1]+1:
				bt[(row,col)] = (row,col-1,'i')
			elif dist[row][col] == dist[row-1][col-1]+1:
				bt[(row,col)] = (row-1,col-1,'s')
			else:
				bt[(row,col)] = (row-1,col-1,'c')

	row=rows-1
	col=cols-1
	d = defaultdict(lambda:[])
	d=[[] for i in range(rows)]
	path = []
	substituted_characters = []
	canonical_characters = []
	deleted_characters = []
	inserted_characters = []
    
	while((row,col) !=(0,0)):
		d[row].append(t[col-1])
		#print(d)
		(row,col,p) = bt[(row,col)]
        # substituted characters s[row] and t[col] when p is 's' 
		if p =='s':
			#print(s[row],t[col])
			canonical_characters.append(s[row])
			substituted_characters.append(t[col])

		if p =='d':
  			#print(s[row],t[col])
  			deleted_characters.append(s[row])
             
		if p =='i':
  			#print(s[row],t[col])
  			inserted_characters.append(t[col])
              
		path.insert(0,p)

	vector=[]
	if path[0]=='i':
		vector.append(1.)
	else:
		vector.append(0.)
	for i in path:
		if i=='c':
			vector.append(0.)
		elif i in ('d','s'):
			vector.append(1.)
		else:
			vector[-1] = 1.	

	return dist[-1][-1], path,vector,corr,correct_characters,canonical_characters,substituted_characters,inserted_characters,deleted_characters

devnagiri_characters=['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न','प','फ','ब','भ','म','क़','ख़','ग़','ज़','ड़','ढ़','फ़','य','र','ल','ळ','व','ह','श','ष','स','ऱ','ऴ','्','अ','आ','इ','ई','उ','ऊ','ा','ि','ी','ु','ू','ऋ','ॠ','ऌ','ॡ','ए','ऐ','ओ','औ','ृ','ॄ','ॢ','ॣ','े','ै','ो','ौ','◌॑','◌॒','॓','॔','ँ','ं','ः','़','।','॥','ऽ','॰','ॐ','०','१','२','३','४','५','६','७','८','९','ऍ','ऑ','ऎ','ऒ','ॅ','ॉ','ॆ','ॊ']

def check_if_has_devnagiri(word):
    word1 = word.replace("_", "")
    for x in word1:
        res = False
        if x in devnagiri_characters:
            res = True
            break
    return res

def check_if_devnagiri(word):
    word1 = word.replace("_", "")
    for x in word1:
        res = True
        if x not in devnagiri_characters:
            res = False
            break
    return res

def get_phone_sequence(word, lexicon_dict):

    phone_sequences = lexicon_dict.get(word)
    #print(phone_sequences)
    if phone_sequences == None:
        words = word.split('_')
        phone_sequence = [lexicon_dict.get(w, '') for w in words]
        phone_sequences = ' '.join(phone_sequence)
    if phone_sequences == None:    
        phone_sequences == ''

    return phone_sequences



languages = ['Hindi','English']
save_files = True
output_folder = "output_files/"
input_folder = "input_files/"
lexicon_folder = "lexicon_files/"

shabd_db_file = f"{lexicon_folder}SHABD_DATABASE.xlsx"
#excel_file_path = 'MT_simple_Hindi.xlsx'
shabd_database = pd.read_excel(shabd_db_file)


lihisto_db_file = f"{lexicon_folder}lihisto_stop_words.xlsx"
lihisto_database = pd.read_excel(lihisto_db_file)

more_hindi_words_file = f"{lexicon_folder}more_hindi_words.xlsx"
more_hindi_words_database = pd.read_excel(more_hindi_words_file)

more_english_words_file = f"{lexicon_folder}more_devnagiri_english_words.xlsx"
more_english_words_database = pd.read_excel(more_english_words_file)

transliterated_hindi_substitutions_file = f"{lexicon_folder}transliterated_hindi_substitutions.xlsx"
transliterated_hindi_substitutions = pd.read_excel(transliterated_hindi_substitutions_file)

transliterated_english_substitutions_file = f"{lexicon_folder}transliterated_english_substitutions.xlsx"
transliterated_english_substitutions = pd.read_excel(transliterated_english_substitutions_file)
   
# First character level    

character_study_table = pd.DataFrame(columns=['lang','filename','Word','Pronunciation','path','correct_characters','subsitution_cannonical','substituted_characters','substituted_character_pairs','inserted_characters','deleted_characters','word_pohoneme_sequence','pronunciation_phoneme_sequence','wav_2_vec_phoneme_sequence','path_p','correct_phonemes','subsitution_cannonical_phoneme_sequence','substituted_phonemes','substituted_phoneme_pairs','inserted_phonemes','deleted_phonemes','path_AT_MT','AT_MT_correct_phonemes','AT_MT_subsitution_cannonical_phoneme_sequence', 'AT_MT_substituted_phonemes', 'AT_MT_substituted_phoneme_pairs', 'AT_MT_inserted_phonemes', 'AT_MT_deleted_phonemes','path_AT_can','AT_canonical_correct_phonemes','AT_canonical_subsitution_cannonical_phoneme_sequence', 'AT_canonical_substituted_phonemes', 'AT_canonical_substituted_phoneme_pairs', 'AT_canonical_inserted_phonemes', 'AT_canonical_deleted_phonemes'])

word_study_table = pd.DataFrame(columns=['lang','filename','Word','Pronunciation','word_phoneme_length','Pronunciatin_phoneme_length','wav2vec_seq_length','W_NW_error','path','per_c','error_type_c','percentage_correct_characters','num_correct_characters','num_inserted_characters','num_deleted_characters','num_substituted_characters','word_phone_seq','pronunciation_phone_seq','path_p','per_p','error_type','error_type_with_similarity','percentage_correct_phonemes','num_correct_phonemes','num_inserted_phonemes','num_deleted_phonemes','num_substituted_phonemes','a',' c','path_AT_can','AT_canonical_per','error_type_AT','error_type_with_similarity_AT','AT_canonical_percentage_correct_phonemes','AT_canonical_num_correct_phonemes','AT_canonical_num_inserted_phonemes','AT_canonical_num_deleted_phonemes','AT_canonical_num_substituted_phonemes','b','c','path_AT_MT','AT_MT_per','AT_MT_percentage_correct_phonemes','AT_MT_num_correct_phonemes','AT_MT_num_inserted_phonemes','AT_MT_num_deleted_phonemes','AT_MT_num_substituted_phonemes'])

utterance_study_table = pd.DataFrame(columns=['lang','fn','student','correct_words','substituted_words','deleted_words','wcpm','accuracy','total_subs','word_non_word_errors','num_word_errors','num_non_word_errors','perc_word_errors','perc_nonword_errors','word_minus_non_word_errors','perc_word_minus_non_word_errors','error_type_MT','num_ip_errors_MT','num_fp_errors_MT','num_sc_errors_MT','num_so_errors_MT','num_ot_errors_MT','perc_ip_errors_MT','perc_fp_errors_MT','perc_sc_errors_MT','perc_so_errors_MT','perc_ot_errors_MT','num_ip_errors_MT_similar','num_fp_errors_MT_similar','num_sc_errors_MT_similar','num_so_errors_MT_similar','num_ip_errors_MT_dissimilar','num_fp_errors_MT_dissimilar','num_sc_errors_MT_dissimilar','num_so_errors_MT_dissimilar','num_ot_errors_MT_dissimilar','perc_ip_errors_MT_similar','perc_fp_errors_MT_similar','perc_sc_errors_MT_similar','perc_so_errors_MT_similar','perc_ip_errors_MT_dissimilar','perc_fp_errors_MT_dissimilar','perc_sc_errors_MT_dissimilar','perc_so_errors_MT_dissimilar','perc_ot_errors_MT_dissimilar','average_per_MT','total_phoneme_insertions_MT','phoneme_insertions_per_word_MT','total_phoneme_deletions_MT','phoneme_deletions_per_word_MT','total_phoneme_substitutions_MT','phoneme_substitutions_per_word_MT','error_type_AT','num_ip_errors_AT','num_fp_errors_AT','num_sc_errors_AT','num_so_errors_AT','num_ot_errors_AT','perc_ip_errors_AT','perc_fp_errors_AT','perc_sc_errors_AT','perc_so_errors_AT','perc_ot_errors_AT','num_ip_errors_AT_similar','num_fp_errors_AT_similar','num_sc_errors_AT_similar','num_so_errors_AT_similar','num_ip_errors_AT_dissimilar','num_fp_errors_AT_dissimilar','num_sc_errors_AT_dissimilar','num_so_errors_AT_dissimilar','num_ot_errors_AT_dissimilar','perc_ip_errors_AT_similar','perc_fp_errors_AT_similar','perc_sc_errors_AT_similar','perc_so_errors_AT_similar','perc_ip_errors_AT_dissimilar','perc_fp_errors_AT_dissimilar','perc_sc_errors_AT_dissimilar','perc_so_errors_AT_dissimilar','perc_ot_errors_AT_dissimilar','average_per_AT','total_phoneme_insertions_AT','phoneme_insertions_per_word_AT','total_phoneme_deletions_AT','phoneme_deletions_per_word_AT','total_phoneme_substitutions_AT','phoneme_substitutions_per_word_AT'])

for lang in languages:
    
    character_study_table_lang = pd.DataFrame(columns=['lang','filename','Word','Pronunciation','path','correct_characters','subsitution_cannonical','substituted_characters','substituted_character_pairs','inserted_characters','deleted_characters','word_pohoneme_sequence','pronunciation_phoneme_sequence','wav_2_vec_phoneme_sequence','path_p','correct_phonemes','subsitution_cannonical_phoneme_sequence','substituted_phonemes','substituted_phoneme_pairs','inserted_phonemes','deleted_phonemes','path_AT_MT','AT_MT_correct_phonemes','AT_MT_subsitution_cannonical_phoneme_sequence', 'AT_MT_substituted_phonemes', 'AT_MT_substituted_phoneme_pairs', 'AT_MT_inserted_phonemes', 'AT_MT_deleted_phonemes','path_AT_can','AT_canonical_correct_phonemes','AT_canonical_subsitution_cannonical_phoneme_sequence', 'AT_canonical_substituted_phonemes', 'AT_canonical_substituted_phoneme_pairs', 'AT_canonical_inserted_phonemes', 'AT_canonical_deleted_phonemes'])

    word_study_table_lang = pd.DataFrame(columns=['lang','filename','Word','Pronunciation','word_phoneme_length','Pronunciatin_phoneme_length','wav2vec_seq_length','W_NW_error','path','per_c','error_type_c','percentage_correct_characters','num_correct_characters','num_inserted_characters','num_deleted_characters','num_substituted_characters','word_phone_seq','pronunciation_phone_seq','path_p','per_p','error_type','error_type_with_similarity','percentage_correct_phonemes','num_correct_phonemes','num_inserted_phonemes','num_deleted_phonemes','num_substituted_phonemes','a',' c','path_AT_can','AT_canonical_per','error_type_AT','error_type_with_similarity_AT','AT_canonical_percentage_correct_phonemes','AT_canonical_num_correct_phonemes','AT_canonical_num_inserted_phonemes','AT_canonical_num_deleted_phonemes','AT_canonical_num_substituted_phonemes','b','c','path_AT_MT','AT_MT_per','AT_MT_percentage_correct_phonemes','AT_MT_num_correct_phonemes','AT_MT_num_inserted_phonemes','AT_MT_num_deleted_phonemes','AT_MT_num_substituted_phonemes'])

    utterance_study_table_lang = pd.DataFrame(columns=['lang','fn','student','correct_words','substituted_words','deleted_words','wcpm','accuracy','total_subs','word_non_word_errors','num_word_errors','num_non_word_errors','perc_word_errors','perc_nonword_errors','word_minus_non_word_errors','perc_word_minus_non_word_errors','error_type_MT','num_ip_errors_MT','num_fp_errors_MT','num_sc_errors_MT','num_so_errors_MT','num_ot_errors_MT','perc_ip_errors_MT','perc_fp_errors_MT','perc_sc_errors_MT','perc_so_errors_MT','perc_ot_errors_MT','num_ip_errors_MT_similar','num_fp_errors_MT_similar','num_sc_errors_MT_similar','num_so_errors_MT_similar','num_ip_errors_MT_dissimilar','num_fp_errors_MT_dissimilar','num_sc_errors_MT_dissimilar','num_so_errors_MT_dissimilar','num_ot_errors_MT_dissimilar','perc_ip_errors_MT_similar','perc_fp_errors_MT_similar','perc_sc_errors_MT_similar','perc_so_errors_MT_similar','perc_ip_errors_MT_dissimilar','perc_fp_errors_MT_dissimilar','perc_sc_errors_MT_dissimilar','perc_so_errors_MT_dissimilar','perc_ot_errors_MT_dissimilar','average_per_MT','total_phoneme_insertions_MT','phoneme_insertions_per_word_MT','total_phoneme_deletions_MT','phoneme_deletions_per_word_MT','total_phoneme_substitutions_MT','phoneme_substitutions_per_word_MT','error_type_AT','num_ip_errors_AT','num_fp_errors_AT','num_sc_errors_AT','num_so_errors_AT','num_ot_errors_AT','perc_ip_errors_AT','perc_fp_errors_AT','perc_sc_errors_AT','perc_so_errors_AT','perc_ot_errors_AT','num_ip_errors_AT_similar','num_fp_errors_AT_similar','num_sc_errors_AT_similar','num_so_errors_AT_similar','num_ip_errors_AT_dissimilar','num_fp_errors_AT_dissimilar','num_sc_errors_AT_dissimilar','num_so_errors_AT_dissimilar','num_ot_errors_AT_dissimilar','perc_ip_errors_AT_similar','perc_fp_errors_AT_similar','perc_sc_errors_AT_similar','perc_so_errors_AT_similar','perc_ip_errors_AT_dissimilar','perc_fp_errors_AT_dissimilar','perc_sc_errors_AT_dissimilar','perc_so_errors_AT_dissimilar','perc_ot_errors_AT_dissimilar','average_per_AT','total_phoneme_insertions_AT','phoneme_insertions_per_word_AT','total_phoneme_deletions_AT','phoneme_deletions_per_word_AT','total_phoneme_substitutions_AT','phoneme_substitutions_per_word_AT'])

     
    master_file_path = f"{input_folder}{lang}_master_file_all.xlsx"
    master_df = pd.read_excel(master_file_path)
    
    
    excel_file_path =  f"{input_folder}{lang}_master_file_single_story_only_s.xlsx"
    df = pd.read_excel(excel_file_path)
    
    canonical_words=df.MT_Canonical.tolist()
    filename = df.Utterance.tolist()
    mt_words=df.MT_Word.tolist()
    AT_phone_sequence = df.w2v_phone_sequence.to_list()
        
    lexicon_file_path = f"{lexicon_folder}{lang}_Lexicon_file_IS2025.txt"
    
    lexicon_dict = {}
    with open(lexicon_file_path, 'r', encoding='utf-8') as lex_file:
        for line in lex_file:
            word, phone_sequence = line.strip().split('\t')
            lexicon_dict[word] = phone_sequence
    
    for i in range(len(canonical_words)):
        
        word_ = canonical_words[i]
        word = canonical_words[i]
        w1_ = mt_words[i]
        w1 = mt_words[i]
        
        w = w1.replace("_", "")
        
        if lang == 'Hindi':
            word_found_in_shabd = len(shabd_database[shabd_database['word'] == w1])
            word_found_in_lihisto = len(lihisto_database[lihisto_database['word'] == w1])
            word_found_in_more_hindi_words = len(more_hindi_words_database[more_hindi_words_database['word'] == w1])            
            
            
            if word_found_in_shabd>0 or word_found_in_lihisto>0 or word_found_in_more_hindi_words>0:
                W_NW_error="word"
            else:
                W_NW_error="nonword"
                        
        if lang == 'English':         
            word_found_in_more_english_words = len(more_english_words_database[more_english_words_database['word'] == w1])   
            dic = enchant.Dict("en_US")
            if dic.check(w1) or word_found_in_more_english_words>0:
                W_NW_error="word"
            else:
                W_NW_error="nonword"
            
    
        a=get_phone_sequence(word, lexicon_dict)
        a1_split = a.split(' ')    
        b=get_phone_sequence(w1, lexicon_dict)
        b1_split = b.split(' ')
        
        a_split = [i for i in a1_split if i != '']
        b_split = [i for i in b1_split if i != '']
        
        if len(b_split)>1 and lang == 'English':
            if b_split[-2]+b_split[-1]=="NG":
                b_split[-2]="NG"
                b_split=b_split[:-1]
                b=b[:-2]+b[-1]
     
            if a_split[-1] == "Z" and b_split[-1] == "S":
                b_split[-1] = "Z"
                b=b[:-1]+"Z"
                
            if a_split[-1] == "S" and b_split[-1] == "Z":
                b_split[-1] = "S"
                b=b[:-1]+"S"
        
        c = AT_phone_sequence[i]
        
        c1_split = c.split(' ') 
        #c_split = c1_split.copy()
        c_split = [i for i in c1_split if (i != '*' and i != 'SIL' and i != '' and i != ' ')]
        # if '*' in c_split:
        #     c_split.remove('*')
     
        fn=filename[i]
               
        e = editdistance(word, w)
         
        if lang == 'English' and (check_if_has_devnagiri(w1)):
            w1_tr = transliterated_english_substitutions[transliterated_english_substitutions['MT'] == w1]['transliteration'].tolist()[0]
            w_tr = w1_tr.replace("_", "")
            e = editdistance(word, w_tr)
            w = w_tr
            #print ('eng tr '+ w1 )
            
            
        
        if lang == 'Hindi' and check_if_devnagiri(w1) == False:
            w1_tr = transliterated_hindi_substitutions[transliterated_hindi_substitutions['MT'] == w1]['transliteration'].tolist()[0]
            w_tr = w1_tr.replace("_", "")
            e = editdistance(word, w_tr)
            w = w_tr
            #print ('hin tr '+ w1 )
            
        path = e[1]  
                         
        correct_characters = e[4]
        subsitution_cannonical = e[5]
        substituted_characters = e[6]
        substituted_character_pairs = [' '.join(x) for x in zip(e[5],e[6])] 
        inserted_characters = e[7]
        deleted_characters = e[8]
        
        percentage_correct_characters = 100*path.count('c')/len(word)
        num_correct_characters = path.count('c')
        num_inserted_characters = path.count('i')
        num_deleted_characters = path.count('d')
        num_substituted_characters = path.count('s')
        per_c = 100*(num_inserted_characters + num_deleted_characters + num_substituted_characters)/len(word)
        
        if len(word)>2 and len(w)>2:
        
            if (path[0] == 'c' and path[-1]!='c'):
                error_type_c = 'initial_correct'
            elif (path[0] != 'c' and path[-1] =='c'):
                error_type_c = 'Final_correct'
            elif (path[0] == 'c' and path[-1] =='c'):
                error_type_c = 'scaffolding_error'
            elif path.count('c')>0:
                error_type_c = 'some_overlap'
            else: 
                error_type_c = 'other'
        else:
            error_type_c = ''
       
               
        if len(b)>0:
            f = editdistance(a_split, b_split)
            path_p = f[1]
            correct_phonemes = f[4]
            subsitution_cannonical_phoneme_sequence = f[5]
            substituted_phonemes = f[6]
            substituted_phoneme_pairs = [' '.join(x) for x in zip(f[5],f[6])] 
            inserted_phonemes = f[7]
            deleted_phonemes = f[8]          
            
                        
            percentage_correct_phonemes = 100*path_p.count('c')/len(a_split)            
            num_correct_phonemes = path_p.count('c')
            num_inserted_phonemes = path_p.count('i')
            num_deleted_phonemes = path_p.count('d')
            num_substituted_phonemes = path_p.count('s')    
            per_p = 100*(num_inserted_phonemes + num_deleted_phonemes + num_substituted_phonemes)/len(a_split)
            
            
            if len(a_split)>2 and len(b_split)>2:

                if (path_p[0] == 'c' and path_p[-1]!='c'):
                    error_type = 'initial_correct'
                elif (path_p[0] != 'c' and path_p[-1] =='c'):
                    error_type = 'Final_correct'
                elif (path_p[0] == 'c' and path_p[-1] =='c'):
                    error_type = 'scaffolding_error'
                elif path_p.count('c')>0:
                    error_type = 'some_overlap'
                else: 
                    error_type = 'other'
                    
                
                if percentage_correct_phonemes > 50:
                    error_type_with_similarity = error_type + '_similar'
                else:
                    error_type_with_similarity = error_type + '_dissimilar'
        
            else:
                error_type = ''
                error_type_with_similarity =''
            
            
        else:
            path_p = []
            correct_phonemes = []
            subsitution_cannonical_phoneme_sequence = []
            substituted_phonemes = []
            substituted_phoneme_pairs = []
            inserted_phonemes = []
            deleted_phonemes = []
            
            percentage_correct_phonemes =''
            num_correct_phonemes = np.nan
            num_inserted_phonemes = np.nan
            num_deleted_phonemes = np.nan
            num_substituted_phonemes = np.nan
            per_p = np.nan

            error_type = ''
            error_type_with_similarity =''    

        


        
        if len(b)>0 and len(c_split)>0:
            g = editdistance(b_split, c_split)
            path_AT_MT = g[1]
            AT_MT_correct_phonemes = g[4]
            AT_MT_subsitution_cannonical_phoneme_sequence = g[5]
            AT_MT_substituted_phonemes = g[6]
            AT_MT_substituted_phoneme_pairs = [' '.join(x) for x in zip(g[5],g[6])] 
            AT_MT_inserted_phonemes = g[7]
            AT_MT_deleted_phonemes = g[8]
        
            
            AT_MT_percentage_correct_phonemes = 100*path_AT_MT.count('c')/len(a_split)            
            AT_MT_num_correct_phonemes = path_AT_MT.count('c')
            AT_MT_num_inserted_phonemes = path_AT_MT.count('i')
            AT_MT_num_deleted_phonemes = path_AT_MT.count('d')
            AT_MT_num_substituted_phonemes = path_AT_MT.count('s') 
            AT_MT_per = 100*(AT_MT_num_inserted_phonemes + AT_MT_num_deleted_phonemes + AT_MT_num_substituted_phonemes)/len(a_split)
            
        else:
            path_AT_MT =[]
            AT_MT_correct_phonemes = []
            AT_MT_subsitution_cannonical_phoneme_sequence = []
            AT_MT_substituted_phonemes = []
            AT_MT_substituted_phoneme_pairs = []
            AT_MT_inserted_phonemes = []
            AT_MT_deleted_phonemes = []
            
            AT_MT_percentage_correct_phonemes =''
            AT_MT_num_correct_phonemes = ''
            AT_MT_num_inserted_phonemes = ''
            AT_MT_num_deleted_phonemes = ''
            AT_MT_num_substituted_phonemes = ''
            AT_MT_per = ''             
            
            
            
            
        if len(b)>0 and len(c_split)>0:
            h = editdistance(a_split, c_split)
            path_AT_can = h[1]
            AT_canonical_correct_phonemes = h[4]
            AT_canonical_subsitution_cannonical_phoneme_sequence = h[5]
            AT_canonical_substituted_phonemes = h[6]
            AT_canonical_substituted_phoneme_pairs = [' '.join(x) for x in zip(h[5],h[6])] 
            AT_canonical_inserted_phonemes = h[7]
            AT_canonical_deleted_phonemes = h[8]
            
            
                       
            AT_canonical_percentage_correct_phonemes = 100*path_AT_can.count('c')/len(a_split)            
            AT_canonical_num_correct_phonemes = path_AT_can.count('c')
            AT_canonical_num_inserted_phonemes = path_AT_can.count('i')
            AT_canonical_num_deleted_phonemes = path_AT_can.count('d')
            AT_canonical_num_substituted_phonemes = path_AT_can.count('s') 
            AT_canonical_per = 100*(AT_canonical_num_inserted_phonemes + AT_canonical_num_deleted_phonemes + AT_canonical_num_substituted_phonemes)/len(a_split)
            
            
            
            if len(a_split)>2 and len(c_split)>2: 
                
                if (path_AT_can[0] == 'c' and path_AT_can[-1]!='c'):
                    error_type_AT = 'initial_correct'
                elif (path_AT_can[0] != 'c' and path_AT_can[-1] =='c'):
                    error_type_AT = 'Final_correct'
                elif (path_AT_can[0] == 'c' and path_AT_can[-1] =='c'):
                    error_type_AT = 'scaffolding_error'
                elif path_AT_can.count('c')>0:
                    error_type_AT = 'some_overlap'
                else: 
                    error_type_AT = 'other'
                    
                if AT_canonical_percentage_correct_phonemes > 50:
                    error_type_with_similarity_AT = error_type_AT + '_similar'
                else:
                    error_type_with_similarity_AT = error_type_AT + '_dissimilar'
        
            else:
                error_type_AT = ''
                error_type_with_similarity_AT = ''
                     
            
            
        else:
            path_AT_can = []
            AT_canonical_correct_phonemes = []
            AT_canonical_subsitution_cannonical_phoneme_sequence = []
            AT_canonical_substituted_phonemes = []
            AT_canonical_substituted_phoneme_pairs = []
            AT_canonical_inserted_phonemes = []
            AT_canonical_deleted_phonemes = []  
            
            AT_canonical_percentage_correct_phonemes =''
            AT_canonical_num_correct_phonemes = np.nan
            AT_canonical_num_inserted_phonemes = np.nan
            AT_canonical_num_deleted_phonemes = np.nan
            AT_canonical_num_substituted_phonemes = np.nan
            AT_canonical_per = np.nan 
            
            error_type_AT = ''
            error_type_with_similarity_AT = ''           
            
            
            
            
        num_of_rows = max(len(correct_characters), len(subsitution_cannonical), len(inserted_characters), len(deleted_characters), len(correct_phonemes), len(subsitution_cannonical_phoneme_sequence), len(inserted_phonemes), len(deleted_phonemes),len(AT_MT_correct_phonemes), len(AT_MT_subsitution_cannonical_phoneme_sequence), len(AT_MT_inserted_phonemes), len(AT_MT_deleted_phonemes))
        
        correct_characters += [''] * (num_of_rows - len(correct_characters))
        subsitution_cannonical += [''] * (num_of_rows - len(subsitution_cannonical))
        substituted_characters += [''] * (num_of_rows - len(substituted_characters)) 
        substituted_character_pairs += [''] * (num_of_rows - len(substituted_character_pairs)) 
        inserted_characters += [''] * (num_of_rows - len(inserted_characters))
        deleted_characters += [''] * (num_of_rows - len(deleted_characters)) 
      
        
        correct_phonemes += [''] * (num_of_rows - len(correct_phonemes))
        subsitution_cannonical_phoneme_sequence += [''] * (num_of_rows - len(subsitution_cannonical_phoneme_sequence))
        substituted_phonemes += [''] * (num_of_rows - len(substituted_phonemes)) 
        substituted_phoneme_pairs += [''] * (num_of_rows - len(substituted_phoneme_pairs)) 
        inserted_phonemes += [''] * (num_of_rows - len(inserted_phonemes))
        deleted_phonemes += [''] * (num_of_rows - len(deleted_phonemes)) 
        
        AT_MT_correct_phonemes += [''] * (num_of_rows - len(AT_MT_correct_phonemes))
        AT_MT_subsitution_cannonical_phoneme_sequence += [''] * (num_of_rows - len(AT_MT_subsitution_cannonical_phoneme_sequence))
        AT_MT_substituted_phonemes += [''] * (num_of_rows - len(AT_MT_substituted_phonemes)) 
        AT_MT_substituted_phoneme_pairs += [''] * (num_of_rows - len(AT_MT_substituted_phoneme_pairs)) 
        AT_MT_inserted_phonemes += [''] * (num_of_rows - len(AT_MT_inserted_phonemes))
        AT_MT_deleted_phonemes += [''] * (num_of_rows - len(AT_MT_deleted_phonemes)) 
        
        AT_canonical_correct_phonemes += [''] * (num_of_rows - len(AT_canonical_correct_phonemes))
        AT_canonical_subsitution_cannonical_phoneme_sequence += [''] * (num_of_rows - len(AT_canonical_subsitution_cannonical_phoneme_sequence))
        AT_canonical_substituted_phonemes += [''] * (num_of_rows - len(AT_canonical_substituted_phonemes)) 
        AT_canonical_substituted_phoneme_pairs += [''] * (num_of_rows - len(AT_canonical_substituted_phoneme_pairs)) 
        AT_canonical_inserted_phonemes += [''] * (num_of_rows - len(AT_canonical_inserted_phonemes))
        AT_canonical_deleted_phonemes += [''] * (num_of_rows - len(AT_canonical_deleted_phonemes)) 

            
        for n in range(num_of_rows):
            
            d = [lang,fn, word_, w1_, path,correct_characters[n], subsitution_cannonical[n], substituted_characters[n], substituted_character_pairs[n], inserted_characters[n], deleted_characters[n],a,b,c,path_p,correct_phonemes[n], subsitution_cannonical_phoneme_sequence[n], substituted_phonemes[n], substituted_phoneme_pairs[n], inserted_phonemes[n], deleted_phonemes[n],path_AT_MT,AT_MT_correct_phonemes[n], AT_MT_subsitution_cannonical_phoneme_sequence[n], AT_MT_substituted_phonemes[n], AT_MT_substituted_phoneme_pairs[n], AT_MT_inserted_phonemes[n], AT_MT_deleted_phonemes[n],path_AT_can,AT_canonical_correct_phonemes[n], AT_canonical_subsitution_cannonical_phoneme_sequence[n], AT_canonical_substituted_phonemes[n], AT_canonical_substituted_phoneme_pairs[n], AT_canonical_inserted_phonemes[n], AT_canonical_deleted_phonemes[n]]
            character_study_table.loc[len(character_study_table.index)] = d            
            character_study_table_lang.loc[len(character_study_table_lang.index)] = d            
            
        
            
    
        sp_count = character_study_table['substituted_phoneme_pairs'].value_counts()
        AT_sp_count = character_study_table['AT_canonical_substituted_phoneme_pairs'].value_counts()
        
        i_count = character_study_table['inserted_phonemes'].value_counts()
        AT_i_count = character_study_table['AT_canonical_inserted_phonemes'].value_counts()
        
        d_count = character_study_table['deleted_phonemes'].value_counts()
        AT_d_count = character_study_table['AT_canonical_deleted_phonemes'].value_counts()
        
        s_count = character_study_table['AT_MT_subsitution_cannonical_phoneme_sequence'].value_counts()
        AT_s_count = character_study_table['AT_canonical_subsitution_cannonical_phoneme_sequence'].value_counts()
    
        dw = [lang,fn, word_, w1_,len(a_split),len(b_split),len(c_split),W_NW_error, path, per_c, error_type_c, percentage_correct_characters,num_correct_characters, num_inserted_characters, num_deleted_characters, num_substituted_characters, a, b, path_p, per_p, error_type, error_type_with_similarity, percentage_correct_phonemes,num_correct_phonemes, num_inserted_phonemes, num_deleted_phonemes, num_substituted_phonemes,a, c,path_AT_can, AT_canonical_per,error_type_AT,error_type_with_similarity_AT, AT_canonical_percentage_correct_phonemes,AT_canonical_num_correct_phonemes, AT_canonical_num_inserted_phonemes, AT_canonical_num_deleted_phonemes, AT_canonical_num_substituted_phonemes, b, c,path_AT_MT, AT_MT_per, AT_MT_percentage_correct_phonemes,AT_MT_num_correct_phonemes, AT_MT_num_inserted_phonemes, AT_MT_num_deleted_phonemes, AT_MT_num_substituted_phonemes]
        word_study_table.loc[len(word_study_table.index)] = dw
    
        word_study_table_lang.loc[len(word_study_table_lang.index)] = dw
    
    
    
    if (save_files): 
        character_study_table_lang.to_excel(f"{output_folder}{lang}_character_study_table.xlsx")    
        word_study_table_lang.to_excel(f"{output_folder}{lang}_word_study_table.xlsx")    
    
        
    filenames = word_study_table_lang['filename'].drop_duplicates().to_list()
    
    
    for fn in filenames:
        student = fn[:-33]
        
        start_time = min(master_df[master_df['Utterance'] == fn]['MT_Start'].to_list())
        end_time = max(master_df[master_df['Utterance'] == fn]['MT_End'].to_list())
        correct_words = len(master_df[(master_df['Utterance'] == fn) & (master_df['MT_Label'] == 'c')])
        substituted_words = len(master_df[(master_df['Utterance'] == fn) & (master_df['MT_Label'] == 's')])
        deleted_words = len(master_df[(master_df['Utterance'] == fn) & (master_df['MT_Label'] == 'd')])
        canonical_WC = {'234_1':65, '234_2':61, '002_1':68, '002_2':65}
        canonical_words = canonical_WC[fn[-5:]]
        accuracy = 100*correct_words/canonical_words
        wcpm = round(60 * correct_words/(end_time - start_time))
        
        
        total_subs = len(word_study_table_lang[word_study_table_lang['filename'] == fn])
        
        word_non_word_errors = {i: len(word_study_table_lang[(word_study_table_lang['W_NW_error'] == i) & (word_study_table_lang['filename'] == fn)]) for i in word_study_table_lang['W_NW_error'].unique()}
        
        num_word_errors = word_non_word_errors['word']
        num_non_word_errors = word_non_word_errors['nonword']
        
        perc_word_errors = 100 * word_non_word_errors['word'] / total_subs
        perc_nonword_errors = 100 * word_non_word_errors['nonword'] / total_subs
        
        #ratio_word_to_non_word_errors = num_word_errors/num_non_word_errors
        word_minus_non_word_errors = num_word_errors - num_non_word_errors
        perc_word_minus_non_word_errors = perc_word_errors - perc_nonword_errors
        
        error_type_MT = {i: len(word_study_table_lang[(word_study_table_lang['error_type'] == i) & (word_study_table_lang['filename'] == fn)]) for i in word_study_table_lang['error_type'].unique()}
        
        num_ip_errors_MT = error_type_MT['initial_correct']
        num_fp_errors_MT = error_type_MT['Final_correct']
        num_sc_errors_MT = error_type_MT['scaffolding_error']
        num_so_errors_MT = error_type_MT['some_overlap']
        num_ot_errors_MT = error_type_MT['other']

        
        
        perc_ip_errors_MT = error_type_MT['initial_correct']*100/total_subs
        perc_fp_errors_MT = error_type_MT['Final_correct']*100/total_subs
        perc_sc_errors_MT = error_type_MT['scaffolding_error']*100/total_subs
        perc_so_errors_MT = error_type_MT['some_overlap']*100/total_subs
        perc_ot_errors_MT = error_type_MT['other']*100/total_subs


        error_type_MT_with_similarity = {i: len(word_study_table_lang[(word_study_table_lang['error_type_with_similarity'] == i) & (word_study_table_lang['filename'] == fn)]) for i in word_study_table_lang['error_type_with_similarity'].unique()}
                 

        num_ip_errors_MT_similar = error_type_MT_with_similarity['initial_correct_similar']
        num_fp_errors_MT_similar = error_type_MT_with_similarity['Final_correct_similar']
        num_sc_errors_MT_similar = error_type_MT_with_similarity['scaffolding_error_similar']
        num_so_errors_MT_similar = error_type_MT_with_similarity['some_overlap_similar']
        num_ip_errors_MT_dissimilar = error_type_MT_with_similarity['initial_correct_dissimilar']
        num_fp_errors_MT_dissimilar = error_type_MT_with_similarity['Final_correct_dissimilar']
        num_sc_errors_MT_dissimilar = error_type_MT_with_similarity['scaffolding_error_dissimilar']
        num_so_errors_MT_dissimilar = error_type_MT_with_similarity['some_overlap_dissimilar']
        num_ot_errors_MT_dissimilar = error_type_MT_with_similarity['other_dissimilar']
        
        perc_ip_errors_MT_similar = error_type_MT_with_similarity['initial_correct_similar']*100/total_subs
        perc_fp_errors_MT_similar = error_type_MT_with_similarity['Final_correct_similar']*100/total_subs
        perc_sc_errors_MT_similar = error_type_MT_with_similarity['scaffolding_error_similar']*100/total_subs
        perc_so_errors_MT_similar = error_type_MT_with_similarity['some_overlap_similar']*100/total_subs
        perc_ip_errors_MT_dissimilar = error_type_MT_with_similarity['initial_correct_dissimilar']*100/total_subs
        perc_fp_errors_MT_dissimilar = error_type_MT_with_similarity['Final_correct_dissimilar']*100/total_subs
        perc_sc_errors_MT_dissimilar = error_type_MT_with_similarity['scaffolding_error_dissimilar']*100/total_subs
        perc_so_errors_MT_dissimilar = error_type_MT_with_similarity['some_overlap_dissimilar']*100/total_subs
        perc_ot_errors_MT_dissimilar = error_type_MT_with_similarity['other_dissimilar']*100/total_subs
                
             
        average_per_MT = word_study_table_lang[word_study_table_lang['filename'] == fn]['per_p'].mean()
        
        total_phoneme_insertions_MT = word_study_table_lang[word_study_table_lang['filename'] == fn]['num_inserted_phonemes'].sum()
        phoneme_insertions_per_word_MT = word_study_table_lang[word_study_table_lang['filename'] == fn]['num_inserted_phonemes'].mean()
        
        total_phoneme_deletions_MT = word_study_table_lang[word_study_table_lang['filename'] == fn]['num_deleted_phonemes'].sum()
        phoneme_deletions_per_word_MT = word_study_table_lang[word_study_table_lang['filename'] == fn]['num_deleted_phonemes'].mean()
    
        total_phoneme_substitutions_MT = word_study_table_lang[word_study_table_lang['filename'] == fn]['num_substituted_phonemes'].sum()
        phoneme_substitutions_per_word_MT = word_study_table_lang[word_study_table_lang['filename'] == fn]['num_substituted_phonemes'].mean()   	     
    
    
    
        error_type_AT = {i: len(word_study_table_lang[(word_study_table_lang['error_type_AT'] == i) & (word_study_table_lang['filename'] == fn)]) for i in word_study_table_lang['error_type_AT'].unique()}
        
         
        num_ip_errors_AT = error_type_AT['initial_correct']
        num_fp_errors_AT = error_type_AT['Final_correct']
        num_sc_errors_AT = error_type_AT['scaffolding_error']
        num_so_errors_AT = error_type_AT['some_overlap']
        num_ot_errors_AT = error_type_AT['other']

        perc_ip_errors_AT = error_type_AT['initial_correct']*100/total_subs
        perc_fp_errors_AT = error_type_AT['Final_correct']*100/total_subs
        perc_sc_errors_AT = error_type_AT['scaffolding_error']*100/total_subs
        perc_so_errors_AT = error_type_AT['some_overlap']*100/total_subs
        perc_ot_errors_AT = error_type_AT['other']*100/total_subs


        error_type_AT_with_similarity = {i: len(word_study_table_lang[(word_study_table_lang['error_type_with_similarity_AT'] == i) & (word_study_table_lang['filename'] == fn)]) for i in word_study_table_lang['error_type_with_similarity_AT'].unique()}
 
        num_ip_errors_AT_similar = error_type_AT_with_similarity['initial_correct_similar']
        num_fp_errors_AT_similar = error_type_AT_with_similarity['Final_correct_similar']
        num_sc_errors_AT_similar = error_type_AT_with_similarity['scaffolding_error_similar']
        num_so_errors_AT_similar = error_type_AT_with_similarity['some_overlap_similar']
        num_ip_errors_AT_dissimilar = error_type_AT_with_similarity['initial_correct_dissimilar']
        num_fp_errors_AT_dissimilar = error_type_AT_with_similarity['Final_correct_dissimilar']
        num_sc_errors_AT_dissimilar = error_type_AT_with_similarity['scaffolding_error_dissimilar']
        num_so_errors_AT_dissimilar = error_type_AT_with_similarity['some_overlap_dissimilar']
        num_ot_errors_AT_dissimilar = error_type_AT_with_similarity['other_dissimilar']
        
        perc_ip_errors_AT_similar = error_type_AT_with_similarity['initial_correct_similar']*100/total_subs
        perc_fp_errors_AT_similar = error_type_AT_with_similarity['Final_correct_similar']*100/total_subs
        perc_sc_errors_AT_similar = error_type_AT_with_similarity['scaffolding_error_similar']*100/total_subs
        perc_so_errors_AT_similar = error_type_AT_with_similarity['some_overlap_similar']*100/total_subs
        perc_ip_errors_AT_dissimilar = error_type_AT_with_similarity['initial_correct_dissimilar']*100/total_subs
        perc_fp_errors_AT_dissimilar = error_type_AT_with_similarity['Final_correct_dissimilar']*100/total_subs
        perc_sc_errors_AT_dissimilar = error_type_AT_with_similarity['scaffolding_error_dissimilar']*100/total_subs
        perc_so_errors_AT_dissimilar = error_type_AT_with_similarity['some_overlap_dissimilar']*100/total_subs
        perc_ot_errors_AT_dissimilar = error_type_AT_with_similarity['other_dissimilar']*100/total_subs


        print(fn)
        average_per_AT = word_study_table_lang[word_study_table_lang['filename'] == fn]['AT_canonical_per'].mean()
        
        total_phoneme_insertions_AT = word_study_table_lang[word_study_table_lang['filename'] == fn]['AT_canonical_num_inserted_phonemes'].sum()
        phoneme_insertions_per_word_AT = word_study_table_lang[word_study_table_lang['filename'] == fn]['AT_canonical_num_inserted_phonemes'].mean()
        
        total_phoneme_deletions_AT = word_study_table_lang[word_study_table_lang['filename'] == fn]['AT_canonical_num_deleted_phonemes'].sum()
        phoneme_deletions_per_word_AT = word_study_table_lang[word_study_table_lang['filename'] == fn]['AT_canonical_num_deleted_phonemes'].mean()
    
        total_phoneme_substitutions_AT = word_study_table_lang[word_study_table_lang['filename'] == fn]['AT_canonical_num_substituted_phonemes'].sum()
        phoneme_substitutions_per_word_AT = word_study_table_lang[word_study_table_lang['filename'] == fn]['AT_canonical_num_substituted_phonemes'].mean()  
        
        d = [lang,fn,student,correct_words,substituted_words,deleted_words,wcpm,accuracy,total_subs,word_non_word_errors,num_word_errors,num_non_word_errors,perc_word_errors,perc_nonword_errors,word_minus_non_word_errors,perc_word_minus_non_word_errors,error_type_MT,num_ip_errors_MT,num_fp_errors_MT,num_sc_errors_MT,num_so_errors_MT,num_ot_errors_MT,perc_ip_errors_MT,perc_fp_errors_MT,perc_sc_errors_MT,perc_so_errors_MT,perc_ot_errors_MT,num_ip_errors_MT_similar,num_fp_errors_MT_similar,num_sc_errors_MT_similar,num_so_errors_MT_similar,num_ip_errors_MT_dissimilar,num_fp_errors_MT_dissimilar,num_sc_errors_MT_dissimilar,num_so_errors_MT_dissimilar,num_ot_errors_MT_dissimilar,perc_ip_errors_MT_similar,perc_fp_errors_MT_similar,perc_sc_errors_MT_similar,perc_so_errors_MT_similar,perc_ip_errors_MT_dissimilar,perc_fp_errors_MT_dissimilar,perc_sc_errors_MT_dissimilar,perc_so_errors_MT_dissimilar,perc_ot_errors_MT_dissimilar,average_per_MT,total_phoneme_insertions_MT,phoneme_insertions_per_word_MT,total_phoneme_deletions_MT,phoneme_deletions_per_word_MT,total_phoneme_substitutions_MT,phoneme_substitutions_per_word_MT,error_type_AT,num_ip_errors_AT,num_fp_errors_AT,num_sc_errors_AT,num_so_errors_AT,num_ot_errors_AT,perc_ip_errors_AT,perc_fp_errors_AT,perc_sc_errors_AT,perc_so_errors_AT,perc_ot_errors_AT,num_ip_errors_AT_similar,num_fp_errors_AT_similar,num_sc_errors_AT_similar,num_so_errors_AT_similar,num_ip_errors_AT_dissimilar,num_fp_errors_AT_dissimilar,num_sc_errors_AT_dissimilar,num_so_errors_AT_dissimilar,num_ot_errors_AT_dissimilar,perc_ip_errors_AT_similar,perc_fp_errors_AT_similar,perc_sc_errors_AT_similar,perc_so_errors_AT_similar,perc_ip_errors_AT_dissimilar,perc_fp_errors_AT_dissimilar,perc_sc_errors_AT_dissimilar,perc_so_errors_AT_dissimilar,perc_ot_errors_AT_dissimilar,average_per_AT,total_phoneme_insertions_AT,phoneme_insertions_per_word_AT,total_phoneme_deletions_AT,phoneme_deletions_per_word_AT,total_phoneme_substitutions_AT,phoneme_substitutions_per_word_AT]
        utterance_study_table.loc[len(utterance_study_table)] = d
        utterance_study_table_lang.loc[len(utterance_study_table_lang)] = d
    
    if (save_files): 
        utterance_study_table_lang.to_excel(f"{output_folder}{lang}_utterance_study_table.xlsx")
        
if (save_files): 
    character_study_table.to_excel(f"{output_folder}character_study_table.xlsx")    
    word_study_table.to_excel(f"{output_folder}word_study_table.xlsx")    
    utterance_study_table.to_excel(f"{output_folder}utterance_study_table.xlsx")

# actual = word_study_table['error_type'].to_list()
# predicted = word_study_table['error_type_AT'].to_list()
# lbl = word_study_table['error_type'].drop_duplicates().sort_values(na_position='first').to_list()

# confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=lbl)
# print(metrics.classification_report(actual, predicted))

# # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
# # cm_display.plot()
# # #plt.savefig("myImagePDF.pdf", format="pdf")
# # plt.show() 

# heatmap = sns.heatmap(confusion_matrix, xticklabels = lbl, yticklabels = lbl, annot=True,fmt='d')
# heatmap.set(xlabel='Hindi_AT_error_type', ylabel='Hindi_MT_error_type')
# heatmap_fig = heatmap.get_figure()
    
    
    
    