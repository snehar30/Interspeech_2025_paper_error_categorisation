This repo contains scripts to perform fine-grained phoneme level analysis of oral reading errors.



See the following paper for more details



https://www.isca-archive.org/interspeech\_2025/raman25\_interspeech.pdf





The repo is organised as follows



1. Scripts

   a. error\_categorisation\_main\_script.py 
      Takes master files and lexicon files as input and gives out output files at the character, word and utterance level that can be analysed 

   b. seaborn\_plots.py
      Has multiple sections within the script to plot data from the output files.
3. Input files
   The master files obtained after postprocessing. The "\_single\_story\_only\_s" versions are filtered versions of the "\_all" versions. 
4. Lexicon files
   The lexicon files required to get phone sequences of word entries.
   Word lists in English and Hindi to identify word and nonword errors
5. Output files

&nbsp;  Three types of output files are generated at the character, word and utterance level, language specific and one file for both languages.

