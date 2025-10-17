# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:49:26 2025

@author: sneha
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib.patches as mpatches


#####################################################################################
#  read output files
#####################################################################################
character_study_file = "output_iles/character_study_table.xlsx"
character_df = pd.read_excel(character_study_file)

word_study_file = "output_iles/word_study_table.xlsx"
word_df = pd.read_excel(word_study_file)

utterance_study_file = "output_iles/utterance_study_table.xlsx"
utterance_df = pd.read_excel(utterance_study_file)



#####################################################################################
#  word-nonword errors
#####################################################################################


a = sns.catplot(hue="W_NW_error", x="lang", hue_order = ['word', 'nonword'],kind="count",palette="colorblind", edgecolor=".6", data=word_df, width=0.5, legend_out=False)
#fig = ax.get_figure()
   
plt.legend(loc='upper right')

plt.xlabel('Language')       
plt.ylabel('Count')

plt.title('Overall counts of word/nonword errors')

a.savefig("word_nonword.png") 

#####################################################################################
#  word-nonword errors by word length
#####################################################################################

b = sns.catplot(hue="W_NW_error",hue_order = ['word', 'nonword'], x="word_phoneme_length", kind="count",palette="colorblind", edgecolor=".6", data=word_df, legend_out=False)

plt.xlabel('Reference word length (number of phonemes)')       
plt.ylabel('Count')
plt.legend(loc=0)

plt.title('Overall counts of word/nonword errors by word length')

b.savefig("word_nonword_by_word_length.png") 


#####################################################################################
#  word-nonword errors by word length by language
#####################################################################################

lng = 'English'

lang_word_df = word_df[word_df['lang'] == lng]

sns.set(font_scale=1.3)
b = sns.catplot(hue="W_NW_error",hue_order = ['word', 'nonword'], x="word_phoneme_length", kind="count",palette="colorblind", edgecolor=".6", data=lang_word_df, legend_out=False)

plt.xlabel('Word length',fontsize=18,fontweight='bold')       
plt.ylabel('Count',fontsize=18,fontweight='bold')
plt.legend('',loc=0)
plt.tight_layout()
#plt.title('Counts of word/nonword errors by word length (' + lng + ')')

# ax = b.axes.flatten() 

# for label in ax.get_xticklabels():
#     label.set_fontsize(20)
    
b.savefig(lng+"word_nonword_by_word_length.pdf") 



#####################################################################################
# combined plot for word non word by word length and lang
#####################################################################################



g = sns.FacetGrid(word_df, row="lang", hue_order = ['word', 'nonword'], palette="colorblind") # Creates a grid based on 'time' and 'day'


def countplot_with_dodge(data, **kwargs):
    counts = data.groupby([kwargs['x'], kwargs['hue']]).size().reset_index(name='count')
    sns.barplot(x=kwargs['x'], y='count', hue=kwargs['hue'], data=counts, dodge=True)
    

g.map_dataframe(countplot_with_dodge, x="word_phoneme_length", hue="W_NW_error")

g.add_legend(None)


axes = g.axes.flatten()  # Get a flattened array of axes

titles = ["Hindi", "English"]  # Your custom titles

for i, ax in enumerate(axes):
    if i < len(titles): # Check if there are enough titles
        ax.set_title(titles[i])
        
plt.tight_layout()
plt.xlabel('Word length',fontsize=18,fontweight='bold')       
plt.ylabel('Count',fontsize=18,fontweight='bold')
g.set_xticklabels(g.get_xticklabels(), fontdict={ 'weight': 'bold'} )
g.set_yticklabels(g.get_yticklabels(), fontdict={ 'weight': 'bold'})


#plt.title('Error categories in orthographic and phonological modalities (' + lng + ')')

plt.savefig(lng+"heatmap_errors_orth_phon.pdf", bbox_inches='tight')

# plt.legend(bbox_to_anchor=(0.8, 0.2))
plt.show()


g.savefig("Word_nonword_by_word_length_and_lang.pdf") 



#####################################################################################
# combined plot for word non word by word length and lang attempt 2
#####################################################################################



# Sample Data (replace with your actual data)

# Facet Plot

sns.set(font_scale=2)
sns.set_style("white")

# 1. Create the FacetGrid and plot
# Increased aspect for wider, more spread-out bars
g = sns.FacetGrid(word_df, row="lang", height=4, aspect=2)
g.map_dataframe(sns.countplot, x="word_phoneme_length", hue="W_NW_error", hue_order=['word', 'nonword'], palette="colorblind")
g.set_axis_labels("", "Count")

# Define hatch patterns
hatches = {'word': '///', 'nonword': 'xx'}

# Define hue labels directly based on your desired order
hue_labels = ['word', 'nonword']
palette_colors = sns.color_palette("colorblind", n_colors=len(hue_labels))

# 2. Apply hatches to the bars using the containers approach
for ax in g.axes.flatten():
    ax.tick_params(labelbottom=True, axis='both')

    # ax.containers is a list of BarContainer objects, one for each hue category
    # The order of containers matches the hue_order: ['word', 'nonword']
    for container_idx, container in enumerate(ax.containers):
        # Determine the hue label for this container based on its index
        current_hue_label = hue_labels[container_idx]
        desired_hatch = hatches[current_hue_label]

        # Apply the hatch to all bars within this container
        for bar in container.patches:
            bar.set_hatch(desired_hatch)

# 3. Create and add custom legend
# Remove any default legend that seaborn might have added
if g.legend is not None:
    g.legend.remove()

# Create custom legend handles (patches) with the correct colors and hatches
custom_legend_handles = []
for i, label in enumerate(hue_labels):
    patch = mpatches.Patch(facecolor=palette_colors[i], hatch=hatches[label], label=label)
    custom_legend_handles.append(patch)

# Add the new legend to the figure, with no title
g.fig.legend(handles=custom_legend_handles, title="",
             bbox_to_anchor=(0.7, 1), loc='upper left', frameon=False, fontsize=20)

# Apply common axis settings and titles to each subplot
axes = g.axes.flatten()
titles = ["Hindi", "English"]

for i, ax in enumerate(axes):
    ax.set_yticks([0,100,200,300,400,500,600])
    ax.set_xticks([0,1,2,3,4,5,6,7,8])
    if i < len(titles):
        ax.set_title(titles[i], fontsize=20, position=[0.4,2])

plt.xlabel('Word length')
plt.ylabel('Count')
plt.tight_layout() # Adjust layout to prevent overlapping
plt.show()
g.savefig("Word_nonword_by_word_length_and_lang.pdf") 



#####################################################################################
#  word-nonword errors percentage
#####################################################################################

x,y = 'lang', 'W_NW_error'

wnw = word_df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()

d = sns.catplot(data = wnw, x=x, y='percent',hue=y,hue_order = ['word', 'nonword'],kind='bar', width=0.5, palette="colorblind", edgecolor=".6",legend_out=False)

plt.legend(loc='upper center')
plt.xlabel('Language')       
plt.ylabel('Percentage')

plt.title('Percentage of word-nonword error')

d.savefig("word_nonword_percentage.png") 


#####################################################################################
#  word-nonword errors percentage by word length 
#####################################################################################


x,y = 'word_phoneme_length', 'W_NW_error'

wnw = word_df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()

d = sns.catplot(data = wnw, x=x, y='percent',hue=y,hue_order = ['word', 'nonword'],kind='bar', palette="colorblind", edgecolor=".6",legend_out=False)


plt.legend(loc='upper center')
plt.xlabel('Reference word length (number of phonemes)')       
plt.ylabel('Percentage')

plt.title('Overall percentage of word-nonword errors by word length')

d.savefig("word_nonword_by_word_length_percentage.png") 

#####################################################################################
#  word-nonword errors percentage by word length by language
#####################################################################################

lng = 'Hindi'

lang_word_df = word_df[word_df['lang'] == lng]

x,y = 'word_phoneme_length', 'W_NW_error'

wnw = lang_word_df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()

d = sns.catplot(data = wnw, x=x, y='percent',hue=y,hue_order = ['word', 'nonword'],kind='bar', palette="colorblind", edgecolor=".6",legend_out=False)



plt.legend(loc='upper center')
plt.xlabel('Reference word length (number of phonemes)')       
plt.ylabel('Percentage')

plt.title('Percentage of word-nonword errors by word length (' + lng + ')')

d.savefig(lng+"word_nonword_by_word_length_percentage.png") 

#####################################################################################
#  Error types percetange by language
#####################################################################################


x,y = 'lang', 'error_type'

err = word_df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()

color_list =  ["#1D418F", '#2A5C2B', '#992C45', '#C99F3A','#7248B5'] 
#color_list = ['#4B678F', '#699463', '#99525C', '#C99D58','#8C74B5'] 

hue_order = ['Scaffolding error','Initial phoneme correct','Final phoneme correct','Some phoneme overlap','Other']

c = sns.catplot(data = err, x=x, y='percent',hue=y,kind='bar', palette=color_list, edgecolor=".6")

plt.xlabel('Language')       
plt.ylabel('Percentage')


# Define your custom labels
custom_labels = ['Scaffolding error','Initial phoneme correct','Final phoneme correct','Some phoneme overlap','Other']




plt.title('Percentage of Each Error Type')
# c._legend.remove() # Remove the old legend
# plt.legend(title="Error Types", labels=hue_labels, loc="upper right") # create new legend


c.savefig("Error_types_by_language_percentage.png") 

#####################################################################################
#  Error types by language
#####################################################################################


# --- Your original data processing ---
x,y = 'lang', 'error_type' # Changed 'error_type_c' to 'error_type'
err = word_df.groupby(x)[y].value_counts().rename('counts').reset_index()

# --- Your defined colors and hue order ---
color_list = ["#1D418F", '#2A5C2B', '#992C45', '#C99F3A','#7248B5']
hue_order = ['scaffolding_error', 'initial_correct', 'Final_correct', 'some_overlap','other']

# Define hatch patterns for each hue category (same as your previous successful one)
hatches = {
    'scaffolding_error': '///',
    'initial_correct': 'xx',
    'Final_correct': '\\\\',
    'some_overlap': '..',
    'other': ''
}

# --- Plotting the catplot ---
# Removed edgecolor=".6" and legend_out=False as these are handled by custom setup
c = sns.catplot(data=err, x=x, y='counts', hue=y, hue_order=hue_order,
                kind='bar', palette=color_list, aspect=1.6) # Keeping your aspect ratio

# --- Apply hatches and white edgecolors to the bars ---
for ax in c.axes.flatten(): # Iterate through each subplot's axes
    for container_idx, container in enumerate(ax.containers):
        # The containers are ordered according to hue_order
        current_hue_label = hue_order[container_idx]
        desired_hatch = hatches[current_hue_label]

        for bar in container.patches:
            bar.set_hatch(desired_hatch)
            bar.set_edgecolor('white') # Set edge color to white for hatches
            # bar.set_width(0.4) # Uncomment and adjust this if you want to explicitly control bar width and spacing

# --- Customize and recreate the legend with hatches ---

# Remove the original legend created by catplot (it doesn't have hatches)
if c._legend: # Check if a legend exists
    c._legend.remove()

# Define your custom labels (updated for 'phoneme' as per your custom_labels)
custom_labels = [
    'Scaffolding error',
    'Initial phoneme correct', # Changed from grapheme to phoneme
    'Final phoneme correct',   # Changed from grapheme to phoneme
    'Some phoneme overlap',    # Changed from grapheme to phoneme
    'Other'
]

# Create new legend handles (patches) that include the colors and hatches
custom_legend_handles = []
for i, label_key in enumerate(hue_order):
    patch = mpatches.Patch(
        facecolor=color_list[i],
        hatch=hatches[label_key],
        label=custom_labels[i],
        edgecolor='white' # Ensure legend hatches are white too
    )
    custom_legend_handles.append(patch)

# Add the new custom legend to the figure
# Set title to empty string for no title, and adjusted bbox_to_anchor for placement
c.fig.legend(handles=custom_legend_handles, title="",
             prop={'size': 18}, bbox_to_anchor=(0.55, 1), loc='upper left')


# --- Common plot aesthetics ---
# Removed plt.title as it's commented out in your original code
plt.xlabel('Language', fontsize=20)
plt.ylabel('Count' , fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1100]) # Keep your original y-limit

# Adjust layout to prevent labels from overlapping
plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust rect to make space for the legend on the right

plt.show()
c.savefig("Error_types_by_language_counts_phonological.pdf") 



#####################################################################################
#  Error types by language
#####################################################################################

# --- Your original data processing ---
x,y = 'lang', 'error_type_c'
err = word_df.groupby(x)[y].value_counts().rename('counts').reset_index()

# --- Your defined colors and hue order ---
color_list = ["#1D418F", '#2A5C2B', '#992C45', '#C99F3A','#7248B5']
hue_order = ['scaffolding_error', 'initial_correct', 'Final_correct', 'some_overlap','other']

# Define hatch patterns for each hue category
hatches = {
    'scaffolding_error': '///',
    'initial_correct': 'xx',
    'Final_correct': '\\\\',
    'some_overlap': '..',
    'other': ''
}

# --- Plotting the catplot ---
c = sns.catplot(data=err, x=x, y='counts', hue=y, hue_order=hue_order,
                kind='bar', palette=color_list, legend_out=False, aspect =2.0)

# --- Apply hatches to the bars ---
for ax in c.axes.flatten(): # Iterate through each subplot's axes
    for container_idx, container in enumerate(ax.containers):
        # The containers are ordered according to hue_order
        current_hue_label = hue_order[container_idx]
        desired_hatch = hatches[current_hue_label]

        for bar in container.patches:
            bar.set_hatch(desired_hatch)
            bar.set_edgecolor('white')
            #bar.set_width(0.2) #

# --- Customize and recreate the legend with hatches ---

# Remove the original legend created by catplot (it doesn't have hatches)
if c._legend: # Check if a legend exists
    c._legend.remove()

# Define your custom labels (must be in the same order as hue_order)
custom_labels = [
    'Scaffolding error',
    'Initial grapheme correct',
    'Final grapheme correct',
    'Some grapheme overlap',
    'Other'
]

# Create new legend handles (patches) that include the colors and hatches
custom_legend_handles = []
for i, label_key in enumerate(hue_order):
    patch = mpatches.Patch(
        facecolor=color_list[i],  # Use the color from your color_list
        hatch=hatches[label_key], # Apply the corresponding hatch
        label=custom_labels[i],    # Use your custom display label
        edgecolor='white'
    )
    custom_legend_handles.append(patch)

# Add the new custom legend to the figure
# For catplot, it's often best to add the legend directly to the figure
# or to one of the axes and then position it globally.
c.fig.legend(handles=custom_legend_handles, title="", # Set title to empty string for no title
             prop={'size': 18}, bbox_to_anchor=(0.55, 1), loc='upper left') # Adjust position as needed


# --- Common plot aesthetics ---
plt.xlabel('Language', fontsize=20)
plt.ylabel('Count' , fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([0,1100]) # Keep your original y-limit

# Adjust layout to prevent labels from overlapping
plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust rect to make space for the legend on the right

plt.show()

c.savefig("Error_types_by_language_counts_orthographic.pdf") 

#####################################################################################
#  Error types similarity percetange by language
#####################################################################################


x,y = 'lang', 'error_type_with_similarity'

err = word_df.groupby(x)[y].value_counts(normalize=True).mul(100).rename('percent').reset_index()

color_list =  ["#1D418F",'#4B678F', '#2A5C2B', '#699463', '#992C45', '#99525C','#C99F3A','#C99D58','#7248B5'] 
hue_order = ['scaffolding_error_similar', 'scaffolding_error_dissimilar', 'initial_phoneme_correct_similar', 'initial_phoneme_correct_dissimilar','Final_phoneme_correct_similar',  'Final_phoneme_correct_dissimilar', 'some_overlap_similar', 'some_overlap_dissimilar', 'other_dissimilar']

c = sns.catplot(data = err, x=x, y='percent',hue=y, kind='bar', hue_order =hue_order, palette=color_list, edgecolor=".6")

c.savefig("Error_types_with_similarity.png") 




#####################################################################################
#  < and > 25 scaffolding errors
#####################################################################################


e = sns.scatterplot(data=utterance_df, x="perc_sc_errors_MT", y="perc_sc_errors_AT")


plt.xlabel('Percentage of scaffolding errors from Manual Transcriptions')       
plt.ylabel('Percentage of scaffolding errors from Automatic Method')
plt.savefig("scatterplot_sc_error_AT_MT.png")



#####################################################################################
#  < and > 25 scaffolding errors
#####################################################################################

lng = 'Hindi'

lang_utterance_df = utterance_df[utterance_df['lang'] == lng]

e = sns.scatterplot(data=lang_utterance_df, x="perc_sc_errors_MT", y="perc_sc_errors_AT")


plt.xlabel('Percentage of scaffolding errors from Manual Transcriptions')       
plt.ylabel('Percentage of scaffolding errors from Automatic Method')
plt.savefig(lng+"scatterplot_sc_error_AT_MT.png")


#####################################################################################
# scaffolding errors orthography and phonology
#####################################################################################

lng = 'English'

lang_word_df = word_df[word_df['lang'] == lng]

#e = sns.scatterplot(data=lang_word_df, x="error_type_c", y="error_type")

actual = lang_word_df['error_type_c'].to_list()
predicted = lang_word_df['error_type'].to_list()
#lbl = lang_word_df['error_type_c'].drop_duplicates().dropna().sort_values(na_position='first').to_list()

lbl=['Final_correct','initial_correct','scaffolding_error','some_overlap', 'other']

confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=lbl)
print(metrics.classification_report(actual, predicted))

lbl=['FC','IC','SC','SO', 'OT']

sns.set(font_scale=2)
heatmap = sns.heatmap(confusion_matrix, cmap="Blues",vmin=-50, vmax=900, xticklabels = lbl, yticklabels = lbl, annot=True,fmt='d')
#heatmap.set(xlabel='AT_error_type', ylabel='MT_error_type')
heatmap_fig = heatmap.get_figure()

plt.xlabel('Phonological',fontsize=24,fontweight='bold')       
plt.ylabel('Orthographic',fontsize=24,fontweight='bold')
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontdict={'color': '#083674', 'weight': 'bold'} )
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontdict={'color': '#083674', 'weight': 'bold'})
#plt.title('Error categories in orthographic and phonological modalities (' + lng + ')')

plt.savefig(lng+"heatmap_errors_orth_phon.pdf", bbox_inches='tight')


# plt.xlabel('Percentage of scaffolding errors from Manual Transcriptions')       
# plt.ylabel('Percentage of scaffolding errors from Automatic Method')
# plt.savefig(lng+"scatterplot_sc_error_AT_MT.png")


#####################################################################################
# scaffolding errors manual and automatic
#####################################################################################

lng = 'Hindi'

lang_word_df = word_df[word_df['lang'] == lng]

actual = lang_word_df['error_type'].to_list()
predicted = lang_word_df['error_type_AT'].to_list()
#lbl = lang_word_df['error_type'].drop_duplicates().dropna().sort_values(na_position='first').to_list()

lbl=['Final_correct','initial_correct','scaffolding_error','some_overlap', 'other']
confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=lbl)
print(metrics.classification_report(actual, predicted))

lbl=['FC','IC','SC','SO', 'OT']

heatmap = sns.heatmap(confusion_matrix,cmap="Blues", vmin=-50, vmax=700, xticklabels = lbl, yticklabels = lbl, annot=True,fmt='d')
#heatmap.set(xlabel='AT_error_type', ylabel='MT_error_type')
heatmap_fig = heatmap.get_figure()

plt.xlabel('Automatic')       
plt.ylabel('Manual')

plt.title('Confusion matrix Manual vs Automatic Method (' + lng + ')')

plt.savefig(lng+"heatmap_error_type_manual_automatic.pdf", bbox_inches='tight')

# plt.xlabel('Percentage of scaffolding errors from Manual Transcriptions')       
# plt.ylabel('Percentage of scaffolding errors from Automatic Method')
# plt.savefig(lng+"scatterplot_sc_error_AT_MT.png")

#####################################################################################
# scaffolding errors 
#####################################################################################

lng = 'Hindi'

lang_word_df = word_df[word_df['lang'] == lng]

actual = lang_word_df['error_type'].to_list()
predicted = lang_word_df['error_type_AT'].to_list()
lbl = lang_word_df['error_type'].drop_duplicates().sort_values(na_position='first').to_list()

confusion_matrix = metrics.confusion_matrix(actual, predicted, labels=lbl)
print(metrics.classification_report(actual, predicted))

heatmap = sns.heatmap(confusion_matrix, xticklabels = lbl, yticklabels = lbl, annot=True,fmt='d')
#heatmap.set(xlabel='AT_error_type', ylabel='MT_error_type')
heatmap_fig = heatmap.get_figure()

plt.xlabel('Automatic')       
plt.ylabel('Manual')
plt.title('Error categories Manual vs Automatic Method (' + lng + ')')

plt.savefig(lng+"heatmap_error_type_manual_automatic.png")

# plt.xlabel('Percentage of scaffolding errors from Manual Transcriptions')       
# plt.ylabel('Percentage of scaffolding errors from Automatic Method')
# plt.savefig(lng+"scatterplot_sc_error_AT_MT.png")

#####################################################################################
# error types combined graph 
#####################################################################################

g = sns.FacetGrid(word_df, row="lang", hue_order = ['word', 'nonword'], palette="colorblind") # Creates a grid based on 'time' and 'day'


def countplot_with_dodge(data, **kwargs):
    counts = data.groupby([kwargs['x'], kwargs['hue']]).size().reset_index(name='count')
    sns.barplot(x=kwargs['x'], y='count', hue=kwargs['hue'], data=counts, dodge=True)

g.map_dataframe(countplot_with_dodge, x="word_phoneme_length", hue="W_NW_error", )

g.add_legend(None)


axes = g.axes.flatten()  # Get a flattened array of axes

titles = ["Hindi", "English"]  # Your custom titles

for i, ax in enumerate(axes):
    if i < len(titles): # Check if there are enough titles
        ax.set_title(titles[i])
        ax.set_yticks([0,100,200,300,400,500,600])
        
plt.tight_layout()
plt.xlabel('Reference word length (number of phonemes)')       
plt.ylabel('Count')

# plt.legend(bbox_to_anchor=(0.8, 0.2))
plt.show()


g.savefig("Word_nonword_by_word_length_and_lang.pdf") 

