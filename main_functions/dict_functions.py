import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pretty_midi
import seaborn as sns
import os

import sys
sys.path.append('/content/drive/Shareddrives/Master Thesis')
import peak_functions
import dict_functions
import freqvibrato_functions
import midi_functions


# Filtering and display for dictionary outputs of the pitch bend values

def filter_nested_dict_by_key_pattern(nested_dict, key_pattern):
    """ 
    Filters a nested dictionary based on whether the key_pattern string is present in the keys of the nested dictionary. 
    It returns a new dictionary containing only the key-value pairs that match the pattern.
    """
    filtered_dict = {}
    for key, value in nested_dict.items():
        if key_pattern in key:
            filtered_dict[key] = value

    return filtered_dict

def filter_and_display(initial_dict, key_pattern, plot_yes=False):
    """
    This function takes a dictionary and a key pattern and returns a filtered dictionary.
    Outliers are also calculated and displayed.
    """
    
    filtered_dict = filter_nested_dict_by_key_pattern(initial_dict, key_pattern)
    #length_of_values = [len(value) if isinstance(value, (list, str, tuple, dict)) else None for value in filtered_dict.values()]

    threshold=3
    if plot_yes:
        plt.figure(figsize=(20,10))
        for key, sub_dict in filtered_dict.items():

            values = np.array(list(sub_dict.values())) # convert to numpy array
            plt.hist(values, bins='auto', alpha=0.4)

            z_scores = np.abs((values - np.mean(values)) / np.std(values))
            outliers = values[z_scores > threshold]

            if len(outliers) > 0:
                print(f"Key: {key}")
                print("Outlier values:", outliers)
                print("Mean:", np.mean(values))
                print("Standard Deviation:", np.std(values))
                outlier_keys = [sub_key for sub_key, sub_value in sub_dict.items() if sub_value in outliers]
                print("Outlier keys:", outlier_keys)
                print("-" * 40)
            
        plt.xlabel('Intonation variation (Cents)') 
        plt.ylabel('Counts')
        plt.title('Histogram of Pitch Bend Values after pre-processing')
        plt.legend(filtered_dict.keys())
        plt.show()

        values = np.asarray(list(sub_dict.values()), dtype=float)
        hist, bins = np.histogram(values, bins='auto')
        mean = np.sum(hist * bins[:-1]) / np.sum(hist)

        print("Mean of the data in the histogram:", mean)        
        threshold=3
        z_scores = np.abs((values - np.mean(values)) / np.std(values))
        outliers = values[z_scores > threshold]
        print("The following values are outliers:", outliers)
    return filtered_dict 

def df_all_players_etude(filtered_dict_tonalities, df_dict, plot_yes = False):
    """
    For short notes!
    This function takes a dictionary with the filtered tonalities and a dictionary with the dataframes of the players.
    It merges the pitch bend value information and the info dataframe of the players.
    It returns a dataframe with the pitch bend values of all the players and midi info of all notes.
    """
    names_new_columns = []
    player_value = pd.DataFrame()
    for key, value in filtered_dict_tonalities.items():
        #print(key)
        data_dict = filtered_dict_tonalities[key]
        last_part = key.split('-')[-1]
        column_name = str('pitchValue'+'_'+last_part)
        names_new_columns.append(column_name)
        filtered_dict_tonalities_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=[column_name])
        player_value = pd.concat([player_value, filtered_dict_tonalities_df], axis=1)

    rafferty_df = df_dict[key]
    merged_df = pd.merge(rafferty_df, player_value, left_index=True, right_index=True)
    if plot_yes:
        display(merged_df.head(25))
    return merged_df, names_new_columns

def df_all_players_etude_long_notes(filtered_dict_eb_n11, pdf_dict_long, plot_yes=False):
    """
    For long notes!
    This function takes a dictionary with the filtered tonalities and a dictionary with the dataframes of the players.
    It merges the pitch bend value information and the info dataframe of the players.
    It returns a dataframe with the pitch bend values of all the players and midi info of the notes.
    """
    names_new_columns = []
    player_value = pd.DataFrame()
    for key, value in filtered_dict_eb_n11.items():
        data_dict = filtered_dict_eb_n11[key]
        last_part = key.split('-')[-1]
        column_name = str('pitchValue'+'_'+last_part)
        names_new_columns.append(column_name)
        filtered_dict_tonalities_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=[column_name])
        player_value = pd.concat([player_value, filtered_dict_tonalities_df], axis=1)

    rafferty_df = pdf_dict_long[key]
    characteristics_df = pd.DataFrame()
    for key, thing in rafferty_df.items():
        characteristics_df[key] = [thing]
    merged_df = pd.concat([characteristics_df, player_value.reset_index()],  axis=1)
    if plot_yes:
        display(merged_df)
    return merged_df, names_new_columns

def see_heatmap_zero_tonic(main_table, direction, overall_coverage_mean):
    """
    This function takes the main intonation deviation from 12-TET table and generates a coloured heatmap.
    It displays both the table without(left) and with (right) tonic normalization.
    Tonic normalization consists on substracting the value of the tonic (I) to the other scale degrees.
    """
    pdf_to_see_1 = main_table.T

    colors = ["darkblue", "white", "darkred"] 
    n_bins = 100  # Number of bins in the colormap
    cmap_name = "custom_cmap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    sns.heatmap(pdf_to_see_1, annot=True, cmap=cm, fmt='.2f', linewidths=0.5, center=0)
    plt.title(f'Trimming by peaks with histogram {overall_coverage_mean:.2f} % coverage mean short {direction} notes')
    
    first_elements = pdf_to_see_1.iloc[:, 0]
    pdf_to_see_1.iloc[:, 0:] = pdf_to_see_1.iloc[:, 0:] - first_elements[:, None]

    plt.subplot(1, 2, 2)
    sns.heatmap(pdf_to_see_1, annot=True, cmap=cm, fmt='.2f', linewidths=0.5, center=0)
    plt.title(f'Trimming by peaks with histogram {overall_coverage_mean:.2f} % coverage mean short {direction} notes')
    plt.tight_layout()
    plt.show()

def define_direction_in_df(merged_df):
    """
    For Backwards Direction!
    This function takes a dataframe and returns the dataframe with the direction of the pitch bend values.
    """
    current_pitch = 0
    past_pitch = 0
    merged_df['pitch'] = merged_df['pitch'].apply(pretty_midi.note_name_to_number)
    directions = ['NaN']

    for i, note in merged_df.iterrows():
        if i > 0:
            current_pitch = note['pitch']
            past_pitch = merged_df.loc[i - 1, 'pitch']
            if current_pitch < past_pitch:
                directions.append('Descending')
            elif current_pitch > past_pitch:
                directions.append('Ascending')
            else:
                directions.append('Equal')

    merged_df['direction'] = directions
    merged_df['pitch'] = merged_df['pitch'].apply(pretty_midi.note_number_to_name)
    return merged_df

def define_next_direction_in_df(merged_df):
    """
    For Forwards Direction!
    This function takes a dataframe and returns the dataframe with the direction of the pitch bend values.
    """
    current_pitch = 0
    past_pitch = 0
    merged_df['pitch'] = merged_df['pitch'].apply(pretty_midi.note_name_to_number)
    directions = []

    for i, note in merged_df.iterrows():
        if i < (len(merged_df) - 1):
            current_pitch = note['pitch']
            next_pitch = merged_df.loc[i + 1, 'pitch']
            if current_pitch < next_pitch:
                directions.append('Ascending')
            elif current_pitch > next_pitch:
                directions.append('Descending')
            else:
                directions.append('Equal')
        else:
            directions.append('Last_Note')

    merged_df['direction'] = directions
    merged_df['pitch'] = merged_df['pitch'].apply(pretty_midi.note_number_to_name)
    return merged_df

def define_2before_direction_in_df(merged_df):
    """
    For Backwards Direction, taking into account TWO notes before the goal note!
    This function takes a dataframe and returns the dataframe with the direction of the pitch bend values.
    """
    current_pitch = 0
    past_pitch = 0
    merged_df['pitch'] = merged_df['pitch'].apply(pretty_midi.note_name_to_number)
    directions = []

    for i, note in merged_df.iterrows():
        if i > 1:
            current_pitch = note['pitch']
            past_pitch = merged_df.loc[i - 1, 'pitch']
            two_past_pitch = merged_df.loc[i - 2, 'pitch']
            if  past_pitch < current_pitch:
                #directions.append('One_Descending')
                if two_past_pitch < past_pitch:
                    directions.append('Two_Ascending')
                elif past_pitch == two_past_pitch:
                    directions.append('Ascending_Equal')
                else:
                    directions.append('Ascending_Descending')
            elif past_pitch > current_pitch:
                #directions.append('Descending')
                if two_past_pitch > past_pitch:
                    directions.append('Two_Descending')
                elif past_pitch == two_past_pitch:
                    directions.append('Descending_Equal')
                else:
                    directions.append('Descending_Ascending')
            else:
                #directions.append('Equal')
                if two_past_pitch > past_pitch:
                    directions.append('Equal_Descending')
                elif past_pitch == two_past_pitch:
                    directions.append('Equal_Equal')
                else:
                    directions.append('Equal_Ascending')
        else:
            directions.append('First_Notes')

    merged_df['direction'] = directions
    merged_df['pitch'] = merged_df['pitch'].apply(pretty_midi.note_number_to_name)
    return merged_df


def get_mean_pitch_value(short_info_method_general, short_time_method_general):
  """
    For short notes (no vibrato)
    This function takes a dictionary with the pitch bend values for each note 
    and returns a dictionary with the mean pitch bend value, the stabilized values, stabilized time and coverage for each note.
  """
  pshort_stabilized_values = {}
  pshort_stabilized_time = {}
  pshort_main_pitch_value = {}
  penergy_rate_comparison = {}
  pnumber_samples = {}
  pno_trimmed_dict = {}
  pshort_coverage_dict = {}

  for key, sub_dict in short_info_method_general.items():
      #print(f"Main key: {key}")
      for sub_key, value in sub_dict.items():
          #print(f"\nSub key: {sub_key}")
          values = np.array(value)
          time = short_time_method_general[key][sub_key]
          trimmed_values, trimmed_time, no_trimmed, coverage = peak_functions.trim_by_peaks(values, time, plot_yes=False)
          if no_trimmed:
              pno_trimmed_dict[key]=sub_key

          if key not in pshort_stabilized_values:
              pshort_stabilized_values[key] = {}
          if key not in pshort_stabilized_time:
              pshort_stabilized_time[key] = {}
          if key not in pshort_main_pitch_value:
              pshort_main_pitch_value[key] = {}
          if key not in pshort_coverage_dict:
              pshort_coverage_dict[key] = {}

          pshort_stabilized_values[key][sub_key] = trimmed_values
          pshort_stabilized_time[key][sub_key] = trimmed_time
          pshort_coverage_dict[key][sub_key] = coverage

          num_bins = 50
          mean_pitch_value_now = peak_functions.hist_pitch_max(num_bins, trimmed_values, plot_yes=False)
          pshort_main_pitch_value[key][sub_key] = mean_pitch_value_now
  return pshort_main_pitch_value, pshort_stabilized_values, pshort_stabilized_time, pshort_coverage_dict

def long_get_mean_pitch_value(long_final_notes, long_final_times, plot_yes=False):
    """
    For long notes (vibrato)
    This function takes a dictionary with the pitch bend values for each note 
    and returns a dictionary with the mean pitch bend value the stabilized values, 
    stabilized time and coverage for each note.
    """
    pshort_stabilized_values = {}
    pshort_stabilized_time = {}
    pshort_main_pitch_value = {}
    pno_trimmed_dict = {}
    pshort_coverage_dict = {}
    long_main_pitch_value = {}

    for key, sub_dict in long_final_notes.items():
        #print(f"Main key: {key}")
        for sub_key, value in sub_dict.items():
            #print(f"\nSub key: {sub_key}")
            values = np.array(value)
            time = long_final_times[key][sub_key]
            trimmed_values, trimmed_time, no_trimmed, coverage = peak_functions.trim_by_peaks(values, time, plot_yes=plot_yes)
            if no_trimmed:
                pno_trimmed_dict[key]=sub_key

            if key not in pshort_stabilized_values:
                pshort_stabilized_values[key] = {}
            if key not in pshort_stabilized_time:
                pshort_stabilized_time[key] = {}
            if key not in pshort_main_pitch_value:
                pshort_main_pitch_value[key] = {}
            if key not in pshort_coverage_dict:
                pshort_coverage_dict[key] = {}
            if key not in long_main_pitch_value:
                long_main_pitch_value[key] = {}
                
            pshort_stabilized_values[key][sub_key] = trimmed_values
            pshort_stabilized_time[key][sub_key] = trimmed_time
            pshort_coverage_dict[key][sub_key] = coverage
            main_freq_sub_key = freqvibrato_functions.find_frrt_frequency(trimmed_values, plot_yes=plot_yes)
            
            if main_freq_sub_key > 4 and main_freq_sub_key < 10:
                my_signal = np.array(trimmed_values)
                my_time = np.array(trimmed_time)
                interpolated_values_upp, interpolated_values_down, overall_median = freqvibrato_functions.find_peaks_and_interpolate(my_signal, my_time, str(sub_key), plot_yes=plot_yes)
                #print(f"Overall median: {overall_median}")
                
                amplitude = (interpolated_values_upp - interpolated_values_down)/2 #time varying amplitude
                mean = (interpolated_values_upp + interpolated_values_down)/2 #time varying mean
                mean_pitch_value_now = np.mean(mean)
                long_main_pitch_value[key][sub_key] = mean_pitch_value_now
            else:
                num_bins = 50
                mean_pitch_value_now = peak_functions.hist_pitch_max(num_bins, values, plot_yes=plot_yes)
                long_main_pitch_value[key][sub_key] = mean_pitch_value_now
    return long_main_pitch_value, pshort_stabilized_values, pshort_stabilized_time, pshort_coverage_dict, pshort_main_pitch_value

def get_midi_info_dict(main_folder= '/content/drive/Shareddrives/Master Thesis/iter1_midi/wohlfahrt_first_30_no_accidentals'):
    """
    Both short and long notes
    This function takes a folder with midi files 
    and returns a dictionary with the pitch bend midi information for each note.
    """
    pdf_dict_short = {}
    pdf_dict_long = {}

    main_folder = '/content/drive/Shareddrives/Master Thesis/iter1_midi/wohlfahrt_first_30_no_accidentals'
    for folder_name in os.listdir(main_folder):
            folder_path = os.path.join(main_folder, folder_name)

            if os.path.isdir(folder_path):  # Check if it's a folder
                files = os.listdir(folder_path)
                for one_file in files:
                    if one_file == ".DS_Store":
                        continue
                    file_path = os.path.join(folder_path, one_file)
                    name_part = one_file.split('_')[0]+'-'+one_file.split('_')[1]+'-'+one_file.split('_')[2]
                    #print(name_part)
                    if file_path.endswith('.mid'):
                        pmidi_data = pretty_midi.PrettyMIDI(file_path)
                        pnotes_df, plongest_notes = midi_functions.look_longest_notes(pmidi_data)
                        pdf_dict_short[name_part] = pnotes_df.iloc[:-1]
                        pdf_dict_long[name_part] = pnotes_df.iloc[-1]
    return pdf_dict_short, pdf_dict_long

def see_main_int_tables(pshort_main_pitch_value, pdf_dict_short, overall_coverage_mean):
  """
    For short notes (no vibrato)
    This function takes a dictionary with the pitch bend values for each note and 
    returns a dictionary with the mean pitch bend value for each note.
    The table of intonation deviation from 12-TET is calculated and displayed.
  """
  tonalities_etudes_notetoGrades = {'Wohlfahrt-Op45-01': {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}, 'Wohlfahrt-Op45-03': {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}, 'Wohlfahrt-Op45-05': {'F':'I', 'G':'II', 'A':'III', 'A#':'IV', 'B':'IV#', 'C':'V', 'D':'VI', 'E':'VII'}, 'Wohlfahrt-Op45-10': {'A':'I', 'B':'II', 'C#':'III', 'D':'IV', 'E':'V', 'F#':'VI', 'G#':'VII'}, 'Wohlfahrt-Op45-11': {'D#':'I', 'F':'II', 'G':'III', 'G#':'IV', 'A':'#IV','A#':'V', 'C':'VI', 'D':'VII'}, 'Wohlfahrt-Op45-15': {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}, 'Wohlfahrt-Op45-26': {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}}
  tonalities_etudes = {'Wohlfahrt-Op45-01': 'C', 'Wohlfahrt-Op45-03': 'G', 'Wohlfahrt-Op45-05': 'F', 'Wohlfahrt-Op45-10': 'A', 'Wohlfahrt-Op45-11': 'Eb', 'Wohlfahrt-Op45-15': 'C', 'Wohlfahrt-Op45-26': 'G'}

  pshort_dict_tonalities_players = {}
  df_value_counts=pd.DataFrame()
  df_value_counts_tonality = pd.DataFrame()

  for element in tonalities_etudes.keys():
      filtered_dict_eb_n11 = dict_functions.filter_and_display(pshort_main_pitch_value, element, plot_yes=False)

      merged_df_eb_n11, names_new_columns = dict_functions.df_all_players_etude(filtered_dict_eb_n11, pdf_dict_short)
      tonalitys_grades = tonalities_etudes_notetoGrades[element]
      
      for index, note in enumerate(merged_df_eb_n11['pitch']):
          if '#' in note:
              full_note = note[:2]
          else:
              full_note = note[0]

          if full_note in tonalitys_grades.keys():
              new_note = tonalitys_grades[full_note]
              merged_df_eb_n11.loc[index, 'note_grade'] = new_note
              
      value_counts = merged_df_eb_n11['note_grade'].value_counts()
      df_value_counts[element] = value_counts

      current_tonality = tonalities_etudes[element]
      if current_tonality in pshort_dict_tonalities_players.keys():
          mean_data = merged_df_eb_n11.groupby('note_grade')[names_new_columns].mean()
          existing_data = pshort_dict_tonalities_players[current_tonality]
    
          updated_mean_data = (mean_data + existing_data) / 2
          pshort_dict_tonalities_players[current_tonality].update(updated_mean_data)

      else:
          pshort_dict_tonalities_players[current_tonality] = merged_df_eb_n11.groupby('note_grade')[names_new_columns].mean()

      if current_tonality in df_value_counts_tonality.keys():
          existing_data = df_value_counts_tonality[current_tonality]
          updated_value_counts = (value_counts + existing_data)
          df_value_counts_tonality[current_tonality].update(updated_value_counts)
      else:
          df_value_counts_tonality[current_tonality] = value_counts

  print('Relationship between the number of notes per scale degree for all etudes and tonalities')
  sorted_df_value_counts = df_value_counts.sort_index()
  display(sorted_df_value_counts)

  sorted_df_value_counts_tonality = df_value_counts_tonality.sort_index()
  display(sorted_df_value_counts_tonality)

  main_table = pd.DataFrame()
  main_table_std = pd.DataFrame()
  other_table = pd.DataFrame()
  for current_tonality in pshort_dict_tonalities_players.keys():
      tonality_df = pshort_dict_tonalities_players[current_tonality]
      mean_values_per_row = tonality_df.mean(axis=1)
      std_per_row = tonality_df.std(axis=1)
      main_table[current_tonality] = mean_values_per_row
      main_table_std[current_tonality] = std_per_row

      formatted_mean_values = mean_values_per_row.apply(lambda x: f"{x:.3f}")
      formatted_std_values = std_per_row.apply(lambda x: f"± {x:.3f}")

      mean_std_combined = formatted_mean_values + ' ' + formatted_std_values
      other_table[current_tonality] = mean_std_combined



  cm = sns.light_palette("green", as_cmap=True)
  cm1 = sns.light_palette("red", as_cmap=True)
  cm2 = sns.light_palette("yellow", as_cmap=True)
  pdf_to_see = main_table.T
  pdf_std = main_table_std.T
  pother_df = other_table.T

  print(f'Trimming by peaks with histogram {overall_coverage_mean:.2f} % coverage mean')
  print(f'Mean for every tonality, overall coverage mean {overall_coverage_mean:.2f}%')
  display(pdf_to_see.style.background_gradient(cmap=cm, axis=1))
  print(f'Std for every tonality, overall coverage mean {overall_coverage_mean:.2f}%')
  display(pdf_std.style.background_gradient(cmap=cm1, axis=1))

  print(f'Mean and Std for every tonality, overall coverage mean {overall_coverage_mean:.2f}% all short notes')
  display(pother_df.style.background_gradient(cmap=cm2))

  return main_table, main_table_std, other_table

def long_see_main_int_tables(long_main_pitch_value, pdf_dict_long, overall_coverage_mean):
    """
    For long notes (vibrato)
    This function takes a dictionary with the pitch bend values for each note and returns a dictionary with the mean pitch bend value for each note.
    The table of intonation deviation from 12-TET is calculated and displayed.
    """
    tonalities_etudes_notetoGrades = {'Wohlfahrt-Op45-01': {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}, 'Wohlfahrt-Op45-03': {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}, 'Wohlfahrt-Op45-05': {'F':'I', 'G':'II', 'A':'III', 'A#':'IV', 'B':'IV#', 'C':'V', 'D':'VI', 'E':'VII'}, 'Wohlfahrt-Op45-10': {'A':'I', 'B':'II', 'C#':'III', 'D':'IV', 'E':'V', 'F#':'VI', 'G#':'VII'}, 'Wohlfahrt-Op45-11': {'D#':'I', 'F':'II', 'G':'III', 'G#':'IV', 'A':'#IV','A#':'V', 'C':'VI', 'D':'VII'}, 'Wohlfahrt-Op45-15': {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}, 'Wohlfahrt-Op45-26': {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}}
    tonalities_etudes = {'Wohlfahrt-Op45-01': 'C', 'Wohlfahrt-Op45-03': 'G', 'Wohlfahrt-Op45-05': 'F', 'Wohlfahrt-Op45-10': 'A', 'Wohlfahrt-Op45-11': 'Eb', 'Wohlfahrt-Op45-15': 'C', 'Wohlfahrt-Op45-26': 'G'}

    pshort_dict_tonalities_players = {}
    df_value_counts=pd.DataFrame()
    df_value_counts_tonality = pd.DataFrame()
    merged_df = pd.DataFrame()
    dict_df_value_count_players = {}
    first = True

    for element in tonalities_etudes.keys():
        filtered_dict_eb_n11 = filter_and_display(long_main_pitch_value, element, plot_yes=False)
        merged_df_eb_n11, names_new_columns = df_all_players_etude_long_notes(filtered_dict_eb_n11, pdf_dict_long)

        tonalitys_grades = tonalities_etudes_notetoGrades[element]
        merged_df_eb_n11['tonality'] = tonalities_etudes[element]
        merged_df_eb_n11['etude'] = element
        for index, note in enumerate(merged_df_eb_n11['pitch']):
            if '#' in note:
                full_note = note[:2]
            else:
                full_note = note[0]

            if full_note in tonalitys_grades.keys():
                new_note = tonalitys_grades[full_note]
                merged_df_eb_n11.loc[index, 'note_grade'] = new_note

        value_counts = merged_df_eb_n11['note_grade'].value_counts()
        df_value_counts[element] = value_counts

        num_violinistas = len(names_new_columns)
        df_value_count_players = num_violinistas * df_value_counts
        print(f'Number of violinists: {num_violinistas} for {element}')
        dict_df_value_count_players[element] = df_value_count_players[element]

        if first:
            merged_df = merged_df_eb_n11
            first = False
        else:
            merged_df = pd.concat([merged_df, merged_df_eb_n11])

        current_tonality = tonalities_etudes[element]
        if current_tonality in df_value_counts_tonality.keys():
            existing_data = df_value_counts_tonality[current_tonality]
            updated_value_counts = (value_counts + existing_data)
            df_value_counts_tonality[current_tonality].update(updated_value_counts)
        else:
            df_value_counts_tonality[current_tonality] = value_counts

    sorted_df_value_counts = df_value_counts.sort_index()

    columns_to_mean = [column for column in merged_df.columns if 'pitchValue' in column]
    mean_per_row = merged_df[columns_to_mean].mean(axis=1)

    final_df = pd.DataFrame()
    final_df['mean_per_row'] = mean_per_row
    final_df['tonality'] = merged_df['tonality']
    final_df['note_grade'] = merged_df['note_grade']

    grouped_df = final_df.groupby(['tonality', 'note_grade']).mean()
    display('grouped_df', grouped_df)

    colors = ["darkblue", "white", "darkred"]
    n_bins = 100 
    cmap_name = "custom_cmap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    sns.heatmap(grouped_df.T, annot=True, cmap=cm, fmt='.2f', linewidths=0.5, center=0)
    plt.title(f'Mean intonation deviation /  {overall_coverage_mean:.2f} % coverage / long notes')
    plt.show()

    return grouped_df, dict_df_value_count_players

def direction_see_main_int_tables(pshort_main_pitch_value, pdf_dict_short, backwards=True, forward=False):
    """
    For short notes regarding Backwards Direction for Mean, Median and STD
    The table of intonation deviation from 12-TET is calculated and displayed.
    This function takes a dictionary with the pitch bend values and midi characteristics for each note and 
    returns the intonation deviation table with the mean, median and STD pitch bend value for each note regarding backwards direction. 
    Each plot will be displayed without(left) and with(right) tonic normalization.
    
    """
    tonalities_etudes_notetoGrades = {'Wohlfahrt-Op45-01': {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}, 'Wohlfahrt-Op45-03': {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}, 'Wohlfahrt-Op45-05': {'F':'I', 'G':'II', 'A':'III', 'A#':'IV', 'B':'IV#', 'C':'V', 'D':'VI', 'E':'VII'}, 'Wohlfahrt-Op45-10': {'A':'I', 'B':'II', 'C#':'III', 'D':'IV', 'E':'V', 'F#':'VI', 'G#':'VII'}, 'Wohlfahrt-Op45-11': {'D#':'I', 'F':'II', 'G':'III', 'G#':'IV', 'A':'#IV','A#':'V', 'C':'VI', 'D':'VII'}, 'Wohlfahrt-Op45-15': {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}, 'Wohlfahrt-Op45-26': {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}}
    tonalities_etudes = {'Wohlfahrt-Op45-01': 'C', 'Wohlfahrt-Op45-03': 'G', 'Wohlfahrt-Op45-05': 'F', 'Wohlfahrt-Op45-10': 'A', 'Wohlfahrt-Op45-11': 'Eb', 'Wohlfahrt-Op45-15': 'C', 'Wohlfahrt-Op45-26': 'G'}


    columns = ['note_grade', 'ascending', 'descending', 'First_Note', 'equal']
    df_final = pd.DataFrame(columns=columns)
    short_dict_tonalities_players = {}
    std_dict_tonalities_players = {}

    for element in tonalities_etudes.keys():
        filtered_dict = dict_functions.filter_and_display(pshort_main_pitch_value, element, plot_yes=False)

        merged_df, names_new_columns = dict_functions.df_all_players_etude(filtered_dict, pdf_dict_short)
        if backwards:
            merged_df = dict_functions.define_direction_in_df(merged_df)
        if forward:
            merged_df = dict_functions.define_next_direction_in_df(merged_df)
        tonalitys_grades = tonalities_etudes_notetoGrades[element]
        
        for index, note in enumerate(merged_df['pitch']):
            if '#' in note:
                full_note = note[:2]
            else:
                full_note = note[0]

            if full_note in tonalitys_grades.keys():
                new_note = tonalitys_grades[full_note]
                merged_df.loc[index, 'note_grade'] = new_note
            else:
                print(f'Note {full_note} not found in tonality {element}')
                print('index', index)
                print('note', note)
                display(merged_df.loc[index])



        current_tonality = tonalities_etudes[element]

        merged_df['direction'] = pd.Categorical(merged_df['direction'], categories=['Ascending', 'Descending', 'Equal'], ordered=False)

        if current_tonality in short_dict_tonalities_players.keys():
            current_df = short_dict_tonalities_players[current_tonality]
            short_dict_tonalities_players[current_tonality] = pd.concat([short_dict_tonalities_players[current_tonality], merged_df])

        else:
            short_dict_tonalities_players[current_tonality] = merged_df

    tonality_series = {}
    tonality_std_series = {}
    tonality_median_series = {}

    # MEAN
    tonality_series_ascending = {}
    tonality_series_descending = {}
    tonality_series_equal = {}

    # STD
    tonality_std_series_ascending = {}
    tonality_std_series_descending = {}
    tonality_std_series_equal = {}

    # MEDIAN
    tonality_median_series_ascending = {}
    tonality_median_series_descending = {}
    tonality_median_series_equal = {}


    # Iterate through each tonality
    for current_tonality in short_dict_tonalities_players.keys():
    
        selected_columns = short_dict_tonalities_players[current_tonality].filter(like='pitchValue')
        selected_columns_mean = selected_columns.mean(axis=1)
        selected_columns_std = selected_columns.std(axis=1)
        selected_columns_median = selected_columns.median(axis=1)
        
        # MEAN
        tonality_series[current_tonality] = pd.DataFrame(selected_columns_mean)
        tonality_series[current_tonality].columns = ['mean pitch value']
        tonality_series[current_tonality]['note_grade'] = short_dict_tonalities_players[current_tonality]['note_grade']
        tonality_series[current_tonality]['pitch'] = short_dict_tonalities_players[current_tonality]['pitch']
        tonality_series[current_tonality]['direction'] = short_dict_tonalities_players[current_tonality]['direction']

        tonality_series_ascending[current_tonality] = tonality_series[current_tonality][tonality_series[current_tonality]['direction'] == 'Ascending']
        direction_columns_to_keep = [column for column in tonality_series_ascending[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_series_ascending[current_tonality] = tonality_series_ascending[current_tonality][direction_columns_to_keep]

        tonality_series_descending[current_tonality] = tonality_series[current_tonality][tonality_series[current_tonality]['direction'] == 'Descending']
        direction_columns_to_keep = [column for column in tonality_series_ascending[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_series_descending[current_tonality] = tonality_series_descending[current_tonality][direction_columns_to_keep]

        tonality_series_equal[current_tonality] = tonality_series[current_tonality][tonality_series[current_tonality]['direction'] == 'Equal']
        direction_columns_to_keep = [column for column in tonality_series_ascending[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_series_equal[current_tonality] = tonality_series_equal[current_tonality][direction_columns_to_keep]


        # STD
        tonality_std_series[current_tonality] = pd.DataFrame(selected_columns_std)
        tonality_std_series[current_tonality].columns = ['std pitch value']
        tonality_std_series[current_tonality]['note_grade'] = short_dict_tonalities_players[current_tonality]['note_grade']
        tonality_std_series[current_tonality]['pitch'] = short_dict_tonalities_players[current_tonality]['pitch']
        tonality_std_series[current_tonality]['direction'] = short_dict_tonalities_players[current_tonality]['direction']

        tonality_std_series_ascending[current_tonality] = tonality_std_series[current_tonality][tonality_std_series[current_tonality]['direction'] == 'Ascending']
        direction_columns_to_keep_std = [column for column in tonality_std_series_ascending[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_std_series_ascending[current_tonality] = tonality_std_series_ascending[current_tonality][direction_columns_to_keep_std]

        tonality_std_series_descending[current_tonality] = tonality_std_series[current_tonality][tonality_std_series[current_tonality]['direction'] == 'Descending']
        direction_columns_to_keep_std = [column for column in tonality_std_series_descending[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_std_series_descending[current_tonality] = tonality_std_series_descending[current_tonality][direction_columns_to_keep_std]

        tonality_std_series_equal[current_tonality] = tonality_std_series[current_tonality][tonality_std_series[current_tonality]['direction'] == 'Equal']
        direction_columns_to_keep_std = [column for column in tonality_std_series_equal[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_std_series_equal[current_tonality] = tonality_std_series_equal[current_tonality][direction_columns_to_keep_std]    

        #MEDIAN
        tonality_median_series[current_tonality] = pd.DataFrame(selected_columns_median)
        tonality_median_series[current_tonality].columns = ['median pitch value']
        tonality_median_series[current_tonality]['note_grade'] = short_dict_tonalities_players[current_tonality]['note_grade']
        tonality_median_series[current_tonality]['pitch'] = short_dict_tonalities_players[current_tonality]['pitch']
        tonality_median_series[current_tonality]['direction'] = short_dict_tonalities_players[current_tonality]['direction']

        tonality_median_series_ascending[current_tonality] = tonality_median_series[current_tonality][tonality_median_series[current_tonality]['direction'] == 'Ascending']
        direction_columns_to_keep_median = [column for column in tonality_median_series_ascending[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_median_series_ascending[current_tonality] = tonality_median_series_ascending[current_tonality][direction_columns_to_keep_median]

        tonality_median_series_descending[current_tonality] = tonality_median_series[current_tonality][tonality_median_series[current_tonality]['direction'] == 'Descending']
        direction_columns_to_keep_median = [column for column in tonality_median_series_descending[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_median_series_descending[current_tonality] = tonality_median_series_descending[current_tonality][direction_columns_to_keep_median]

        tonality_median_series_equal[current_tonality] = tonality_median_series[current_tonality][tonality_median_series[current_tonality]['direction'] == 'Equal']
        direction_columns_to_keep_median = [column for column in tonality_median_series_equal[current_tonality].columns if 'direction' not in column or 'direction' not in column.lower()]
        tonality_median_series_equal[current_tonality] = tonality_median_series_equal[current_tonality][direction_columns_to_keep_median]



        tonality_series[current_tonality] =  tonality_series[current_tonality].groupby(['note_grade', 'direction']).mean()
        tonality_std_series[current_tonality] = tonality_series[current_tonality].groupby(['note_grade', 'direction']).mean()
        tonality_series_ascending[current_tonality] = tonality_series_ascending[current_tonality].groupby(['note_grade']).mean()
        tonality_series_descending[current_tonality] = tonality_series_descending[current_tonality].groupby(['note_grade']).mean()
        tonality_series_equal[current_tonality] = tonality_series_equal[current_tonality].groupby(['note_grade']).mean()

        tonality_std_series_ascending[current_tonality] = tonality_std_series_ascending[current_tonality].groupby(['note_grade']).mean()
        tonality_std_series_descending[current_tonality] = tonality_std_series_descending[current_tonality].groupby(['note_grade']).mean()
        tonality_std_series_equal[current_tonality] = tonality_std_series_equal[current_tonality].groupby(['note_grade']).mean()

        tonality_median_series[current_tonality] = tonality_median_series[current_tonality].groupby(['note_grade', 'direction']).mean()
        tonality_median_series_ascending[current_tonality] = tonality_median_series_ascending[current_tonality].groupby(['note_grade']).mean()
        tonality_median_series_descending[current_tonality] = tonality_median_series_descending[current_tonality].groupby(['note_grade']).mean()
        tonality_median_series_equal[current_tonality] = tonality_median_series_equal[current_tonality].groupby(['note_grade']).mean()


    main_table = pd.concat(tonality_series, axis=1)
    main_table_std = pd.concat(tonality_std_series, axis=1)
    main_table_median = pd.concat(tonality_median_series, axis=1)


    main_ascending = pd.concat(tonality_series_ascending, axis=1)
    main_descending = pd.concat(tonality_series_descending, axis=1)
    main_equal = pd.concat(tonality_series_equal, axis=1)

    main_ascending_std = pd.concat(tonality_std_series_ascending, axis=1)
    main_descending_std = pd.concat(tonality_std_series_descending, axis=1)
    main_equal_std = pd.concat(tonality_std_series_equal, axis=1)

    main_median_ascending = pd.concat(tonality_median_series_ascending, axis=1)
    main_median_descending = pd.concat(tonality_median_series_descending, axis=1)
    main_median_equal = pd.concat(tonality_median_series_equal, axis=1)
        
    cm = sns.light_palette("green", as_cmap=True)
    cm1 = sns.light_palette("red", as_cmap=True)
    cm2 = sns.light_palette("blue", as_cmap=True)
    cm3 = sns.light_palette("yellow", as_cmap=True)


    display(main_ascending.T.style.background_gradient(cmap=cm1, axis=1))
    display(main_descending.T.style.background_gradient(cmap=cm2, axis=1))
    display(main_equal.T.style.background_gradient(cmap=cm3, axis=1))

    display(main_ascending_std.T.style.background_gradient(cmap=cm1, axis=1))
    display(main_descending_std.T.style.background_gradient(cmap=cm2, axis=1))
    display(main_equal_std.T.style.background_gradient(cmap=cm3, axis=1))

    display(main_median_ascending.T.style.background_gradient(cmap=cm1, axis=1))
    display(main_median_descending.T.style.background_gradient(cmap=cm2, axis=1))
    display(main_median_equal.T.style.background_gradient(cmap=cm3, axis=1))
    return main_table, main_table_std, main_table_median, main_ascending, main_descending, main_equal, main_ascending_std, main_descending_std, main_equal_std, main_median_ascending, main_median_descending, main_median_equal

def get_overall_coverage(pshort_coverage_dict):
    """
    This function takes a dictionary with the coverage of the pitch bend values for each note and returns the overall coverage mean.
    """
    mean_values_by_key = {}
    for key, inner_dict in pshort_coverage_dict.items():
        mean_values_by_key[key] = sum(inner_dict.values()) / len(inner_dict)

    overall_coverage_mean = sum(mean_values_by_key.values()) / len(mean_values_by_key)

    print(f"Overall mean: {overall_coverage_mean}")
    return overall_coverage_mean
