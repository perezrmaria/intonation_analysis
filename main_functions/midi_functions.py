# format from MIDI to dict/df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pretty_midi

import os
from tqdm import tqdm

def look_longest_notes(midi_data):
    """
    This function takes a midi file and returns a dataframe with the start time, end time, duration and pitch of each note.
    It also returns a dataframe with the 10 longest notes.
    """
    all_notes = []
    for instrument in midi_data.instruments:
        all_notes += instrument.notes

    all_notes_sorted = sorted(all_notes, key=lambda note: note.start, reverse=False)
    notes_dict = {
        'start_time': [note.start for note in all_notes_sorted],
        'end_time': [note.end for note in all_notes_sorted],
        'duration': [note.end - note.start for note in all_notes_sorted],
        'pitch': [note.pitch for note in all_notes_sorted]
    }

    notes_df = pd.DataFrame(notes_dict)
    notes_df['pitch'] = notes_df['pitch'].apply(pretty_midi.note_number_to_name)

    # Initialize the previous_notes list
    previous_notes1 = [None] * len(notes_df)
    previous_notes2 = [None] * len(notes_df)
    previous_notes3 = [None] * len(notes_df)
    next_notes1 = [None] * len(notes_df)
    next_notes2 = [None] * len(notes_df)

    # Iterate over the notes and update the previous_notes list for each note
    for i, note in notes_df.iterrows():
        # Check if index is within bounds
        if i<(len(notes_df)-1):
            next_notes1[i] = notes_df.loc[i+1, 'pitch']
        if i<(len(notes_df)-2):
            next_notes2[i] = notes_df.loc[i+2, 'pitch']
        if i >= 1:
            previous_notes1[i] = notes_df.loc[i-1, 'pitch']
        if i >= 2:
            previous_notes2[i] = notes_df.loc[i-2, 'pitch']
        if i >= 3:
            previous_notes3[i] = notes_df.loc[i-3, 'pitch']

    notes_df['previous_notes_1'] = previous_notes1
    notes_df['previous_notes_2'] = previous_notes2
    notes_df['previous_notes-3'] = previous_notes3
    notes_df['next_notes_1'] = next_notes1
    notes_df['next_notes_2'] = next_notes2

    longest_notes = notes_df.sort_values('duration', ascending=False).head(10)
    return notes_df, longest_notes

def pitch_bend_to_cents(pitch_bend_value):
    """
    This function takes a pitch bend value 
    and returns the corresponding pitch bend in cents.
    """
    pitch_bend_value += 8192
    cents = (pitch_bend_value - 8192) / 8192 * 100
    return cents

def create_dic_all_method(main_folder, low_duration = 0, high_duration = 1000):
    """
    This function takes a folder with midi files 
    and returns a dictionary with the pitch bend and time values for each note, separated into short and long.
    It diferenciates into short and long notes automatically 
    following the duration parameters defined in the master thesis memory.
    It also returns the length of every midi.
    """
    all_info_method = {}
    all_time_method = {}
    len_every_mid ={}
    long_final_notes = {}
    long_final_times = {}

    for folder_name in tqdm(os.listdir(main_folder)):
        folder_path = os.path.join(main_folder, folder_name)

        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for one_file in files:
                if one_file == ".DS_Store":
                    continue
                file_path = os.path.join(folder_path, one_file)
                name_part = one_file.split('_')[0]+'-'+one_file.split('_')[1]+'-'+one_file.split('_')[2]
                #print(name_part)
                if file_path.endswith('.mid'):
                    midi_data = pretty_midi.PrettyMIDI(file_path)
                    notes_df, longest_notes = look_longest_notes(midi_data)
                    notes_df = notes_df[(notes_df['duration'] < high_duration) & (notes_df['duration'] > low_duration)]

                    dic_pitch_bends = {}
                    dic_time_bends = {}
                    dic_long_final_notes = {}
                    dic_long_final_times = {}
                    all_pitch_bends_one_mid = []
                    len_mysignal = []

                    for instrument in midi_data.instruments:
                        all_pitch_bends_one_mid += instrument.pitch_bends

                    for note in notes_df.itertuples():
                        pitch_bends_in_longest_note = [event for event in all_pitch_bends_one_mid if note.start_time <= event.time < note.end_time]

                        pitch_bends_in_longest_note = sorted(pitch_bends_in_longest_note, key=lambda event: event.time, reverse=False)
                        pitch_bend_times = [event.time for event in pitch_bends_in_longest_note]
                        pitch_bend_values = [event.pitch for event in pitch_bends_in_longest_note]

                        pitch_bend_values = [pitch_bend_values] if isinstance(pitch_bend_values, int) else pitch_bend_values
                        for i in range(len(pitch_bend_values)):
                            pitch_bend_values[i] = np.array(pitch_bend_to_cents(pitch_bend_values[i])).astype(np.float64)


                        my_signal = list(pitch_bend_values)
                        my_time = list(pitch_bend_times)
                        len_mysignal.append(len(my_signal))

                        # diference between short and long notes (by following the score)
                        if note.Index == notes_df.index[-1]:
                            dic_long_final_notes[note.Index] = my_signal
                            dic_long_final_times[note.Index] = my_time
                        else:
                            dic_pitch_bends[note.Index] = my_signal
                            dic_time_bends[note.Index] = my_time
                            len_every_mid[note.Index] = len_mysignal

                    all_info_method[name_part] = dic_pitch_bends
                    all_time_method[name_part] = dic_time_bends
                    len_every_mid[name_part] = len_mysignal
                    long_final_notes[name_part] = dic_long_final_notes
                    long_final_times[name_part] = dic_long_final_times

    return all_info_method, all_time_method, long_final_notes, long_final_times, len_every_mid

def create_dic_all_method_original(main_folder, low_duration = 0 , high_duration = 1000):
    """
    This function takes a folder with midi files 
    and returns a dictionary with the pitch bend values for each note.
    It diferenciates into short and long notes following 
    the duration parameters of low and high duration.
    """
    all_info_method = {}
    all_time_method = {}
    len_every_mid ={}

    for folder_name in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder_name)

        if os.path.isdir(folder_path): 
            files = os.listdir(folder_path)
            for one_file in files:
                if one_file == ".DS_Store":
                    continue
                file_path = os.path.join(folder_path, one_file)
                name_part = one_file.split('_')[0]+'-'+one_file.split('_')[1]+'-'+one_file.split('_')[2]
                #print(name_part)
                if file_path.endswith('.mid'):
                    midi_data = pretty_midi.PrettyMIDI(file_path)
                    notes_df, longest_notes = look_longest_notes(midi_data)
                    notes_df = notes_df[(notes_df['duration'] < high_duration) & (notes_df['duration'] > low_duration)]
                    dic_pitch_bends_zero_mean_one_mid = {}
                    dic_pitch_bends = {}
                    dic_time_bends = {}
                    all_pitch_bends_one_mid = []
                    len_mysignal = []

                    for instrument in midi_data.instruments:
                        all_pitch_bends_one_mid += instrument.pitch_bends

                    for note in notes_df.itertuples():
                        pitch_bends_in_longest_note = [event for event in all_pitch_bends_one_mid if note.start_time <= event.time < note.end_time]
                        #sort pitch bend events by time
                        pitch_bends_in_longest_note = sorted(pitch_bends_in_longest_note, key=lambda event: event.time, reverse=False)
                        pitch_bend_times = [event.time for event in pitch_bends_in_longest_note]
                        pitch_bend_values = [event.pitch for event in pitch_bends_in_longest_note]

                        pitch_bend_values = [pitch_bend_values] if isinstance(pitch_bend_values, int) else pitch_bend_values
                        for i in range(len(pitch_bend_values)):
                            pitch_bend_values[i] = np.array(pitch_bend_to_cents(pitch_bend_values[i])).astype(np.float64)

                        my_signal = list(pitch_bend_values)
                        my_time = list(pitch_bend_times)
                        len_mysignal.append(len(my_signal))

                        #zero_mean_sinusoid = list(my_signal - mean_value)
                        #dic_pitch_bends_zero_mean_one_mid[note.Index] = zero_mean_sinusoid

                        dic_pitch_bends[note.Index] = my_signal
                        dic_time_bends[note.Index] = my_time

                    all_info_method[name_part] = dic_pitch_bends
                    all_time_method[name_part] = dic_time_bends
                    len_every_mid[name_part] = len_mysignal
    return all_info_method, all_time_method, len_every_mid

def formatting_all_info(all_info_method, max_num_samples=180, min_num_samples=509):    
    """
    For clustering
    This function takes a dictionary with the pitch bend values for each note 
    and returns a formatted array.
    """
    list_of_lists = []
    list_more_than25 = []
    max_len = 0

    for key, value in all_info_method.items():
        for note, inner_list in value.items():
            if len(inner_list) >= min_num_samples and len(inner_list) <= max_num_samples:
                list_of_lists.append(inner_list)
            else:
                list_more_than25.append(inner_list)
            if len(inner_list) > max_len:
                max_len = len(inner_list)
                max_key = key
                max_value = value

    max_length = max(len(sublist) for sublist in list_of_lists if sublist)
    indices = [i for i, sublist in enumerate(list_of_lists) if len(sublist) == max_length]

    interpolated_time_series = []
    for sublist in list_of_lists:
        original_length = len(sublist)
        new_x = np.linspace(0, 1, max_length)
        interpolated_sublist = []
        for item in sublist:
            interpolated_item = np.array(item, dtype=float)
            interpolated_sublist.append(interpolated_item)
        interpolated_sublist = np.array(interpolated_sublist)
        x = np.linspace(0, 1, original_length)
        
        interpolated = np.interp(new_x, x, interpolated_sublist)
        interpolated_time_series.append(interpolated)

    #reshape the array to match the required input format
    X = np.array(interpolated_time_series)
    print('Shape before formatting',X.shape)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print('Shape after formatting',X.shape)
    return X
# Conversion details from grades to notes for all studied tonalities
gradestoNote_C_Maj = {'I':'C', 'II':'D', 'III':'E', 'IV':'F', 'V':'G', 'VI':'A', 'VII':'B'}
gradestoNote_G_Maj = {'I':'G', 'II':'A', 'III':'B', 'IV':'C', 'V':'D', 'VI':'E', 'VII':'F#'}
gradestoNote_F_Maj = {'I':'F', 'II':'G', 'III':'A', 'IV':'A#', 'V':'C', 'VI':'D', 'VII':'E'}
gradestoNote_Eb_Maj = {'I':'D#', 'II':'F', 'III':'G', 'IV':'G#', 'V':'A#', 'VI':'C', 'VII':'D'}
gradestoNote_A_Maj = {'I':'A', 'II':'B', 'III':'C#', 'IV':'D', 'V':'E', 'VI':'F#', 'VII':'G#'}
# Conversion details from notes to grades for all studied tonalities
notetoGrades_C_Maj = {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}
notetoGrades_G_Maj = {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}
notetoGrades_F_Maj = {'F':'I', 'G':'II', 'A':'III', 'A#':'IV', 'C':'V', 'D':'VI', 'E':'VII'}
notetoGrades_Eb_Maj = {'D#':'I', 'F':'II', 'G':'III', 'G#':'IV', 'A#':'V', 'C':'VI', 'D':'VII'}
notetoGrades_A_Maj = {'A':'I', 'B':'II', 'C#':'III', 'D':'IV', 'E':'V', 'F#':'VI', 'G#':'VII'}

# Relationships between etudes and tonalities
tonalities = {'C':notetoGrades_C_Maj, 'G':notetoGrades_G_Maj, 'F':notetoGrades_F_Maj, 'Eb':notetoGrades_Eb_Maj, 'A':notetoGrades_A_Maj}
tonalities_etudes = {'Wohlfahrt-Op45-01': 'C', 'Wohlfahrt-Op45-03': 'G', 'Wohlfahrt-Op45-05': 'F', 'Wohlfahrt-Op45-10': 'A', 'Wohlfahrt-Op45-11': 'Eb', 'Wohlfahrt-Op45-15': 'C', 'Wohlfahrt-Op45-26': 'G'}
tonalities_etudes_no26 = {'Wohlfahrt-Op45-01': 'C', 'Wohlfahrt-Op45-03': 'G', 'Wohlfahrt-Op45-05': 'F', 'Wohlfahrt-Op45-10': 'A', 'Wohlfahrt-Op45-11': 'Eb', 'Wohlfahrt-Op45-15': 'C'}

tonalities_etudes_gradestoNote = {'Wohlfahrt-Op45-01': {'I':'C', 'II':'D', 'III':'E', 'IV':'F', 'V':'G', 'VI':'A', 'VII':'B'}, 'Wohlfahrt-Op45-03': {'I':'G', 'II':'A', 'III':'B', 'IV':'C', 'V':'D', 'VI':'E', 'VII':'F#'}, 'Wohlfahrt-Op45-05': {'I':'F', 'II':'G', 'III':'A', 'IV':'A#', 'V':'C', 'VI':'D', 'VII':'E'}, 'Wohlfahrt-Op45-10': {'I':'A', 'II':'B', 'III':'C#', 'IV':'D', 'V':'E', 'VI':'F#', 'VII':'G#'}, 'Wohlfahrt-Op45-11': {'I':'D#', 'II':'F', 'III':'G', 'IV':'G#', 'V':'A#', 'VI':'C', 'VII':'D'}, 'Wohlfahrt-Op45-15': {'I':'C', 'II':'D', 'III':'E', 'IV':'F', 'V':'G', 'VI':'A', 'VII':'B'}, 'Wohlfahrt-Op45-26': {'I':'G', 'II':'A', 'III':'B', 'IV':'C', 'V':'D', 'VI':'E', 'VII':'F#'}}
tonalities_etudes_notetoGrades = {'Wohlfahrt-Op45-01': {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}, 'Wohlfahrt-Op45-03': {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}, 'Wohlfahrt-Op45-05': {'F':'I', 'G':'II', 'A':'III', 'A#':'IV', 'B':'IV#', 'C':'V', 'D':'VI', 'E':'VII'}, 'Wohlfahrt-Op45-10': {'A':'I', 'B':'II', 'C#':'III', 'D':'IV', 'E':'V', 'F#':'VI', 'G#':'VII'}, 'Wohlfahrt-Op45-11': {'D#':'I', 'F':'II', 'G':'III', 'G#':'IV', 'A':'#IV','A#':'V', 'C':'VI', 'D':'VII'}, 'Wohlfahrt-Op45-15': {'C':'I', 'D':'II', 'E':'III', 'F':'IV', 'G':'V', 'A':'VI', 'B':'VII'}, 'Wohlfahrt-Op45-26': {'G':'I', 'A':'II', 'B':'III', 'C':'IV', 'D':'V', 'E':'VI', 'F#':'VII'}}
