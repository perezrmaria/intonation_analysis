import pandas as pd
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
import statistics

# Functions used to develop the trimming algorithm for the segmentation process

def midi_work(midi_file, plot_yes=True, chroma=True, df_sub=False):
    """Returns a DataFrame with mean and standard deviation for each note in a midi file.
    If plot is True, a plot is created.
    If chroma is True, the DataFrame is also created for the chroma.
    If df_sub is True, a DataFrame is created for all notes between C3 and C5 
    (df_sub is specially for Paganini scores where much higher notes are played).
    """
    pitch = 0
    dict_instrument = {}
    dict_instrument_chroma = {}

    midi_data = pretty_midi.PrettyMIDI(midi_file)
    for instrument in midi_data.instruments:
        notes = instrument.notes
        note = [note.pitch for note in notes][0]
        
        pitch_bends = instrument.pitch_bends

        mean_pitch_bend = sum([pitch_bend_to_cents(pb.pitch) for pb in pitch_bends])/len(pitch_bends)
        std_pitch_bend = statistics.stdev([pitch_bend_to_cents(pb.pitch) for pb in pitch_bends])
        dict_instrument[note] = {'mean_pitch_bend': mean_pitch_bend, 'std_pitch_bend': std_pitch_bend}

        if chroma:
            note_class = [note.pitch % 12 for note in notes][0] 
            dict_instrument_chroma[note_class] = {'mean_pitch_bend': mean_pitch_bend, 'std_pitch_bend': std_pitch_bend}
    if plot_yes:
        print('Pitch bends for all notes:')
        print(dict_instrument)
    df = dict_from_df(dict_instrument, plot_yes=plot_yes)
    max_min_mean_std(df, plot_yes=plot_yes)
    if plot_yes:
        plot_mean_and_std(df)

    if chroma:
        df_chroma = dict_from_df(dict_instrument_chroma, plot_yes=plot_yes)
        max_min_mean_std(df_chroma, plot_yes=plot_yes)
        if plot_yes:
            print('Pitch bends for chroma:')
            print(dict_instrument_chroma)
            plot_mean_and_std(df_chroma)
    if df_sub:
        df_sub = dict_from_df(dict_instrument, df_sub=True, plot_yes=plot_yes)
        max_min_mean_std(df_sub, plot_yes=plot_yes)
        if plot_yes:
            print('Pitch bends for all notes between C3 and C5:')
            plot_mean_and_std(df_sub)

    return df, df_chroma, df_sub

def folder_work(folder_path, chroma=True, sub=False, return_yes = True, plot_yes=True):
    """Returns a DataFrame with mean and standard deviation for each note in a folder.
    The DataFrame is ordered by note name.
    """

    df = pd.DataFrame()
    df_chroma = pd.DataFrame()
    df_sub = pd.DataFrame()

    pieces_name = np.sort(next(os.walk(folder_path))[2])
    for piece in pieces_name:
        if plot_yes:
            print(piece.split("_")[2])
        player = piece.split("_")[2]

        df_piece, df_chroma_piece, df_sub_piece = midi_work(folder_path + '/' + piece, plot_yes=plot_yes, chroma=chroma, df_sub=sub)
        df = migrate_def(df_piece, df, player)
        order_df_by_note_name(df)
        if chroma:
            df_chroma = migrate_def(df_chroma_piece, df_chroma, player)
            order_df_by_note_name(df_chroma)
        if sub:
            df_sub = migrate_def(df_sub_piece, df_sub, player)
            order_df_by_note_name(df_sub)
        if plot_yes:
            display(df.head(20))
            if chroma:
                display(df_chroma.head(20))
            if sub:
                display(df_sub.head(20))
    if return_yes:
        return df, df_chroma, df_sub

def pitch_bend_to_cents(pitch_bend_value):
    """Converts pitch bend value to cents."""
    pitch_bend_value += 8192
    cents = (pitch_bend_value - 8192) / 8192 * 100
    return cents

def plot_mean_and_std(df):
    """Plot mean and standard deviation for each note in a DataFrame."""
    plt.figure(figsize=(20,10))
    df.plot(kind='bar', y='mean_pitch_bend', yerr='std_pitch_bend', legend=False, figsize=(20, 10))
    plt.legend()
    plt.title('Mean and standard deviation for all notes in the piece')
    plt.show()

def max_min_mean_std(df, plot_yes=True):
    """Prints the maximum and minimum values for mean and standard deviation."""
    max_mean_pitch_bend = df.loc[df['mean_pitch_bend'].idxmax()]
    min_mean_pitch_bend = df.loc[df['mean_pitch_bend'].idxmin()]
    max_std_pitch_bend = df.loc[df['std_pitch_bend'].idxmax()]
    min_std_pitch_bend = df.loc[df['std_pitch_bend'].idxmin()]
    if plot_yes:
        display('max_mean_pitch_bend',max_mean_pitch_bend)
        display('min_mean_pitch_bend',min_mean_pitch_bend)
        display('max_std_pitch_bend',max_std_pitch_bend)
        display('min_std_pitch_bend',min_std_pitch_bend)

def dict_from_df(dict_instrument,note_from_midi=True, df_sub=False, plot_yes=True):
    """Creates a DataFrame from a dictionary.
    If note_from_midi is True, the index is associated with the note name."""
    df = pd.DataFrame.from_dict(dict_instrument, orient='index')
    if df_sub:
        df = df[(df.index >= 55) & (df.index <= 84)]
    df = df.sort_index()
    if note_from_midi:
        df.index = [pretty_midi.note_number_to_name(note) for note in df.index] # associate index with note name
    if plot_yes:
        display(df.head(15))
    return df

def order_df_by_note_name(df_player_note):
    """Specific for the functions midi_work & folder_work.
    Orders the DataFrame by note name."""
    df_player_note = df_player_note.explode('Note')
    df_player_note['Note'] = df_player_note['Note'].apply(lambda x: pretty_midi.note_name_to_number(x))
    df_player_note['Note'] = df_player_note['Note'].astype(int)
    df_player_note = df_player_note.sort_values(by=['Note'])
    df_player_note['Note'] = df_player_note['Note'].apply(lambda x: pretty_midi.note_number_to_name(x))

def migrate_def(df_piece, df, player):
    """Migrates the DataFrame df_piece to df.
    The player name is added to the DataFrame."""
    df_piece.insert(0, 'Player', player)
    df_piece = df_piece.reset_index(drop=False)
    df_piece = df_piece.rename(columns={'index': 'Note'})
    for i in range(len(df_piece)):
        df = df.append(df_piece.iloc[[i]], ignore_index=True)
    return df
