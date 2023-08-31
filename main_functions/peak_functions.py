import numpy as np
import pandas as pd
import scipy
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# pre-processing and trimming functions & peak energy and comparison functions

def find_next_peak(peaks, next_peaks, values, prominence_threshold=1.5):
    """
    This function takes a list of peaks, a list of next peaks, the values of the signal and a prominence threshold.
    It returns the index of the next peak that has a prominence higher than the threshold.
    """
    for next_peak in next_peaks:
        prominence = values[next_peak] - values[peaks[0]]
        print('next_peak', next_peak, 'prominence', prominence)
        if prominence >= prominence_threshold:
            return next_peak
    return None

def look_for_coverage(values, trimmed_values):
    """
    This function takes the values of a signal and the trimmed values and returns the coverage of the trimmed values.
    """
    return (len(trimmed_values) / len(values)) * 100

def trim_by_peaks_prominence(values, time, plot_yes=True):
    """
    This function takes the values of a signal and the time 
    and returns the trimmed values and time, trimmed by the prominence of the peaks.
    It works via peaks and prominences.
    It also returns the coverage of the trimmed values.
    """
    #no_trimmed = False
    positive_peaks, _ = find_peaks(values)
    really_high_positive_peaks, _ = find_peaks(values, prominence=5)
    negative_peaks, _ = find_peaks(-values)
    really_high_negative_peaks, _ = find_peaks(-values, prominence=5)

    display('positive_peaks', positive_peaks, 'really_high_positive_peaks', really_high_positive_peaks)
    display('negative_peaks', negative_peaks, 'really_high_negative_peaks', really_high_negative_peaks)
    if len(really_high_positive_peaks) > 0:
        positive_peaks = np.setdiff1d(positive_peaks, really_high_positive_peaks)
        next_peak_index = find_next_peak(really_high_positive_peaks, positive_peaks, values)
    elif len(really_high_negative_peaks) > 0:
        negative_peaks = np.setdiff1d(negative_peaks, really_high_negative_peaks)
        next_peak_index = find_next_peak(really_high_negative_peaks, negative_peaks, values)
    else:
        next_peak_index = None

    if next_peak_index is not None and len(positive_peaks) > 2 and len(negative_peaks) > 2:
        trimmed_values = values[next_peak_index:positive_peaks[-1]]
        trimmed_time = time[next_peak_index:positive_peaks[-1]]
    elif next_peak_index is None and len(positive_peaks) > 2 and len(negative_peaks) > 2:
        trimmed_values = values[positive_peaks[0]:positive_peaks[-1]]
        trimmed_time = time[positive_peaks[0]:positive_peaks[-1]]
    else:
        trimmed_values = values
        trimmed_time = time
        #no_trimmed = True

    coverage = look_for_coverage(values, trimmed_values)
    
    if plot_yes:
        plt.figure(figsize=(15, 5))
        plt.plot(time, values)
        plt.plot(trimmed_time, trimmed_values)
        if len(really_high_positive_peaks) > 0 or len(really_high_negative_peaks) > 0:
            if len(really_high_positive_peaks) > 0:
                for i in really_high_positive_peaks:
                    plt.plot(time[i], values[i], "x")
                plt.plot(time[i+1], values[i+1], "o")
                plt.plot(time[i+2], values[i+2], "+")
            if len(really_high_negative_peaks) > 0:
                for i in really_high_negative_peaks:
                    plt.plot(time[i], values[i], "x")
                plt.plot(time[i+1], values[i+1], "o")
                plt.plot(time[i+2], values[i+2], "+")
        if len(positive_peaks) > 0 or len(negative_peaks) > 0:
            for i in positive_peaks:
                plt.plot(time[i], values[i], "x")
            plt.plot(time[positive_peaks[-1]], values[positive_peaks[-1]], "*")
            for i in negative_peaks:
                plt.plot(time[i], values[i], "x")
            plt.plot(time[negative_peaks[-1]], values[negative_peaks[-1]], "*")
            
        plt.legend(['values', 'trimmed_values'])
        plt.show()
    return trimmed_values, trimmed_time, coverage

def calculate_slope(x1, y1, x2, y2):
    """
    This function calculates the slope between two points.
    """
    return (y2 - y1) / (x2 - x1)

def trim_start_end_slope(values, time, window_size, threshold_slope, plot_yes=True, index_number=0):
    """
    This function takes the values of a signal, the time, the window size and the threshold slope.
    It returns the stabilized values and time after trimming the original signal by looking at the slope of its start and end.
    """
    stabilized_index_start = None

    for i in range(len(values) - window_size):
        window_slope = calculate_slope(time[i], values[i], time[i + window_size], values[i + window_size])
        
        if abs(window_slope) < threshold_slope :
            stabilized_index_start = i + window_size
            break

    stabilized_index_end = None
    if stabilized_index_start is not None:
        for i in range(len(values) - window_size - 1, stabilized_index_start, -1):
            window_slope = calculate_slope(time[i - window_size], values[i - window_size], time[i], values[i])
            
            if window_slope > -threshold_slope:
                stabilized_index_end = i - window_size + 1 
                break

    if stabilized_index_end is not None:
        values_trimmed = values[stabilized_index_start:stabilized_index_end]
        time_trimmed = time[stabilized_index_start:stabilized_index_end]
    elif stabilized_index_start is not None:
        values_trimmed = values[stabilized_index_start:]
        time_trimmed = time[stabilized_index_start:]

    if plot_yes:
        plt.figure(figsize=(8, 5))
        plt.plot(time, values, 'bo-')
        plt.xlabel('Samples')
        plt.ylabel('Values')
        plt.title(f'Original Time Series Wohlfahrt-Op45-26-ZYZ {index_number}')

        if stabilized_index_start is not None:
            plt.plot(time[stabilized_index_start:], values[stabilized_index_start:], 'r--', label='Stabilized Time Series (Start)')
            if stabilized_index_end is not None:
                plt.plot(time[stabilized_index_start:stabilized_index_end], values[stabilized_index_start:stabilized_index_end], 'm--', label='Stabilized Time Series (End)')
            plt.axvline(time[stabilized_index_start], color='g', linestyle='--', label='Stabilized Index (Start)')
            if stabilized_index_end is not None:
                plt.axvline(time[stabilized_index_end], color='y', linestyle='--', label='Stabilized Index (End)')
            plt.legend()

        plt.grid(True)
        plt.show()

    stabilized_values = values[stabilized_index_start:stabilized_index_end]
    stabilized_time = time[stabilized_index_start:stabilized_index_end]
    return stabilized_values, stabilized_time

def trim_by_peaks(values, time, plot_yes=True):
    """
    This function takes the values of a signal and the time and returns the trimmed values and time.
    It works via peaks.
    It also returns the coverage of the trimmed values.
    """
    no_trimmed = False
    positive_peaks, _ = find_peaks(values)
    really_high_positive_peaks, _ = find_peaks(values, prominence=5)
    negative_peaks, _ = find_peaks(-values)
    really_high_negative_peaks, _ = find_peaks(-values, prominence=5)

    if len(positive_peaks) > 2 and len(negative_peaks) > 2:
        if negative_peaks[0] == (positive_peaks[0] + 1):
            trimmed_values = values[positive_peaks[1]:positive_peaks[-1]]
            trimmed_time = time[positive_peaks[1]:positive_peaks[-1]]
            coverage = look_for_coverage(values, trimmed_values)
        else:
            trimmed_values = values[positive_peaks[0]:positive_peaks[-1]]
            trimmed_time = time[positive_peaks[0]:positive_peaks[-1]]
            coverage = look_for_coverage(values, trimmed_values)

        if len(trimmed_values)==0:
            trimmed_values = values
            trimmed_time = time
            coverage = look_for_coverage(values, trimmed_values)

        if plot_yes:
            plt.figure(figsize=(8, 6))
            plt.plot(time, values)
            if len(really_high_positive_peaks) > 0 or len(really_high_negative_peaks) > 0:
                if len(really_high_positive_peaks) > 0:
                    for i in really_high_positive_peaks:
                        plt.plot(time[i], values[i], "x")
                    plt.plot(time[i+1], values[i+1], "o")
                    plt.plot(time[i+2], values[i+2], "+")
                if len(really_high_negative_peaks) > 0:
                    for i in really_high_negative_peaks:
                        plt.plot(time[i], values[i], "x")
                    plt.plot(time[i+1], values[i+1], "o")
                    plt.plot(time[i+2], values[i+2], "+")

            for i in positive_peaks:
                plt.plot(time[i], values[i], "x")
            for i in negative_peaks:
                plt.plot(time[i], values[i], "x")
            plt.plot(trimmed_time, trimmed_values, 'red', label='Trimmed Time Series')
            plt.xlabel('Time')
            plt.ylabel('Intonation deviation (cents)')
            plt.show()
    else:
        trimmed_values = values
        trimmed_time = time
        no_trimmed = True
        coverage = look_for_coverage(values, trimmed_values)
        if plot_yes:
            plt.figure(figsize=(8, 6))
            plt.plot(time, values)
            plt.title('No trimmed values')
            plt.show()
    
    return trimmed_values, trimmed_time, no_trimmed, coverage

def hist_pitch_max(num_bins, stabilized_values, plot_yes=False):
    """
    This function takes the number of bins, the stabilized values and a boolean to plot the histogram.
    It returns the mean of the bins with the highest counts.
    It converts a pitch bend time series into a single representing value.
    """
    max_value = -1
    i_anterior = 0
    first = 1
    count_value_anterior = -1
    count_same_value = 0
    count_second_value = 0
    count_third_value = 0

    hist, bins = np.histogram(stabilized_values, bins=num_bins)
    bin_index_with_highest_count = np.argmax(hist)
    value_with_highest_count = bins[bin_index_with_highest_count]
    #print('Value with highest count:', value_with_highest_count)

    if plot_yes:
        plt.figure(figsize=(8, 5))
        plt.hist(stabilized_values, bins=num_bins)
        plt.xlabel('Bins')
        plt.ylabel('Counts')
        plt.title('Histogram of Pitch Bend Values')
        plt.grid(True)
        plt.show()

    sorted_indices = np.argsort(-hist)
    
    for count_value in hist[sorted_indices]:
        if first:
            count_value_anterior = count_value
            first = 0
        if count_value == count_value_anterior:
            count_same_value += 1
            count_value_anterior = count_value
        elif count_value == (count_value_anterior-1) and count_value>=2:
            count_second_value +=1
        elif count_value == (count_value_anterior-1) and count_value>=2:
            count_third_value +=1
        else:
            break
    
    #print('hist[sorted_indices[:count_same_value]]', hist[sorted_indices[:count_same_value]])
    first = 1
    sum_all_indexes = 0

    for bin_index in sorted_indices[:count_same_value]:
        sum_all_indexes += bins[bin_index]
    mean_all_indexes = sum_all_indexes / count_same_value 

    return mean_all_indexes

def calculate_peak_relative_energy(signal, plot_yes=False):
    """
    This function takes a signal and returns the peak relative energy.
    """ 
    x = (np.fft.rfftfreq(len(signal), d = 1/172.41))
    y = np.abs(np.fft.rfft(signal))

    peaks, _ = find_peaks(y)
    if len(peaks) == 0:
        print("No peaks found")
        main_frequency = 0
        peak_relative_energy = -1
    else:

        power_spectrum = y ** 2
        frequency_range = (4, 10)

        indices_of_interest = np.where((x >= frequency_range[0]) & (x <= frequency_range[1]))
        peak_relative_energy = np.sum(power_spectrum[indices_of_interest])
        power_spectrum_db = 10 * np.log10(power_spectrum)

        if plot_yes:
            plt.figure(figsize=(10, 5))
            plt.plot(x, power_spectrum_db, label='Power Spectrum dB')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectrum dB')
            plt.title('Power Spectrum of the Signal (rFFT)')
            plt.grid()

            ax = plt.gca()
            ax.axvspan(4, 10, facecolor='green', alpha=0.3)
            plt.legend()
            plt.show()
            
    return peak_relative_energy

def look_energy_comparison(signal_data):
    """
    This function takes a signal and returns the energy comparison ratio 
    from the energy in vibrato range vs the total energy.
    """
    rfft_result = np.fft.rfft(signal_data)
    total_energy_entire_range = np.sum(np.abs(rfft_result)**2)  #Total energy in the entire rfft range

    freqs = np.fft.rfftfreq(len(signal_data), d=1/172.41) #indices corresponding to the vibrato range
    total_energy_vibrato_range = np.sum(np.abs(rfft_result[(freqs >= 4) & (freqs <= 10)])**2) #total energy in the vibrato range

    energy_comparison_ratio = total_energy_vibrato_range / total_energy_entire_range # total energy comparison ratio

    print("Total energy in the vibrato range (4-10 Hz):", total_energy_vibrato_range)
    print("Total energy in the entire rfft curve:", total_energy_entire_range)
    print("Energy comparison ratio:", energy_comparison_ratio)
    return energy_comparison_ratio

def draw_vibrato_plots(my_time, signal_interpolated, mean, amplitude):
    """
    This function takes the time and signal arrays of a pitch bend (usually with vibrato)
    and displays the interpolated signals among their peaks, the mean value and amplitude
    """
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(my_time, signal_interpolated, label='Signal (Interpolated)')
    plt.plot(my_time, mean, label='Mean', color='green')
    plt.legend()
    plt.title("Interpolated Signal and Mean")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (cents)")

    plt.subplot(2, 1, 2)
    plt.plot(my_time, signal_interpolated, label='Signal (Interpolated)')
    plt.plot(my_time, amplitude, label='Amplitude')
    plt.legend()
    plt.title("Interpolated Signal and Amplitude")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (cents)")
    plt.show()
