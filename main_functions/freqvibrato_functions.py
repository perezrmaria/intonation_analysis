import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Frequency, interpolation and vibrato modeling functions

def find_frrt_frequency(signal, plot_yes=False):
    """
    This function takes a signal and returns the main frequency.
    Specifically, it returns the frequency with the greatest peak.
    Takes into account the sampling rate of the signal: 5.8 ms.
    """
    x = (np.fft.rfftfreq(len(signal), d = 1/172.41))
    y = np.abs(np.fft.rfft(signal))
    peaks, _ = find_peaks(y)

    if len(peaks) == 0:
        print("No peaks found")
        main_frequency = 0
    else:
        greatest_peak_index = np.argmax(y[peaks])
        greatest_peak_value = y[peaks][greatest_peak_index]

        if plot_yes:
            plt.figure(figsize=(20, 5))
            plt.plot(x,y)
            plt.plot(x[peaks], y[peaks], "x")

            plt.axhline(y=greatest_peak_value, color='r', linestyle='--')
            plt.axvline(x=x[peaks[greatest_peak_index]], color='r', linestyle='--')

            plt.title("Peaks on the rFFT")
            plt.show()
        main_frequency = x[peaks[greatest_peak_index]]
    return main_frequency

def frequency_from_peaks(peaks, limit):
    """
    This function takes a list of peaks and returns the frequency.
    Takes into account the sampling rate of the signal: 5.8 ms.
    """
    result = []
    noisy_values = []
    for i in range(len(peaks) - 1):
        difference = peaks[i+1] - peaks[i]
        period = (difference*5.8)/1000
        vibrato_freq_peaks = 1/period
        if vibrato_freq_peaks > limit:
            noisy_values.append(peaks[i])
        result.append(vibrato_freq_peaks)
    return result, noisy_values

def find_peaks_and_interpolate(my_signal, my_time, name_part, limit=0, plot_yes=False):
    """
    This function takes a signal and the time and returns the interpolated values 
    of the signal around the found positive and negativepeaks.
    It also returns the median of the interpolated values.
    """
    peaks_upp, _ = find_peaks(my_signal, prominence=1.5)
    negative_peaks, _ = find_peaks(-my_signal, prominence=2.5)


    peaks_upp_freq, _ = frequency_from_peaks(peaks_upp, limit)
    negative_peaks_freq, _ = frequency_from_peaks(negative_peaks, limit)
    overall_median = np.median([np.median(peaks_upp_freq), np.median(negative_peaks_freq)])

    if plot_yes:
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        title = 'Peaks on the Signal: ' + name_part
        plt.plot(my_time, my_signal)
        plt.plot(my_time[peaks_upp], my_signal[peaks_upp], "x")
        plt.plot(my_time[negative_peaks], my_signal[negative_peaks], "x")

        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (cents)')
        plt.show()

    if len(peaks_upp) >= 2 and len(negative_peaks) >= 2:
        interp_func_upp = interp1d(peaks_upp, my_signal[peaks_upp], kind='linear')
        interp_func_down = interp1d(negative_peaks, my_signal[negative_peaks], kind='linear')

        interpolation_points_upp = np.linspace(peaks_upp[0], peaks_upp[-1], num=len(my_signal))
        interpolation_points_down = np.linspace(negative_peaks[0], negative_peaks[-1], num=len(my_signal))

        interpolated_values_upp = interp_func_upp(interpolation_points_upp)
        interpolated_values_down = interp_func_down(interpolation_points_down)

        if plot_yes: 
            plt.subplot(1, 2, 2)
            title = 'Interpolation of Peaks: ' + name_part

            plt.plot(my_signal, label='Original Signal')
            plt.plot(interpolation_points_upp, interpolated_values_upp, label='Interpolated Values (Positive Peaks)')
            plt.plot(interpolation_points_down, interpolated_values_down, label='Interpolated Values (Negative Peaks)')

            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel('Amplitude (cents)')
            plt.title(title)
            plt.show()
            plt.close()

    else:
        #print("Not enough peaks for interpolation.")
        interpolated_values_upp = np.zeros(35)
        interpolated_values_down = np.zeros(35)
        overall_median = np.zeros(35)
    
    return interpolated_values_upp, interpolated_values_down, overall_median

def model_vibrato_sinusoid(y, t, name_part, interpolated_values_upp, interpolated_values_down, rfft_frequency, phase_shift, plot_yes=False):
    """
    This function takes a signal, the corresponding time series, frequency, phase shift and interpolated values 
    and returns the sinusoid of the vibrato signal.
    It also returns the mean, amplitude and frequency of the sinusoid.
    """
    mean = np.mean(y)
    amplitude = (np.mean(interpolated_values_upp) - np.mean(interpolated_values_down))/2
    frequency = np.round(rfft_frequency,2)

    print('mean:', mean)
    print('amplitude:', amplitude)
    print('frequency:', frequency)

    sinusoid = mean + amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)
    if plot_yes: 
        
        plt.figure(figsize=(20, 5))
        title = 'Sinusoid vs Signal: ' + name_part

        plt.plot(t[:len(y)], y, label='Signal y')
        plt.plot(t[:len(y)], sinusoid, label='Sinusoid')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (cents)')
        plt.title(title)
        plt.show()

    return sinusoid, mean, amplitude, frequency

def model_time_varying_vibrato_sinusoid(y, t, name_part, mean, amplitude, rfft_frequency, phase_shift, plot_yes=False):
    """
    This function takes a signal, time series, mean, amplitude, rFFT frequency and phase shift 
    and returns the sinusoid of the time-varying vibrato signal.
    """

    freq_list = np.full_like(y, rfft_frequency)
    phase_shift_list = np.full_like(y, phase_shift)

    sinusoid = mean + amplitude * np.sin(2 * np.pi * freq_list * t + phase_shift_list)
    print('sinusoid', len(sinusoid))

    if plot_yes:
        plt.figure(figsize=(20, 5))
        title = 'Sinusoid vs Signal: ' + name_part

        plt.plot(t, y, label='Signal y')
        plt.plot(t, sinusoid, label='Sinusoid')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (cents)')
        plt.title(title)
        plt.show()
    return sinusoid

def apply_lowpass_filter(input_signal, cutoff_frequency, sampling_frequency):
    """
    This function takes a signal, the cutoff frequency and the sampling frequency 
    and returns the filtered signal.
    """
    nyquist_frequency = 0.5 * sampling_frequency
    normalized_cutoff = cutoff_frequency / nyquist_frequency

    b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
    filtered_signal = signal.lfilter(b, a, input_signal)

    return filtered_signal
