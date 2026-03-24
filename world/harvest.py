#build-in imports
from decimal import Decimal, ROUND_HALF_UP
import copy
import math

# 3rd-party imports
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import lfilter
from scipy import signal
from scipy.fftpack import fft
import numba

EPS = 2.220446049250313e-16

def hv_cand_num(x: np.ndarray, fs: int, f0_floor: int=71, f0_ceil: int=800, frame_period: int=5) -> int:
    basic_frame_period: int = 1
    target_fs = 8000
    num_samples = int(1000 * len(x) / fs / basic_frame_period + 1)
    basic_temporal_positions = np.arange(0, num_samples) * basic_frame_period / 1000
    channels_in_octave = 40
    f0_floor_adjusted = f0_floor * 0.9
    f0_ceil_adjusted = f0_ceil * 1.1

    boundary_f0_list = np.arange(np.ceil(np.log2(f0_ceil_adjusted / f0_floor_adjusted) * channels_in_octave)) + 1
    boundary_f0_list = boundary_f0_list / channels_in_octave
    boundary_f0_list = 2.0 ** boundary_f0_list
    boundary_f0_list *= f0_floor_adjusted

    # down - sampling to target_fs Hz
    [y, actual_fs] = CalculateDownsampledSignal(x, fs, target_fs)
    fft_size = int(2 ** np.ceil(np.log2(len(y) + int(fs / f0_floor_adjusted * 4 + 0.5) + 1)))
    y_spectrum = np.fft.fft(y, fft_size)

    # Calculate from 4 points
    raw_f0_candidates = CalculateCandidates(len(basic_temporal_positions), boundary_f0_list, len(y),
                                            basic_temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil)
    # 
    f0_candidates, number_of_candidates = DetectCandidates(raw_f0_candidates)

    return number_of_candidates


############################################################################################
def sawed_harvest(x: np.ndarray, fs: int, f0_floor: int=71, f0_ceil: int=800, frame_period: int=5) -> dict:
    basic_frame_period: int = 1
    target_fs = 8000
    num_samples = int(1000 * len(x) / fs / basic_frame_period + 1)
    basic_temporal_positions = np.arange(0, num_samples) * basic_frame_period / 1000
    channels_in_octave = 40
    f0_floor_adjusted = f0_floor * 0.9
    f0_ceil_adjusted = f0_ceil * 1.1

    boundary_f0_list = np.arange(np.ceil(np.log2(f0_ceil_adjusted / f0_floor_adjusted) * channels_in_octave)) + 1
    boundary_f0_list = boundary_f0_list / channels_in_octave
    boundary_f0_list = 2.0 ** boundary_f0_list
    boundary_f0_list *= f0_floor_adjusted

    # down - sampling to target_fs Hz
    [y, actual_fs] = CalculateDownsampledSignal(x, fs, target_fs)
    fft_size = int(2 ** np.ceil(np.log2(len(y) + int(fs / f0_floor_adjusted * 4 + 0.5) + 1)))
    y_spectrum = np.fft.fft(y, fft_size)

    # Calculate from 4 points
    raw_f0_candidates = CalculateCandidates(len(basic_temporal_positions), boundary_f0_list, len(y),
                                            basic_temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil)
    # 
    f0_candidates, number_of_candidates = DetectCandidates(raw_f0_candidates)
    f0_candidates = OverlapF0Candidates(f0_candidates, number_of_candidates)
    f0_candidates, f0_candidates_score = RefineCandidates(y, actual_fs,
                                                          basic_temporal_positions, f0_candidates, f0_floor, f0_ceil)
    # f0_candidates, f0_candidates_score = RemoveUnreliableCandidates(f0_candidates, f0_candidates_score)

    return f0_candidates, f0_candidates_score


############################################################################################
def long_rough_harvest(x: np.ndarray, fs: int, f0_floor: int=71, f0_ceil: int=800, frame_period: int=5) -> dict:
    basic_frame_period: int = 1
    target_fs = 8000
    num_samples = int(1000 * len(x) / fs / basic_frame_period + 1)
    basic_temporal_positions = np.arange(0, num_samples) * basic_frame_period / 1000
    channels_in_octave = 40
    f0_floor_adjusted = f0_floor * 0.9
    f0_ceil_adjusted = f0_ceil * 1.1

    boundary_f0_list = np.arange(np.ceil(np.log2(f0_ceil_adjusted / f0_floor_adjusted) * channels_in_octave)) + 1
    boundary_f0_list = boundary_f0_list / channels_in_octave
    boundary_f0_list = 2.0 ** boundary_f0_list
    boundary_f0_list *= f0_floor_adjusted

    # down - sampling to target_fs Hz
    [y, actual_fs] = CalculateDownsampledSignal(x, fs, target_fs)
    fft_size = int(2 ** np.ceil(np.log2(len(y) + int(fs / f0_floor_adjusted * 4 + 0.5) + 1)))
    y_spectrum = np.fft.fft(y, fft_size)

    raw_f0_candidates = CalculateCandidates(len(basic_temporal_positions), boundary_f0_list, len(y),
                                            basic_temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil)
    f0_candidates, number_of_candidates = DetectCandidates(raw_f0_candidates)
    f0_candidates = OverlapF0Candidates(f0_candidates, number_of_candidates)
    f0_candidates, f0_candidates_score = RefineCandidates(y, actual_fs,
                                                          basic_temporal_positions, f0_candidates, f0_floor, f0_ceil)
    f0_candidates, f0_candidates_score = RemoveUnreliableCandidates(f0_candidates, f0_candidates_score)

    connected_f0, vuv = FixF0Contour(f0_candidates, f0_candidates_score)
    
    return connected_f0, vuv


############################################################################################
def harvest(x: np.ndarray, fs: int, f0_floor: int=71, f0_ceil: int=800, frame_period: int=5) -> dict:
    # Basic frame period: it is 1 (ms) not 1(s)
    basic_frame_period: int = 1
    # They will resample it into 8 Hz
    target_fs = 8000
    # num_sample? number of sample / timeframe
    # 1st = len of x / sampling rate    : length of audio (in second)
    # 2nd = basic_frame_period / 1000   : basic_frame_period (in second/frame, not ms/frame)
    # 3rd = 1st / 2nd                   : number of frame or sample per audio
    num_samples = int(1000 * len(x) / fs / basic_frame_period + 1)
    # basic_temporal_position
    # temporal position for each timeframe, in second
    basic_temporal_positions = np.arange(0, num_samples) * basic_frame_period / 1000
    channels_in_octave = 40
    f0_floor_adjusted = f0_floor * 0.9
    f0_ceil_adjusted = f0_ceil * 1.1

    # Boundary F0 list? What is that?
    # From the steps, it seem that it tried to have a ascending list of F0 candidate that scales exponentially.
    # First step, get the amount of F0 list by getting the difference between the ceil and floor range of F0.
    boundary_f0_list = np.arange(np.ceil(np.log2(f0_ceil_adjusted / f0_floor_adjusted) * channels_in_octave)) + 1
    # 2nd step: Normalization? Making sure the end is the ceiling?
    boundary_f0_list = boundary_f0_list / channels_in_octave
    # 3rd step: rescaling it exponentially (coz earlier we applied log2())
    boundary_f0_list = 2.0 ** boundary_f0_list
    # 4th step: making sure list start with floor
    boundary_f0_list *= f0_floor_adjusted

    # down - sampling to target_fs Hz
    [y, actual_fs] = CalculateDownsampledSignal(x, fs, target_fs)
    # Get propoer FFT_size
    fft_size = int(2 ** np.ceil(np.log2(len(y) + int(fs / f0_floor_adjusted * 4 + 0.5) + 1)))
    # get the spectrogram
    y_spectrum = np.fft.fft(y, fft_size)

    # Calculate candidates. 
    # For each temp_pos, each boundary_f0 are tested if the F0 cand received based on it still in the boundary of logic.
    # Return shape: (boundary_f0_len, temp_pos_len)
    raw_f0_candidates = CalculateCandidates(len(basic_temporal_positions), boundary_f0_list, len(y),
                                            basic_temporal_positions, actual_fs, y_spectrum, f0_floor, f0_ceil)
    # Detect Candidates
    # It is like a concensus: on a cluster of boundary_f0, averaging is done to get a new F0_cand.
    # output shape: (n_f0_cand, n_timeframe), max(n_f0_cand)
    f0_candidates, number_of_candidates = DetectCandidates(raw_f0_candidates)
    # Overlap F0 Candidates
    # I think the name implies itself
    f0_candidates = OverlapF0Candidates(f0_candidates, number_of_candidates)
    # Refine Candidates
    f0_candidates, f0_candidates_score = RefineCandidates(y, actual_fs,
                                                          basic_temporal_positions, f0_candidates, f0_floor, f0_ceil)
    f0_candidates, f0_candidates_score = RemoveUnreliableCandidates(f0_candidates, f0_candidates_score)

    connected_f0, vuv = FixF0Contour(f0_candidates, f0_candidates_score)
    smoothed_f0 = SmoothF0(connected_f0)
    num_samples = int(1000 * len(x) / fs / frame_period + 1)
    temporal_positions = np.arange(0, num_samples) * frame_period / 1000
    temporal_positions_sampe = np.minimum(len(smoothed_f0) - 1, round_matlab(temporal_positions * 1000))
    temporal_positions_sampe = np.array(temporal_positions_sampe, dtype=np.int32)
    return {
        'temporal_positions': temporal_positions,
        'f0': smoothed_f0[temporal_positions_sampe],
        'vuv': vuv[temporal_positions_sampe]
    }


############################################################################################
def CalculateDownsampledSignal(x: np.ndarray, fs: int, target_fs: int) -> tuple:
    # Ratio between the orig_fs and target_fs
    decimation_ratio = int(fs / target_fs + 0.5) 

    if fs <= target_fs: # Ignore upsampling
        y = copy.deepcopy(x)
        actual_fs = fs
    else:
        offset = int(np.ceil(140 / decimation_ratio) * decimation_ratio)
        x = np.append(np.append(np.ones(offset) * x[0], x), np.ones(offset) * x[-1])
        y0 = decimate_matlab(x, decimation_ratio, n = 3)
        actual_fs = fs / decimation_ratio
        y = y0[int(offset / decimation_ratio) : int(-offset / decimation_ratio)]
    y -= np.mean(y)
    return y, actual_fs


###################################################################################################
def CalculateCandidates(number_of_frames: int,
                        boundary_f0_list: np.ndarray, y_length: int, temporal_positions: np.ndarray, actual_fs: int, y_spectrum: np.ndarray,
                        f0_floor: int, f0_ceil: int) -> np.ndarray:
    # Prepare the output
    # Shape: number of f0 to be tested, number of timeframes
    raw_f0_candidates = np.zeros((len(boundary_f0_list), number_of_frames))

    # Looping thru boundary F0 list (list of num from f0floor --> f0ceil)
    for i in range(len(boundary_f0_list)):
        # Calculate raw event
        raw_f0_candidates[i, :] = \
                CalculateRawEvent(boundary_f0_list[i], actual_fs, y_spectrum,
                              y_length, temporal_positions, f0_floor, f0_ceil)
    return raw_f0_candidates


####################################################################################################
def DetectCandidates(raw_f0_candidates: np.ndarray):
    # Get n_boundary_f0 and n_timeframe
    number_of_channels, number_of_frames = raw_f0_candidates.shape
    # Prepare var for F0_candidates for each number of frames
    # shape: (round(n_boundary_f0/10), n_timeframe)
    f0_candidates = np.zeros((int(number_of_channels / 10 + 0.5), number_of_frames))
    # Counter of candidates
    number_of_candidates = 0
    # THreshold?
    threshold = 10

    # For each timeframe:
    for i in np.arange(number_of_frames):
        # Get all raw_f0_cands on timeframe i
        tmp = np.array(raw_f0_candidates[:, i])
        # Making a mask: 1 for candidate, 0 for nothing.
        tmp[tmp > 0] = 1
        tmp[0] = 0
        tmp[-1] = 0
        # Get the transition point: when v-->uv and vice versa.
        tmp = np.diff(tmp)
        # Indices where uv --> v
        st = np.where(tmp == 1)[0]
        # Indices where v --> uv
        ed = np.where(tmp == -1)[0]
        # Counter for something
        count = 0
        # looping thru uv>v transition
        for j in np.arange(len(st)):
            # Checking the distance between j-th v>uv and uv>v (in index)
            dif = ed[j] - st[j]
            # If the distance exceed threshold
            if dif >= threshold:
                # Get all the candidates between st[j] and ed[j]
                tmp_f0 = raw_f0_candidates[st[j] + 1: ed[j] + 1, i]
                # Calculate their mean, and use it as a F0 candidate.
                # What if the F0 cand between st[j] and ed[j] are too far?
                # It cannot be the case, because on a surrounding boundary_f0, the f0_cand must have a similar value.
                f0_candidates[count, i] = np.mean(tmp_f0)
                count += 1
        number_of_candidates = max(number_of_candidates, count)
    return f0_candidates, number_of_candidates


####################################################################################################
def OverlapF0Candidates(f0_candidates: np.ndarray, max_candidates: int) -> np.ndarray:
    n = 3 # This is the optimzied parameter.

    number_of_candidates = n * 2 + 1
    # Prepare var for new_f0_cand
    new_f0_candidates = np.zeros((number_of_candidates * max_candidates, f0_candidates.shape[1]))
    # Assign the first candidate as the number_of_candidates-1-th of the f0_cand
    new_f0_candidates[0, :] = f0_candidates[number_of_candidates - 1, :]
    # Loop thru number_of_cands
    for i in np.arange(number_of_candidates):
        st1 = max(-(i - n) + 1, 1)
        ed1 = min(-(i - n), 0)
        # Assign the f0_cand from surrounding timeframe as the f0_cand in the current timeframe
        new_f0_candidates[np.arange(max_candidates) + i * max_candidates, st1 - 1 : new_f0_candidates.shape[1] + ed1] = \
            f0_candidates[np.arange(max_candidates), -ed1 : new_f0_candidates.shape[1] - (st1 - 1)]
    return new_f0_candidates


####################################################################################################
import multiprocessing as mp

def RefineCandidates(x: np.ndarray, fs: float, temporal_positions: np.ndarray,
                     f0_candidates: np.ndarray, f0_floor: float, f0_ceil: float) -> tuple:
    # Prepare var
    new_f0_candidates = copy.deepcopy(f0_candidates)
    f0_candidates_score = f0_candidates * 0
    # Get the n_candidates, n_timeframe
    N, f = f0_candidates.shape
    if 1:  # parallel
        # Prepare the data for parallel computing
        # data:
        ## x = audio
        ## fs = sampling rate
        ## temporal_position[i]
        ## f0_candidates[j,i]
        ## f0floor
        ## f0ceil
        frame_candidate_data = [(x, fs, temporal_positions[i], f0_candidates[j, i], f0_floor, f0_ceil)
                                for j in np.arange(N)
                                for i in np.arange(f)]
        # Creating Pool object, which can be used in parallel computing
        with mp.Pool(mp.cpu_count()) as pool:
            # Mapping the array of input from frame_candidate_data with GetRefinedF0()
            results = np.array(pool.starmap(GetRefinedF0, frame_candidate_data))
        new_f0_candidates = np.reshape(results[:, 0], [N, f])
        f0_candidates_score = np.reshape(results[:, 1], [N, f])
    else:
        # old one
        for i in range(f):
            for j in range(N):
                new_f0_candidates[j,i], f0_candidates_score[j,i] = GetRefinedF0(x, fs, temporal_positions[i], f0_candidates[j,i], f0_floor, f0_ceil)

    return new_f0_candidates, f0_candidates_score


#################################################################################################
@numba.jit((numba.float64[:],), nopython=True, cache=True)
def round_matlab(x: np.ndarray) -> np.ndarray:
    '''
    round function works as matlab round
    :param x: input vector
    :return: rounded vector
    '''
    #return int(Decimal(n).quantize(0, ROUND_HALF_UP))
    y = x.copy()
    y[x > 0] += 0.5
    y[x <= 0] -= 0.5
    return y

####################################################################################################
#@numba.jit((numba.float64[:], numba.float64, numba.float64, numba.float64, numba.float64, numba.float64), nopython=True, cache=True)
def GetRefinedF0(x: np.ndarray, fs: float, current_time: float, current_f0: float, f0_floor: float, f0_ceil: float) -> tuple:
    # Cannot refine, if it is unvoiced
    if current_f0 == 0:
        return 0, 0
    #####
    # Step #0: prepare needed vars
    #####
    # half_window_length (in sample)
    half_window_length = np.ceil(3 * fs / current_f0 / 2)
    # (in second)
    window_length_in_time = (2 * half_window_length + 1) / fs
    # range between -hwl and hwl (in second)
    base_time = np.arange(-half_window_length, half_window_length + 1) / fs
    fft_size = int(2 ** np.ceil(np.log2((half_window_length * 2 + 1)) + 1))

    ######
    # Step #1: Get instantaneous freq
    ######

    # First-aid treatment
    # Get the index representataion of base_time for specific audio x.
    index_raw = round_matlab((current_time + base_time) * fs + 0.001)

    common = math.pi * ((index_raw - 1) / fs - current_time) / window_length_in_time
    main_window = 0.42 + 0.5 * np.cos(2 * common) + 0.08 * np.cos(4 * common)

    diff_window = np.empty_like(main_window)
    diff_window[0] = - main_window[1] / 2
    diff_window[-1] = main_window[-2] / 2
    diff = np.diff(main_window)
    diff_window[1:-1] = - (diff[1:] + diff[:-1]) / 2

    index = (np.maximum(1, np.minimum(len(x), index_raw)) - 1).astype(np.int32)

    spectrum = fft(x[index] * main_window, fft_size)
    diff_spectrum = fft(x[index] * diff_window, fft_size)

    numerator_i = spectrum.real * diff_spectrum.imag - spectrum.imag * diff_spectrum.real
    power_spectrum = np.abs(spectrum) ** 2
    instantaneous_frequency = (np.arange(fft_size) / fft_size + numerator_i / power_spectrum / 2 / math.pi) * fs

    #######
    # Step #2: F0 candidate refinement
    #######

    number_of_harmonics = min(np.floor(fs / 2 / current_f0), 6)  # with safe guard
    harmonic_index = np.arange(1, number_of_harmonics + 1)

    index = round_matlab(current_f0 * fft_size / fs * harmonic_index).astype(np.int32)
    instantaneous_frequency_list = instantaneous_frequency[index]
    amplitude_list = np.sqrt(power_spectrum[index])
    refined_f0 = np.sum(amplitude_list * instantaneous_frequency_list) / np.sum(amplitude_list * harmonic_index)

    variation = np.abs(((instantaneous_frequency_list / harmonic_index) - current_f0) / current_f0)
    refined_score = 1 / (0.000000000001 + np.mean(variation))
    if refined_f0 < f0_floor or refined_f0 > f0_ceil or refined_score < 2.5:
        refined_f0 = 0
        refined_score = 0
    return refined_f0, refined_score


####################################################################################################
# Remove Unreliable Candidates ?
# There is no such things as this step in the paper
# It is between GetRefinedF0() and FixContour()
def RemoveUnreliableCandidates(f0_candidates: np.ndarray, f0_candidates_score: np.ndarray) -> tuple:
    # Copying input to be used as output
    # Prevent accidental input value change
    new_f0_candidates = np.array(f0_candidates)
    new_f0_candidates_score = np.array(f0_candidates_score)
    threshold = 0.05

    f0_length = f0_candidates.shape[1]              # Length of audio
    number_of_candidates = f0_candidates.shape[0]   # Numbe of candidate for each timeframe

    for i in np.arange(1, f0_length - 1):               # Looping thru all audio with head and tail as exception.
        for j in np.arange(0, number_of_candidates):    # Looping thru all F0 candidate.
            reference_f0 = f0_candidates[j, i]          # Set curr loop F0 candidate as reference
            if reference_f0 == 0:                       # Ignore if the current F0 cand is 0, and continue the loop
                continue
            _, min_error1 = SelectBestF0(reference_f0, f0_candidates[:, i + 1], 1)      # SelectBestF0(), by looking at the next timeframe
            _, min_error2 = SelectBestF0(reference_f0, f0_candidates[:, i - 1], 1)      # SelectBestF0(), by looking at the prev timeframe
            min_error = min([min_error1, min_error2])   # Find the minimum error between 2 above
            if min_error > threshold:                   # If the error is exceeding the threshold: Make the 
                new_f0_candidates[j, i] = 0             
                new_f0_candidates_score[j, i] = 0
    return new_f0_candidates, new_f0_candidates_score


###################################################################################################
@numba.jit((numba.float64, numba.float64[:], numba.float64), nopython=True, cache=True)  # eager compilation through function signature
def SelectBestF0(reference_f0: float, f0_candidates: np.ndarray, allowed_range: float) -> tuple:
    best_f0 = 0                                                         # Initiate var
    best_error = allowed_range                                          # Initiate var
    for i in np.arange(len(f0_candidates)):                             # Loop thru all F0 candidate
        tmp = np.abs(reference_f0 - f0_candidates[i]) / reference_f0    # 
        if tmp > best_error:
            continue
        best_f0 = f0_candidates[i]
        best_error = tmp
    return best_f0, best_error


####################################################################################################
def CalculateRawEvent(boundary_f0: float, fs: int, y_spectrum: np.ndarray, y_length: int, temporal_positions: np.ndarray, f0_floor: int, f0_ceil: int) -> np.ndarray:
    #######
    # First stage: Band-pass filter
    #######
    # Get the filter length
    filter_length_half = int(Decimal(fs / boundary_f0 * 2).quantize(0, ROUND_HALF_UP))
    # Get the nuttall window for transformation
    band_pass_filter_base = nuttall(filter_length_half * 2 + 1)
    # Get the shifter: cos(w_c . t)
    shifter = np.cos(2 * math.pi * boundary_f0 * np.arange(-filter_length_half, filter_length_half + 1) / fs)
    # Get the band_pass_filter
    band_pass_filter = band_pass_filter_base * shifter

    index_bias = filter_length_half + 1
    # Transform the filter into spectrogram
    spectrum_low_pass_filter = np.fft.fft(band_pass_filter, len(y_spectrum))

    # Apply filter to y_spectrum
    filtered_signal = np.real(np.fft.ifft(spectrum_low_pass_filter * y_spectrum))
    # Take the filtered signal only at a specific sectors.
    filtered_signal = filtered_signal[index_bias + np.arange(y_length)]

    ########
    # second stage: zero-cross analysis on 4 points, to get the F0 candidates
    ########
    # calculate 4 kinds of event
    # Still a bit confused, but conceptually understandable
    # Let's skip it for now
    neg_loc, neg_f0 = ZeroCrossingEngine(filtered_signal, fs)
    pos_loc, pos_f0 = ZeroCrossingEngine(-filtered_signal, fs)
    peak_loc, peak_f0 = ZeroCrossingEngine(np.diff(filtered_signal), fs)
    dip_loc, dip_f0 = ZeroCrossingEngine(-np.diff(filtered_signal), fs)

    # Get F0 Candidates
    # What is temporal position again? It is list of position of timeframe, in second.
    # Assigning F0 candidates for each temporal_position.
    f0_candidates = GetF0Candidates(neg_loc, neg_f0, pos_loc, pos_f0,
                                    peak_loc, peak_f0, dip_loc, dip_f0, temporal_positions)

    # Since this func is used to test F0 cand on a particular band (the band-pass filter, remember?),
    # we need to make sure that the F0 cand are still in a logical range, relative to the boundary f0.
    f0_candidates[f0_candidates > boundary_f0 * 1.1] = 0
    f0_candidates[f0_candidates < boundary_f0 * 0.9] = 0
    f0_candidates[f0_candidates > f0_ceil] = 0
    f0_candidates[f0_candidates < f0_floor] = 0

    return f0_candidates


###################################################################################################
# negative zero crossing: going from positive to negative
@numba.jit((numba.float64[:], numba.float64), nopython=True, cache=True)
def ZeroCrossingEngine(x: np.ndarray, fs: int) -> tuple:
    # Copying x, but shift it to t - 1
    # It is done so we can truly get the zeroCrossing without thinking about the t=0
    y = np.empty_like(x)
    y[:-1] = x[1:]
    y[-1] = x[-1]
    
    # Finding points where zero-crossing happened. But, what with these weird operations?
    # 1st line: creating list of index: num from 1 to len(x)+1
    # 2nd line: checking if neighboring signal point has diff sign: x[t-1] * x[t] < 0
    # 3rd line: checking if current trend is descending: x[t-1] < x[t]
    # Combining 2nd and 3rd, we can find if current point is zero-crossing or not.
    negative_going_points = np.arange(1, len(x) + 1) * \
                            ((y * x < 0) * \
                             (y < x))

    # Throwing away the non-zero-crossing stuffs
    edge_list = negative_going_points[negative_going_points > 0]

    # ...what?
    # Got it. Operation is done to get the fractional precision on where does zero-crossing happen
    fine_edge_list = (edge_list) - x[edge_list - 1] / (x[edge_list] - x[edge_list - 1])

    # It add the fine_edge_list by itself[t+1], then divide it by 2, and by sampling rate
    # Oooh... rather than having the exact point where zero-crossing happen, we can have the separator of each zero-crossing.
    # So, between 2 interval_location, there will only be 1 zero-crossing event.
    # It can also mean the "center" of full wave.
    # Dividing it by sampling_rate means the time is in second, rather than index.
    interval_locations = (fine_edge_list[:len(fine_edge_list) - 1] + fine_edge_list[1:]) / 2 / fs

    # How about this? sampling rate divided by difference of fine_edge_list
    # Fine_edge_list are also not in second, but in index
    # Difference between 2 Zero-crossing moment means how long the interval between 2 ZC
    # sampling_rate are in index/second
    # Dividing sampling_rate with difference means getting the interval between 2 ZC, but in Hz
    # Since interval between 2 ZC are 1 full period, the operation tries to get the F0 in the interval.
    interval_based_f0 = fs / np.diff(fine_edge_list)
    return interval_locations, interval_based_f0


####################################################################################################
def FixF0Contour(f0_candidates: np.ndarray, f0_candidates_score: np.ndarray) -> tuple:
    f0_base = SearchF0Base(f0_candidates, f0_candidates_score)
    f0_step1 = FixStep1(f0_base, 0.008) # optimized
    f0_step2 = FixStep2(f0_step1, 6) # optimized
    f0_step3 = FixStep3(f0_step2, f0_candidates, 0.18, f0_candidates_score) # optimized
    f0 = FixStep4(f0_step3, 9) # optimized
    vuv = copy.deepcopy(f0)
    vuv[vuv != 0] = 1
    return f0, vuv


####################################################################################################
# F0s with the highest score are selected as a basic f0 contour.
def SearchF0Base(f0_candidates: np.ndarray, f0_candidates_score: np.ndarray) -> np.ndarray:
    f0_base = np.zeros((f0_candidates.shape[1]))
    for i in range(len(f0_base)):
        max_index = np.argmax(f0_candidates_score[:, i])
        f0_base[i] = f0_candidates[max_index, i]
    return f0_base


####################################################################################################
# Step 1: Rapid change of f0 contour is replaced by 0
@numba.jit((numba.float64[:], numba.float64), nopython=True, cache=True)
def FixStep1(f0_base: np.ndarray, allowed_range: float) -> np.ndarray:
    f0_step1 = np.empty_like(f0_base)
    f0_step1[:] = f0_base
    f0_step1[0] = 0
    f0_step1[1] = 0

    for i in np.arange(2, len(f0_base)):
        if f0_base[i] == 0:
            continue
        reference_f0 = f0_base[i - 1] * 2 - f0_base[i - 2]
        if np.abs((f0_base[i] - reference_f0) / (reference_f0 + EPS)) > allowed_range and \
                        np.abs((f0_base[i] - f0_base[i - 1]) / (f0_base[i - 1] + EPS)) > allowed_range:
            f0_step1[i] = 0
    return f0_step1


####################################################################################################
# Step 2: Voiced sections with a short period are removed
def FixStep2(f0_step1: np.ndarray, voice_range_minimum: float):
    f0_step2 = np.empty_like(f0_step1)
    f0_step2[:] = f0_step1
    boundary_list = GetBoundaryList(f0_step1)

    for i in np.arange(1, len(boundary_list) // 2 + 1):
        distance = boundary_list[2 * i - 1] - boundary_list[(2 * i) - 2]
        if distance < voice_range_minimum:
            f0_step2[boundary_list[(2 * i) - 2] : boundary_list[2 * i - 1] + 1] = 0
    return f0_step2


####################################################################################################
# Step 3: Voiced sections are extended based on the continuity of F0 contour
def FixStep3(f0_step2: np.ndarray, f0_candidates: np.ndarray, allowed_range: float, f0_candidates_score: np.ndarray) -> np.ndarray:
    f0_step3 = np.array(f0_step2)
    boundary_list = GetBoundaryList(f0_step2)
    multi_channel_f0 = GetMultiChannelF0(f0_step2, boundary_list)
    range = np.zeros((len(boundary_list) // 2, 2))
    threshold1 = 100
    threshold2 = 2200

    count = -1
    for i in np.arange(1, len(boundary_list) // 2 + 1):
        tmp_range = np.zeros(2)
        # Value 100 is optimized.
        extended_f0, tmp_range[1] = ExtendF0(multi_channel_f0[i - 1, :], boundary_list[i * 2 - 1], min(len(f0_step2) - 2,
                                    boundary_list[i * 2 - 1] + threshold1), 1, f0_candidates, allowed_range)
        tmp_f0_sequence, tmp_range[0] = ExtendF0(extended_f0, boundary_list[(i * 2) - 2],
            max(1, boundary_list[(i * 2) - 2] - threshold1), -1, f0_candidates, allowed_range)

        mean_f0 = np.mean(tmp_f0_sequence[int(tmp_range[0]) : int(tmp_range[1]) + 1])
        if threshold2 / mean_f0 < tmp_range[1] - tmp_range[0]:
            count += 1
            multi_channel_f0[count, :] = tmp_f0_sequence
            range[count, :] = tmp_range
    multi_channel_f0 = multi_channel_f0[0 : count + 1, :]
    range = range[0 : count + 1, :]
    if count > -1:
        f0_step3 = MergeF0(multi_channel_f0, range, f0_candidates, f0_candidates_score)
    return f0_step3


####################################################################################################
# Step 4: F0s in short unvoiced section are faked
def FixStep4(f0_step3: np.ndarray, threshold:float) -> np.ndarray:
    f0_step4 = np.empty_like(f0_step3)
    f0_step4[:] = f0_step3
    boundary_list = GetBoundaryList(f0_step3)

    for i in np.arange(1, len(boundary_list) // 2 ):
        distance = boundary_list[2 * i] - boundary_list[2 * i - 1] - 1
        if distance >= threshold:
            continue
        tmp0 = f0_step3[boundary_list[2 * i - 1]] + 1
        tmp1 = f0_step3[boundary_list[2 * i]] - 1
        c = (tmp1 - tmp0) / (distance + 1)
        count = 1
        for j in np.arange(boundary_list[2 * i - 1] + 1, boundary_list[2 * i]):
            f0_step4[j] = tmp0 + c * count
            count += 1
    return f0_step4


####################################################################################################
def ExtendF0(f0, origin, last_point, shift, f0_candidates, allowed_range):
    threshold = 4
    extended_f0 = np.array(f0)
    tmp_f0 = extended_f0[origin]
    shifted_origin = origin

    count = 0
    if shift == 1:
        last_point += 1
    elif shift == -1:
        last_point -= 1
    for i in np.arange(origin, last_point, shift):
        extended_f0[i + shift], _ = SelectBestF0(tmp_f0, f0_candidates[:, i + shift], allowed_range)
        if extended_f0[i + shift] != 0:
            tmp_f0 = extended_f0[i + shift]
            count = 0
            shifted_origin = i + shift
        else:
            count += + 1
        if count == threshold:
            break
    return extended_f0, shifted_origin


####################################################################################################
def GetMultiChannelF0(f0: np.ndarray, boundary_list: np.ndarray) -> np.ndarray:
    multi_channel_f0 = np.zeros((len(boundary_list) // 2, len(f0)))
    for i in np.arange(1, len(boundary_list) // 2 + 1):
        multi_channel_f0[i - 1, boundary_list[(i * 2) - 2] : boundary_list[i * 2 - 1] + 1] =\
            f0[boundary_list[(i * 2) - 2] : boundary_list[(i * 2) - 1] + 1]
    return multi_channel_f0


####################################################################################################
def MergeF0(multi_channel_f0: np.ndarray, range_: np.ndarray, f0_candidates: np.ndarray, f0_candidates_score: np.ndarray) -> np.ndarray:
    number_of_channels = multi_channel_f0.shape[0]
    sorted_order = np.argsort(range_[:, 0], axis=0, kind='quicksort')
    f0 = multi_channel_f0[sorted_order[0], :]
    range_ = range_.astype(int)

    for i in np.arange(1, number_of_channels):
        # without overlapping
        if range_[sorted_order[i], 0] - range_[sorted_order[0], 1] > 0:
            f0[range_[sorted_order[i], 0] : range_[sorted_order[i], 1] + 1] = multi_channel_f0[sorted_order[i],\
                                                        range_[sorted_order[i], 0] : range_[sorted_order[i], 1] + 1]
            range_[sorted_order[0], 0] = range_[sorted_order[i], 0]
            range_[sorted_order[0], 1] = range_[sorted_order[i], 1]
        else: # with overlapping
            f0, range_[sorted_order[0], 1] = MergeF0Sub(f0, range_[sorted_order[0], 0], range_[sorted_order[0], 1],
              multi_channel_f0[sorted_order[i], :], range_[sorted_order[i], 0],
              range_[sorted_order[i], 1], f0_candidates, f0_candidates_score)
    return f0


####################################################################################################
def MergeF0Sub(f0_1: np.ndarray, st1: int, ed1: int,
               f0_2: np.ndarray, st2: int, ed2: int,
               f0_candidates: np.ndarray, f0_candidates_score: np.ndarray) -> tuple:
    merged_f0 = copy.deepcopy(f0_1)
    st1 = int(st1)
    st2 = int(st2)
    ed1 = int(ed1)
    ed2 = int(ed2)
    # Completely overlapping section
    if st1 <= st2 and ed1 >= ed2:
        new_ed = ed1
        return merged_f0, new_ed
    new_ed = ed2

    score1 = 0
    score2 = 0
    for i in np.arange(st2, ed1 + 1):
        score1 = score1 + SerachScore(f0_1[i], f0_candidates[:, i], f0_candidates_score[:, i])
        score2 = score2 + SerachScore(f0_2[i], f0_candidates[:, i], f0_candidates_score[:, i])
    if score1 > score2:
        merged_f0[ed1 : ed2 + 1] = f0_2[ed1 : ed2 + 1]
    else:
        merged_f0[st2 : ed2 + 1] = f0_2[st2 : ed2 + 1]
    return merged_f0, new_ed


####################################################################################################
def SerachScore(f0: float, f0_candidates: np.ndarray, f0_candidates_score: np.ndarray) -> float:
    score = 0
    for i in range(f0_candidates.shape[0]):
        if f0 == f0_candidates[i] and score < f0_candidates_score[i]:
            score = f0_candidates_score[i]
    return score


####################################################################################################
def GetF0Candidates(neg_loc: np.ndarray, neg_f0: np.ndarray,
                    pos_loc: np.ndarray, pos_f0: np.ndarray,
                    peak_loc: np.ndarray, peak_f0: np.ndarray,
                    dip_loc: np.ndarray, dip_f0: np.ndarray, temporal_positions: np.ndarray):
    # test this one
    # np.size() get the number of element
    # Because no axis provided, all singular element is counted.
    # usable_channel? what is this?
    # Check if all the ZC has at least 3 sector to be analyzed.
    usable_channel = max(0, np.size(neg_loc) - 2) * \
                     max(0, np.size(pos_loc) - 2) * \
                     max(0, np.size(peak_loc) - 2) * \
                     max(0, np.size(dip_f0) - 2)

    # Preparing the F0 candidate for each timeframe, based on each ZC.
    interpolated_f0_list = np.zeros((4, np.size(temporal_positions)))

    # Filling the F0 candidate based on ZC F0 list.
    # Coz of different resolution or scale, interpolation is used.
    if usable_channel > 0:
        interpolated_f0_list[0, :] = interp1d(neg_loc,
                                              neg_f0,
                                              fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[1, :] = interp1d(pos_loc,
                                              pos_f0,
                                              fill_value='extrapolate')(temporal_positions)

        interpolated_f0_list[2, :] = interp1d(peak_loc,
                                              peak_f0,
                                              fill_value='extrapolate')(temporal_positions)
        interpolated_f0_list[3, :] = interp1d(dip_loc,
                                              dip_f0,
                                              fill_value='extrapolate')(temporal_positions)

        interpolated_f0 = np.mean(interpolated_f0_list, axis=0)
    else:
        interpolated_f0 = temporal_positions * 0
    return interpolated_f0


###################################################################################################
def SmoothF0(f0: np.ndarray) -> np.ndarray:
    b = np.array([0.0078202080334971724, 0.015640416066994345, 0.0078202080334971724])
    a = np.array([1.0, -1.7347257688092754, 0.76600660094326412])

    smoothed_f0 = np.append(np.append(np.zeros(300), f0), np.zeros(300))
    boundary_list = GetBoundaryList(smoothed_f0)
    multi_channel_f0 = GetMultiChannelF0(smoothed_f0, boundary_list)
    for i in np.arange(1, len(boundary_list) // 2 + 1):
        tmp_f0_contour = FilterF0(multi_channel_f0[i - 1, :], boundary_list[i * 2 - 2], boundary_list[i * 2 - 1], b, a)
        smoothed_f0[boundary_list[i * 2 - 2] : boundary_list[i * 2 - 1] + 1] = \
            tmp_f0_contour[boundary_list[i * 2 - 2] : boundary_list[i * 2 - 1] + 1]

    smoothed_f0 = smoothed_f0[300 : len(smoothed_f0) - 300]
    return smoothed_f0


####################################################################################################
def FilterF0(f0_contour: np.ndarray, st: int, ed: int, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    smoothed_f0 = copy.deepcopy(f0_contour)
    smoothed_f0[0 : st] = smoothed_f0[st]
    smoothed_f0[ed + 1: ] = smoothed_f0[ed]
    aaa = lfilter(b, a, smoothed_f0, axis=0)
    bbb = lfilter(b, a, aaa[-1 : : -1], axis=0)
    smoothed_f0 = bbb[-1 : : -1]
    smoothed_f0[0 : st] = 0
    smoothed_f0[ed + 1: ] = 0
    return smoothed_f0


####################################################################################################
# Get the impulse response for nuttall window transformation
def nuttall(N: int) -> np.ndarray:
    t = np.asmatrix(np.arange(N) * 2 * math.pi / (N-1))
    coefs = np.array([0.355768, -0.487396, 0.144232, -0.012604])
    window = coefs @ np.cos(np.matrix([0,1,2,3]).T @ t)
    return np.squeeze(np.asarray(window))


#######################################################################################################
# Note: vuv(1) and vuv(end) are set to 0.
def GetBoundaryList(f0: np.ndarray) -> np.ndarray:
    vuv = np.array(f0)
    vuv[vuv != 0] = 1
    vuv[0] = 0
    vuv[-1] = 0
    diff_vuv = np.diff(vuv)
    boundary_list = np.where(diff_vuv != 0)[0]
    boundary_list[0:: 2] += 1
    return boundary_list


#############################################################################################################
def decimate_matlab(x, q, n=None, axis=-1):
    """
    :param x: signal
    :param q: decimation ration
    :param n: order of filter
    :param axis:
    :return: resampled signal
    """

    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is not None and not isinstance(n, int):
        raise TypeError("n must be an integer")

    system = signal.dlti(*signal.cheby1(n, 0.05, 0.8 / q))

    #zero_phase = True

    y = signal.filtfilt(system.num, system.den, x, axis=axis, padlen=3 * (max(len(system.den), len(system.num)) - 1))

    # make it the same as matlab
    nd = len(y)
    n_out = np.ceil(nd / q)
    n_beg = int(q - (q * n_out - nd))
    return y[n_beg - 1::q]
