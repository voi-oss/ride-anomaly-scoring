from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import filtfilt, firwin


def reshape_to_numpy(
    data: pd.DataFrame, features: List[str], max_timestamps: Optional[int] = None
) -> np.ndarray:
    """Reshape grouped time series data into a 3D numpy array.

    Returns array of shape (n_instances, n_timestamps, n_features).
    """
    grouped_data = (
        data.groupby("ride_id")[features].apply(lambda x: x.values).reset_index()
    )

    time_series = grouped_data.iloc[:, 1].values

    if max_timestamps is None:
        max_timestamps = max(len(ts) for ts in time_series)

    n_features = len(features)

    reshaped_data = np.full((len(time_series), max_timestamps, n_features), np.nan)

    for i, ts in enumerate(time_series):
        reshaped_data[i, : len(ts), :] = ts

    return reshaped_data


def interpolate_missing_values(
    data: np.ndarray,
    method: str,
    limit: Optional[int] = None,
) -> np.ndarray:
    """Interpolate missing values in time series data.

    Only interpolates up to the last valid value to avoid trailing NaNs.
    """
    interpolated_data = data.copy()

    for ride_idx in range(data.shape[0]):
        for feat_idx in range(data.shape[2]):
            series = pd.Series(interpolated_data[ride_idx, :, feat_idx])

            last_valid_idx = series.last_valid_index()
            if last_valid_idx is not None:
                series_to_interpolate = series.iloc[: last_valid_idx + 1]
                interpolated_series = series_to_interpolate.interpolate(
                    method=method, limit=limit
                )
                interpolated_data[
                    ride_idx, : last_valid_idx + 1, feat_idx
                ] = interpolated_series.values

    return interpolated_data


def denoise_data(
    data: np.ndarray,
    accel_indices: List[int],
    gyro_indices: List[int],
    accel_cutoff: float,
    accel_order: int,
    gyro_cutoff: float,
    gyro_order: int,
) -> np.ndarray:
    """Apply FIR lowpass filters to denoise accelerometer and gyroscope data."""
    denoised_data = data.copy()

    if accel_indices:
        filter_coeffs = firwin(accel_order + 1, accel_cutoff, window="hamming")
        denoised_data[:, :, accel_indices] = filtfilt(
            filter_coeffs, 1.0, data[:, :, accel_indices], axis=1
        )

    if gyro_indices:
        filter_coeffs = firwin(gyro_order + 1, gyro_cutoff, window="hamming")
        denoised_data[:, :, gyro_indices] = filtfilt(
            filter_coeffs, 1.0, data[:, :, gyro_indices], axis=1
        )

    return denoised_data


def compute_spectrograms(
    data_np_clean: np.ndarray, sampling_rate: float, nperseg: int, noverlap: int
) -> np.ndarray:
    """Compute spectrograms for all features of all rides.

    Returns array of shape (n_rides, n_time_windows, n_features, n_frequencies).
    """
    n_rides = data_np_clean.shape[0]
    n_features = data_np_clean.shape[2]

    all_rides_spectrograms = []

    for ride_idx in range(n_rides):
        ride_data = data_np_clean[ride_idx]

        spectrograms_list = []

        for feat_idx in range(n_features):
            signal_data = ride_data[:, feat_idx]

            frequencies, times, Sxx = signal.spectrogram(
                signal_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap
            )
            spectrograms_list.append(Sxx.T)

        ride_spectrograms = np.stack(spectrograms_list, axis=1)
        all_rides_spectrograms.append(ride_spectrograms)

    spectrograms_array = np.stack(all_rides_spectrograms, axis=0)
    return spectrograms_array


def time_avg_pooling(spectrograms_array: np.ndarray) -> np.ndarray:
    """Average spectrograms over time dimension and flatten to feature vector.

    Returns array of shape (n_rides, n_features * n_frequencies).
    """
    X_sum = np.nansum(spectrograms_array, axis=1)
    X_count = np.sum(~np.isnan(spectrograms_array), axis=1)
    X_mean_t = np.divide(X_sum, X_count, where=X_count > 0, out=np.zeros_like(X_sum))

    X_feat = X_mean_t.reshape(spectrograms_array.shape[0], -1)
    return X_feat
