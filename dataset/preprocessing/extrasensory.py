import os
import numpy as np
import gzip
from io import BytesIO
from collections import defaultdict
from tqdm import tqdm

DATA_PATH = '/data/path/to/ExtraSensory'

def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index(b'\n')]
    columns = headline.split(b',')

    # The first column should be timestamp:
    assert columns[0] == b'timestamp'
    # The last column should be label_source:
    assert columns[-1] == b'label_source'

    # Search for the column of the first label:
    for (ci, col) in enumerate(columns):
        if col.startswith(b'label:'):
            first_label_ind = ci
            break
        pass

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith(b'label:')
        label_names[li] = label.replace(b'label:', b'')
        pass

    return (feature_names, label_names)


def parse_body_of_csv(csv_str, n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(BytesIO(csv_str), delimiter=',', skiprows=1)

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int)

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)]

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1]  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.  # Y is the label matrix

    return (X, Y, M, timestamps)


'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
def read_user_data(user_data_file):
    # user_data_file = os.path.join(DATA_PATH, f"{uuid}.features_labels.csv.gz")
    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rb') as fid:
        csv_str = fid.read()
        pass

    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return (X, Y, M, timestamps, feature_names, label_names)


def extract_subsequences():
    uuid_list = [p for p in os.listdir(DATA_PATH) if p[-7:] == '.csv.gz']

    data = {
        'X': [],
        'Y': [],
        'M': [],
        'T': [],
        'feature_names': [b'raw_acc:magnitude_stats:mean', b'raw_acc:magnitude_stats:std', b'raw_acc:magnitude_stats:moment3', b'raw_acc:magnitude_stats:moment4', b'raw_acc:magnitude_stats:percentile25', b'raw_acc:magnitude_stats:percentile50', b'raw_acc:magnitude_stats:percentile75', b'raw_acc:magnitude_stats:value_entropy', b'raw_acc:magnitude_stats:time_entropy', b'raw_acc:magnitude_spectrum:log_energy_band0', b'raw_acc:magnitude_spectrum:log_energy_band1', b'raw_acc:magnitude_spectrum:log_energy_band2', b'raw_acc:magnitude_spectrum:log_energy_band3', b'raw_acc:magnitude_spectrum:log_energy_band4', b'raw_acc:magnitude_spectrum:spectral_entropy', b'raw_acc:magnitude_autocorrelation:period', b'raw_acc:magnitude_autocorrelation:normalized_ac', b'raw_acc:3d:mean_x', b'raw_acc:3d:mean_y', b'raw_acc:3d:mean_z', b'raw_acc:3d:std_x', b'raw_acc:3d:std_y', b'raw_acc:3d:std_z', b'raw_acc:3d:ro_xy', b'raw_acc:3d:ro_xz', b'raw_acc:3d:ro_yz', b'proc_gyro:magnitude_stats:mean', b'proc_gyro:magnitude_stats:std', b'proc_gyro:magnitude_stats:moment3', b'proc_gyro:magnitude_stats:moment4', b'proc_gyro:magnitude_stats:percentile25', b'proc_gyro:magnitude_stats:percentile50', b'proc_gyro:magnitude_stats:percentile75', b'proc_gyro:magnitude_stats:value_entropy', b'proc_gyro:magnitude_stats:time_entropy', b'proc_gyro:magnitude_spectrum:log_energy_band0', b'proc_gyro:magnitude_spectrum:log_energy_band1', b'proc_gyro:magnitude_spectrum:log_energy_band2', b'proc_gyro:magnitude_spectrum:log_energy_band3', b'proc_gyro:magnitude_spectrum:log_energy_band4', b'proc_gyro:magnitude_spectrum:spectral_entropy', b'proc_gyro:magnitude_autocorrelation:period', b'proc_gyro:magnitude_autocorrelation:normalized_ac', b'proc_gyro:3d:mean_x', b'proc_gyro:3d:mean_y', b'proc_gyro:3d:mean_z', b'proc_gyro:3d:std_x', b'proc_gyro:3d:std_y', b'proc_gyro:3d:std_z', b'proc_gyro:3d:ro_xy', b'proc_gyro:3d:ro_xz', b'proc_gyro:3d:ro_yz', b'raw_magnet:magnitude_stats:mean', b'raw_magnet:magnitude_stats:std', b'raw_magnet:magnitude_stats:moment3', b'raw_magnet:magnitude_stats:moment4', b'raw_magnet:magnitude_stats:percentile25', b'raw_magnet:magnitude_stats:percentile50', b'raw_magnet:magnitude_stats:percentile75', b'raw_magnet:magnitude_stats:value_entropy', b'raw_magnet:magnitude_stats:time_entropy', b'raw_magnet:magnitude_spectrum:log_energy_band0', b'raw_magnet:magnitude_spectrum:log_energy_band1', b'raw_magnet:magnitude_spectrum:log_energy_band2', b'raw_magnet:magnitude_spectrum:log_energy_band3', b'raw_magnet:magnitude_spectrum:log_energy_band4', b'raw_magnet:magnitude_spectrum:spectral_entropy', b'raw_magnet:magnitude_autocorrelation:period', b'raw_magnet:magnitude_autocorrelation:normalized_ac', b'raw_magnet:3d:mean_x', b'raw_magnet:3d:mean_y', b'raw_magnet:3d:mean_z', b'raw_magnet:3d:std_x', b'raw_magnet:3d:std_y', b'raw_magnet:3d:std_z', b'raw_magnet:3d:ro_xy', b'raw_magnet:3d:ro_xz', b'raw_magnet:3d:ro_yz', b'raw_magnet:avr_cosine_similarity_lag_range0', b'raw_magnet:avr_cosine_similarity_lag_range1', b'raw_magnet:avr_cosine_similarity_lag_range2', b'raw_magnet:avr_cosine_similarity_lag_range3', b'raw_magnet:avr_cosine_similarity_lag_range4', b'watch_acceleration:magnitude_stats:mean', b'watch_acceleration:magnitude_stats:std', b'watch_acceleration:magnitude_stats:moment3', b'watch_acceleration:magnitude_stats:moment4', b'watch_acceleration:magnitude_stats:percentile25', b'watch_acceleration:magnitude_stats:percentile50', b'watch_acceleration:magnitude_stats:percentile75', b'watch_acceleration:magnitude_stats:value_entropy', b'watch_acceleration:magnitude_stats:time_entropy', b'watch_acceleration:magnitude_spectrum:log_energy_band0', b'watch_acceleration:magnitude_spectrum:log_energy_band1', b'watch_acceleration:magnitude_spectrum:log_energy_band2', b'watch_acceleration:magnitude_spectrum:log_energy_band3', b'watch_acceleration:magnitude_spectrum:log_energy_band4', b'watch_acceleration:magnitude_spectrum:spectral_entropy', b'watch_acceleration:magnitude_autocorrelation:period', b'watch_acceleration:magnitude_autocorrelation:normalized_ac', b'watch_acceleration:3d:mean_x', b'watch_acceleration:3d:mean_y', b'watch_acceleration:3d:mean_z', b'watch_acceleration:3d:std_x', b'watch_acceleration:3d:std_y', b'watch_acceleration:3d:std_z', b'watch_acceleration:3d:ro_xy', b'watch_acceleration:3d:ro_xz', b'watch_acceleration:3d:ro_yz', b'watch_acceleration:spectrum:x_log_energy_band0', b'watch_acceleration:spectrum:x_log_energy_band1', b'watch_acceleration:spectrum:x_log_energy_band2', b'watch_acceleration:spectrum:x_log_energy_band3', b'watch_acceleration:spectrum:x_log_energy_band4', b'watch_acceleration:spectrum:y_log_energy_band0', b'watch_acceleration:spectrum:y_log_energy_band1', b'watch_acceleration:spectrum:y_log_energy_band2', b'watch_acceleration:spectrum:y_log_energy_band3', b'watch_acceleration:spectrum:y_log_energy_band4', b'watch_acceleration:spectrum:z_log_energy_band0', b'watch_acceleration:spectrum:z_log_energy_band1', b'watch_acceleration:spectrum:z_log_energy_band2', b'watch_acceleration:spectrum:z_log_energy_band3', b'watch_acceleration:spectrum:z_log_energy_band4', b'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range0', b'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range1', b'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range2', b'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range3', b'watch_acceleration:relative_directions:avr_cosine_similarity_lag_range4', b'watch_heading:mean_cos', b'watch_heading:std_cos', b'watch_heading:mom3_cos', b'watch_heading:mom4_cos', b'watch_heading:mean_sin', b'watch_heading:std_sin', b'watch_heading:mom3_sin', b'watch_heading:mom4_sin', b'watch_heading:entropy_8bins', b'location:num_valid_updates', b'location:log_latitude_range', b'location:log_longitude_range', b'location:min_altitude', b'location:max_altitude', b'location:min_speed', b'location:max_speed', b'location:best_horizontal_accuracy', b'location:best_vertical_accuracy', b'location:diameter', b'location:log_diameter', b'location_quick_features:std_lat', b'location_quick_features:std_long', b'location_quick_features:lat_change', b'location_quick_features:long_change', b'location_quick_features:mean_abs_lat_deriv', b'location_quick_features:mean_abs_long_deriv', b'audio_naive:mfcc0:mean', b'audio_naive:mfcc1:mean', b'audio_naive:mfcc2:mean', b'audio_naive:mfcc3:mean', b'audio_naive:mfcc4:mean', b'audio_naive:mfcc5:mean', b'audio_naive:mfcc6:mean', b'audio_naive:mfcc7:mean', b'audio_naive:mfcc8:mean', b'audio_naive:mfcc9:mean', b'audio_naive:mfcc10:mean', b'audio_naive:mfcc11:mean', b'audio_naive:mfcc12:mean', b'audio_naive:mfcc0:std', b'audio_naive:mfcc1:std', b'audio_naive:mfcc2:std', b'audio_naive:mfcc3:std', b'audio_naive:mfcc4:std', b'audio_naive:mfcc5:std', b'audio_naive:mfcc6:std', b'audio_naive:mfcc7:std', b'audio_naive:mfcc8:std', b'audio_naive:mfcc9:std', b'audio_naive:mfcc10:std', b'audio_naive:mfcc11:std', b'audio_naive:mfcc12:std', b'audio_properties:max_abs_value', b'audio_properties:normalization_multiplier', b'discrete:app_state:is_active', b'discrete:app_state:is_inactive', b'discrete:app_state:is_background', b'discrete:app_state:missing', b'discrete:battery_plugged:is_ac', b'discrete:battery_plugged:is_usb', b'discrete:battery_plugged:is_wireless', b'discrete:battery_plugged:missing', b'discrete:battery_state:is_unknown', b'discrete:battery_state:is_unplugged', b'discrete:battery_state:is_not_charging', b'discrete:battery_state:is_discharging', b'discrete:battery_state:is_charging', b'discrete:battery_state:is_full', b'discrete:battery_state:missing', b'discrete:on_the_phone:is_False', b'discrete:on_the_phone:is_True', b'discrete:on_the_phone:missing', b'discrete:ringer_mode:is_normal', b'discrete:ringer_mode:is_silent_no_vibrate', b'discrete:ringer_mode:is_silent_with_vibrate', b'discrete:ringer_mode:missing', b'discrete:wifi_status:is_not_reachable', b'discrete:wifi_status:is_reachable_via_wifi', b'discrete:wifi_status:is_reachable_via_wwan', b'discrete:wifi_status:missing', b'lf_measurements:light', b'lf_measurements:pressure', b'lf_measurements:proximity_cm', b'lf_measurements:proximity', b'lf_measurements:relative_humidity', b'lf_measurements:battery_level', b'lf_measurements:screen_brightness', b'lf_measurements:temperature_ambient', b'discrete:time_of_day:between0and6', b'discrete:time_of_day:between3and9', b'discrete:time_of_day:between6and12', b'discrete:time_of_day:between9and15', b'discrete:time_of_day:between12and18', b'discrete:time_of_day:between15and21', b'discrete:time_of_day:between18and24', b'discrete:time_of_day:between21and3'],
        'label_names': [b'LYING_DOWN', b'SITTING', b'FIX_walking', b'FIX_running', b'BICYCLING', b'SLEEPING', b'LAB_WORK', b'IN_CLASS', b'IN_A_MEETING', b'LOC_main_workplace', b'OR_indoors', b'OR_outside', b'IN_A_CAR', b'ON_A_BUS', b'DRIVE_-_I_M_THE_DRIVER', b'DRIVE_-_I_M_A_PASSENGER', b'LOC_home', b'FIX_restaurant', b'PHONE_IN_POCKET', b'OR_exercise', b'COOKING', b'SHOPPING', b'STROLLING', b'DRINKING__ALCOHOL_', b'BATHING_-_SHOWER', b'CLEANING', b'DOING_LAUNDRY', b'WASHING_DISHES', b'WATCHING_TV', b'SURFING_THE_INTERNET', b'AT_A_PARTY', b'AT_A_BAR', b'LOC_beach', b'SINGING', b'TALKING', b'COMPUTER_WORK', b'EATING', b'TOILET', b'GROOMING', b'DRESSING', b'AT_THE_GYM', b'STAIRS_-_GOING_UP', b'STAIRS_-_GOING_DOWN', b'ELEVATOR', b'OR_standing', b'AT_SCHOOL', b'PHONE_IN_HAND', b'PHONE_IN_BAG', b'PHONE_ON_TABLE', b'WITH_CO-WORKERS', b'WITH_FRIENDS']
    }

    missing_cnt = 0
    full_label_cnt = 0
    for uuid in tqdm(uuid_list, total=len(uuid_list)):
        user_data_file = os.path.join(DATA_PATH, uuid)
        (X, Y, M, timestamps, feature_names, label_names) = read_user_data(user_data_file)

        y_subseq = None
        m_subseq = None
        x_subseq = []
        t_subseq = []

        for example_x, example_y, example_m, example_t in zip(X, Y, M, timestamps):
            if sum(example_m) == len(example_m) or sum(example_y) == 0:
                missing_cnt += 1
                continue
            if sum(example_m) == 0:
                full_label_cnt += 1

            if y_subseq is None:
                y_subseq = example_y
                m_subseq = example_m
                x_subseq = [example_x]
                t_subseq = [example_t]

            elif list(example_y) == list(y_subseq) and list(example_m) == list(m_subseq):
                x_subseq.append(example_x)
                t_subseq.append(example_t)

            else:
                data['Y'].append(y_subseq)
                data['M'].append(m_subseq)
                data['X'].append(x_subseq)
                data['T'].append(t_subseq)

                y_subseq = example_y
                m_subseq = example_m
                x_subseq = [example_x]
                t_subseq = [example_t]

        data['Y'].append(y_subseq)
        data['M'].append(m_subseq)
        data['X'].append(x_subseq)
        data['T'].append(t_subseq)

    print(f"in total {len(data['Y'])} subseqences, {sum([len(t) for t in data['T']])} minutes.")
    print(f'{missing_cnt} subsequences miss all labels.')
    print(f'{full_label_cnt} subsequences have all labels.')

    print('Y.shape:', np.shape(data['Y']))
    print('T.shape:', np.shape(data['T']))
    print('M.shape:', np.shape(data['M']))
    print('X.shape:', np.shape(data['X']))

    np.savez(os.path.join(DATA_PATH, 'data.npz'),
             X=np.array(data['X']),
             Y=np.array(data['Y'], dtype=int),
             M=np.array(data['M'], dtype=int),
             T=np.array(data['T']),
             feature_names=data['feature_names'],
             label_names=data['label_names']
             )


extract_subsequences()
