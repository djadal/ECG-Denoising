import numpy as np
import yaml
from pathlib import Path
import _pickle as pickle

def Data_Preparation(noise_version=1):

    print('Getting the Data ready ... ')
    
    
    config_path = Path("./config") / "data_prepare.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    # Load QT Database
    with open(config['qt_path'], 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open(config['nstdb_path'], 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    [bw_signals,_,_] = nstdb
    #[_, em_signals, _ ] = nstdb
    #[_, _, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    
    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]


    #####################################
    # Data split
    #####################################
    if noise_version == 1:
        noise_test = bw_noise_channel2_b
        noise_train = bw_noise_channel1_a
    elif noise_version == 2:
        noise_test = bw_noise_channel1_b
        noise_train = bw_noise_channel2_a
    else:
        raise Exception("Sorry, noise_version should be 1 or 2")

    #####################################
    # QTDatabase
    #####################################

    beats_train = []
    beats_test = []

    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database
                'sel233',  # Record from MIT-BIH Arrhythmia Database

                'sel302',  # Record from MIT-BIH ST Change Database
                'sel307',  # Record from MIT-BIH ST Change Database

                'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database

                'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database

                'sele0106',  # Record from European ST-T Database
                'sele0121',  # Record from European ST-T Database

                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',  # Record from ``sudden death'' patients from BIH

                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]
    
    skip_beats = 0
    samples = 512
    qtdb_keys = list(qtdb.keys())
    
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        
        for b in qtdb[signal_name]:
            b_np = np.zeros(samples)
            b_sq = np.array(b)
            
            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)
    
    sn_train = []
    sn_test = []

    noise_index = 0
    
    # Adding noise to train
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i in range(len(beats_train)):
        noise = noise_train[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / Ase
        signal_noise = beats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_train) - samples):
            noise_index = 0

    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100

    # Saving the random array so we can use it on the amplitude segmentation tables
    np.save(config['rnd_test_path'], rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + samples]
        beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / Ase
        signal_noise = beats_test[i] + alpha * noise
        
        sn_test.append(signal_noise)
        noise_index += samples

        if noise_index > (len(noise_test) - samples):
            noise_index = 0


    X_train = np.array(sn_train)
    y_train = np.array(beats_train)
    
    X_test = np.array(sn_test)
    y_test = np.array(beats_test)
    
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)


    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset

def Data_Preparation_RMN(noise_version=1):

    print('Getting the Data ready ... ')
    
    config_path = Path("./config") / "data_prepare.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    # Load QT Database
    with open(config['qt_path'], 'rb') as input:
        # dict {register_name: beats_list}
        qtdb = pickle.load(input)

    # Load NSTDB
    with open(config['nstdb_path'], 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB - Extract three types of noise
    #####################################

    [bw_signals, em_signals, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    em_signals = np.array(em_signals)
    ma_signals = np.array(ma_signals)
    
    # BW noise extraction
    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]
    
    # EM noise extraction
    em_noise_channel1_a = em_signals[0:int(em_signals.shape[0]/2), 0]
    em_noise_channel1_b = em_signals[int(em_signals.shape[0]/2):-1, 0]
    em_noise_channel2_a = em_signals[0:int(em_signals.shape[0]/2), 1]
    em_noise_channel2_b = em_signals[int(em_signals.shape[0]/2):-1, 1]
    
    # MA noise extraction
    ma_noise_channel1_a = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    ma_noise_channel1_b = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    ma_noise_channel2_a = ma_signals[0:int(ma_signals.shape[0]/2), 1]
    ma_noise_channel2_b = ma_signals[int(ma_signals.shape[0]/2):-1, 1]

    #####################################
    # Data split
    #####################################
    if noise_version == 1:
        bw_noise_test = bw_noise_channel2_b
        bw_noise_train = bw_noise_channel1_a
        em_noise_test = em_noise_channel2_b
        em_noise_train = em_noise_channel1_a
        ma_noise_test = ma_noise_channel2_b
        ma_noise_train = ma_noise_channel1_a
    elif noise_version == 2:
        bw_noise_test = bw_noise_channel1_b
        bw_noise_train = bw_noise_channel2_a
        em_noise_test = em_noise_channel1_b
        em_noise_train = em_noise_channel2_a
        ma_noise_test = ma_noise_channel1_b
        ma_noise_train = ma_noise_channel2_a
    else:
        raise Exception("Sorry, noise_version should be 1 or 2")

    #####################################
    # QTDatabase
    #####################################
    beats_train = []
    beats_test = []

    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database
                'sel233',  # Record from MIT-BIH Arrhythmia Database
                
                'sel302',  # Record from MIT-BIH ST Change Database
                'sel307',  # Record from MIT-BIH ST Change Database
                
                'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel853',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                
                'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sel16795',  # Record from MIT-BIH Normal Sinus Rhythm Database
                
                'sele0106',  # Record from European ST-T Database
                'sele0121',  # Record from European ST-T Database
                
                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',  # Record from ``sudden death'' patients from BIH
                
                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',  # Record from MIT-BIH Long-Term ECG Database
                ]
    
    skip_beats = 0
    samples = 512
    qtdb_keys = list(qtdb.keys())
    
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        
        for b in qtdb[signal_name]:
            b_np = np.zeros(samples)
            b_sq = np.array(b)
            
            init_padding = 16
            if b_sq.shape[0] > (samples - init_padding):
                skip_beats += 1
                continue

            b_np[init_padding:b_sq.shape[0] + init_padding] = b_sq - (b_sq[0] + b_sq[-1]) / 2

            if signal_name in test_set:
                beats_test.append(b_np)
            else:
                beats_train.append(b_np)

    def add_mixed_noise(beats, bw_noise, em_noise, ma_noise, is_test=False):
        """
        Add mixed noise combinations to ECG beats
        Args:
            beats: clean ECG beats
            bw_noise, em_noise, ma_noise: three types of noise
            is_test: whether this is for test set
        Returns:
            sn_list: noisy signals
            noise_combinations: noise type combinations used
            snr_values: SNR values used
        """
        sn_list = []
        noise_combinations = []
        snr_values = []

        noise_index_bw = 0
        noise_index_em = 0
        noise_index_ma = 0
        
        for i in range(len(beats)):
            noise_type = np.random.randint(1, 8)
            noise_combinations.append(noise_type)
            
            # Randomly select SNR value (-6 to 18 dB)
            snr_db = np.random.randint(-6, 19)
            snr_values.append(snr_db)
            snr_linear = 10**(snr_db / 10.0)
            
            # Determine noise combination based on binary representation
            use_bw = (noise_type & 1) != 0  # bit 0: BW noise
            use_em = (noise_type & 2) != 0  # bit 1: EM noise
            use_ma = (noise_type & 4) != 0  # bit 2: MA noise
            
            # Initialize mixed noise
            mixed_noise = np.zeros(samples)
            noise_count = 0
            
            # Add BW noise if selected
            if use_bw:
                bw_segment = bw_noise[noise_index_bw:noise_index_bw + samples]
                mixed_noise += bw_segment
                noise_count += 1
                noise_index_bw += samples
                if noise_index_bw > (len(bw_noise) - samples):
                    noise_index_bw = 0
            
            # Add EM noise if selected
            if use_em:
                em_segment = em_noise[noise_index_em:noise_index_em + samples]
                mixed_noise += em_segment
                noise_count += 1
                noise_index_em += samples
                if noise_index_em > (len(em_noise) - samples):
                    noise_index_em = 0
            
            # Add MA noise if selected
            if use_ma:
                ma_segment = ma_noise[noise_index_ma:noise_index_ma + samples]
                mixed_noise += ma_segment
                noise_count += 1
                noise_index_ma += samples
                if noise_index_ma > (len(ma_noise) - samples):
                    noise_index_ma = 0

            if noise_count == 0:
                bw_segment = bw_noise[noise_index_bw:noise_index_bw + samples]
                mixed_noise = bw_segment
                noise_index_bw += samples
                if noise_index_bw > (len(bw_noise) - samples):
                    noise_index_bw = 0
                noise_count = 1
            
            # Scale noise based on SNR
            signal_power = np.mean(beats[i]**2)
            noise_power = np.mean(mixed_noise**2)
            
            if noise_power > 0:
                noise_scaling = np.sqrt(signal_power / (snr_linear * noise_power))
                scaled_noise = mixed_noise * noise_scaling
            else:
                scaled_noise = mixed_noise
            
            # Generate noisy signal
            signal_noise = beats[i] + scaled_noise
            sn_list.append(signal_noise)
        
        return sn_list, noise_combinations, snr_values

    # Add mixed noise to training set
    sn_train, train_noise_combinations, train_snr_values = add_mixed_noise(
        beats_train, bw_noise_train, em_noise_train, ma_noise_train, is_test=False)

    # Add mixed noise to test set
    sn_test, test_noise_combinations, test_snr_values = add_mixed_noise(
        beats_test, bw_noise_test, em_noise_test, ma_noise_test, is_test=True)

    np.save(config['rnd_test_path'], test_snr_values)
    np.save(config['rnd_test_path'].replace('.npy', '_noise_combinations.npy'), test_noise_combinations)
    print('test_snr_values shape: ' + str(np.array(test_snr_values).shape))
    print('test_noise_combinations shape: ' + str(np.array(test_noise_combinations).shape))

    # Convert to numpy arrays and add channel dimension
    X_train = np.array(sn_train)
    y_train = np.array(beats_train)
    
    X_test = np.array(sn_test)
    y_test = np.array(beats_test)
    
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset