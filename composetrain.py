import sys
import os
from func_train_f import train_pretrain
from func_train import train_finetune

def main():
    '''
    target_features = [
        'get_rms',
    ]
    '''
    '''
    target_features = [
        'get_sum_sq_diff',
        'get_rms',
        'get_energy',
        'get_zcr',
        'get_env_mean_max_ratio',
        'get_env_median_max_ratio',
        'get_rise_fall_ratio',
        'get_raw_kurtosis',
        'get_env_kurtosis',
        'get_raw_skewness',
        'get_env_skewness',
        'get_crest_factor',
        'get_autocorr_peaks',
        'get_autocorr_energy_ratio',
        'get_linear_decay_error',
        'get_spec_mean',
        'get_dom_freq',
        'get_quartile_freq',
        'get_spec_peaks_count',
        'get_nyquist_band_energy',
        'get_spec_centroid',
        'get_gyration_radius',
        'get_spec_bandwidth',
    ]
    
    '''
    target_features = [
        'get_sum_sq_diff',
        #'get_env_mean_max_ratio',       
        'get_raw_kurtosis',     
        #'get_env_median_max_ratio',     
    ]


    total_features = len(target_features)

    for index, phy_name in enumerate(target_features):
        print(f"[{index + 1}/{total_features}] Processing attribute: {phy_name}")
        print("=" * 50)
        
        try:
            print(f"--- Step 1: Pre-training ({phy_name}) ---")
            pretrain_weights = train_pretrain(phy_name)
            #pretrain_weights = "./modelnews/model_get_rms_pretrain/model_save/model_pretrain.ckpt"
            if pretrain_weights:
                print(f"Pre-training completed. Weights: {pretrain_weights}")
                
                print(f"--- Step 2: Fine-tuning ({phy_name}) ---")
                train_finetune(phy_name, pretrain_weights)
                print(f"Fine-tuning completed for {phy_name}")
            else:
                print(f"Skipping Fine-tuning for {phy_name}: Pre-training failed or data missing.")
        
        except Exception as e:
            print(f"Error occurred while processing {phy_name}: {e}")
        
        print("\n")

if __name__ == "__main__":
    main()