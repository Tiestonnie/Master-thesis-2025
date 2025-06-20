"""decadal variability of subseasonal predictability experimental settings

"""

__author__ = "Marybeth Arcodia; modified from Libby Barnes"
__date__   = "24 May 2022"


def get_settings(experiment_name):
    experiments = {     
        
        "exp_1/exp_100": {
            "PREDICTOR_VAR": 'SLP',
            "PREDICTAND_VAR": 'NAO',
            "REGION_TOR": 'north_atlantic',
            "REGION_TAND": 'north_atlantic',
            "training_ens": (0,1,2,3,4,5,6,7),
            "validation_ens": (8),
            "testing_ens": (9),
            "train_list": '01234567',
            "train_val_list": '012345678',
            "lead": 14,          #days of lead for first day of predictions
            "days_average": 14,   #number of days predictand data was averaged over 
            "GLOBAL_SEED": 147483648,
            "HIDDENS": [512, 256, 64],
            "DROPOUT": 0.2,
            "RIDGE1": 5.0,            
            "LR_INIT": 0.001,
            "BATCH_SIZE": 512,
            "RANDOM_SEED": [210, 47, 33, 133, 410], # 692, 64, 99, 910, 92]
            "act_fun": "relu",
            "N_EPOCHS": 300,
            "PATIENCE": 25,
        },          
        "exp_4/exp_400": {
            "PREDICTOR_VAR": 'SLP',
            "PREDICTAND_VAR": 'NAO',
            "REGION_TOR": 'north_atlantic',
            "REGION_TAND": 'north_atlantic',
            "training_ens": (0,1,2,3,4,5,6,7),
            "validation_ens": (8),
            "testing_ens": (9),
            "train_list": '01234567',
            "train_val_list": '012345678',
            "lead": 21,          #days of lead for first day of predictions
            "days_average": 14,   #number of days predictand data was averaged over 
            "GLOBAL_SEED": 147483648,
            "HIDDENS": [512, 256, 64],
            "DROPOUT": 0.2,
            "RIDGE1": 5.0,            
            "LR_INIT": 0.001,
            "BATCH_SIZE": 512,
            "RANDOM_SEED": [210, 47], #, 33, 133, 410, 692, 64, 99, 910, 92]
            "act_fun": "relu",
            "N_EPOCHS": 300,
            "PATIENCE": 25,
        },
        "exp_5/exp_500": {
            "PREDICTOR_VAR": 'SLP',
            "PREDICTAND_VAR": 'NAO',
            "REGION_TOR": 'north_atlantic',
            "REGION_TAND": 'north_atlantic',
            "training_ens": (0,1,2,3,4,5,6,7),
            "validation_ens": (8),
            "testing_ens": (9),
            "train_list": '01234567',
            "train_val_list": '012345678',
            "lead": 28,          #days of lead for first day of predictions
            "days_average": 14,   #number of days predictand data was averaged over 
            "GLOBAL_SEED": 147483648,
            "HIDDENS": [512, 256, 64],
            "DROPOUT": 0.2,
            "RIDGE1": 5.0,            
            "LR_INIT": 0.001,
            "BATCH_SIZE": 512,
            "RANDOM_SEED": [210, 47], #, 33, 133, 410, 692, 64, 99, 910, 92]
            "act_fun": "relu",
            "N_EPOCHS": 300,
            "PATIENCE": 25,
        },
        "exp_6/exp_600": {
            "PREDICTOR_VAR": 'SLP',
            "PREDICTAND_VAR": 'NAO',
            "REGION_TOR": 'north_atlantic',
            "REGION_TAND": 'north_atlantic',
            "training_ens": (0,1,2,3,4,5,6,7),
            "validation_ens": (8),
            "testing_ens": (9),
            "train_list": '01234567',
            "train_val_list": '012345678',
            "lead": 35,          #days of lead for first day of predictions
            "days_average": 14,   #number of days predictand data was averaged over 
            "GLOBAL_SEED": 147483648,
            "HIDDENS": [512, 256, 64],
            "DROPOUT": 0.2,
            "RIDGE1": 5.0,            
            "LR_INIT": 0.001,
            "BATCH_SIZE": 512,
            "RANDOM_SEED": [210, 47], #, 33, 133, 410, 692, 64, 99, 910, 92]
            "act_fun": "relu",
            "N_EPOCHS": 300,
            "PATIENCE": 25,
        },
        "exp_7/exp_700": {
            "PREDICTOR_VAR": 'SLP',
            "PREDICTAND_VAR": 'NAO',
            "REGION_TOR": 'north_atlantic',
            "REGION_TAND": 'north_atlantic',
            "training_ens": (0,1,2,3,4,5,6,7),
            "validation_ens": (8),
            "testing_ens": (9),
            "train_list": '01234567',
            "train_val_list": '012345678',
            "lead": 42,          #days of lead for first day of predictions
            "days_average": 14,   #number of days predictand data was averaged over 
            "GLOBAL_SEED": 147483648,
            "HIDDENS": [512, 256, 64],
            "DROPOUT": 0.2,
            "RIDGE1": 5.0,            
            "LR_INIT": 0.001,
            "BATCH_SIZE": 512,
            "RANDOM_SEED": [210, 47], #, 33, 133, 410, 692, 64, 99, 910, 92]
            "act_fun": "relu",
            "N_EPOCHS": 300,
            "PATIENCE": 25,
        },
        "exp_8/exp_800": {
            "PREDICTOR_VAR": 'SLP',
            "PREDICTAND_VAR": 'NAO',
            "REGION_TOR": 'north_atlantic',
            "REGION_TAND": 'north_atlantic',
            "training_ens": (0,1,2,3,4,5,6,7),
            "validation_ens": (8),
            "testing_ens": (9),
            "train_list": '01234567',
            "train_val_list": '012345678',
            "lead": 49,          #days of lead for first day of predictions
            "days_average": 14,   #number of days predictand data was averaged over 
            "GLOBAL_SEED": 147483648,
            "HIDDENS": [512, 256, 64],
            "DROPOUT": 0.2,
            "RIDGE1": 5.0,            
            "LR_INIT": 0.001,
            "BATCH_SIZE": 512,
            "RANDOM_SEED": [210, 47], #, 33, 133, 410, 692, 64, 99, 910, 92]
            "act_fun": "relu",
            "N_EPOCHS": 300,
            "PATIENCE": 25,
        },
        "exp_9/exp_900": {
            "PREDICTOR_VAR": 'SLP',
            "PREDICTAND_VAR": 'NAO',
            "REGION_TOR": 'north_atlantic',
            "REGION_TAND": 'north_atlantic',
            "training_ens": (0,1,2,3,4,5,6,7),
            "validation_ens": (8),
            "testing_ens": (9),
            "train_list": '01234567',
            "train_val_list": '012345678',
            "lead": 56,          #days of lead for first day of predictions
            "days_average": 14,   #number of days predictand data was averaged over 
            "GLOBAL_SEED": 147483648,
            "HIDDENS": [512, 256, 64],
            "DROPOUT": 0.2,
            "RIDGE1": 5.0,            
            "LR_INIT": 0.001,
            "BATCH_SIZE": 512,
            "RANDOM_SEED": [210, 47], #, 33, 133, 410, 692, 64, 99, 910, 92]
            "act_fun": "relu",
            "N_EPOCHS": 300,
            "PATIENCE": 25,
        },                   
    }

    return experiments[experiment_name]