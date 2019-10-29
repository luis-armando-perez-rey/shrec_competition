
from modules.shrec import cvae_parameters, cvae_basic, cvae_vgg, cvae_shared, cvae_pose
import pandas as pd
import os

"""
Functions for loading the parameters of a trained CVAE. 
"""

def retrieve_parameters_dataframe(path, print= False):
    stored_models_file = os.path.join(path, "diffusion_vae_experiments.csv")
    df = pd.read_csv(stored_models_file)
    if print:
        print(df)
    return df


def load_trained_vae(path, num_experiment):
    # Read parameters from file
    df = retrieve_parameters_dataframe(path)
    dict_parameters = {"image_shape": eval(df["image_shape"][num_experiment]),
                       "num_classes": int(df["num_classes"][num_experiment]),
                       "r_loss": str(df["r_loss"][num_experiment]),
                       "intermediate_dim": int(df["intermediate_dim"][num_experiment]),
                       "type_layers": str(df["type_layers"][num_experiment]),
                       "num_encoding_layers": int(df["num_encoding_layers"][num_experiment]),
                       "num_decoding_layers": int(df["num_decoding_layers"][num_experiment]),
                       "kernel_size": int(df["kernel_size"][num_experiment]),
                       "stride": int(df["stride"][num_experiment]),
                       "latent_dim": int(df["latent_dim"][num_experiment])}
    print(dict_parameters)

    vae_params = cvae_parameters.CVAEParams(**dict_parameters)
    if df["type"][num_experiment]=="basic":
        vae = cvae_basic.CVAE_Basic(vae_params)
    elif df["type"][num_experiment]=="VGG":
        vae = cvae_vgg.CVAE_VGG(vae_params)
    elif df["type"][num_experiment]=="shared":
        vae = cvae_shared.CVAE_Shared(vae_params)
    elif df["type"][num_experiment]=="pose":
        vae = cvae_pose.CVAEPose(vae_params)
    else:
        vae = None
    vae.vae.summary()
    weight_path = os.path.join(path, "weights_folder", df["timestamp"][num_experiment] + '.h5')
    vae.load_model(weight_path)
    return vae
