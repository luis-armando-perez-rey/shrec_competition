
import numpy as np

class DatabaseObjectsViews():
    def __init__(self, images, identifiers, models,  vae):
        self.vae = vae

        # Basic properties of the images
        self.images = images
        self.num_images = len(images)
        self.images_size = images.shape[1]
        self.mean, self.log_var_z, self.pi_cat = self.image_latent()
        self.sigmas = np.exp(0.5 * self.log_var_z)
        self.objects_identifiers = identifiers
        self.num_classes = len(np.unique(self.objects_identifiers))
        self.models = models
        self.unique_identifiers, self.unique_models= self.calculate_unique_identifiers()



        # Images by identifier
        self.identifiers_by_model = self.separate_by_identifier(models, identifiers)
        self.images_by_identifier  = self.separate_by_identifier(images, identifiers)
        self.mean_by_identifier = self.separate_by_identifier(self.mean, identifiers)
        self.log_var_by_identifier= self.separate_by_identifier(self.log_var_z, identifiers)




    def image_latent(self):
        mean, log_var_z, pi_cat = self.vae.encode(self.images, batch_size = 128)
        return mean, log_var_z, pi_cat


    def calculate_unique_identifiers(self):
        objects_identifiers = []
        unique_models = []
        for num_model, model in enumerate(self.models):
            if model not in unique_models:
                unique_models.append(model)
                objects_identifiers.append(self.objects_identifiers[num_model])
        objects_identifiers = np.array(objects_identifiers)
        unique_models = np.array(unique_models)
        return objects_identifiers, unique_models

    def separate_by_identifier(self,images, identifiers):
        """
        Separate the images with respect to an identifier
        :identifiers: identifiers that describe the images
        :return: array with the separated images first dimension corresponds to the number of the view
        """
        unique_identifiers = np.unique(identifiers)
        boolean_flag = np.zeros(len(images), dtype=bool)
        for identifier in unique_identifiers:
            boolean_flag = boolean_flag + (identifiers == identifier)
        separated = images[boolean_flag]
        return separated




