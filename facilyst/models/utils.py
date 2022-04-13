"""Utility functions for all model types."""
from facilyst.models.model_base import ModelBase
from facilyst.utils import _get_subclasses


def get_models(model):
    """Return all models that correspond to either the name or type passed.

    A model can be selected either by its name, or by its primary, secondary, or tertiary type.

    :param model: The name or type of model(s) to return.
    :type model: str
    :return: A list of all models found.
    :rtype list:
    """
    if not model:
        raise ValueError("No model name passed.")

    all_models = _get_subclasses(ModelBase)
    if " " not in model:
        if model.lower() in ["all", "any"]:
            return all_models
        subset_models_primary = [
            each_model
            for each_model in all_models
            if model.lower() in each_model.primary_type.lower()
        ]
        subset_models_secondary = [
            each_model
            for each_model in all_models
            if model.lower() in each_model.secondary_type.lower()
        ]
        subset_models_tertiary = [
            each_model
            for each_model in all_models
            if model.lower() in each_model.tertiary_type.lower()
        ]
        subset_models = (
            subset_models_primary or subset_models_secondary or subset_models_tertiary
        )

        if not subset_models:
            raise ValueError(
                f"That model type doesn't exist. Available model types are: \n"
                f"Primary types: {set(each_model.primary_type for each_model in all_models)} \n"
                f"Secondary types: {set(each_model.secondary_type for each_model in all_models)} \n"
                f"Tertiary types: {set(each_model.tertiary_type for each_model in all_models)}"
            )
        else:
            return subset_models

    for each_model in all_models:
        if each_model.name.lower() == model.lower():
            return [each_model]

    raise ValueError(
        f"That model doesn't exist. This is the list of all available models: {[each_model.name for each_model in all_models]}"
    )
