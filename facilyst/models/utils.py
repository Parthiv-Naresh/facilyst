"""Utility functions for all model types."""
from typing import Optional

from facilyst.models.model_base import ModelBase
from facilyst.utils import _get_subclasses
from facilyst.utils.gen_utils import handle_problem_type

all_models = set(_get_subclasses(ModelBase))


def get_models(
    name_or_tag: Optional[str] = None,
    problem_type: Optional[str] = None,
    exclude: Optional[str] = None,
) -> set:
    """Return all models that correspond to either the name or type passed.

    A model can be selected either by its name, or by its primary, secondary, or tertiary type. If problem type is passed,
    then only models belonging to that type will be returned. If the name of a model is passed that conflicts with the
    problem type passed, then an error will be raised.

    name_or_tag (str): The name or tag of model(s) to return.
    problem_type (str): The problem type to which the models should belong, `regression`, `classification`, or `time series`.
    exclude (str): The name or tag to exclude.
    return (set): A set of all models found.
    """
    all_models = set(_get_subclasses(ModelBase))

    if _is_any_allowed(name_or_tag) and _is_any_allowed(problem_type):
        if exclude is None or exclude == "":
            return all_models
        else:
            return all_models - _get_models_by_name(exclude, None).union(
                _get_models_by_tag(exclude)
            )

    if _is_any_allowed(name_or_tag):
        problem_type = handle_problem_type(problem_type)
        primary_tagged = _get_models_by_primary_tag(primary_tag=problem_type)
        if exclude is None or exclude == "":
            return primary_tagged
        else:
            return primary_tagged - _get_models_by_name(exclude, None).union(
                _get_models_by_tag(exclude)
            )

    if _is_any_allowed(problem_type):
        try:
            problem_type = handle_problem_type(name_or_tag)
            primary_tagged = _get_models_by_primary_tag(primary_tag=problem_type)
            if exclude is None or exclude == "":
                return primary_tagged
            else:
                return primary_tagged - _get_models_by_name(exclude, None).union(
                    _get_models_by_tag(exclude)
                )
        except ValueError:
            pass
        name_tagged = _get_models_by_name(name=name_or_tag, problem_type=None)
        secondary_tagged = _get_models_by_secondary_tag(secondary_tag=name_or_tag)
        tertiary_tagged = _get_models_by_tertiary_tag(tertiary_tag=name_or_tag)
        no_models_found = ValueError(
            f"No models were found for that name/tag. Available model names are: \n"
            f"All model names: {sorted(set(each_model.name for each_model in all_models))} \n"
            f"Available model tags are: \n"
            f"Primary tags: {sorted(set(each_model.primary_type for each_model in all_models))} \n"
            f"Secondary tags: {sorted(set(each_model.secondary_type for each_model in all_models))} \n"
            f"Tertiary tags: {sorted(set(each_model.tertiary_type for each_model in all_models))}"
        )
        if name_tagged or secondary_tagged or tertiary_tagged:
            all_name_tagged_models = name_tagged.union(secondary_tagged).union(
                tertiary_tagged
            )
            if exclude is None or exclude == "":
                return all_name_tagged_models
            else:
                return all_name_tagged_models - _get_models_by_name(
                    exclude, None
                ).union(_get_models_by_tag(exclude))
        else:
            raise no_models_found

    problem_type = handle_problem_type(problem_type)
    name_tagged = _get_models_by_name(name=name_or_tag, problem_type=problem_type)
    primary_tagged = _get_models_by_primary_tag(primary_tag=problem_type)
    secondary_tagged = _get_models_by_secondary_tag(secondary_tag=name_or_tag)
    tertiary_tagged = _get_models_by_tertiary_tag(tertiary_tag=name_or_tag)
    no_models_found = ValueError(
        f"No models were found for that name/tag. Available model names are: \n"
        f"All model names: {sorted(set(each_model.name for each_model in all_models))} \n"
        f"Available model tags are: \n"
        f"Primary tags: {sorted(set(each_model.primary_type for each_model in all_models))} \n"
        f"Secondary tags: {sorted(set(each_model.secondary_type for each_model in all_models))} \n"
        f"Tertiary tags: {sorted(set(each_model.tertiary_type for each_model in all_models))}"
    )
    if name_tagged or secondary_tagged or tertiary_tagged:
        all_name_tagged_models = primary_tagged.intersection(
            name_tagged.union(secondary_tagged).union(tertiary_tagged)
        )
        if exclude is None or exclude == "":
            return all_name_tagged_models
        else:
            return all_name_tagged_models - _get_models_by_name(exclude, None).union(
                _get_models_by_tag(exclude)
            )
    else:
        raise no_models_found


def _is_any_allowed(tag):
    if tag is None or tag.lower() in ["any", "all", ""]:
        return True


def _get_models_by_name(name, problem_type):
    subset_models_name = []
    subset_models_problem_type = _get_models_by_primary_tag(primary_tag=problem_type)
    for model in subset_models_problem_type:
        if name.lower() in model.name.lower():
            subset_models_name.append(model)
    return set(subset_models_name)


def _get_models_by_tag(tag):
    try:
        problem_type = handle_problem_type(tag)
        subset_models_tagged = _get_models_by_primary_tag(primary_tag=problem_type)
    except ValueError:
        secondary_tagged = _get_models_by_secondary_tag(secondary_tag=tag)
        tertiary_tagged = _get_models_by_tertiary_tag(tertiary_tag=tag)
        subset_models_tagged = secondary_tagged.union(tertiary_tagged)
    return subset_models_tagged


def _get_models_by_primary_tag(primary_tag):
    if _is_any_allowed(primary_tag):
        return all_models
    else:
        subset_models_primary = {
            each_model
            for each_model in all_models
            if primary_tag.lower() == each_model.primary_type.lower()
        }
        return subset_models_primary


def _get_models_by_secondary_tag(secondary_tag):
    subset_models_secondary = {
        each_model
        for each_model in all_models
        if secondary_tag.lower() == each_model.secondary_type.lower()
    }
    return subset_models_secondary


def _get_models_by_tertiary_tag(tertiary_tag):
    subset_models_tertiary = {
        each_model
        for each_model in all_models
        if tertiary_tag.lower() == each_model.tertiary_type.lower()
    }
    return subset_models_tertiary
