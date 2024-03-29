Release Notes
-------------

Future Release
==============
    * Enhancements
        * Added ``FTDatetime`` as a preprocessor to perform feature engineering on datetime columns :pr:`55`
        * Added ``TimeSeriesModelPipeline`` to support time series models from ``statsforecast`` :pr:`73`
        * Added ``ADIDA``, ``AutoARIMA``, ``AutoETS``, ``AuthoTheta``, ``CrostonOptimized``, ``IMAPA``, and ``TSB`` time series models :pr:`73`
    * Fixes
        * Removed automatic writing of files during training for ``CatBoost`` models :pr:`79`
    * Changes
        * Improved the logic for ``get_models`` to include an ``exclude`` parameter :pr:`73`
    * Documentation Changes
    * Testing Changes
    * CI/CD Changes
        * Update ``latest_dependency_checker`` to edit ``Keras-Preprocessing`` :pr:`65`
        * Formatted ``dependency_diff.sh`` to include Keras_Preprocessing dependency :pr:`59`
        * Added ``windows_unit_tests`` GitHub Action to check that all tests pass on MRs in a windows environment :pr:`59`
        * Update ``latest_dependency_checker`` to remove the correct folders :pr:`55`
        * Update ``latest_dependency_checker`` to find source of egg-info folder :pr:`56`


v0.0.4 May 1, 2022
==================
    * Enhancements
    * Fixes
    * Changes
    * Documentation Changes
    * Testing Changes
    * CI/CD Changes
        * Added ``linux_unit_tests_without_extra_dependencies`` GitHub Action to check that all tests pass on MRs without extra dependencies :pr:`52`
        * Made ``Hyperopt`` a required dependency :pr:`48`


v0.0.3 May 1, 2022
==================
    * Enhancements
        * Added ``AggregateDatetime`` as a preprocessor to aggregate datetime components into a datetime column :pr:`29`
        * Added ``ADABoost``, ``Bagging``, ``CatBoost``, ``DecisionTree``, ``Extra Trees``, ``MultiLayerPerceptron``, and ``XGBoost`` classifiers :pr:`25`
        * Added type annotations :pr:`21`
        * Added ``BERTBinaryClassifier`` and ``BERTQuestionAnswering`` for NLP problems :pr:`20`
        * Added ``ADABoost``, ``Bagging``, ``CatBoost``, ``DecisionTree``, ``Extra Trees``, and ``XGBoost`` regressors :pr:`15`
        * Added Hyperopt optimizer :pr:`13`
        * Added primary, secondary, and tertiary types to models :pr:`13`
        * Added the ``RandomForestRegressor`` :pr:`13`
        * Added the ``get_models`` utility function to easily retrieve models by name or type :pr:`13`
        * Added common regression and time series regression datasets via git lfs :pr:`6`
        * Added ``MultiLayerPerceptronRegressor`` as a model :pr:`6`
        * Added nullable booleans, phone numbers, addresses, email addresses, URLs, currencies, file paths, full names, IPAddresses, and latitude/longitude as data types in ``Features`` :pr:`6`
        * Added ``Woodwork`` initialization for ``Features`` mock data :pr:`5`
        * Added ``Wave`` mock type :pr:`4`
        * Added convenience functions ``make_features``, ``make_dates``, and ``make_wave`` for making ``Features``, ``Dates``, and ``Wave`` mock types respectively :pr:`4`
        * Added ``Features``, ``Target``, and ``Dates`` mock types :pr:`3`
        * Added ``Scatter`` and ``Line`` graph types :pr:`3`
        * Added ``MockBase`` parent class :pr:`3`
        * Added ``GraphBase`` parent class :pr:`3`
        * Added ``create_data`` utility to make mock data creation easier :pr:`3`
        * Added ``handle_data_and_library_type`` and ``mock_dtypes`` helper functions :pr:`3`
    * Fixes
    * Changes
        * Removed ``Target`` mock data type :pr:`4`
    * Documentation Changes
        * Added status badges to README.md :pr:`35`
        * Added black profile to isort command to get parity between the two for new lint errors :pr:`6`
        * Updated docstrings for numerous files :pr:`6`
        * Added and updated docstrings for a variety of classes and functions :pr:`4`
    * Testing Changes
        * Updated test requirements to ``pytest>=7.1.1``, ``pytest-xdist>=2.5.0``, ``pytest-timeout>=2.1.0``, and ``pytest-cov>=3.0.0`` :pr:`6`
        * Added tests for locally stored datasets :pr:`6`
        * Added and updated testing for subclasses :pr:`4`
    * CI/CD Changes
        * Pinned ``setup.cfg`` test dependencies :pr:`43`
        * Updated ``check_latest_dependencies`` to handle extra dependencies :pr:`41`
        * Removed ``catboost`` from `latest_dependencies`` :pr:`40`
        * Separated ``extra-requirements`` from ``requirements`` :pr:`35`
        * Added ``pyproject.toml`` and simplified ``setup.py`` :pr:`35`
        * Added ``check_latest_dependencies`` GitHub Action to check for updates to the latest dependencies :pr:`23`
        * Added ``type_hint_tests`` GitHub Action to check for type hint errors on MRs :pr:`21`
        * Added and pinned ``pytype==2022.4.22`` :pr:`21`
        * Removed testing datasets from CI :pr:`7`
        * Parallelized ``linux_unit_tests`` testing :pr:`6`
        * Pinned to ``numpy==1.22.3``, ``scipy==1.8.0``, and ``plotly==5.7.0`` :pr:`6`
        * Added ``setup.py`` to enable the build process :pr:`3`
        * Added ``lint_tests`` GitHub Action to check for lint errors on MRs :pr:`3`
        * Added ``linux_unit_tests`` GitHub Action to check that all tests pass on MRs :pr:`3`
        * Added ``release_notes_updated`` GitHub Action to check that every MRs is associated with an update in the release notes :pr:`3`
        * Added ``codecov`` support :pr:`3`
