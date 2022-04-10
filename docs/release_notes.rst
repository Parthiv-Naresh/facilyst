Release Notes
-------------

**Future Releases**
    * Enhancements
        *  Added common regression and time series regression datasets via git lfs :pr:`6`
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
        * Pinned to ``numpy==1.22.3``, ``scipy==1.8.0``, and ``plotly==5.7.0`` :pr:`6`
        * Removed ``Target`` mock data type :pr:`4`
    * Documentation Changes
        * Added black profile to isort command to get parity between the two for new lint errors :pr:`6`
        * Updated docstrings for numerous files :pr:`6`
        * Added and updated docstrings for a variety of classes and functions :pr:`4`
    * Testing Changes
        * Updated test requirements to ``pytest>=7.1.1``, ``pytest-xdist>=2.5.0``, ``pytest-timeout>=2.1.0``, and ``pytest-cov>=3.0.0`` :pr:`6`
        * Added tests for locally stored datasets :pr:`6`
        * Added and updated testing for subclasses :pr:`4`
    * CI/CD Changes
        * Parallelized ``linux_unit_tests`` testing :pr:`6`
        * Added ``setup.py`` to enable the build process :pr:`3`
        * Added ``lint_tests`` GitHub Action to check for lint errors on PRs :pr:`3`
        * Added ``linux_unit_tests`` GitHub Action to check that all tests pass on PRs :pr:`3`
        * Added ``release_notes_updated`` GitHub Action to check that every PR is associated with an update in the release notes :pr:`3`
        * Added ``codecov`` support :pr:`3`

.. warning::

    **Breaking Changes**
