.. highlight:: shell

===================
Project Maintenance
===================

This page covers tips for project maintenance.

Releasing a New Version
-----------------------

1. Checkout the latest ``main`` branch.
2. Checkout a branch with the name of the version.

  ::

      # Prepend "v" to <version>
      # For release candidates, append "rc" to <version>
      git checkout -b v<version>
      git push --set-upstream origin v<version>

3. Bump version using tbump.

  ::

      # Exclude "v" and <version> should match step 3.
      # --no-tag is required since tagging is handled in step 5.
      tbump <version> --no-tag

4. Add updates to ``HISTORY.rst``
5. Create a pull request to the main repo and merge it.
6. Create a GitHub release.
7. Open a PR to the xcdat conda-forge feedstock with the latest changes.

Continuous Integration / Continuous Delivery (CI/CD)
-----------------------------------------------------

This project uses `GitHub Actions <https://github.com/XCDAT/xcdat/actions>`_ to run the CI/CD build workflow.

This workflow is triggered by Git ``pull_request`` and ``push`` (merging PRs) events to the the main repo's ``main`` branch.

Jobs:
    1. Run ``pre-commit`` for formatting, linting, and type checking
    2. Build conda development environment and run test suite
