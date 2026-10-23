"""Host-side scripts for Memory OS (retrieval fallbacks, ingest, maintenance).

WHY THIS FILE EXISTS
--------------------
``icarus/hooks.py::_load_context_enhancer`` imports
``scripts.context_enhancer``. That is a *package* import: it requires ``scripts``
to be a package, i.e. to have an ``__init__.py``. Without one, the import raised
``ModuleNotFoundError: No module named 'scripts'`` and the Qdrant recall layer
was disabled for the whole process — logged at ERROR, but only visible if you
read the log. The four-level fallback chain that makes this project worth using
was therefore never reachable.

This package marker is what makes ``import scripts.context_enhancer`` work when
the repo root is on sys.path (which ``_plugin_candidate_paths`` arranges). The
individual modules remain runnable as bare scripts
(``python3 scripts/wiki_continuous_ingest.py``) because they do their own
sys.path manipulation at the top.
"""
