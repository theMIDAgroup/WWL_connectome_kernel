Quickstart
==========

The pipeline needs three NumPy objects::

   SC : (S, N, N)  structural connectivity, one matrix per subject
   FC : (S, N, N)  functional connectivity, one matrix per subject
   y  : (S,)       integer group label per subject

The full runnable example is ``examples/quickstart.py``:

.. literalinclude:: ../examples/quickstart.py
   :language: python
   :pyobject: main
