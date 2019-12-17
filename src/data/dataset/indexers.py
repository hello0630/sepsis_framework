class LocIndexer():
    """ Emulates the pandas loc behaviour to work with TimeSeriesDataset. """
    def __init__(self, dataset):
        """
        Args:
            dataset (class): A TimeSeriesDataset class instance.
        """
        self.dataset = dataset

    def _tuple_loc(self, query):
        """ Loc getter if query is specified as an (id, column) tuple. """
        idx, cols = query

        # Handle single column case
        if isinstance(cols, str):
            cols = [cols]

        # Ensure cols exist and get integer locations
        col_mask = self.dataset._col_indexer(cols)

        if not isinstance(idx, slice):
            assert isinstance(idx, int), 'Either index with a slice (a:b) or an integer.'
            idx = slice(idx, idx+1)

        return self.dataset.data[idx, :, col_mask]

    def __getitem__(self, query):
        """
        Args:
            query [slice, list]: Works like the loc indexer, e.g. [1:5, ['col1', 'col2']].
        """
        if isinstance(query, tuple):
            output = self._tuple_loc(query)
        else:
            output = self.dataset.data[query, :, :]
        return output
