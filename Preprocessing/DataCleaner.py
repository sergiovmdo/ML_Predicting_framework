class DataCleaner:
    """
    Parent class that will deal with the first step of the pipeline. Standardization of columns, duplicate checking,
    data formatting, imputation, and outlier removal are performed in this step.

    There are two classes that inheritate from this class, Imputer and OutlierRemover that will be in charge of this
    particular tasks of the pipeline.
    """
    def __init__(self):
        pass