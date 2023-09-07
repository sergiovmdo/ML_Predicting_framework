class Output:
    """
    Class that represents the Output of the program, that will allocate all the information related to it.
    """
    def __init__(self, parameters):
        """
        Initialize a new instance of Output

        Args:
            parameters (dictionary): Set of parameters and results that have been collected during execution.

        """
        self.parameters = parameters

    def write_parameters_to_file(self):
        """
        Creates the output file and calls the recursive method that will perform the writing operation.
        """
        with open(self.parameters['output_file'], 'w') as file:
            self.write_dict_recursive(self.parameters, file)

    def write_dict_recursive(self, dictionary, file, indent=0, is_nested=False):
        """
        Iterates through all the parameters and results and writes them into a file.

        Args:
            dictionary (dictionary): set of parameters and results.
            file (file): file where everything it's written.
            indent (int): parameter used for indenting the text file.
            is_nested (boolean): parameter used for adding extra spaces inside the text file.
        """
        for key, value in dictionary.items():
            # If it is a dictionary we need to write all the values inside it, we will indent the results so it is
            # clear that the elements of the dictionary belong to it.
            if isinstance(value, dict):
                file.write(f"{'  ' * indent}{key}:\n")
                self.write_dict_recursive(value, file, indent + 1, is_nested=True)
            else:
                file.write(f"{'  ' * indent}{key}: {value}\n")

            # We add an extra space between parameters
            if not is_nested:
                file.write('\n')



