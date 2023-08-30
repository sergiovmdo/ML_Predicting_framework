class Output:
    def __init__(self, parameters):
        self.parameters = parameters

    def write_parameters_to_file(self):
        with open(self.parameters['filename'], 'w') as file:
            self.write_dict_recursive(self.parameters, file)

    def write_dict_recursive(self, dictionary, file, indent=0):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                file.write(f"{'  ' * indent}{key}:\n")
                self._write_dict_recursive(value, file, indent + 1)
            else:
                file.write(f"{'  ' * indent}{key}: {value}\n")

