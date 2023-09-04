class Output:
    def __init__(self, parameters):
        self.parameters = parameters

    def write_parameters_to_file(self):
        with open(self.parameters['output_file'], 'w') as file:
            self.write_dict_recursive(self.parameters, file)

    def write_dict_recursive(self, dictionary, file, indent=0, is_nested=False):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                file.write(f"{'  ' * indent}{key}:\n")
                self.write_dict_recursive(value, file, indent + 1, is_nested=True)
            else:
                file.write(f"{'  ' * indent}{key}: {value}\n")

            if not is_nested:
                file.write('\n')



