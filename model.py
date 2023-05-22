import tensorflow as tf
from dataclasses import dataclass
from typing import Union, List, Dict
import numpy as np
from copy import deepcopy

class HyperParameter:
    def __init__(self, value: Union[int, float, str], possible_values: Dict[str, Union[str, List[Union[int, float]]]]):
        """
        Initializes a HyperParameter instance.
        
        Args:
        value: Initial value for this hyperparameter.
        possible_values: Dictionary specifying the type and possible values for this hyperparameter.
        """
        self.value = value
        self.possible_values = possible_values

    def randomize(self):
        """
        Randomizes this hyperparameter based on its possible_values.
        """
        value_type = self.possible_values['type']
        possible_values = self.possible_values['values']
        
        if value_type == 'list':
            self.value = np.random.choice(possible_values)
        elif value_type == 'power_2_range':
            power_low, power_high = map(int, np.log2(self.possible_values['values']))
            power = np.random.randint(power_low, power_high + 1)
            self.value = 2**power
        elif value_type == 'int_range':
            low, high = self.possible_values['values']
            self.value = np.random.randint(low, high + 1)
        elif value_type == 'float_range':
            self.value = np.random.uniform(*possible_values)
        elif self.possible_values == 'this':
            self.value == self.possible_values

    def mutate(self, mutation_rate: float):
        """
        Returns a new HyperParameter with a mutated value, based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_param: New HyperParameter instance with potentially mutated value.
        """
        mutated_param = deepcopy(self) 
        if np.random.random() < mutation_rate:
            value_type = self.possible_values['type']
            possible_values = self.possible_values['values']
            if value_type in ['int_range', 'power_2_range', 'float_range']:
                mutation = np.random.normal(scale=(possible_values[1] - possible_values[0]) / 10)
                new_value = self.value + mutation
                # Make sure new value is in the allowed range
                new_value = min(max(new_value, *possible_values))
                mutated_param.value = new_value
            else:
                mutated_param.randomize()
            return mutated_param
        else:
            return deepcopy(self)


@dataclass
class BaseLayer:
    layer_type: HyperParameter
    activation: HyperParameter
    mutable_attributes: List
    
    def randomize(self):
        """
        Randomizes all mutable attributes of this layer.
        """
        for attribute in self.mutable_attributes:
            attribute.randomize()
            
    def mutate(self, mutation_rate: float):
        """
        Returns a new layer with mutated hyperparameters based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_layer: New BaseLayer instance with potentially mutated hyperparameters.
        """
        mutated_layer = deepcopy(self)
        for attribute_name in mutated_layer.mutable_attributes:
            mutated_value = getattr(mutated_layer, attribute_name).mutate(mutation_rate)
            setattr(mutated_layer, attribute_name, mutated_value)
        return mutated_layer


@dataclass
class DenseLayer(BaseLayer):
    units: HyperParameter

    def __init__(self, units: HyperParameter, activation: HyperParameter):
        """
        Initializes a DenseLayer instance.
        
        Args:
        units: HyperParameter specifying the number of units in this layer.
        activation: HyperParameter specifying the activation function for this layer.
        """
        self.layer_type = "Dense"
        self.activation = activation
        self.units = units
        self.mutable_attributes = [self.activation, self.units]


@dataclass
class ConvLayer(BaseLayer):
    filters: HyperParameter
    kernel_size: HyperParameter
    strides: HyperParameter = HyperParameter(1, {'type': 'int', 'values': [1]})
    padding: HyperParameter = HyperParameter('valid', {'type': 'str', 'values': ['valid', 'same']})
    
    def __init__(self, filters: HyperParameter, kernel_size: HyperParameter, activation: HyperParameter, strides: HyperParameter):
        """
        Initializes a ConvLayer instance.
        
        Args:
        filters: HyperParameter specifying the number of filters in this layer.
        kernel_size: HyperParameter specifying the kernel size in this layer.
        activation: HyperParameter specifying the activation function for this layer.
        strides: HyperParameter specifying the stride length for this layer.
        """
        self.layer_type = "Convolutional"
        self.activation = activation
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = HyperParameter("same", {"type" : "this", "values" : "same"})
        self.mutable_attributes = [self.activation, self.filters, self.kernel_size, self.strides]


def randomizeLayer(layer_types: List[str], default_layers: Dict[str, BaseLayer]) -> BaseLayer:
    """
    Returns a randomized layer of a random type.
    
    Args:
    layer_types: List of possible layer types.
    default_layers: Dictionary mapping layer types to default layers of that type.
    
    Returns:
    layer: New BaseLayer instance of a random type, with randomized hyperparameters.
    """
    layer_type = np.random.choice(layer_types)
    layer = default_layers[layer_type]
    layer.randomize()
    return layer


class ModelBuilder:
    def __init__(self, layers: List[BaseLayer], optimizer: str, loss: str, batch_size: int):
        """
        Initializes a ModelBuilder instance.
        
        Args:
        layers: List of BaseLayer instances making up the model.
        optimizer: Optimizer to use when training the model.
        loss: Loss function to use when training the model.
        batch_size: Batch size to use when training the model.
        """        
        self.layers = layers
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        
        self.metrics = []

    def build_model(self, input_shape: Union[int, tuple], output_shape: int):
        """
        Builds the model.
        
        Args:
        input_shape: Shape of the input data.
        output_shape: Shape of the output data.
        """        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Reshape((-1, 1)))

        for layer in self.layers:
            if layer.layer_type == "Dense":
                model.add(
                    tf.keras.layers.Dense(
                        layer.units.value, 
                        activation=layer.activation.value
                    )
                )
            elif layer.layer_type == "Convolutional":
                model.add(tf.keras.layers.Conv1D(
                    layer.filters.value, 
                    layer.kernel_size.value, 
                    strides=layer.strides.value, 
                    padding=layer.padding.value, 
                    activation=layer.activation.value)
                )
            else:
                raise ValueError(f"Layer type '{layer.layer_type.value}' not recognized")
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
        model.compile(optimizer=self.optimizer, loss=self.loss)
        
        self.model = model

    def train_model(self, train_dataset: tf.data.Dataset, num_epochs: int):
        """
        Trains the model.
        
        Args:
        train_dataset: Dataset to train on.
        num_epochs: Number of epochs to train for.
        """
        self.metrics.append(
            self.model.fit(
                train_dataset, 
                epochs=num_epochs, 
                batch_size=self.batch_size
            )
        )
        
        print(self.metrics)

    def test_model(self, validation_datasets: tf.data.Dataset, batch_size: int):
        """
        Tests the model.
        
        Args:
        validation_datasets: Dataset to test on.
        batch_size: Batch size to use when testing.
        """
        self.model.evaluate(validation_datasets, batch_size=batch_size)
        
    def summary(self):
        """
        Prints a summary of the model.
        """
        self.model.summary()
        
    @staticmethod
    def crossover(parent1: 'ModelBuilder', parent2: 'ModelBuilder') -> 'ModelBuilder':
        """
        Creates a new model whose hyperparameters are a combination of two parent models.
        The child model is then returned.
        """
        # Determine the shorter and longer layers lists
        short_layers, long_layers = (parent1.layers, parent2.layers) if len(parent1.layers) < len(parent2.layers) else (parent2.layers, parent1.layers)

        # Choose a random split point in the shorter layers list
        split_point = np.random.randint(1, len(short_layers))

        # Choose which parent to take the first part from
        first_part, second_part = (short_layers[:split_point], long_layers[split_point:]) if np.random.random() < 0.5 else (long_layers[:split_point], short_layers[split_point:])
        child_layers = first_part + second_part

        child_model = ModelBuilder(child_layers, parent1.optimizer, parent1.loss, parent1.batch_size)

        return child_model
    
    def mutate(self, mutation_rate: float) -> 'ModelBuilder':
        """
        Returns a new model with mutated layers based on the mutation_rate.
        
        Args:
        mutation_rate: Probability of mutation.
        
        Returns:
        mutated_model: New ModelBuilder instance with potentially mutated layers.
        """
        mutated_layers = [layer.mutate(mutation_rate) for layer in self.layers]
        mutated_model = ModelBuilder(mutated_layers, self.optimizer, self.loss, self.batch_size)

        return mutated_model

class Population:
    def __init__(
        self, 
        initial_population_size: int, 
        max_population_size: int,
        genome_template: int,
        input_size : int,
        output_size : int 
    ):
        
        self.initial_population_size = initial_population_size
        self.current_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.population = \
            self.initilize_population(
                genome_template, 
                input_size, 
                output_size
            )
        self.fitnesses = np.ones(self.current_population_size)
        
    def initilize_population(self, genome_template, input_size, output_size):
    
        population = []
        for j in range(self.initial_population_size):   
            layers = []
            
            genome_template['base']['num_layers'].randomize()
            genome_template['base']['optimizer'].randomize()
            genome_template['base']['batch_size'].randomize()
            
            for i in range(genome_template['base']['num_layers'].value):
                layers.append(
                    randomizeLayer(*genome_template['layers'][i])
                )
                
            # Create an instance of DenseModelBuilder with num_neurons list
            builder = ModelBuilder(
                layers, 
                optimizer = genome_template['base']['optimizer'].value, 
                loss ='sparse_categorical_crossentropy', 
                batch_size = genome_template['base']['batch_size'].value
            )

            # Build the model with input shape (input_dim,)
            builder.build_model(
                input_shape=(input_size,), 
                output_shape = output_size
            )
            population.append(builder)
            builder.summary()
            
        return population
    
    def roulette_wheel_selection(self):
        """
        Performs roulette wheel selection on the population.

        Args:
            population (list): The population of individuals.
            fitnesses (list): The fitness of each individual in the population.

        Returns:
            The selected individual from the population.
        """

        # Convert the fitnesses to probabilities.
        total_fit = sum(self.fitnesses)
        prob = [fit/total_fit for fit in self.fitnesses]

        # Calculate the cumulative probabilities.
        cumulative_probs = np.cumsum(prob)

        # Generate a random number in the range [0, 1).
        r = np.random.rand()

        # Find the index of the individual to select.
        for i in range(len(self.population)):
            if r <= cumulative_probs[i]:
                return self.population[i]

        # If we've gotten here, just return the last individual in the population.
        # This should only happen due to rounding errors, and should be very rare.
        return self.population[-1]
    
    def train_population(self, num_generations, training_ds):
        for i in range(self.current_population_size*num_generations):
            self.roulette_wheel_selection().train_model(training_ds, 1)
        
        
        
                    