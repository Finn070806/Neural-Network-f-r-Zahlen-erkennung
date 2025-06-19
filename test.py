import numpy as np # type: ignore
# Funktion für Ausgabe begrenzung
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_input = np.array([ [0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1],
                            [1,0,0]])

training_outputs = np.array([[0,1,1,0,1]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Random starting synaptic weights: = ")
print(synaptic_weights)

for iteration in range(100000):

    input_layer = training_input

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synapric weights after training = ")
print(synaptic_weights)
#Ausgabe für die ergebnisse nach dem training
print("outputs after training: ")
print( outputs)
# Ausgabe für die erwarteten antworten
print("erwarteter output: ")
print(training_outputs)