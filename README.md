# backprop


A neural network for [Crystal], with one hidden layer and backprop learning.
## Usage

```crystal
# Creating a new network with two input layers, three hidden layers, and one
# output layer, with a learning rate of 1.0:
back = Backprop::Backprop.new([2, 3, 1], 1.0)

# Saving and loading current network weights (as YAML files):
back.save_weights("./wt.yaml")
back.load_weights("./wy.yaml")

# Of course, you can also load weights directly upon initialization:
back = Backprop::Backprop.new("./wy.yaml", 1.0)

# Use the network to calculate output from a a new input:
back.forward_calc(

## Contributing

1. Fork it ( https://github.com/[papaskrobe]/backprop/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## TODO

1. Modify input/output methods to use arrays instead of matrix classes
2. Error handling:
  -"Couldn't find/load file"
  -"Layer mismatch" (layers <> 3)
3. Separate the test code out of the modules
	-Is the "module" framework necessary?
