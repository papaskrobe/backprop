require "./backprop/*"
require "yaml"
require "../lib/matrix/src/matrix.cr"

module Backprop
	def sigmoid(val : Float64)
		return 1.0 / (1.0 + (2.71828 ** (-val)))
	end

	def sigmoid(val : Int32)
		return 1.0 / (1.0 + (2.71828 ** (-(val.to_f))))
	end

	def sigmoid(val : Matrix(Float64))
		out_matrix = Matrix.new(val.rows.size, val.columns.size) {0.0}
		(val.rows.size * val.columns.size).times do |x|
			out_matrix[x] = sigmoid(val[x])
		end
		out_matrix
	end

	def generate(rows, cols, epsilon)
		return Matrix.new(rows, cols) { (rand * 2 * epsilon) - epsilon }
	end

	def bias(matrix)
		return Matrix.rows([matrix.to_a.unshift(1.0)])
	end

	def total_error(target : Array(Float64), net_out : Array(Float64))
		diffs = target.clone
		diffs.size.times { |x| diffs[x] = 0.5 * ((diffs[x] - net_out[x]) ** 2) }
		return diffs.sum
	end

	class Trainer
	
		def initialize(data : Array(Int32|Float64|String), outputs : Int32)
			@data = [] of Float64
			@data = data[0...(data.size - outputs)].map { |x| x.to_f64 }
			@outputs = [] of Float64
			@outputs = data[-outputs..-1].map { |x| x.to_f64 }
		end
	
		def initialize(data : Array(Array(Int32|Float64|String)))
			@data = [] of Float64
			@data = data[0].map { |x| x.to_f64 }
			@outputs = [] of Float64
			@outputs = data[1].map { |x| x.to_f64 }
		end

		def data
			return @data
		end
	
		def outputs
			return @outputs
		end
	
	end #of class Trainer


class Backprop

	def initialize(layers : Array(Int32), alpha = 0.5)
		@learning_rate = alpha
		@weights = [] of Matrix(Float64)
		(layers.size - 1).times do |x|
			epsilon = (6.0 ** 0.5) / ((layers[x] + layers[x + 1]).to_f ** 0.5)
			@weights.push generate(layers[x] + 1, layers[x + 1], epsilon)
		end
	end
	
	def weights
		return @weights
	end
	
	def load_weights(file_name)
		new_wts = YAML.parse(File.read(file_name))
		@weights = new_wts.to_a.map do |mats|
			Matrix.rows(mats.to_a.map { |row| row.to_a.map { |val| val.to_s.chomp.to_f } } )
		end
	end

	def save_weights(file_name)
		File.write(file_name, @weights.map { |x| x.rows }.to_yaml)
	end
	
	def forward_calc(input_layer : Matrix(Float64))
		layer = input_layer
		layers = [] of Matrix(Float64)
		layers.push input_layer
		@weights.each do |x|
			layer = sigmoid(bias(layer) * x)
			layers.push(layer)
		end
		return layers
	end

	def delta(target : Matrix(Float64), outs : Matrix(Float64), position : Int32)
		targ = target[position]
		o = outs[position]
		return (-(targ - o)) * (o * (1 - o))
	end

	def delta(target : Matrix(Float64)|Array(Float64), outs : Matrix(Float64)|Array(Float64))
		accumulator = [] of Float64
		target.size.times do |x|
			targ = target[x]
			o = outs[x]
			accumulator.push ((-(targ - o)) * (o * (1 - o)))
		end
		return accumulator
	end

	def delta(target : Float64, outs : Float64)
		return (-(target - outs)) * (outs * (1 - outs))
	end

	def backprop(trainer : Trainer)
		layers = forward_calc(Matrix.rows([trainer.data]))
layers[0] = bias(layers[0])
layers[1] = bias(layers[1])
		new_weights = @weights.map { |x| x.clone }
		#First, set the weights from the hidden layer to the output layer
		@weights[-1].column_count.times do |col|
			@weights[-1].row_count.times do |row|
				new_weights[-1][((@weights[-1].column_count * row) + col)] -= @learning_rate * delta(trainer.outputs[col], layers[-1][col]) * layers[1][row]
			end
		end
		#Next, set weights from input to hidden layer
		@weights[0].column_count.times do |col|
			@weights[0].row_count.times do |row|
				accumulator = (Matrix.rows([delta(trainer.outputs, layers[-1])]) * Matrix.columns([@weights[-1].rows[col + 1]])).to_a[0]
				accumulator = accumulator * layers[0][row] * (layers[1][col + 1] * (1 - layers[1][col + 1]))
				new_weights[0][((@weights[0].column_count * row) + col)] -= @learning_rate * accumulator
			end
		end
		@weights = new_weights.map { |x| x.clone }
	end

end #of class Backprop

end #of module Backprop

include Backprop

#back = Backprop::Backprop.new([256, 256, 10], 2.0)
#ary = [] of Backprop::Trainer
#File.each_line("data.txt") do |x|
#	ary.push(Backprop::Trainer.new(x[0..-2].split(" ").map { |w| w.to_f }, 10))
#end
#ary.shuffle!
#
#2.times do
#	1400.times do |w|
#		puts "#{Time.now.to_s("%T")} - #{w} - #{ary[w].outputs.index { |z| z == 1.0 }}"
#		back.backprop(ary[w])
#	end
#end
#
#back.save_weights("weights.txt")
#
#200.times do |w|
#	16.times do |x|
#		16.times do |y|
#			print ary[w].data[(16 * x) + y] == 1.0 ? "X" : " "
#		end
#		print "\n"
#	end
#	result = back.forward_calc(Matrix.rows([ary[w + 1300].data]))[-1].to_a
#	puts result
#	puts "Result: #{result.index { |z| z == result.max }}"
#	puts "Target: #{ary[w].outputs.index { |n| n == 1.0 }}"
#	gets
# end




#back = Backprop::Backprop.new([784, 300, 10], 0.5)
#
#io_label = IO::Memory.new(File.read("./data/labels"))
#slice_label = Bytes.new(1)
#io_label.read(Bytes.new(8))
#
#io_image = IO::Memory.new(File.read("./data/images"))
#slice_image = Bytes.new(784)
#io_image.read(Bytes.new(16))
#
#10000.times do |w|
#	io_label.read(slice_label)
#	io_image.read(slice_image)
#	output_layer = [0.0]*10
#	output_layer[slice_label[0]] = 1.0
#	input_layer = slice_image.to_a.map { |x| x / 256.0 }
#	train = Trainer.new([input_layer, output_layer])
#	back.backprop(train)
#	puts "#{Time.now.to_s("%T")} - #{w} - #{slice_label[0]}"
#end
#
#back.save_weights("weights.txt")
#
#50.times do
#	io_image.read(slice_image)
#	slice = Bytes.new(1)
#	28.times do |x|
#		28.times do |y|
#			print (slice_image[y + (28 * x)] > 75 ? "X" : " ")
#		end
#		print "\n"
#	end
#	io_label.read(slice_label)
#	result = back.forward_calc(Matrix.rows([slice_image.to_a.map { |w| w / 256.0 }]))[-1].to_a
#	puts result
#	puts "Result: #{result.index { |z| z == result.max }}"
#	puts "Target: #{slice_label[0]}"
#	gets
#end
