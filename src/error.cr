require "./backprop/*"
require "yaml"
require "../lib/matrix/src/matrix.cr"
require "./backprop.cr"

include Backprop

back = Backprop::Backprop.new([784, 300, 10], 0.5)
back.load_weights("./weights.txt")

io_label = IO::Memory.new(File.read("./data/test_labels"))
slice_label = Bytes.new(1)
io_label.read(Bytes.new(8))

io_image = IO::Memory.new(File.read("./data/test_images"))
slice_image = Bytes.new(784)
io_image.read(Bytes.new(16))

correct = 0

10000.times do |w|
	io_image.read(slice_image)
slice = Bytes.new(1)
28.times do |x|
28.times do |y|
print (slice_image[y + (28 * x)] < 75 ? "X" : " ")
end
print "\n"
end
	io_label.read(slice_label)
	result = back.forward_calc(Matrix.rows([slice_image.to_a.map { |w| w / 256.0 }]))[-1].to_a
puts "Result: #{result.index { |z| z == result.max }}"
puts "Target: #{slice_label[0]}"
gets
	if result.index { |z| z == result.max } == slice_label[0]
		correct += 1
	end
end

puts "#{correct} correct out of 10,000"
