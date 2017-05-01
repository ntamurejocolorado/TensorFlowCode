
#TensorFlow
#Example 1: Computational Graph
import tensorflow as tfw

#A computational graph is a series of TensorFlow operations arranged
# into a graph of nodes. Like all TensorFlow constants, it takes
# no inputs, and it outputs a value it stores internally.

node1 = tfw.constant(3.0,tfw.float32)
node2 = tfw.constant(4.0)

print(node1, node2)

#To actually evaluate the nodes, we must run the computational graph
# within a session. A session encapsulates the control and state of
# the TensorFlow runtime.

sess = tfw.Session()
print(sess.run([node1, node2]))

#Example: Add two constant nodes and produce a new graph:

node3 = tfw.add(node1,node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

#TensorFlow provide a utility called TensorBoard that can display a picture
# of the computational graph.


#A graph can be parameterized to accept external inputs
#known as placeholders. A placesholders is a promise to provide a value later.

a = tfw.placeholder(tfw.float32)
b = tfw.placeholder(tfw.float32)
adder_node = a + b #Similar a function or a lambda in which we define two inputs

sess2 = tfw.Session()
print(sess2.run(adder_node,{a:3,b:4.5}))
print(sess2.run(adder_node,{a:[1,3], b:[2,4]}))

add_and_triple = adder_node * 3.
print(sess2.run(add_and_triple,{a:3,b:4.5}))

W = tfw.Variable([.3],tfw.float32)
b = tfw.Variable([-.3],tfw.float32)
x = tfw.placeholder(tfw.float32)
linear_model = W * x + b
#constants are initialized when you call tf.constant, and their value can never
# change. By contrast, variables are not initialized when you call tf.Variable.
#To initialize all the variables in a TensorFlow program, you must explicitly
#call a special operations:
init = tfw.global_variables_initializer()
sess3 = tfw.Session()
sess3.run(init)

print(sess3.run(linear_model,{x:[1,2,3,4]}))

#To EVALUATE the model on training data, we need a Y placeholder to provide
# the desired values, and we need to write a loss function.

#LOSS FUNCTION: measures how far apart the current model is from the provided data
# We'll use a standard loss model for linear regression, which sums the squares
# of the deltas between the current model and the provided data.
#linear_model-y creates a vector where each element is the corresponding
#example's error delta. We call tfw.square to square that error. Then, we sum
# all the squared errors to create a single scalar that abstracts the error of
#all examples using tfw.reduce_sum

y = tfw.placeholder(tfw.float32)
squared_deltas = tfw.square(linear_model - y)
loss = tfw.reduce_sum(squared_deltas)
print(sess3.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

#Result is 23.66. We could improve this manually by reassinging the values of
#W and b to the perfect values of -1 and 1. The final result is 0.0
#Reducimos el error a 0.

#Hemos adivinado el valor perfecto para W y b,pero el punto fuerte de machine
#learning es encontrar el modelo correcto automáticamente. Es decir, encontrar
# el valor de W y b de forma automatica y solo.

#We will show how to accomplish this int the next section.
