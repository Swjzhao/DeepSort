# DeepSort

Please install Keras and Tensorflow for the program to run smoothly

This algorithm trains the program to sort numbers from in a set array size. The amount of numbers in an array or the max value
of the number can be set in the variables: len and num. num is max value

Sources:
Code to generate input and test data
https://github.com/drforester/Sequence_to_Sequence_Sorting/blob/master/sort_seq.py

I implemented my own seq2seq model and modified so it was able to sort arrays len (2 to max len) instead of just one set length. However this does not work too well with bigger arrays.

Please run the DeepSort.py file to train the model. You can modilify the length and max value to make it train faster.
After code will save the model once completed training. you can access it later on in test_sort.py. 
I currently have random numbers to test, but you can set your own arrays. 

