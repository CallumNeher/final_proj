Final Project Writeup
Callum Neher

For my final project in DS210, I decided to create a 2 layer character level language model. At a high level, what this model does is take set of starting characters (the ‘context’), and output a distribution of possible next characters, including an end character, with different probabilities for each character that are based upon it’s training. The model then selects the next character by randomly sampling from the probability distribution, and repeats the process for that character after updating the context to include the new character. In doing this it generates new characters iteratively, until it selects to end the word. The user has the option to input starting characters (a name that starts with _), or to generate names with any starting character. This creates bespoke, new, and often quite interesting names. There are 3 csv files included in the repository. ‘Alphabet Set’ is simply a csv where each line is a letter of the alphabet, and the first line is the placeholder character ‘.’. This csv is used to quickly create the tokenizers for the network. The other two are different data sets of names, ‘Big Names’ is a csv with ~32k rows, each corresponding to a different name (https://raw.githubusercontent.com/karpathy/makemore/master/names.txt). ‘1900 Names’ is a csv of the top 1000 baby boy names and top 1000 baby girl names from the 1900 census (https://www.ssa.gov/OACT/babynames/index.html). 

To run this code, open the repository in an environment with rust installed, adjust any hyperparameters you want, change the input dataset if desired (make sure any data used is in csv format, with one word per row), and then input ‘cargo run –release’ in the terminal. You can also keep the hyperparameters as they are and use the provided dataset if you wish. The network will now begin training. It will print an initial loss and a loss report every 10,000 training epochs. After it is done training, It will output a final cross entropy loss value based on the loss calculated by the test data (which the network has never seen). Then it will ask you to provide starting letters for name generation, or type ‘end’ to end the program. After the user chooses to end the program, it will create a graph representation of the final lookup table, in which the vowels will be green and he consonants will be red. This can provide some interesting insight on the spatial relationships between these letters. Ideally, I would be able to display the actual letter corresponding to each point, as was done for makemore, but I have been unable to find a way to easily do this in rust.
	
The best final loss value I have been able to come up with using the “Big Names” dataset is 2.24, with Hidden Size = 200, block size (context) = 3, Dimensions = 10, Batch Size = 32, and 200,000 epochs (batches) of training. If you adjust the number of Dimensions or the Block Size, make sure to change it for the main file and the Network file. Lower numbers for loss is theoretically a better model, but too low can mean that the model is overfitting the dataset. Some names I have gotten out of the model without specifying starting letters include: “Bellangeton”, “Parie”, “Haerian”, “Jamis”, “Zarari”, and “Briani”.

This code implementation is inspired by Andrej Karpathy’s lessons on neural networks, in which he created a similar neural network called makemore, meant to iteratively generate names. His implementation was in python, and has some structural differences to my implementation, but his youtube videos explaining neural networks and walking through his own code implementation helped me understand how to execute on my vision for this project. His github repo can be found here. I have not seen a rust interpretation of Karpathy’s makemore anywhere on the internet, so I’m fairly certain that the network I have created is the first of it’s kind. The differences in data structures and modules between rust and python led to my network implementation being quite different than makemore. 

My code is split into my main.rs file and network.rs file, which hold the implementation for my custom struct Network. Network has various helper functions implemented to allow it to work, which I will cover shortly. The code in my main file helps load the data in from the csv files, construct the dataset, and create training batches of data. There is a helper function for taking inputs from the command line. I also have a function to plot the final correlations between letters in the words. My main function runs the training loop, and operates the command line interface, which is complete with instructions for a new user to operate the program without any understanding of rust (other than cargo run –release).

Next, I will explain how the network functions, including the initialization, forwards pass, backwards pass, and updates.

The Network struct itself holds 4 arrays, a 2D embedding lookup table, 2D input weights, 2D output weights, and 1D output bias. The network has tweakable number of hidden nodes, which can be tweaked by changing the Hidden Size constant in network.rs, I have found that 200 works pretty well. To initialize the network I got initial values for each of my 2D arrays by generating values in a range between a lower and upper limit. I tweaked this limit over different training iterations, but I found that 5/(3 * rows^0.5) was a limit that produced reliable initialization values for my tanh non linearity, this limit is also used in makemore, but in a different implementation. My code loads the data from the csvs into a vector of strings, one for each name. Then, my code tokenizes each word, by using a tokenizer HashMap that converts a letter to it’s corresponding number (0=’.’, 1=’a’, 2=’b’, etc.). The dataset consists of a training set and test set, each of which has x and y values. The x values are VecDeque’s of context, and the y values are the actual next letter in the word. The placeholder value stands in for characters at the start of the word. For example, “bob” would be encoded: (x,y): ( [0,0,0],[2]), ([0,0,2],[15]), ([0,2,15],[2]), ([2,15,2],[0]) ‘b’=2, ‘o’=15, ‘.’=0. The data is split into test and training sets by just taking every 10th pair of x and y and adding it to the test set. This could be done randomly instead, but I was unable to find a straightforward way of doing this in rust. The words are not in any particular order already so this method is fine. 

The forward pass of my network runs based off of randomly sampled batches of the data, not the whole dataset at once. This can help to avoid overfitting of my model to my dataset. First, the batch is created by taking a random sample of a hyperparameter batch size from the x and y training set. Then the input data is embedded based on the lookup table in the neural network. In this step each embedded letter in each context in the training data is turned from an integer (‘a’ = 1, etc.) to a multidimensional representation (something like [0.456723, -0.73480, … , 0.78303]) the number of dimensions is also one of the hyperparameters. Then these embeddings for each letter are concatenated into one vector for each context. If we are using 5 dimensional embeddings and 3 letters of context, then the vector will have 15 values. This is done for each set of context in the input data, resulting in an array that has dimensions (Batch size, Dimensions * Context length). Then the model takes the dot product of this array and the input weights (W1) array. This is the first linear layer. Next we pass the values through an activation function. I used Tanh for this, because it allows for negative values and is easy to backpropagate over. The activation function squeezes all the data into the range between -1 and 1, which keeps the values flowing through the network in a desirable range. Next, I take the dot product of the tanh output and the output weights (W2), and add the output bias vector (b2). This is the second linear layer. I also subtract the largest value in each row from the entire row, to keep the next step stable. Next, I set each value such that x -> e^x, to get the “counts” of my softmax. The final step is normalizing each row (dividing each value by the row sum) to get a probability distribution for the row. This probability distribution is essentially the probability of each of the 27 characters coming next, for each context in the input data batch. In an equation form, the forwards pass looks like this for any given datum flowing through the network: prob dist. = (e^(tanh(x @ W1) @ W2 + b2))/sum(given row). 

The backwards pass is where the network learns. I use gradient descent for this. Gradient descent is done by calculating the loss for each output, and then taking the derivative at each step of the froward pass (starting from the final step), and doing the chain rule to work backwards from the probability distribution to the embedding lookup table. Then, we adjust the weights, bias, and embeddings of the network by a small amount, in the direction of the gradient with respect to each of them. I used cross entropy loss, because it is very easy to backpropagate over, and is most applicable for multiple classification. To get the Gradient Descent started, the network compares each value in each probability distribution, to the expected output of that distribution given the corresponding y-value. If the distribution is perfectly trained, it would output 1 for the ground truth y-value, and 0 for the rest. Thus, this step is simply subtracting 1 from the probability distribution at the index of the correct y value. This could also be done using 1 hot encoding, but it is very quick and easy to just subtract 1 at each index instead. This backpropagates over our loss and softmax, we will call it gradient of logits. I won’t get into all of the calculus in this explanation, but it essentially just uses chain rule to unpack and iterate through the forward pass at each step. The gradient of the output weights is calculated by taking the gradient of logits @ h where h is th output of the tanh step in the forward pass. The gradient of the output bias is the sum along each row of the gradient of the logits. The rest of the gradients are calculated in similar fashion using chain rule. The backwards pass function outputs these gradients, to be used in the update functions.

The weights, bias and lookup arrays are all updated by taking the gradient with respect to the given matrix, grad, and the learning rate, lr, and executing this equation for the matrix. Matrix = matrix - grad * lr. The lookup table is a slightly special case because of the way that the inputs were embedded before being passed into the network. To update this the network just indexes into the lookup table for the given encoding in the batch of x values, and updates each dimension of the embedding by the corresponding dimensions of the gradient. For the other matrices it is sufficient to just subtract the gradient * lr element wise. The learning rate is set to 0.01, and it is reduced to 0.001 after half of the training epochs have elapsed. I did this because it allows the network to effectively train and make big gains at the start, and make finer tuned gains in accuracy after the initial gains have been made. 
