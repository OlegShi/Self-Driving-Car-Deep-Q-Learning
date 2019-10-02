# Self-Driving-Car-Deep-Q-Learning

 Self-Driving Car implemented with PyTorch, open source deep learning framework.
 we are combining Q-Learning with an Artificial Neural Network. The states of the environment are encoded by a vector which is passed as
 input into the Neural Network. Then the Neural Network will try to predict which action should be played, by returning as outputs a 
 Q-value for each of the possible actions. Eventually, the action to play is chosen using a Softmax function.
 the goal of the car is to reach the upper left corner and the lower right corner of the map interchangeably and avoid the sand.
 map.py file is the environment containing the map, the car and all the features that go with it.
 ai.py file is the AI, the Deep Q-Learning model.
 
 Demo:
 https://youtu.be/cyOXuiGAY9k
 
