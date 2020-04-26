### Notes on RNNs:-  

RNN cells emulate latent Autoregressive models. The prediction at 't' time step is conditioned on the input observed at time 't' and history(*a sophisticated function*) until time 't'. RNNs have shortcomings like vanishing gradient that do not qualify them as good candidates for interesting applications involving long term dependencies of the input.  

RNN's shortcomings are addressed by LSTMs and GRUs by filtering the history of an RNN cell by retaining only the part of history that is relevant to the scenario and forgetting the rest.  

### GRU(Gated Recurrent Unit)  
GRUs have mechanisms, deciding when a hidden state(history) should be updated and when it should be reset. These mechanisms are learned during training.  
With respect to programing, a reset variable decides how much of the previous state will be retained, and an update variable decides how much of the state that has been modified by the reset variable will be used. These variables are vectors with values between 0 & 1.  

#### Reset gate  
To reduce the influence of the previous states we  multiply  hidden state(vector) with reset state(vector), elementwise.  The elements of hidden state corresponding to values of the reset state that are close to 1 are more likely to be retained than those which are close to 0 leading to a new modified hidden state.  

However, there are 2 special cases depending on elements of reset state(vector),    
  1. If all elements = 1, then GRU behaves like a Vanilla RNN-- everything in hidden state is remembered.  
  2. If all elements = 0, then GRU behaves like a feed-forward neural network-- nothing in hidden state is remembered.  

#### Update gate  
This gate assigns the weight between the new modified hidden state from the reset gate and the old state. If the values of update state(vector) are close to 1, then old state is retained, skipping the input at timestep 't'; else if the update state values are close to 0, then modified state from reset state is used.  

To summarize,  
Reset gate captures short-term dependencies.  
Update gate captures long-term dependencies.
