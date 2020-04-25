### Notes on RNNs:-  

RNN cells emulate latent Autoregressive models. The prediction at 't' th time step is conditioned on the input observed at time 't' and history(*a sophisticated function*) until time 't'. RNNs have shortcomings like vanishing gradient that do not qualify them as good candidates for interesting applications involving long term dependencies of the input.  

RNN's shortcomings are addressed by LSTMs and GRUs by filtering the history in an RNN cell by retaining only the part of history that is relevant to the scenario and forgetting the rest.  

### GRU(Gated Recurrent Unit)  
GRUs have mechanisms, deciding when a hidden state(history) should be updated and when it should be reset. These mechanisms are learned during training.  
During programing, a reset variable would decide how much of the previous state will be retained, and an update variable would decide how much of the new state will inherit from the old state. These variables are vectors with values between 0 & 1.  

