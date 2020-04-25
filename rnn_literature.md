### Notes on RNNs:-  

RNNs emulate latent Autoregressive models. The prediction at 't' th time step is conditioned on the input observed at time 't' and history(*a sophisticated function*) until time 't'. RNNs have shortcomings like vanishing gradient that do not qualify them as good candidates for interesting applications involving long term dependencies of the input.  

RNNs shortcomings are addressed by LSTMs and GRUs by filtering the history in an RNN cell by retaining only the part of history that is relevant to the scenario and forgetting the rest.  

### GRU(Gated Recurrent Unit)  
* 
