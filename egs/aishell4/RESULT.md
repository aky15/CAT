Unified multi-channel streaming/non-streaming model results. The streaming model is based on CUSIDE. 
The results are evaluated on the non-overlapped part of Aishell4 test set.

Streaming model configuration:
- left context: 800ms
- chunk size: 400ms
- simulated right context: 400ms

The non-streaming model uses full-context encoder in the front-end and AM.
 
|         model       |  Aishell-4 CER |
|---------------------|----------------|   
|      streaming      |      36.2      |
|     non-streaming   |      31.2      |


