Apart from the CBOW, GRU based RNN and attention mechanism on bi-GRU, I also implemented several models as you can see in my report. To use this model, you can type
1. `--model lstm` to use bi-LSTM model
1. `--model gru_lstm` to use GRU fowrad LSTM backward model.
1. `--model lstm_gru` to use LSTM forward GRU backward model.
1. `lstm_att` to use attention mechanism on bi-LSTM.
1. `att2rnn` to use History attention model.

The directory structures for me to store the output file is:
`out/{MODEL_NAME}/kp_{KEEP_PROB}_{TOTAL_STEP}` if the total step is not 20k. As for the case of total step not equal 20k, the `{TOTAL_STEP}` is empty and the last low dash is removed.
