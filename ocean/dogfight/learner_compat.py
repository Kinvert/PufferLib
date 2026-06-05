"""Dogfight learner compatibility facts for the 5.0 port.

This module is intentionally data-only. It documents the known df36 learner
contract and the current 5.0 diagnostic/native learner split without changing
training behavior.
"""

DF36_POLICY_NAME = "DogfightPolicy"
DF36_RNN_NAME = "DogfightRecurrent"
DF36_RECURRENT_TYPE = "LSTMWrapper"

DOGFIGHT5_NATIVE_NETWORK = "MinGRU"
DOGFIGHT5_NATIVE_LEARNER_MATCHES_DF36 = False

DOGFIGHT5_LSTM_DIAGNOSTIC_BACKEND_FLAG = "--slowly"
DOGFIGHT5_LSTM_DIAGNOSTIC_NETWORK = "LSTM"
