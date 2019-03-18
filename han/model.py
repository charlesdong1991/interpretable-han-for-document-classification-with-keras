from keras.models import Model
from keras.layers import Input, Embedding, Dense, GRU, Bidirectional, TimeDistributed, Lambda
from keras.callbacks import ModelCheckpoint

from .attention import Attention

DENSE_SIZE = 100


class HAN(Model):
    def __init__(self, embedding_matrix, max_sent_length=100,
                 max_sent_num=10, word_embed_dim=100, sent_embed_dim=100):
        """Implementation of Hierarchical Attention Networks for document classification.

        Args:
            embedding_matrix: embedding matrix used to represent words.
            max_sent_length: int, maximum number of words per sentence, default is 100.
            max_sent_num: int, maximum number of sentences accepted, default is 10.
            word_embed_dim: int, dimension of word encoder, default is 100.
            sent_embed_dim: int, dimension of sentence encoder, default is 100.
        """
        self.embedding_matrix = embedding_matrix
        self.max_sent_length = max_sent_length
        self.max_sent_num = max_sent_num
        self.word_embed_dim = word_embed_dim
        self.sent_embed_dim = sent_embed_dim

        super(HAN, self).__init__(name='han')
        self.build_model()

    def build_model(self):
        """Build the embed and encode models for word and sentence.

        For the word model, the sequence of Layers is: Embedding ->
        Bidirectional(GRU) -> TimeDistributed(Dense) -> Attention

        For the sentence model, it takes the word level model as input
        for TimeDistributed Layer to make sentence encoder. And the
        sequence is: TimeDistributed(WordModel) -> Bidirectional(GRU)
        -> TimeDistributed(Dense) -> Attention -> Dense

        There is no output, but will save the word and sentence models.
        """
        sent_input = Input(shape=(self.max_sent_length,), dtype='float32')
        embedded_sent = Embedding(
            self.embeddings_matrix.shape[0], self.embeddings_matrix.shape[1],
            weights=[self.embeddings_matrix], input_length=self.max_sent_length,
            trainable=False
        )(sent_input)

        # For Bidirectional, devide by 2
        encoded_sent = Bidirectional(
            GRU(int(self.word_embed_dim / 2), return_sequences=True)
        )(embedded_sent)
        dense_sent = TimeDistributed(Dense(DENSE_SIZE))(encoded_sent)

        word_att = Attention(name='word_attention')(dense_sent)
        # create Word-level Model
        self.model_word = Model(sent_input, word_att)

        text_input = Input(shape=(self.max_sent_num, self.max_sent_length))
        sent_encoder = TimeDistributed(self.model_word)(text_input)

        # For Bidirectional, devide by 2
        encoded_text = Bidirectional(
            GRU(int(self.sent_embed_dim / 2), return_sequences=True)
        )(sent_encoder)
        dense_text = TimeDistributed(Dense(DENSE_SIZE))(encoded_text)
        doc_att = Attention(name='sent_attention')(dense_text)
        # dense the output to 2 because the result is a binary classification.
        output_tensor = Dense(2, activation='softmax', name='classification')(doc_att)
        # Create Sentence-level Model
        self.model = Model(text_input, output_tensor)

    def print_summary(self):
        print("Word Level")
        self.model_word.summary()

        print("Sentence Level")
        self.model.summary()

    def train_model(self, checkpoint_path, X_train, y_train, X_test, y_test,
                    optimizer='adagrad', loss='categorical_crossentropy',
                    metric='acc', monitor='val_loss', batch_size=20, epochs=10):
        """Train the HAN model.

        Args:
            checkpoint_path: str, the path to save checkpoint file.
            X_train: training dataset.
            y_train: target of training dataset.
            X_test: testing dataset.
            y_test: target of testing dataset.
            optimizer: optimizer for compiling, default is adagrad.
            loss: loss function, default is categorical_crossentropy.
            metric: measurement metric, default is acc (accuracy).
            monitor: monitor of metric to pick up best weights, default is val_loss.
            batch_size: batch size, default is 20.
            epochs: number of epoch, default is 10.
        """
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric]
        )
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            verbose=1, save_best_only=True
        )

        self.model.fit(
            X_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint]
        )

    def show_word_attention(self, x):
        """Show the prediction of the word level attention.

        Args:
            x: the input array with size of (max_sent_length,).

        Returns:
            Attention weights.
        """
        att_layer = self.model_word.get_layer('word_attention')
        prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.model_word.input, dummy_layer).predict(x)

    def show_sent_attention(self, x):
        """Show the prediction of the sentence level attention.

        Args:
            x: the input array with the size of (max_sent_num, max_sent_length).

        Returns:
            Attention weights.
        """
        att_layer = self.model.get_layer('sent_attention')
        prev_tensor = att_layer.input

        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.model.input, dummy_layer).predict(x)
