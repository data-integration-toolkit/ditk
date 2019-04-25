import abc


class TextSimilarity(abc.ABC):

    @classmethod
    def read_dataset(self, fileNames, *args, **kwargs):
        """
        Reads a dataset that is a CSV/Excel File.
        Args:
            fileName : With it's absolute path
        Returns:
            training_data_list : List of Lists that containes 2 sentences and it's similarity score
            Note :
                Format of the output : [[S1,S2,Sim_score],[T1,T2,Sim_score]....]
        Raises:
            None
        """
        # parse files to obtain the output
        return training_data_list

    @abc.abstractmethod
    def train(self, *args, **kwargs):  # <--- implemented PER class

        # some individuals don't need training so when the method is extended, it can be passed

        pass

    @abc.abstractmethod
    def predict(self, data_X, data_Y, *args, **kwargs):
        """
        Predicts the similarity score on the given input data(2 sentences). Assumes model has been trained with train()
        Args:
            data_X: Sentence 1(Non Tokenized).
            data_Y: Sentence 2(Non Tokenized)
        Returns:
            prediction_score: Similarity Score ( Float )

        Raises:
            None
        """
        return prediction_score

    @classmethod
    def evaluate(self, actual_values, predicted_values, *args, **kwargs):
        """
        Returns the correlation score(0-1) between the actual and predicted similarity scores
        Args:
            actual_values : List of actual similarity scores
            predicted_values : List of predicted similarity scores
        Returns:
            correlation_coefficient : Value between 0-1 to show the correlation between the values(actual and predicted)
        Raises:
            None
        """
        return evaluation_score

