from src.AttentionBasedModel import AttentionBasedBiLstmModel

def main(input_file_path):
    attnBasedModel = AttentionBasedBiLstmModel()
    output_file_path = attnBasedModel.main(input_file_path)
    return output_file_path
