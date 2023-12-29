from word_embeddings.word2vec_loader import Word2VecLoader
from word_embeddings.phrase_similarity_calculator import PhraseSimilarityCalculator
from utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)

try:
    # Load Phrases from CSV
    phrases_df = pd.read_csv('project\phrases.csv', encoding='latin1')
    
    # Load Word2Vec Vectors
    word2vec_loader = Word2VecLoader('project\google.bin',limit=1000000)

    # Create Phrase Similarity Calculator
    phrase_similarity_calculator = PhraseSimilarityCalculator(phrases_df, word2vec_loader)

    # Calculate Distances
    
    phrase_similarity_calculator.calculate_distances()
    k=input("Enter the input here:")
    b,c=phrase_similarity_calculator.find_closest_match(k)
    print("The Closest Phrase to the input is : ",end="")
    print(b)
    print("The Distance is : ",end="")
    print(c)
    
    # Your additional logic here

except Exception as e:
    logger.exception(f"An error occurred: {str(e)}")
