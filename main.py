from modules.semantic import SemanticSearch
import pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PegasusTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./model_gen/")
tokenizer = PegasusTokenizer.from_pretrained('./tokenizer_gen/')
USE = tensorflow_hub.load("./USE/")


def generation(key_words):
        learn_file = open("learner/learner.pkl","rb")
        learner = pickle.load(learn_file)
        learn_file.close()
        chunk_file = open("data/chunks","rb")
        chunks = pickle.load(chunk_file)
        chunk_file.close()
        for i in key_words:
            key_chunks = learner(USE([i]),chunks)
            result = []
            for j in range(5): 
                    result.append(chunks[random.randint(0, 6)])
            q_list= []

            for i in outputs:
                    input_text = i
                    input_ = tokenizer.batch_encode_plus([input_text], max_length=1024, pad_to_max_length=True,
                                    truncation=True, padding='longest', return_tensors='pt')
                    input_ids = input_['input_ids']
                    input_mask = input_['attention_mask']
                    questions = model.generate(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            num_beams=32,
                                            no_repeat_ngram_size=2,
                                            early_stopping=True,
                                            num_return_sequences=10)

                    questions = tokenizer.batch_decode(questions, skip_special_tokens=True)
                    for j in questions:

                        q_list.append(j)
                    return set(q_list)