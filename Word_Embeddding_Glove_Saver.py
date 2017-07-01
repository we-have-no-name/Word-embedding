import tensorflow as tf 
import numpy as np 
import pickle, json
import os, time
import re

class EmbeddingAdapter():
	'''
	Uses the twitter glove embedding available at https://nlp.stanford.edu/projects/glove/
	can convert them into:
		a dictionary that maps a word to its vector's index in an embedding array
		a numpy array for the words' embeddings
		a TensorFlow variable file for the embeddings
	'''
	vocab_size=int(1.2e6)
	embedding_dim=0
	def __init__(self, user_config_filename='config.json'):
		'''Set the path that has the glove folder and will have the exported folders'''
		try:
			json_config_file = open(user_config_filename)
		except FileNotFoundError:
			json_config_file = self.set_config_file(user_config_filename)
			
		user_config = json.load(json_config_file)
		self.data_path = user_config.get('data_path', os.getcwd())
		
	def set_config_file(self, user_config_filename):
		'''Set a user config file with the path that has the glove folder'''
		user_config = {}
		user_config['data_path'] = input("Enter the path to the folder that has the glove folder:\n")

		json_config_file = open(user_config_filename, 'w+')
		json.dump(user_config, json_config_file)
		json_config_file.seek(0)
		return json_config_file

	def read_embedding(self, embedding_dim=200):
		'''Read the embedding from the glove files'''
		self.embedding_dim=embedding_dim
		embedding_file = open(os.path.join(self.data_path, "glove.twitter.27B", "glove.twitter.27B." + str(embedding_dim) + "d.txt"),"r", encoding="utf-8")
		self.np_embedding = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float16)
		self.word_dict = dict()
		self.word_dict2 = dict()
		
		for i in range(self.vocab_size):
			line_r=embedding_file.readline()
			if line_r is None or len(line_r)==0 : break
			line = line_r[:-1].split(" ")
			# replace the angle brackets '<>' in special-use tags with the special characters 'ᐸᐳ'
			tags_pattern = '^<(url|user|smile|lolface|sadface|neutralface|heart|number|repeat|elong|hashtag|allcaps|)>$'
			word=re.sub(tags_pattern, r'ᐸ\1ᐳ', line[0])
			self.word_dict[line[0]]=i
			self.word_dict2[word]=i
			wordvec=np.array(line[1:],dtype=np.float16)
			self.np_embedding[i]=wordvec
		return True

	def save_embedding_dict(self, embedding_dim=200, save_vocab=False):
		'''Save the dictionary that maps words to their indices in the embedding numpy array'''
		if embedding_dim!=self.embedding_dim: self.read_embedding(embedding_dim)
		self.embedding_folder = os.path.join(self.data_path, "d" + str(embedding_dim) +"_word_embedding")
		if not os.path.exists(self.embedding_folder): os.makedirs(self.embedding_folder)
		
		word_dict_file=open(os.path.join(self.embedding_folder, "word_dict.pickle"), 'wb')
		pickle.dump(self.word_dict, word_dict_file)
		word_dict2_file=open(os.path.join(self.embedding_folder, "word_dict2.pickle"), 'wb')
		pickle.dump(self.word_dict2, word_dict2_file)
		word_dict_file.close(); word_dict2_file.close()
		
		# save all the words in the vocabulary to a file (optional)
		if not save_vocab: return True
		words_file = open(os.path.join(self.data_path, "vocabulary.txt"), 'w', encoding='utf-8')
		for key, _ in sorted(self.word_dict2.items(), key=lambda i: i[1]):
			words_file.write(key + '\n')
		words_file.close()
		return True
		
		
	def save_embedding_np(self, embedding_dim=200):
		'''Save the embedding to pickle files'''
		# save the dictionary that maps words to their indices in the np_embedding array
		self.save_embedding_dict(embedding_dim)
		
		# save the embedding numpy array
		word_embedding_file = open(os.path.join(self.embedding_folder, "word_embeddings_ndarray.pickle"), 'wb')
		pickle.dump(self.np_embedding, word_embedding_file)
		word_embedding_file.close()
		return True

	def save_embedding_tf(self, embedding_dim=200, save_dict=True):
		'''Save the embedding as a TensorFlow session variable'''
		if embedding_dim!=self.embedding_dim: self.read_embedding(embedding_dim)
		# save the dictionary that maps words to their indices in the np_embedding array
		if save_dict: self.save_embedding_dict(embedding_dim)
		
		# create the graph and store the embedding numpy array in the *embedding* session variable
		tf.reset_default_graph()
		embedding=tf.Variable(tf.constant(0, dtype=tf.float32, shape=(self.vocab_size, self.embedding_dim)), trainable=False, name='embedding')
		embedding_ph = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_dim])
		embedding_init= embedding.assign(embedding_ph)

		sess = tf.Session()
		_ = sess.run(embedding_init, feed_dict={embedding_ph: self.np_embedding})

		# save the embedding session variable to a file
		sess_save_folder = os.path.join(self.embedding_folder, "TF_Variables")
		if not os.path.exists(sess_save_folder): os.makedirs(sess_save_folder)
		saver=tf.train.Saver({'embedding': embedding})
		saver.save(sess, os.path.join(sess_save_folder, "Embedding"))
		sess.close()
		return True

	def run(self, embedding_dim=None):
		'''Run the class interactively'''
		ve=False
		try:
			if embedding_dim is None:
				embedding_dim = input("Choose an embedding dimensionality (25, 50, 100, 200, all) [200]: ")
				if embedding_dim != 'all': embedding_dim = int(embedding_dim)
		except ValueError:
			ve = True
		if ve or embedding_dim not in [25, 50, 100, 200, 'all']:
			print("Invalid choice, using the default value of 200")
			embedding_dim = 200
		print('Please wait')
		if embedding_dim == 'all': embedding_dims = [25, 50, 100, 200]
		else: embedding_dims = [embedding_dim]
		
		for embedding_dim in embedding_dims:
			self.save_embedding_np(embedding_dim)
			self.save_embedding_tf(embedding_dim)
		print('Done')
		return True
def main():
	ea = EmbeddingAdapter()
	ea.run()

if __name__ == "__main__": main()
