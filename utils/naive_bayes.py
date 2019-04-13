class NaiveBayes:

    def __init__(self):
        """ P(c) = Nc / N
        # Nc = class_freq
        # N = total_freq
        """
        self.class_freq = {}
        self.total_freq = 0

        """P(w|c) = (count(w,c) + 1) / (count(c) = |V|)
        # count(w,c) = word_freq_in_class
        # count(c) = word_in_class
        # |V| = unique_word
        """

        self.word_freq_in_class = {}
        self.word_in_class = {}
        self.unique_word = []

    def reset(self):
        self.class_freq = {}
        self.total_freq = 0
        self.word_freq_in_class = {}
        self.word_in_class = {}
        self.unique_word = []

    def update_model(self, words, class_name):
        # Add class occurence to Nc
        if class_name not in self.class_freq:
            self.class_freq[class_name] = 0
        self.class_freq[class_name] += 1

        # Add occurence to N
        self.total_freq += 1

        # Updating the count(w,c) and |V|
        for word in words:
            # Adding occurence to count(w,c)
            if word not in self.word_freq_in_class:
                self.word_freq_in_class[word] = {}
            if class_name not in self.word_freq_in_class[word]:
                self.word_freq_in_class[word][class_name] = 0
            self.word_freq_in_class[word][class_name] += 1

            # Adding unique words if its unique
            if word not in self.unique_word:
                self.unique_word.append(word)

        # Add total word to count(c)
        if class_name not in self.word_in_class:
            self.word_in_class[class_name] = 0
        self.word_in_class[class_name] += len(words)

    def refit(self, X, y):
        for idx, x in enumerate(X):
            self.update_model(x, y[idx])

    def fit(self, X, y):
        self.reset()
        self.refit(X, y)

    def get_class_prob(self, class_name):
        return self.class_freq[class_name] / self.total_freq

    def get_cond_prob(self, word, class_name):
        quantifier = 1
        if word in self.word_freq_in_class:
            if class_name in self.word_freq_in_class[word] :
                quantifier = self.word_freq_in_class[word][class_name] + 1
        divisor = self.word_in_class[class_name] + len(self.unique_word)

        return quantifier/divisor

    def calc_single_prob(self, x, class_name):
        prob = self.get_class_prob(class_name)
        for word in x:
            prob *= self.get_cond_prob(word, class_name)
        return prob

    def calc_prob(self, x):
        result = {}
        for y in self.class_freq.keys():
            result[y] = self.calc_single_prob(x, y)
        return sorted(result.items(), key=lambda keyval: keyval[1], reverse=True)

    def predict_single(self, x):
        return self.calc_prob(x)[0][0]

    def predict(self, X):
        return [self.predict_single(x) for x in X]


if __name__ == "__main__":
    nb = NaiveBayes()
    X = [['chinese', 'beijing', 'chinese'],
         ['chinese', 'chinese', 'shanghai'],
         ['chinese', 'macao'],
         ['tokyo', 'japan', 'chinese']]
    y = [1, 1, 1, 0]

    nb.fit(X, y)
    print(nb.predict_single(["chinese", "chinese", "chinese", "tokyo", "japan"])) # should be 1
    #print(nb.calc_single_prob(['asdfasdfasdf','asdfasdfasdf','chinese', 'chinese', 'chinese', 'tokyo', 'japan'], 1))
    #print(nb.calc_single_prob(["asdf"], 1))
