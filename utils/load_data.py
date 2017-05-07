import h5py

def load_img(path):
    with h5py.File(path, 'r') as f:
        return f['train_set'][()], f['validation_set'][()], f['test_set'][()]

def load_text(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        ids, sentences = [], []
        idx = 0
        for i, line in enumerate(f):
            s = line.strip()
            try:
                idx = int(s)
            except ValueError:
                ids.append(idx)
                sentences.append(s)
        return ids, sentences
